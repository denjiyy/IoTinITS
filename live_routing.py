from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from app_config import AppConfig
from app_logging import get_logger
from route_engine import VEHICLE_PROFILES, VehicleProfile, clamp


LOGGER = get_logger(__name__)


class LiveRoutingError(RuntimeError):
    """Raised when the live routing backend cannot return a usable route."""


@dataclass(frozen=True, slots=True)
class NavigationStep:
    instruction: str
    maneuver: str
    street: str
    distance_km: float
    time_min: float


@dataclass(slots=True)
class LiveRouteResult:
    name: str
    provider: str
    route_type: str
    coordinates: list[list[float]]
    ordered_stops: list[str]
    guidance_steps: list[NavigationStep]
    total_distance_km: float
    total_time_min: float
    total_emissions_g: float
    total_signal_delay_min: float
    average_green_score: float | None
    total_cost: float
    strict_green_applied: bool
    traffic_delay_min: float
    no_traffic_time_min: float | None
    historic_traffic_time_min: float | None
    live_incident_time_min: float | None
    departure_time: str | None
    arrival_time: str | None
    legs_count: int


@dataclass(frozen=True, slots=True)
class GeocodeCandidate:
    label: str
    lat: float
    lon: float
    score: float


def live_routing_available(config: AppConfig) -> bool:
    return bool(config.tomtom_api_key.strip())


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "message" in value and isinstance(value["message"], str):
            return value["message"]
    return str(value)


def _minutes_or_none(seconds: Any) -> float | None:
    if seconds is None:
        return None
    return float(seconds) / 60.0


def _join_points(route: dict[str, Any]) -> list[list[float]]:
    joined: list[list[float]] = []
    route_points = route.get("points")
    if isinstance(route_points, list):
        for point in route_points:
            joined.append([float(point["longitude"]), float(point["latitude"])])
        return joined

    for leg in route.get("legs", []):
        for index, point in enumerate(leg.get("points", [])):
            pair = [float(point["longitude"]), float(point["latitude"])]
            if joined and index == 0 and joined[-1] == pair:
                continue
            joined.append(pair)
    return joined


def _ordered_stop_labels(stop_labels: list[str], optimized_waypoints: list[dict[str, Any]]) -> list[str]:
    if not stop_labels:
        return []
    if not optimized_waypoints:
        return list(stop_labels)

    indexed_pairs: list[tuple[int, int]] = []
    for waypoint in optimized_waypoints:
        provided_index = waypoint.get("providedIndex")
        optimized_index = waypoint.get("optimizedIndex")
        if not isinstance(provided_index, int) or not isinstance(optimized_index, int):
            continue
        if 0 <= provided_index < len(stop_labels):
            indexed_pairs.append((optimized_index, provided_index))

    if not indexed_pairs:
        return list(stop_labels)

    indexed_pairs.sort()
    ordered = [stop_labels[provided_index] for _, provided_index in indexed_pairs]
    remaining = [label for label in stop_labels if label not in ordered]
    ordered.extend(remaining)
    return ordered


def _estimate_live_emissions_g(
    *,
    vehicle: VehicleProfile,
    distance_km: float,
    total_time_min: float,
    no_traffic_time_min: float | None,
    route_type: str,
) -> float:
    if distance_km <= 0:
        return 0.0

    baseline_time = max(no_traffic_time_min or total_time_min, 1.0)
    delay_ratio = max(total_time_min - baseline_time, 0.0) / baseline_time
    congestion_penalty = 1.0 + min(delay_ratio, 1.6) * (0.45 + vehicle.stop_go_penalty)
    eco_bonus = 0.92 if route_type == "eco" else 1.0
    return vehicle.base_g_per_km * distance_km * congestion_penalty * eco_bonus


def _parse_guidance_steps(route: dict[str, Any]) -> list[NavigationStep]:
    instructions = route.get("guidance", {}).get("instructions", [])
    if not isinstance(instructions, list):
        return []

    steps: list[NavigationStep] = []
    previous_offset_m = 0.0
    previous_time_s = 0.0
    for instruction in instructions:
        route_offset_m = float(instruction.get("routeOffsetInMeters", 0.0))
        travel_time_s = float(instruction.get("travelTimeInSeconds", 0.0))
        distance_delta_km = max(route_offset_m - previous_offset_m, 0.0) / 1000.0
        time_delta_min = max(travel_time_s - previous_time_s, 0.0) / 60.0
        previous_offset_m = route_offset_m
        previous_time_s = travel_time_s

        message = (
            _text_value(instruction.get("message"))
            or _text_value(instruction.get("combinedMessage"))
            or _text_value(instruction.get("street"))
            or "Continue"
        )
        street = _text_value(instruction.get("street"))
        maneuver = _text_value(instruction.get("maneuver")) or _text_value(instruction.get("instructionType")) or "FOLLOW"

        steps.append(
            NavigationStep(
                instruction=message,
                maneuver=maneuver,
                street=street,
                distance_km=round(distance_delta_km, 2),
                time_min=round(time_delta_min, 1),
            )
        )

    return steps


class TomTomRoutingClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return live_routing_available(self._config)

    def calculate_route(
        self,
        *,
        scenario_name: str,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        stop_points: list[tuple[float, float]],
        stop_labels: list[str],
        vehicle_key: str,
        route_type: str,
    ) -> LiveRouteResult:
        if vehicle_key not in VEHICLE_PROFILES:
            raise KeyError(f"Unknown vehicle profile: {vehicle_key}")
        if not self.enabled:
            raise LiveRoutingError("TomTom live routing is not configured. Set TOMTOM_API_KEY to enable it.")

        payload = self._request_route(
            points=[start_point, *stop_points, end_point],
            vehicle_key=vehicle_key,
            route_type=route_type,
            optimize_stops=bool(stop_points),
        )

        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise LiveRoutingError("TomTom returned an empty route response.")

        route = routes[0]
        summary = route.get("summary", {})
        coordinates = _join_points(route)
        if len(coordinates) < 2:
            raise LiveRoutingError("TomTom did not return usable route geometry.")

        ordered_stops = _ordered_stop_labels(stop_labels, payload.get("optimizedWaypoints", []))
        vehicle = VEHICLE_PROFILES[vehicle_key]
        distance_km = float(summary.get("lengthInMeters", 0.0)) / 1000.0
        total_time_min = float(summary.get("travelTimeInSeconds", 0.0)) / 60.0
        no_traffic_time_min = _minutes_or_none(summary.get("noTrafficTravelTimeInSeconds"))
        traffic_delay_min = _minutes_or_none(summary.get("trafficDelayInSeconds")) or 0.0
        estimated_emissions = _estimate_live_emissions_g(
            vehicle=vehicle,
            distance_km=distance_km,
            total_time_min=total_time_min,
            no_traffic_time_min=no_traffic_time_min,
            route_type=route_type,
        )

        return LiveRouteResult(
            name=scenario_name,
            provider="TomTom Routing API",
            route_type=route_type,
            coordinates=coordinates,
            ordered_stops=ordered_stops,
            guidance_steps=_parse_guidance_steps(route),
            total_distance_km=distance_km,
            total_time_min=total_time_min,
            total_emissions_g=estimated_emissions,
            total_signal_delay_min=traffic_delay_min,
            average_green_score=None,
            total_cost=total_time_min,
            strict_green_applied=False,
            traffic_delay_min=traffic_delay_min,
            no_traffic_time_min=no_traffic_time_min,
            historic_traffic_time_min=_minutes_or_none(summary.get("historicTrafficTravelTimeInSeconds")),
            live_incident_time_min=_minutes_or_none(summary.get("liveTrafficIncidentsTravelTimeInSeconds")),
            departure_time=_text_value(summary.get("departureTime")) or None,
            arrival_time=_text_value(summary.get("arrivalTime")) or None,
            legs_count=len(route.get("legs", [])),
        )

    def _request_route(
        self,
        *,
        points: list[tuple[float, float]],
        vehicle_key: str,
        route_type: str,
        optimize_stops: bool,
    ) -> dict[str, Any]:
        if len(points) < 2:
            raise LiveRoutingError("At least a start and destination point are required for live routing.")

        vehicle_params = _vehicle_params(vehicle_key)
        locations = ":".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
        params = {
            "key": self._config.tomtom_api_key,
            "traffic": "true",
            "departAt": "now",
            "routeType": route_type,
            "travelMode": vehicle_params["travelMode"],
            "vehicleCommercial": "true" if vehicle_params["vehicleCommercial"] else "false",
            "vehicleEngineType": vehicle_params["vehicleEngineType"],
            "instructionsType": "text",
            "language": "en-GB",
            "sectionType": "traffic",
            "routeRepresentation": "polyline",
            "computeTravelTimeFor": "all",
            "report": "effectiveSettings",
            "computeBestOrder": "true" if optimize_stops else "false",
        }

        url = f"{self._config.tomtom_base_url}/routing/1/calculateRoute/{locations}/json?{urlencode(params)}"
        LOGGER.info("Requesting live route from TomTom: routeType=%s points=%s", route_type, len(points))
        request = Request(url, headers={"accept": "application/json"})

        try:
            with urlopen(request, timeout=self._config.tomtom_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            LOGGER.warning("TomTom HTTP error %s: %s", exc.code, detail[:400])
            raise LiveRoutingError(f"TomTom request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            LOGGER.warning("TomTom connection error: %s", exc)
            raise LiveRoutingError("TomTom live routing could not be reached from this machine.") from exc
        except json.JSONDecodeError as exc:
            raise LiveRoutingError("TomTom returned an invalid JSON response.") from exc

    def geocode(self, query: str, *, around_point: tuple[float, float] | None = None, limit: int = 5) -> list[GeocodeCandidate]:
        if not self.enabled:
            raise LiveRoutingError("TomTom geocoding is not configured. Set TOMTOM_API_KEY to enable it.")
        query = query.strip()
        if not query:
            return []

        encoded_query = quote(query, safe="")
        params = {
            "key": self._config.tomtom_api_key,
            "limit": max(1, min(limit, 10)),
            "language": "en-GB",
            "countrySet": "BG",
        }
        if around_point is not None:
            params["lat"] = f"{around_point[0]:.6f}"
            params["lon"] = f"{around_point[1]:.6f}"

        url = f"{self._config.tomtom_base_url}/search/2/geocode/{encoded_query}.json?{urlencode(params)}"
        request = Request(url, headers={"accept": "application/json"})

        try:
            with urlopen(request, timeout=self._config.tomtom_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            LOGGER.warning("TomTom geocode HTTP error %s: %s", exc.code, detail[:400])
            raise LiveRoutingError(f"TomTom geocode request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            LOGGER.warning("TomTom geocode connection error: %s", exc)
            raise LiveRoutingError("TomTom geocoding could not be reached from this machine.") from exc
        except json.JSONDecodeError as exc:
            raise LiveRoutingError("TomTom geocoding returned an invalid JSON response.") from exc

        candidates: list[GeocodeCandidate] = []
        for result in payload.get("results", []):
            position = result.get("position", {})
            address = result.get("address", {})
            lat = position.get("lat")
            lon = position.get("lon")
            if lat is None or lon is None:
                continue
            label = (
                _text_value(address.get("freeformAddress"))
                or _text_value(address.get("streetName"))
                or _text_value(result.get("type"))
                or query
            )
            score = float(result.get("score", 0.0))
            candidates.append(GeocodeCandidate(label=label, lat=float(lat), lon=float(lon), score=score))
        return candidates


def _vehicle_params(vehicle_key: str) -> dict[str, Any]:
    if vehicle_key == "Passenger EV":
        return {"travelMode": "car", "vehicleCommercial": False, "vehicleEngineType": "electric"}
    if vehicle_key == "Heavy Truck":
        return {"travelMode": "truck", "vehicleCommercial": True, "vehicleEngineType": "combustion"}
    if vehicle_key == "Delivery Van":
        return {"travelMode": "car", "vehicleCommercial": True, "vehicleEngineType": "combustion"}
    return {"travelMode": "car", "vehicleCommercial": False, "vehicleEngineType": "combustion"}


def live_route_scenarios(
    *,
    config: AppConfig,
    start_label: str,
    start_point: tuple[float, float],
    end_label: str,
    end_point: tuple[float, float],
    stop_labels: list[str],
    stop_points: list[tuple[float, float]],
    vehicle_key: str,
) -> dict[str, LiveRouteResult]:
    client = TomTomRoutingClient(config)
    scenarios = {
        "Traffic-aware fastest": client.calculate_route(
            scenario_name="Traffic-aware fastest",
            start_point=start_point,
            end_point=end_point,
            stop_points=stop_points,
            stop_labels=stop_labels,
            vehicle_key=vehicle_key,
            route_type="fastest",
        ),
        "Live eco": client.calculate_route(
            scenario_name="Live eco",
            start_point=start_point,
            end_point=end_point,
            stop_points=stop_points,
            stop_labels=stop_labels,
            vehicle_key=vehicle_key,
            route_type="eco",
        ),
    }

    for result in scenarios.values():
        if not result.ordered_stops:
            continue
        LOGGER.info(
            "Live route stop order for %s -> %s via %s: %s",
            start_label,
            end_label,
            result.name,
            " -> ".join(result.ordered_stops),
        )
    return scenarios


def geocode_candidates(
    *,
    config: AppConfig,
    query: str,
    around_point: tuple[float, float] | None = None,
    limit: int = 5,
) -> list[GeocodeCandidate]:
    return TomTomRoutingClient(config).geocode(query, around_point=around_point, limit=limit)


def resolve_backend(config: AppConfig, requested_backend: str) -> str:
    normalized = requested_backend.strip().lower()
    if normalized not in {"auto", "local", "tomtom"}:
        normalized = "auto"

    if normalized == "local":
        return "local"
    if normalized == "tomtom":
        return "tomtom" if live_routing_available(config) else "local"
    return "tomtom" if live_routing_available(config) else "local"


def live_backend_status_message(config: AppConfig, requested_backend: str) -> str:
    effective_backend = resolve_backend(config, requested_backend)
    if effective_backend == "tomtom":
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        return f"TomTom live routing is active for departure now. Provider refresh reference: {timestamp}."
    return "Live routing is not active. The app is using the validated local Sofia road model."


def live_delay_confidence(route: LiveRouteResult) -> float:
    no_traffic = route.no_traffic_time_min or route.total_time_min
    if no_traffic <= 0:
        return 0.0
    ratio = route.traffic_delay_min / no_traffic
    return clamp(1.0 - ratio * 0.6, 0.2, 1.0)
