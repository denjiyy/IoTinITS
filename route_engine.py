from __future__ import annotations

import csv
import heapq
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ml_models import MLSegmentScore
from traffic_calibration import CalibrationBundle


DATA_PATH = Path(__file__).with_name("sofia_route_network.csv")

HUB_COORDINATES = {
    "Lyulin Center": (42.7178, 23.2680),
    "Central Railway Station": (42.7109, 23.3226),
    "Lions Bridge": (42.7060, 23.3218),
    "Serdika": (42.6977, 23.3225),
    "National Palace of Culture": (42.6864, 23.3196),
    "Sofia University": (42.6932, 23.3348),
    "Orlov Most": (42.6939, 23.3365),
    "Krasno Selo": (42.6850, 23.2842),
    "Paradise Center": (42.6648, 23.3159),
    "Hotel Pliska": (42.6762, 23.3571),
    "Mladost 1": (42.6517, 23.3790),
    "The Mall": (42.6598, 23.3956),
    "Sofia Airport Terminal 2": (42.6887, 23.4080),
}


@dataclass(frozen=True, slots=True)
class Edge:
    segment_id: str
    from_node: str
    from_name: str
    from_lat: float
    from_lon: float
    to_node: str
    to_name: str
    to_lat: float
    to_lon: float
    road_name: str
    road_class: str
    direction: str
    lanes: int
    length_km: float
    speed_limit_kmh: int
    free_flow_speed_kmh: float
    base_congestion_index: float
    peak_sensitivity: float
    traffic_volume_aadt: int
    heavy_vehicle_share: float
    signal_count: int
    green_wave_score: float
    green_corridor: bool
    avg_intersection_delay_s: float
    grade_percent: float
    eco_factor: float
    congestion_profile_3h: tuple[float, ...]
    green_profile_3h: tuple[float, ...]
    curb_activity_profile_3h: tuple[float, ...]
    weekday_volume_profile: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Node:
    node_id: str
    name: str
    lat: float
    lon: float


@dataclass(slots=True)
class RouteSegment:
    edge: Edge
    travel_time_min: float
    emissions_g: float
    signal_delay_s: float
    current_speed_kmh: float
    green_penalty_min: float
    composite_cost: float
    ml_predicted_time_min: float | None = None
    ml_high_congestion_probability: float | None = None
    ml_green_corridor_probability: float | None = None
    ml_observed_correction_factor: float | None = None


@dataclass(slots=True)
class RouteResult:
    name: str
    segments: list[RouteSegment]
    ordered_stops: list[str]
    total_distance_km: float
    total_time_min: float
    total_emissions_g: float
    average_green_score: float
    total_signal_delay_min: float
    total_cost: float
    strict_green_applied: bool
    uses_ml_scoring: bool
    uses_empirical_calibration: bool
    uses_observed_target_model: bool


@dataclass(frozen=True, slots=True)
class NetworkSummary:
    node_count: int
    edge_count: int
    hub_count: int
    reachable_hub_count: int
    green_corridor_count: int
    major_road_count: int


@dataclass(frozen=True, slots=True)
class VehicleProfile:
    label: str
    base_g_per_km: float
    stop_go_penalty: float


VEHICLE_PROFILES = {
    "Passenger EV": VehicleProfile("Passenger EV", 52.0, 0.10),
    "Passenger Petrol": VehicleProfile("Passenger Petrol", 168.0, 0.22),
    "Delivery Van": VehicleProfile("Delivery Van", 228.0, 0.28),
    "Heavy Truck": VehicleProfile("Heavy Truck", 590.0, 0.36),
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0
    lat1 = math.radians(lat_a)
    lat2 = math.radians(lat_b)
    dlat = math.radians(lat_b - lat_a)
    dlon = math.radians(lon_b - lon_a)
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))


def parse_profile(raw: str | None, expected_size: int, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return fallback
    try:
        values = tuple(float(part) for part in raw.split("|"))
    except ValueError:
        return fallback
    if len(values) != expected_size:
        return fallback
    return values


DEFAULT_CONGESTION_PROFILE = (0.38, 0.44, 0.88, 1.22, 1.00, 1.08, 1.28, 0.74)
DEFAULT_GREEN_PROFILE = (0.76, 0.78, 0.72, 0.66, 0.68, 0.64, 0.60, 0.74)
DEFAULT_CURB_ACTIVITY_PROFILE = (0.06, 0.05, 0.12, 0.24, 0.32, 0.28, 0.20, 0.10)
DEFAULT_WEEKDAY_VOLUME_PROFILE = (1.05, 1.08, 1.09, 1.07, 1.10, 0.86, 0.73)


def load_network(csv_path: Path | str = DATA_PATH) -> tuple[dict[str, Node], dict[str, list[Edge]], list[Edge]]:
    nodes: dict[str, Node] = {}
    adjacency: dict[str, list[Edge]] = {}
    edges: list[Edge] = []

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edge = Edge(
                segment_id=row["segment_id"],
                from_node=row["from_node"],
                from_name=row["from_name"],
                from_lat=float(row["from_lat"]),
                from_lon=float(row["from_lon"]),
                to_node=row["to_node"],
                to_name=row["to_name"],
                to_lat=float(row["to_lat"]),
                to_lon=float(row["to_lon"]),
                road_name=row["road_name"],
                road_class=row["road_class"],
                direction=row["direction"],
                lanes=int(row["lanes"]),
                length_km=float(row["length_km"]),
                speed_limit_kmh=int(row["speed_limit_kmh"]),
                free_flow_speed_kmh=float(row["free_flow_speed_kmh"]),
                base_congestion_index=float(row["base_congestion_index"]),
                peak_sensitivity=float(row["peak_sensitivity"]),
                traffic_volume_aadt=int(row["traffic_volume_aadt"]),
                heavy_vehicle_share=float(row["heavy_vehicle_share"]),
                signal_count=int(row["signal_count"]),
                green_wave_score=float(row["green_wave_score"]),
                green_corridor=row["green_corridor"] == "1",
                avg_intersection_delay_s=float(row["avg_intersection_delay_s"]),
                grade_percent=float(row["grade_percent"]),
                eco_factor=float(row["eco_factor"]),
                congestion_profile_3h=parse_profile(row.get("congestion_profile_3h"), 8, DEFAULT_CONGESTION_PROFILE),
                green_profile_3h=parse_profile(row.get("green_profile_3h"), 8, DEFAULT_GREEN_PROFILE),
                curb_activity_profile_3h=parse_profile(
                    row.get("curb_activity_profile_3h"), 8, DEFAULT_CURB_ACTIVITY_PROFILE
                ),
                weekday_volume_profile=parse_profile(
                    row.get("weekday_volume_profile"), 7, DEFAULT_WEEKDAY_VOLUME_PROFILE
                ),
            )
            nodes.setdefault(edge.from_node, Node(edge.from_node, edge.from_name, edge.from_lat, edge.from_lon))
            nodes.setdefault(edge.to_node, Node(edge.to_node, edge.to_name, edge.to_lat, edge.to_lon))
            adjacency.setdefault(edge.from_node, []).append(edge)
            edges.append(edge)

    return nodes, adjacency, edges


def nearest_node_id(nodes: dict[str, Node], lat: float, lon: float) -> tuple[str, float]:
    best_id = ""
    best_distance = float("inf")
    for node_id, node in nodes.items():
        distance = haversine_km(lat, lon, node.lat, node.lon)
        if distance < best_distance:
            best_id = node_id
            best_distance = distance
    return best_id, best_distance


def available_hubs(nodes: dict[str, Node]) -> dict[str, str]:
    snapped: dict[str, str] = {}
    for label, (lat, lon) in HUB_COORDINATES.items():
        node_id, distance = nearest_node_id(nodes, lat, lon)
        if distance <= 1.0:
            snapped[label] = node_id
    return dict(sorted(snapped.items(), key=lambda item: item[0]))


def reachable_nodes(adjacency: dict[str, list[Edge]], start_node: str) -> set[str]:
    if start_node not in adjacency:
        return set()

    seen = {start_node}
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        for edge in adjacency.get(node, []):
            if edge.to_node not in seen:
                seen.add(edge.to_node)
                queue.append(edge.to_node)
    return seen


def build_network_summary(
    nodes: dict[str, Node],
    adjacency: dict[str, list[Edge]],
    edges: list[Edge],
    hubs: dict[str, str],
) -> NetworkSummary:
    green_corridor_count = sum(edge.green_corridor for edge in edges)
    major_road_count = sum(edge.road_class in {"motorway", "trunk", "primary", "secondary"} for edge in edges)

    if hubs:
        seed_hub_node = next(iter(hubs.values()))
        reachable = reachable_nodes(adjacency, seed_hub_node)
        reachable_hub_count = sum(node_id in reachable for node_id in hubs.values())
    else:
        reachable_hub_count = 0

    return NetworkSummary(
        node_count=len(nodes),
        edge_count=len(edges),
        hub_count=len(hubs),
        reachable_hub_count=reachable_hub_count,
        green_corridor_count=green_corridor_count,
        major_road_count=major_road_count,
    )


def validate_network(
    nodes: dict[str, Node],
    adjacency: dict[str, list[Edge]],
    edges: list[Edge],
    hubs: dict[str, str],
) -> NetworkSummary:
    issues: list[str] = []
    summary = build_network_summary(nodes, adjacency, edges, hubs)

    if summary.node_count < 1000:
        issues.append(f"node count too low: {summary.node_count}")
    if summary.edge_count < 5000:
        issues.append(f"edge count too low: {summary.edge_count}")
    if summary.hub_count < 5:
        issues.append(f"too few selectable hubs: {summary.hub_count}")
    if summary.reachable_hub_count < min(5, summary.hub_count):
        issues.append(
            f"insufficient connected hubs in main graph: reachable {summary.reachable_hub_count} of {summary.hub_count}"
        )
    if summary.green_corridor_count == 0:
        issues.append("no green corridor segments available")

    for label, node_id in hubs.items():
        if node_id not in nodes:
            issues.append(f"hub '{label}' does not map to a known node")
        elif node_id not in adjacency:
            issues.append(f"hub '{label}' has no outbound road segments")

    if issues:
        raise ValueError("Invalid road network: " + "; ".join(issues))

    return summary


def hour_multiplier(hour: int) -> float:
    if 7 <= hour <= 9:
        return 1.24
    if 16 <= hour <= 18:
        return 1.28
    if 10 <= hour <= 15:
        return 0.96
    if 19 <= hour <= 22:
        return 0.76
    if 0 <= hour <= 5:
        return 0.48
    return 0.82


def profile_bucket(hour: int) -> int:
    return max(0, min(7, hour // 3))


def distance_to_center(lat: float, lon: float) -> float:
    center_lat = 42.6940
    center_lon = 23.3250
    return math.hypot(lat - center_lat, lon - center_lon)


def directional_peak_bias(edge: Edge, hour: int) -> float:
    from_distance = distance_to_center(edge.from_lat, edge.from_lon)
    to_distance = distance_to_center(edge.to_lat, edge.to_lon)
    moving_inward = to_distance < from_distance
    if 7 <= hour <= 9 and moving_inward:
        return 1.12
    if 16 <= hour <= 18 and not moving_inward:
        return 1.14
    return 1.0


def edge_metrics(
    edge: Edge,
    *,
    hour: int,
    vehicle: VehicleProfile,
    weights: tuple[float, float, float],
    ml_score: MLSegmentScore | None = None,
    calibration: CalibrationBundle | None = None,
) -> RouteSegment:
    bucket_index = profile_bucket(hour)
    profile_congestion = edge.congestion_profile_3h[bucket_index]
    profile_green = edge.green_profile_3h[bucket_index]
    curb_activity = edge.curb_activity_profile_3h[bucket_index]

    peak_load = profile_congestion * edge.peak_sensitivity * directional_peak_bias(edge, hour)
    dynamic_congestion = clamp(edge.base_congestion_index * peak_load, 0.10, 0.96)

    speed_drop = 0.50 * dynamic_congestion
    current_speed = max(12.0, edge.free_flow_speed_kmh * (1.0 - speed_drop))
    current_green_score = clamp((edge.green_wave_score * 0.45) + (profile_green * 0.55), 0.20, 0.98)
    signal_delay = edge.avg_intersection_delay_s * (0.60 + dynamic_congestion * 0.92) * (1.08 - current_green_score * 0.34)
    curbside_delay = 0.0
    if vehicle.stop_go_penalty >= 0.22 and edge.road_class not in {"motorway", "motorway_link"}:
        curbside_delay = curb_activity * (3.5 + edge.signal_count * 1.4) * (0.8 + vehicle.stop_go_penalty)

    modeled_travel_time_min = edge.length_km / current_speed * 60.0 + signal_delay / 60.0 + curbside_delay / 60.0
    travel_time_min = modeled_travel_time_min

    grade_factor = 1.0 + max(edge.grade_percent, 0.0) * 0.02
    congestion_factor = 1.0 + dynamic_congestion * 0.42 + vehicle.stop_go_penalty * (1 - current_green_score)
    congestion_factor += curb_activity * vehicle.stop_go_penalty * 0.16
    if ml_score and ml_score.high_congestion_probability is not None:
        congestion_factor *= 1.0 + ml_score.high_congestion_probability * 0.10
    emissions_g = vehicle.base_g_per_km * edge.length_km * edge.eco_factor * grade_factor * congestion_factor

    green_penalty = signal_delay / 60.0 + curbside_delay / 120.0 + (1.0 - current_green_score) * 1.35
    if ml_score and ml_score.high_congestion_probability is not None:
        green_penalty += ml_score.high_congestion_probability * 0.25
    if ml_score and ml_score.predicted_time_min is not None:
        hybrid_predicted_time = max(0.01, float(ml_score.predicted_time_min))
        travel_time_min = modeled_travel_time_min * 0.35 + hybrid_predicted_time * 0.65
        current_speed = max(6.0, edge.length_km / max(travel_time_min, 0.02) * 60.0)
    observed_factor = ml_score.observed_correction_factor if ml_score else None
    empirical_factor = calibration.time_factor(edge.segment_id, edge.road_class, hour) if calibration and calibration.active else None
    correction_factor = 1.0
    if observed_factor is not None and empirical_factor is not None:
        correction_factor = observed_factor * 0.40 + empirical_factor * 0.60
    elif observed_factor is not None:
        correction_factor = observed_factor
    elif empirical_factor is not None:
        correction_factor = empirical_factor
    if correction_factor != 1.0:
        travel_time_min *= correction_factor
        emissions_g *= 1.0 + max(correction_factor - 1.0, -0.18) * 0.30
        green_penalty += max(correction_factor - 1.0, 0.0) * 0.22

    weight_time, weight_emissions, weight_green = weights
    composite_cost = (
        weight_time * travel_time_min
        + weight_emissions * (emissions_g / 140.0)
        + weight_green * green_penalty
    )

    return RouteSegment(
        edge=edge,
        travel_time_min=travel_time_min,
        emissions_g=emissions_g,
        signal_delay_s=signal_delay,
        current_speed_kmh=current_speed,
        green_penalty_min=green_penalty,
        composite_cost=composite_cost,
        ml_predicted_time_min=ml_score.predicted_time_min if ml_score else None,
        ml_high_congestion_probability=ml_score.high_congestion_probability if ml_score else None,
        ml_green_corridor_probability=ml_score.green_corridor_probability if ml_score else None,
        ml_observed_correction_factor=ml_score.observed_correction_factor if ml_score else None,
    )


def normalize_weights(time_weight: int, emissions_weight: int, green_weight: int) -> tuple[float, float, float]:
    if any(weight < 0 for weight in (time_weight, emissions_weight, green_weight)):
        raise ValueError("Objective weights must be non-negative.")
    total = max(time_weight + emissions_weight + green_weight, 1)
    return time_weight / total, emissions_weight / total, green_weight / total


def edge_allowed(edge: Edge, *, strict_green: bool) -> bool:
    if not strict_green:
        return True
    if edge.green_corridor:
        return True
    return edge.green_wave_score >= 0.84 and edge.road_class in {"motorway", "trunk", "primary", "secondary"}


def shortest_path(
    *,
    adjacency: dict[str, list[Edge]],
    start_node: str,
    end_node: str,
    hour: int,
    vehicle: VehicleProfile,
    weights: tuple[float, float, float],
    strict_green: bool,
    ml_segment_scores: dict[str, MLSegmentScore] | None = None,
    calibration: CalibrationBundle | None = None,
) -> list[RouteSegment] | None:
    queue: list[tuple[float, str]] = [(0.0, start_node)]
    costs = {start_node: 0.0}
    previous: dict[str, tuple[str, RouteSegment]] = {}

    while queue:
        current_cost, node = heapq.heappop(queue)
        if node == end_node:
            break
        if current_cost > costs.get(node, float("inf")):
            continue

        for edge in adjacency.get(node, []):
            if not edge_allowed(edge, strict_green=strict_green):
                continue
            segment = edge_metrics(
                edge,
                hour=hour,
                vehicle=vehicle,
                weights=weights,
                ml_score=ml_segment_scores.get(edge.segment_id) if ml_segment_scores else None,
                calibration=calibration,
            )
            new_cost = current_cost + segment.composite_cost
            if new_cost < costs.get(edge.to_node, float("inf")):
                costs[edge.to_node] = new_cost
                previous[edge.to_node] = (node, segment)
                heapq.heappush(queue, (new_cost, edge.to_node))

    if end_node not in previous and start_node != end_node:
        return None

    path: list[RouteSegment] = []
    node = end_node
    while node != start_node:
        prior_node, segment = previous[node]
        path.append(segment)
        node = prior_node
    path.reverse()
    return path


def optimize_delivery_order(
    *,
    adjacency: dict[str, list[Edge]],
    start_node: str,
    end_node: str,
    stops: list[str],
    hour: int,
    vehicle: VehicleProfile,
    weights: tuple[float, float, float],
    strict_green: bool,
    ml_segment_scores: dict[str, MLSegmentScore] | None = None,
    calibration: CalibrationBundle | None = None,
) -> list[str]:
    remaining = list(dict.fromkeys(stop for stop in stops if stop not in {start_node, end_node}))
    ordered: list[str] = []
    current = start_node

    while remaining:
        best_stop = None
        best_cost = float("inf")
        for candidate in remaining:
            path = shortest_path(
                adjacency=adjacency,
                start_node=current,
                end_node=candidate,
                hour=hour,
                vehicle=vehicle,
                weights=weights,
                strict_green=strict_green,
                ml_segment_scores=ml_segment_scores,
                calibration=calibration,
            )
            if not path:
                continue
            cost = sum(segment.composite_cost for segment in path)
            if cost < best_cost:
                best_cost = cost
                best_stop = candidate
        if best_stop is None:
            return []
        ordered.append(best_stop)
        remaining.remove(best_stop)
        current = best_stop

    return ordered


def build_trip(
    *,
    name: str,
    adjacency: dict[str, list[Edge]],
    start_node: str,
    end_node: str,
    stops: list[str],
    hour: int,
    vehicle: VehicleProfile,
    weights: tuple[float, float, float],
    strict_green: bool,
    ml_segment_scores: dict[str, MLSegmentScore] | None = None,
    calibration: CalibrationBundle | None = None,
) -> RouteResult | None:
    requested_stops = list(dict.fromkeys(stop for stop in stops if stop not in {start_node, end_node}))
    ordered_stops = optimize_delivery_order(
        adjacency=adjacency,
        start_node=start_node,
        end_node=end_node,
        stops=requested_stops,
        hour=hour,
        vehicle=vehicle,
        weights=weights,
        strict_green=strict_green,
        ml_segment_scores=ml_segment_scores,
        calibration=calibration,
    )
    if len(ordered_stops) != len(requested_stops):
        return None

    itinerary = [start_node, *ordered_stops, end_node]
    segments: list[RouteSegment] = []

    for leg_start, leg_end in zip(itinerary, itinerary[1:]):
        leg = shortest_path(
            adjacency=adjacency,
            start_node=leg_start,
            end_node=leg_end,
            hour=hour,
            vehicle=vehicle,
            weights=weights,
            strict_green=strict_green,
            ml_segment_scores=ml_segment_scores,
            calibration=calibration,
        )
        if leg is None:
            return None
        segments.extend(leg)

    if not segments:
        return RouteResult(
            name=name,
            segments=[],
            ordered_stops=ordered_stops,
            total_distance_km=0.0,
            total_time_min=0.0,
            total_emissions_g=0.0,
            average_green_score=1.0,
            total_signal_delay_min=0.0,
            total_cost=0.0,
            strict_green_applied=strict_green,
            uses_ml_scoring=ml_segment_scores is not None,
            uses_empirical_calibration=calibration is not None and calibration.active,
            uses_observed_target_model=ml_segment_scores is not None
            and any(score.observed_correction_factor is not None for score in ml_segment_scores.values()),
        )

    total_distance = sum(segment.edge.length_km for segment in segments)
    total_time = sum(segment.travel_time_min for segment in segments)
    total_emissions = sum(segment.emissions_g for segment in segments)
    total_signal_delay = sum(segment.signal_delay_s for segment in segments) / 60.0
    average_green = sum(segment.edge.green_wave_score for segment in segments) / len(segments)
    total_cost = sum(segment.composite_cost for segment in segments)

    return RouteResult(
        name=name,
        segments=segments,
        ordered_stops=ordered_stops,
        total_distance_km=total_distance,
        total_time_min=total_time,
        total_emissions_g=total_emissions,
        average_green_score=average_green,
        total_signal_delay_min=total_signal_delay,
        total_cost=total_cost,
        strict_green_applied=strict_green,
        uses_ml_scoring=ml_segment_scores is not None,
        uses_empirical_calibration=calibration is not None and calibration.active,
        uses_observed_target_model=any(segment.ml_observed_correction_factor is not None for segment in segments),
    )


def route_scenarios(
    *,
    adjacency: dict[str, list[Edge]],
    start_node: str,
    end_node: str,
    stops: list[str],
    hour: int,
    vehicle_key: str,
    user_weights: tuple[int, int, int],
    strict_green: bool,
    ml_segment_scores: dict[str, MLSegmentScore] | None = None,
    calibration: CalibrationBundle | None = None,
) -> dict[str, RouteResult | None]:
    if vehicle_key not in VEHICLE_PROFILES:
        raise KeyError(f"Unknown vehicle profile: {vehicle_key}")
    if hour < 0 or hour > 23:
        raise ValueError("Departure hour must be in the range 0-23.")
    if start_node == end_node:
        raise ValueError("Start and destination nodes must be different.")
    if start_node not in adjacency:
        raise ValueError(f"Unknown or disconnected start node: {start_node}")
    if end_node not in adjacency:
        raise ValueError(f"Unknown or disconnected destination node: {end_node}")
    invalid_stops = [stop for stop in stops if stop not in adjacency]
    if invalid_stops:
        raise ValueError(f"Unknown or disconnected stop nodes: {', '.join(invalid_stops)}")

    vehicle = VEHICLE_PROFILES[vehicle_key]
    balanced_weights = normalize_weights(*user_weights)

    return {
        "Balanced": build_trip(
            name="Balanced",
            adjacency=adjacency,
            start_node=start_node,
            end_node=end_node,
            stops=stops,
            hour=hour,
            vehicle=vehicle,
            weights=balanced_weights,
            strict_green=strict_green,
            ml_segment_scores=ml_segment_scores,
            calibration=calibration,
        ),
        "Fastest": build_trip(
            name="Fastest",
            adjacency=adjacency,
            start_node=start_node,
            end_node=end_node,
            stops=stops,
            hour=hour,
            vehicle=vehicle,
            weights=(1.0, 0.0, 0.0),
            strict_green=False,
            ml_segment_scores=ml_segment_scores,
            calibration=calibration,
        ),
        "Lowest emissions": build_trip(
            name="Lowest emissions",
            adjacency=adjacency,
            start_node=start_node,
            end_node=end_node,
            stops=stops,
            hour=hour,
            vehicle=vehicle,
            weights=(0.0, 1.0, 0.0),
            strict_green=False,
            ml_segment_scores=ml_segment_scores,
            calibration=calibration,
        ),
        "Green corridor": build_trip(
            name="Green corridor",
            adjacency=adjacency,
            start_node=start_node,
            end_node=end_node,
            stops=stops,
            hour=hour,
            vehicle=vehicle,
            weights=(0.15, 0.10, 0.75),
            strict_green=False,
            ml_segment_scores=ml_segment_scores,
            calibration=calibration,
        ),
    }


def grouped_steps(route: RouteResult) -> list[dict[str, str | float]]:
    if not route.segments:
        return []

    groups: list[dict[str, float | str]] = []
    current = None

    for segment in route.segments:
        key = (segment.edge.road_name, segment.edge.direction)
        if current and current["road_name"] == key[0] and current["direction"] == key[1]:
            current["distance_km"] += segment.edge.length_km
            current["time_min"] += segment.travel_time_min
            current["emissions_g"] += segment.emissions_g
            current["green_score_total"] += segment.edge.green_wave_score
            current["segment_count"] += 1
        else:
            if current:
                groups.append(current)
            current = {
                "road_name": segment.edge.road_name,
                "direction": segment.edge.direction,
                "distance_km": segment.edge.length_km,
                "time_min": segment.travel_time_min,
                "emissions_g": segment.emissions_g,
                "green_score_total": segment.edge.green_wave_score,
                "segment_count": 1,
            }

    if current:
        groups.append(current)

    return [
        {
            "road_name": group["road_name"],
            "direction": group["direction"],
            "distance_km": round(float(group["distance_km"]), 2),
            "time_min": round(float(group["time_min"]), 1),
            "emissions_g": round(float(group["emissions_g"]), 0),
            "avg_green_score": round(float(group["green_score_total"]) / int(group["segment_count"]), 2),
        }
        for group in groups
    ]


def route_path_coordinates(route: RouteResult) -> list[list[float]]:
    coords: list[list[float]] = []
    if not route.segments:
        return coords

    coords.append([route.segments[0].edge.from_lon, route.segments[0].edge.from_lat])
    for segment in route.segments:
        coords.append([segment.edge.to_lon, segment.edge.to_lat])
    return coords


def route_waypoint_names(route: RouteResult, node_label_lookup: dict[str, str], start_label: str, end_label: str) -> list[str]:
    labels = [start_label]
    labels.extend(node_label_lookup.get(node_id, node_id) for node_id in route.ordered_stops)
    labels.append(end_label)
    return labels
