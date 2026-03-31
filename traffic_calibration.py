from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_models import modeled_segment_travel_time


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


def profile_bucket(hour: int) -> int:
    return max(0, min(7, hour // 3))


def _bearing_deg(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat1 = math.radians(lat_a)
    lat2 = math.radians(lat_b)
    dlon = math.radians(lon_b - lon_a)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _heading_delta_deg(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def _grid_key(lat: float, lon: float, cell_size_deg: float) -> tuple[int, int]:
    return int(math.floor(lat / cell_size_deg)), int(math.floor(lon / cell_size_deg))


def _route_signature(route: Any, start_label: str, end_label: str, vehicle_key: str) -> str:
    digest = hashlib.sha1()
    parts = [
        str(getattr(route, "provider", "")),
        str(getattr(route, "name", "")),
        str(getattr(route, "route_type", "")),
        start_label,
        end_label,
        vehicle_key,
        str(getattr(route, "departure_time", "")),
        str(round(float(getattr(route, "total_distance_km", 0.0)), 3)),
        str(round(float(getattr(route, "total_time_min", 0.0)), 3)),
        str(getattr(route, "legs_count", 0)),
    ]
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"|")
    coordinates = getattr(route, "coordinates", []) or []
    if coordinates:
        first = coordinates[0]
        last = coordinates[-1]
        digest.update(f"{first[0]:.6f},{first[1]:.6f}|{last[0]:.6f},{last[1]:.6f}".encode("utf-8"))
    return digest.hexdigest()


def _local_departure_hour(route: Any) -> int:
    departure_time = getattr(route, "departure_time", None)
    if departure_time:
        try:
            return datetime.fromisoformat(str(departure_time)).hour
        except ValueError:
            pass
    return datetime.now().astimezone().hour


def _vehicle_stop_go_penalty(vehicle_key: str) -> float:
    return {
        "Passenger EV": 0.10,
        "Passenger Petrol": 0.22,
        "Delivery Van": 0.28,
        "Heavy Truck": 0.36,
    }.get(vehicle_key, 0.22)


@dataclass(frozen=True, slots=True)
class EdgeMatcher:
    edge_map: dict[str, Any]
    grid_index: dict[tuple[int, int], tuple[str, ...]]
    cell_size_deg: float
    max_match_distance_km: float


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    segment_time_factors: dict[str, float]
    road_class_bucket_factors: dict[tuple[str, int], float]
    route_count: int
    observation_rows: int
    calibrated_segments: int
    calibrated_groups: int
    updated_at_utc: str | None

    def time_factor(self, segment_id: str, road_class: str, hour: int) -> float:
        segment_factor = self.segment_time_factors.get(segment_id)
        bucket_factor = self.road_class_bucket_factors.get((road_class, profile_bucket(hour)))
        if segment_factor is not None and bucket_factor is not None:
            return clamp(segment_factor * 0.7 + bucket_factor * 0.3, 0.75, 1.45)
        if segment_factor is not None:
            return clamp(segment_factor, 0.75, 1.45)
        if bucket_factor is not None:
            return clamp(bucket_factor, 0.78, 1.40)
        return 1.0

    @property
    def active(self) -> bool:
        return self.calibrated_segments > 0 or self.calibrated_groups > 0


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    observation_rows: int
    route_count: int
    calibrated_segments: int
    calibrated_groups: int
    updated_at_utc: str | None

    @property
    def active(self) -> bool:
        return self.observation_rows > 0


OBSERVATION_FIELDS = [
    "route_signature",
    "recorded_at_utc",
    "provider",
    "scenario_name",
    "route_type",
    "start_label",
    "end_label",
    "vehicle_key",
    "departure_hour_local",
    "segment_id",
    "road_class",
    "matched_distance_km",
    "modeled_time_min",
    "observed_time_min",
    "observed_to_modeled_ratio",
    "traffic_delay_min",
    "no_traffic_time_min",
]


def build_edge_matcher(
    edges: list[Any],
    *,
    cell_size_deg: float = 0.005,
    max_match_distance_km: float = 0.22,
) -> EdgeMatcher:
    grid: dict[tuple[int, int], list[str]] = defaultdict(list)
    edge_map: dict[str, Any] = {}

    for edge in edges:
        segment_id = str(getattr(edge, "segment_id"))
        edge_map[segment_id] = edge
        min_lat = min(float(getattr(edge, "from_lat")), float(getattr(edge, "to_lat")))
        max_lat = max(float(getattr(edge, "from_lat")), float(getattr(edge, "to_lat")))
        min_lon = min(float(getattr(edge, "from_lon")), float(getattr(edge, "to_lon")))
        max_lon = max(float(getattr(edge, "from_lon")), float(getattr(edge, "to_lon")))
        row_start, col_start = _grid_key(min_lat, min_lon, cell_size_deg)
        row_end, col_end = _grid_key(max_lat, max_lon, cell_size_deg)
        for row in range(row_start - 1, row_end + 2):
            for col in range(col_start - 1, col_end + 2):
                grid[(row, col)].append(segment_id)

    return EdgeMatcher(
        edge_map=edge_map,
        grid_index={key: tuple(value) for key, value in grid.items()},
        cell_size_deg=cell_size_deg,
        max_match_distance_km=max_match_distance_km,
    )


def _candidate_segment_ids(matcher: EdgeMatcher, lat: float, lon: float) -> list[str]:
    row, col = _grid_key(lat, lon, matcher.cell_size_deg)
    segment_ids: list[str] = []
    seen: set[str] = set()
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            for segment_id in matcher.grid_index.get((row + row_offset, col + col_offset), ()):
                if segment_id not in seen:
                    seen.add(segment_id)
                    segment_ids.append(segment_id)
    return segment_ids


def match_live_segment(matcher: EdgeMatcher, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Any | None:
    midpoint_lat = (start_lat + end_lat) / 2
    midpoint_lon = (start_lon + end_lon) / 2
    live_heading = _bearing_deg(start_lat, start_lon, end_lat, end_lon)
    live_distance = haversine_km(start_lat, start_lon, end_lat, end_lon)

    best_edge = None
    best_score = float("inf")
    for segment_id in _candidate_segment_ids(matcher, midpoint_lat, midpoint_lon):
        edge = matcher.edge_map[segment_id]
        edge_mid_lat = (float(getattr(edge, "from_lat")) + float(getattr(edge, "to_lat"))) / 2
        edge_mid_lon = (float(getattr(edge, "from_lon")) + float(getattr(edge, "to_lon"))) / 2
        midpoint_distance = haversine_km(midpoint_lat, midpoint_lon, edge_mid_lat, edge_mid_lon)
        if midpoint_distance > matcher.max_match_distance_km:
            continue
        edge_heading = _bearing_deg(
            float(getattr(edge, "from_lat")),
            float(getattr(edge, "from_lon")),
            float(getattr(edge, "to_lat")),
            float(getattr(edge, "to_lon")),
        )
        heading_penalty = _heading_delta_deg(live_heading, edge_heading) / 180.0
        length_penalty = abs(float(getattr(edge, "length_km")) - live_distance)
        score = midpoint_distance + heading_penalty * 0.18 + length_penalty * 0.10
        if score < best_score:
            best_score = score
            best_edge = edge
    return best_edge


def record_live_route_observations(
    *,
    observation_path: Path | str,
    matcher: EdgeMatcher,
    route: Any,
    start_label: str,
    end_label: str,
    vehicle_key: str,
    min_match_ratio: float = 0.35,
) -> int:
    coordinates = getattr(route, "coordinates", None) or []
    if len(coordinates) < 2:
        return 0

    observation_path = Path(observation_path)
    route_signature = _route_signature(route, start_label, end_label, vehicle_key)
    if observation_path.exists():
        with observation_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("route_signature") == route_signature:
                    return 0

    departure_hour = _local_departure_hour(route)
    stop_go_penalty = _vehicle_stop_go_penalty(vehicle_key)
    matched: dict[str, dict[str, float | str]] = {}
    total_live_distance = 0.0
    matched_distance = 0.0

    for start, end in zip(coordinates, coordinates[1:]):
        start_lon, start_lat = float(start[0]), float(start[1])
        end_lon, end_lat = float(end[0]), float(end[1])
        live_distance = haversine_km(start_lat, start_lon, end_lat, end_lon)
        if live_distance <= 0.003:
            continue
        total_live_distance += live_distance
        edge = match_live_segment(matcher, start_lat, start_lon, end_lat, end_lon)
        if edge is None:
            continue
        matched_distance += live_distance
        segment_id = str(getattr(edge, "segment_id"))
        entry = matched.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "road_class": str(getattr(edge, "road_class")),
                "matched_distance_km": 0.0,
                "modeled_time_min": 0.0,
            },
        )
        entry["matched_distance_km"] = float(entry["matched_distance_km"]) + live_distance
        entry["modeled_time_min"] = float(entry["modeled_time_min"]) + modeled_segment_travel_time(
            edge,
            hour=departure_hour,
            stop_go_penalty=stop_go_penalty,
        )

    if total_live_distance <= 0.0 or matched_distance / total_live_distance < min_match_ratio or not matched:
        return 0

    recorded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    total_time_min = float(getattr(route, "total_time_min", 0.0))
    traffic_delay_min = float(getattr(route, "traffic_delay_min", 0.0))
    no_traffic_time_min = getattr(route, "no_traffic_time_min", None)
    matched_denominator = max(matched_distance, 0.001)

    rows = []
    for entry in matched.values():
        observed_time = total_time_min * (float(entry["matched_distance_km"]) / matched_denominator)
        modeled_time = max(float(entry["modeled_time_min"]), 0.01)
        ratio = clamp(observed_time / modeled_time, 0.60, 1.90)
        rows.append(
            {
                "route_signature": route_signature,
                "recorded_at_utc": recorded_at,
                "provider": str(getattr(route, "provider", "")),
                "scenario_name": str(getattr(route, "name", "")),
                "route_type": str(getattr(route, "route_type", "")),
                "start_label": start_label,
                "end_label": end_label,
                "vehicle_key": vehicle_key,
                "departure_hour_local": str(departure_hour),
                "segment_id": str(entry["segment_id"]),
                "road_class": str(entry["road_class"]),
                "matched_distance_km": f"{float(entry['matched_distance_km']):.4f}",
                "modeled_time_min": f"{modeled_time:.4f}",
                "observed_time_min": f"{observed_time:.4f}",
                "observed_to_modeled_ratio": f"{ratio:.4f}",
                "traffic_delay_min": f"{traffic_delay_min:.4f}",
                "no_traffic_time_min": "" if no_traffic_time_min is None else f"{float(no_traffic_time_min):.4f}",
            }
        )

    observation_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = observation_path.exists()
    with observation_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_calibration_bundle(
    observation_path: Path | str,
    *,
    min_segment_samples: int = 3,
    min_group_samples: int = 8,
) -> CalibrationBundle:
    observation_path = Path(observation_path)
    if not observation_path.exists():
        return CalibrationBundle({}, {}, 0, 0, 0, 0, None)

    segment_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"weighted_ratio": 0.0, "weight": 0.0, "count": 0.0})
    group_stats: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"weighted_ratio": 0.0, "weight": 0.0, "count": 0.0}
    )
    route_signatures: set[str] = set()
    updated_at: str | None = None
    observation_rows = 0

    with observation_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ratio = clamp(float(row["observed_to_modeled_ratio"]), 0.60, 1.90)
                matched_distance = max(float(row["matched_distance_km"]), 0.01)
                hour = int(row["departure_hour_local"])
            except (KeyError, ValueError):
                continue
            observation_rows += 1
            route_signatures.add(row.get("route_signature", ""))
            if row.get("recorded_at_utc"):
                updated_at = row["recorded_at_utc"]
            segment_id = row.get("segment_id", "")
            road_class = row.get("road_class", "unclassified")
            group_key = (road_class, profile_bucket(hour))

            segment_entry = segment_stats[segment_id]
            segment_entry["weighted_ratio"] += ratio * matched_distance
            segment_entry["weight"] += matched_distance
            segment_entry["count"] += 1.0

            group_entry = group_stats[group_key]
            group_entry["weighted_ratio"] += ratio * matched_distance
            group_entry["weight"] += matched_distance
            group_entry["count"] += 1.0

    segment_factors = {
        segment_id: clamp(values["weighted_ratio"] / max(values["weight"], 0.01), 0.75, 1.45)
        for segment_id, values in segment_stats.items()
        if values["count"] >= float(min_segment_samples)
    }
    group_factors = {
        group_key: clamp(values["weighted_ratio"] / max(values["weight"], 0.01), 0.78, 1.40)
        for group_key, values in group_stats.items()
        if values["count"] >= float(min_group_samples)
    }

    return CalibrationBundle(
        segment_time_factors=segment_factors,
        road_class_bucket_factors=group_factors,
        route_count=len([value for value in route_signatures if value]),
        observation_rows=observation_rows,
        calibrated_segments=len(segment_factors),
        calibrated_groups=len(group_factors),
        updated_at_utc=updated_at,
    )


def calibration_status(
    observation_path: Path | str,
    *,
    min_segment_samples: int = 3,
    min_group_samples: int = 8,
) -> CalibrationStatus:
    bundle = load_calibration_bundle(
        observation_path,
        min_segment_samples=min_segment_samples,
        min_group_samples=min_group_samples,
    )
    return CalibrationStatus(
        observation_rows=bundle.observation_rows,
        route_count=bundle.route_count,
        calibrated_segments=bundle.calibrated_segments,
        calibrated_groups=bundle.calibrated_groups,
        updated_at_utc=bundle.updated_at_utc,
    )
