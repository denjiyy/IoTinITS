from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from app_config import load_app_config
from app_logging import get_logger, setup_logging


APP_CONFIG = load_app_config()
SOURCE_PATH = APP_CONFIG.source_path
OUTPUT_PATH = APP_CONFIG.data_path
LOGGER = get_logger(__name__)

CSV_FIELDS = [
    "segment_id",
    "from_node",
    "from_name",
    "from_lat",
    "from_lon",
    "to_node",
    "to_name",
    "to_lat",
    "to_lon",
    "road_name",
    "road_class",
    "direction",
    "lanes",
    "length_km",
    "speed_limit_kmh",
    "free_flow_speed_kmh",
    "base_congestion_index",
    "peak_sensitivity",
    "traffic_volume_aadt",
    "heavy_vehicle_share",
    "signal_count",
    "green_wave_score",
    "green_corridor",
    "avg_intersection_delay_s",
    "grade_percent",
    "eco_factor",
    "congestion_profile_3h",
    "green_profile_3h",
    "curb_activity_profile_3h",
    "weekday_volume_profile",
]

EXCLUDED_HIGHWAYS = {"service", "services"}
GREEN_CORRIDORS = {
    "tsarigradsko shose",
    "bulgaria",
    "cherni vrah",
    "dragan tsankov",
    "evlogi i hristo georgievi",
    "aleksandar malinov",
    "brussels",
    "slivnitsa",
    "nikola petkov",
}

DEFAULT_SPEEDS = {
    "motorway": 90,
    "motorway_link": 65,
    "trunk": 80,
    "trunk_link": 60,
    "primary": 60,
    "primary_link": 50,
    "secondary": 50,
    "secondary_link": 40,
    "tertiary": 40,
    "tertiary_link": 35,
    "unclassified": 35,
    "residential": 30,
    "living_street": 20,
}

DEFAULT_LANES = {
    "motorway": 3,
    "motorway_link": 2,
    "trunk": 3,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 2,
    "secondary": 2,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 1,
    "unclassified": 1,
    "residential": 1,
    "living_street": 1,
}

PROFILE_BUCKETS = tuple(range(0, 24, 3))
WEEKDAY_BUCKETS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


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


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def stable_wave(*values: float) -> float:
    total = 0.0
    for index, value in enumerate(values, start=1):
        total += value * (1.19 + index * 0.21)
    return (math.sin(total) + 1.0) / 2.0


def gaussian_peak(value: float, center: float, width: float) -> float:
    return math.exp(-((value - center) / width) ** 2)


def serialize_profile(values: list[float]) -> str:
    return "|".join(f"{value:.3f}" for value in values)


def centrality(lat: float, lon: float) -> float:
    center_lat = 42.6940
    center_lon = 23.3250
    lat_span = 0.050
    lon_span = 0.120
    lat_distance = abs(lat - center_lat) / lat_span
    lon_distance = abs(lon - center_lon) / lon_span
    return clamp(1.0 - (lat_distance + lon_distance) / 2.0, 0.0, 1.0)


def direction_label(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
    dlat = to_lat - from_lat
    dlon = to_lon - from_lon
    if abs(dlon) >= abs(dlat):
        return "eastbound" if dlon > 0 else "westbound"
    return "northbound" if dlat > 0 else "southbound"


def parse_speed(tags: dict[str, str], highway: str) -> int:
    raw_speed = tags.get("maxspeed", "")
    match = re.search(r"(\d+)", raw_speed)
    if match:
        return int(match.group(1))
    return DEFAULT_SPEEDS.get(highway, 30)


def parse_lanes(tags: dict[str, str], highway: str) -> int:
    raw_lanes = tags.get("lanes", "")
    match = re.search(r"(\d+)", raw_lanes)
    if match:
        return max(1, int(match.group(1)))
    return DEFAULT_LANES.get(highway, 1)


def road_name_from(tags: dict[str, str], highway: str) -> str:
    if tags.get("name:en"):
        return tags["name:en"]
    if tags.get("name"):
        return tags["name"]
    return highway.replace("_", " ").title()


def is_green_corridor(road_name: str) -> bool:
    normalized = normalize_name(road_name)
    return any(keyword in normalized for keyword in GREEN_CORRIDORS)


def build_metrics(
    *,
    highway: str,
    road_name: str,
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    length_km: float,
    speed_limit: int,
    lanes: int,
    segment_seed: float,
) -> dict[str, str]:
    midpoint_lat = (from_lat + to_lat) / 2
    midpoint_lon = (from_lon + to_lon) / 2
    center_score = centrality(midpoint_lat, midpoint_lon)
    green_corridor = is_green_corridor(road_name)
    variation = stable_wave(segment_seed, midpoint_lat * 100, midpoint_lon * 100)

    if highway in {"motorway", "trunk", "motorway_link", "trunk_link"}:
        volume = 42000 + 23000 * variation + 12000 * center_score
        heavy_share = 0.16 + 0.05 * variation
        peak_sensitivity = 1.15 + 0.12 * variation
        signal_count = 0 if highway.startswith("motorway") else 1
        green_base = 0.78
    elif highway in {"primary", "primary_link"}:
        volume = 25000 + 18000 * variation + 12000 * center_score
        heavy_share = 0.10 + 0.03 * variation
        peak_sensitivity = 1.05 + 0.14 * variation
        signal_count = 1 if length_km < 0.5 else 2
        green_base = 0.70
    elif highway in {"secondary", "secondary_link"}:
        volume = 16000 + 12000 * variation + 9000 * center_score
        heavy_share = 0.08 + 0.02 * variation
        peak_sensitivity = 0.92 + 0.16 * variation
        signal_count = 2
        green_base = 0.60
    elif highway in {"tertiary", "tertiary_link"}:
        volume = 9000 + 8000 * variation + 6000 * center_score
        heavy_share = 0.05 + 0.02 * variation
        peak_sensitivity = 0.84 + 0.14 * variation
        signal_count = 2
        green_base = 0.52
    else:
        volume = 5000 + 4500 * variation + 3500 * center_score
        heavy_share = 0.03 + 0.015 * variation
        peak_sensitivity = 0.72 + 0.12 * variation
        signal_count = 1
        green_base = 0.44

    if green_corridor:
        green_base += 0.16
    if "ring road" in normalize_name(road_name):
        volume *= 1.15
        green_base += 0.08
        signal_count = min(signal_count, 1)

    if length_km < 0.045:
        signal_count = 0
    elif length_km < 0.120:
        signal_count = min(signal_count, 1)

    base_congestion = clamp(0.18 + volume / (max(lanes, 1) * 50000), 0.16, 0.88)
    green_wave_score = clamp(green_base - 0.10 * center_score + 0.05 * variation, 0.30, 0.96)
    free_flow_speed = speed_limit * (0.80 + 0.08 * variation)
    junction_factor = clamp(length_km / 0.18, 0.0, 1.0)
    avg_delay = clamp(signal_count * (6 + (1 - green_wave_score) * 17 + base_congestion * 10) * junction_factor, 0, 40)

    east_gain = (to_lon - from_lon) * 170
    south_gain = (from_lat - to_lat) * 240
    grade_percent = clamp((east_gain + south_gain) / max(length_km * 1000, 150) * 100, -4.0, 5.5)
    eco_factor = clamp(
        0.88
        + base_congestion * 0.52
        + heavy_share * 0.95
        + max(grade_percent, 0) * 0.05
        + signal_count * (1 - green_wave_score) * 0.08,
        0.80,
        1.95,
    )

    highway_bias = {
        "motorway": 0.08,
        "motorway_link": 0.04,
        "trunk": 0.10,
        "trunk_link": 0.05,
        "primary": 0.12,
        "primary_link": 0.08,
        "secondary": 0.10,
        "secondary_link": 0.06,
        "tertiary": 0.06,
        "tertiary_link": 0.04,
    }.get(highway, 0.02)

    congestion_profile: list[float] = []
    green_profile: list[float] = []
    curb_activity_profile: list[float] = []
    for hour in PROFILE_BUCKETS:
        bucket_hour = hour + 1.5
        morning_peak = gaussian_peak(bucket_hour, 8.0, 2.1)
        evening_peak = gaussian_peak(bucket_hour, 17.5, 2.5)
        midday_peak = gaussian_peak(bucket_hour, 12.5, 3.0)
        overnight = gaussian_peak(bucket_hour, 2.0, 2.5)

        congestion_multiplier = clamp(
            0.52
            + morning_peak * (0.44 + 0.18 * center_score + highway_bias)
            + evening_peak * (0.50 + 0.16 * center_score + highway_bias)
            + midday_peak * (0.10 + 0.08 * variation)
            - overnight * (0.18 + 0.05 * variation)
            + (base_congestion - 0.35) * 0.30,
            0.35,
            1.72,
        )

        green_hour = clamp(
            green_wave_score
            - morning_peak * (0.06 + 0.03 * center_score)
            - evening_peak * (0.08 + 0.03 * center_score)
            + overnight * 0.04
            + (0.03 if green_corridor else 0.0),
            0.24,
            0.98,
        )

        curb_activity = clamp(
            0.08
            + midday_peak * (0.26 + 0.18 * center_score)
            + evening_peak * (0.16 + 0.10 * center_score)
            + (0.10 if highway in {"residential", "living_street", "tertiary", "unclassified"} else 0.03)
            - overnight * 0.10,
            0.02,
            0.92,
        )

        congestion_profile.append(congestion_multiplier)
        green_profile.append(green_hour)
        curb_activity_profile.append(curb_activity)

    weekday_volume_profile = [
        1.05,
        1.08,
        1.09,
        1.07,
        1.10,
        0.86,
        0.73,
    ]

    return {
        "road_class": highway,
        "lanes": str(lanes),
        "speed_limit_kmh": str(speed_limit),
        "free_flow_speed_kmh": f"{free_flow_speed:.1f}",
        "base_congestion_index": f"{base_congestion:.3f}",
        "peak_sensitivity": f"{peak_sensitivity:.3f}",
        "traffic_volume_aadt": str(int(round(volume))),
        "heavy_vehicle_share": f"{heavy_share:.3f}",
        "signal_count": str(signal_count),
        "green_wave_score": f"{green_wave_score:.3f}",
        "green_corridor": "1" if green_corridor and green_wave_score >= 0.68 else "0",
        "avg_intersection_delay_s": f"{avg_delay:.1f}",
        "grade_percent": f"{grade_percent:.2f}",
        "eco_factor": f"{eco_factor:.3f}",
        "congestion_profile_3h": serialize_profile(congestion_profile),
        "green_profile_3h": serialize_profile(green_profile),
        "curb_activity_profile_3h": serialize_profile(curb_activity_profile),
        "weekday_volume_profile": serialize_profile(weekday_volume_profile),
    }


def node_id(lat: float, lon: float) -> str:
    return f"N{lat:.6f}_{lon:.6f}"


def node_name(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


def append_segment(
    rows: list[dict[str, str]],
    *,
    way_id: int,
    segment_index: int,
    road_name: str,
    highway: str,
    from_point: dict[str, float],
    to_point: dict[str, float],
    speed_limit: int,
    lanes: int,
) -> None:
    from_lat = float(from_point["lat"])
    from_lon = float(from_point["lon"])
    to_lat = float(to_point["lat"])
    to_lon = float(to_point["lon"])
    length_km = haversine_km(from_lat, from_lon, to_lat, to_lon)
    if length_km <= 0.001:
        return

    direction = direction_label(from_lat, from_lon, to_lat, to_lon)
    metrics = build_metrics(
        highway=highway,
        road_name=road_name,
        from_lat=from_lat,
        from_lon=from_lon,
        to_lat=to_lat,
        to_lon=to_lon,
        length_km=length_km,
        speed_limit=speed_limit,
        lanes=lanes,
        segment_seed=way_id * 0.01 + segment_index,
    )

    rows.append(
        {
            "segment_id": f"W{way_id}_{segment_index:03d}_{direction[0].upper()}",
            "from_node": node_id(from_lat, from_lon),
            "from_name": node_name(from_lat, from_lon),
            "from_lat": f"{from_lat:.6f}",
            "from_lon": f"{from_lon:.6f}",
            "to_node": node_id(to_lat, to_lon),
            "to_name": node_name(to_lat, to_lon),
            "to_lat": f"{to_lat:.6f}",
            "to_lon": f"{to_lon:.6f}",
            "road_name": road_name,
            "direction": direction,
            "length_km": f"{length_km:.3f}",
            **metrics,
        }
    )


def load_source_data(source_path: Path = SOURCE_PATH) -> dict[str, object]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")
    return json.loads(source_path.read_text())


def generate_rows_from_elements(elements: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for way in elements:
        tags = way.get("tags", {})
        highway = tags.get("highway", "")
        if highway in EXCLUDED_HIGHWAYS or "geometry" not in way:
            continue

        geometry = way["geometry"]
        if len(geometry) < 2:
            continue

        road_name = road_name_from(tags, highway)
        speed_limit = parse_speed(tags, highway)
        lanes = parse_lanes(tags, highway)

        oneway = tags.get("oneway", "").lower()
        reverse_only = oneway in {"-1", "reverse"}
        bidirectional = oneway not in {"yes", "1", "true", "-1", "reverse"}
        ordered_geometry = list(reversed(geometry)) if reverse_only else geometry

        for index, (from_point, to_point) in enumerate(zip(ordered_geometry, ordered_geometry[1:]), start=1):
            append_segment(
                rows,
                way_id=int(way["id"]),
                segment_index=index,
                road_name=road_name,
                highway=highway,
                from_point=from_point,
                to_point=to_point,
                speed_limit=speed_limit,
                lanes=lanes,
            )
            if bidirectional:
                append_segment(
                    rows,
                    way_id=int(way["id"]),
                    segment_index=index + 500,
                    road_name=road_name,
                    highway=highway,
                    from_point=to_point,
                    to_point=from_point,
                    speed_limit=speed_limit,
                    lanes=lanes,
                )

    neighbor_map: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        neighbor_map[row["from_node"]].add(row["to_node"])
        neighbor_map[row["to_node"]].add(row["from_node"])

    for row in rows:
        degree = max(len(neighbor_map[row["from_node"]]), len(neighbor_map[row["to_node"]]))
        signal_count = int(row["signal_count"])
        avg_delay = float(row["avg_intersection_delay_s"])

        if degree <= 2:
            signal_count = 0
            avg_delay = 0.0
        elif degree == 3:
            signal_count = min(signal_count, 1)
            avg_delay *= 0.35
        else:
            avg_delay *= 0.55

        if row["road_class"] in {"motorway", "motorway_link", "trunk", "trunk_link"}:
            avg_delay *= 0.15
            signal_count = 0

        row["signal_count"] = str(signal_count)
        row["avg_intersection_delay_s"] = f"{avg_delay:.1f}"

    return rows


def generate_rows(source_path: Path = SOURCE_PATH) -> list[dict[str, str]]:
    data = load_source_data(source_path)
    elements = data.get("elements", [])
    if not isinstance(elements, list) or not elements:
        raise ValueError("Source OSM file does not contain any road elements.")
    rows = generate_rows_from_elements(elements)
    if not rows:
        raise ValueError("Generated road network is empty.")
    return rows


def main() -> None:
    setup_logging(APP_CONFIG.log_level)
    LOGGER.info("Generating route network from %s", SOURCE_PATH)
    rows = generate_rows(SOURCE_PATH)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s road segments to %s", len(rows), OUTPUT_PATH)
    print(f"Wrote {len(rows)} road segments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
