from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    return max(minimum, min(maximum, value))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = float(raw)
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str
    data_path: Path
    source_path: Path
    observation_log_path: Path
    map_height: int
    route_padding_deg: float
    max_delivery_stops: int
    log_level: str
    routing_backend: str
    tomtom_api_key: str
    tomtom_base_url: str
    tomtom_timeout_s: float
    calibration_min_segment_samples: int
    calibration_min_group_samples: int
    observation_min_match_ratio: float


def load_app_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent
    return AppConfig(
        app_name="Urban Route Optimizer",
        data_path=base_dir / "sofia_route_network.csv",
        source_path=base_dir / "sofia_osm_overpass.json",
        observation_log_path=base_dir / "live_route_observations.csv",
        map_height=_bounded_int("IOTINITS_MAP_HEIGHT", 660, 480, 1200),
        route_padding_deg=_bounded_float("IOTINITS_ROUTE_PADDING_DEG", 0.012, 0.004, 0.05),
        max_delivery_stops=_bounded_int("IOTINITS_MAX_DELIVERY_STOPS", 2, 0, 5),
        log_level=os.getenv("IOTINITS_LOG_LEVEL", "INFO").upper(),
        routing_backend=os.getenv("IOTINITS_ROUTING_BACKEND", "auto").lower(),
        tomtom_api_key=os.getenv("TOMTOM_API_KEY", ""),
        tomtom_base_url=os.getenv("TOMTOM_BASE_URL", "https://api.tomtom.com").rstrip("/"),
        tomtom_timeout_s=_bounded_float("TOMTOM_TIMEOUT_S", 12.0, 2.0, 60.0),
        calibration_min_segment_samples=_bounded_int("IOTINITS_CALIBRATION_MIN_SEGMENT_SAMPLES", 3, 1, 30),
        calibration_min_group_samples=_bounded_int("IOTINITS_CALIBRATION_MIN_GROUP_SAMPLES", 8, 2, 100),
        observation_min_match_ratio=_bounded_float("IOTINITS_OBSERVATION_MIN_MATCH_RATIO", 0.35, 0.10, 0.95),
    )
