from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROAD_CLASSES = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
)

DIRECTIONS = ("northbound", "southbound", "eastbound", "westbound")
CONGESTION_CLASS_NAMES = ("low", "medium", "high")

DEFAULT_CONGESTION_PROFILE = (0.38, 0.44, 0.88, 1.22, 1.00, 1.08, 1.28, 0.74)
DEFAULT_GREEN_PROFILE = (0.76, 0.78, 0.72, 0.66, 0.68, 0.64, 0.60, 0.74)
DEFAULT_CURB_ACTIVITY_PROFILE = (0.06, 0.05, 0.12, 0.24, 0.32, 0.28, 0.20, 0.10)
DEFAULT_WEEKDAY_VOLUME_PROFILE = (1.05, 1.08, 1.09, 1.07, 1.10, 0.86, 0.73)

MODEL_DIR = Path(__file__).resolve().parent / "ml_demo"
TRAVEL_TIME_MODEL_PATH = MODEL_DIR / "travel_time_random_forest.joblib"
CONGESTION_MODEL_PATH = MODEL_DIR / "congestion_random_forest.joblib"
GREEN_CORRIDOR_MODEL_PATH = MODEL_DIR / "green_corridor_logreg_model.npz"
OBSERVED_CORRECTION_MODEL_PATH = MODEL_DIR / "observed_time_correction_random_forest.joblib"

SEGMENT_MODEL_FEATURE_NAMES = (
    "from_lat",
    "from_lon",
    "to_lat",
    "to_lon",
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
    "green_corridor_flag",
    "avg_intersection_delay_s",
    "grade_percent",
    "eco_factor",
    *[f"road_class={value}" for value in ROAD_CLASSES],
    *[f"direction={value}" for value in DIRECTIONS],
    *[f"congestion_profile_3h[{index}]" for index in range(8)],
    *[f"green_profile_3h[{index}]" for index in range(8)],
    *[f"curb_activity_profile_3h[{index}]" for index in range(8)],
    *[f"weekday_volume_profile[{index}]" for index in range(7)],
    "hour_of_day",
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def profile_bucket(hour: int) -> int:
    return max(0, min(7, hour // 3))


def distance_to_center(lat: float, lon: float) -> float:
    center_lat = 42.6940
    center_lon = 23.3250
    return math.hypot(lat - center_lat, lon - center_lon)


def directional_peak_bias(from_lat: float, from_lon: float, to_lat: float, to_lon: float, hour: int) -> float:
    moving_inward = distance_to_center(to_lat, to_lon) < distance_to_center(from_lat, from_lon)
    if 7 <= hour <= 9 and moving_inward:
        return 1.12
    if 16 <= hour <= 18 and not moving_inward:
        return 1.14
    return 1.0


def parse_profile(raw: Any, expected_size: int, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        try:
            values = tuple(float(part) for part in raw.split("|"))
        except ValueError:
            return fallback
        return values if len(values) == expected_size else fallback
    if isinstance(raw, (list, tuple, np.ndarray)):
        try:
            values = tuple(float(part) for part in raw)
        except (TypeError, ValueError):
            return fallback
        return values if len(values) == expected_size else fallback
    return fallback


def segment_value(segment: Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def segment_feature_map(segment: Any, *, hour: int) -> dict[str, float]:
    road_class = str(segment_value(segment, "road_class", "unclassified"))
    direction = str(segment_value(segment, "direction", "northbound"))
    raw_green_corridor = segment_value(segment, "green_corridor", 0)
    if isinstance(raw_green_corridor, str):
        green_corridor_flag = 1.0 if raw_green_corridor == "1" else 0.0
    else:
        green_corridor_flag = 1.0 if raw_green_corridor else 0.0

    congestion_profile = parse_profile(
        segment_value(segment, "congestion_profile_3h"), 8, DEFAULT_CONGESTION_PROFILE
    )
    green_profile = parse_profile(segment_value(segment, "green_profile_3h"), 8, DEFAULT_GREEN_PROFILE)
    curb_activity_profile = parse_profile(
        segment_value(segment, "curb_activity_profile_3h"), 8, DEFAULT_CURB_ACTIVITY_PROFILE
    )
    weekday_volume_profile = parse_profile(
        segment_value(segment, "weekday_volume_profile"), 7, DEFAULT_WEEKDAY_VOLUME_PROFILE
    )

    feature_map = {
        "from_lat": float(segment_value(segment, "from_lat", 0.0)),
        "from_lon": float(segment_value(segment, "from_lon", 0.0)),
        "to_lat": float(segment_value(segment, "to_lat", 0.0)),
        "to_lon": float(segment_value(segment, "to_lon", 0.0)),
        "lanes": float(segment_value(segment, "lanes", 1.0)),
        "length_km": float(segment_value(segment, "length_km", 0.0)),
        "speed_limit_kmh": float(segment_value(segment, "speed_limit_kmh", 30.0)),
        "free_flow_speed_kmh": float(segment_value(segment, "free_flow_speed_kmh", 24.0)),
        "base_congestion_index": float(segment_value(segment, "base_congestion_index", 0.3)),
        "peak_sensitivity": float(segment_value(segment, "peak_sensitivity", 1.0)),
        "traffic_volume_aadt": float(segment_value(segment, "traffic_volume_aadt", 0.0)),
        "heavy_vehicle_share": float(segment_value(segment, "heavy_vehicle_share", 0.0)),
        "signal_count": float(segment_value(segment, "signal_count", 0.0)),
        "green_wave_score": float(segment_value(segment, "green_wave_score", 0.5)),
        "green_corridor_flag": green_corridor_flag,
        "green_corridor": green_corridor_flag,
        "avg_intersection_delay_s": float(segment_value(segment, "avg_intersection_delay_s", 0.0)),
        "grade_percent": float(segment_value(segment, "grade_percent", 0.0)),
        "eco_factor": float(segment_value(segment, "eco_factor", 1.0)),
        "hour_of_day": float(hour),
    }

    for value in ROAD_CLASSES:
        feature_map[f"road_class={value}"] = 1.0 if road_class == value else 0.0
    for value in DIRECTIONS:
        feature_map[f"direction={value}"] = 1.0 if direction == value else 0.0
    for index, value in enumerate(congestion_profile):
        feature_map[f"congestion_profile_3h[{index}]"] = value
    for index, value in enumerate(green_profile):
        feature_map[f"green_profile_3h[{index}]"] = value
    for index, value in enumerate(curb_activity_profile):
        feature_map[f"curb_activity_profile_3h[{index}]"] = value
    for index, value in enumerate(weekday_volume_profile):
        feature_map[f"weekday_volume_profile[{index}]"] = value
    return feature_map


def feature_vector(feature_names: tuple[str, ...], feature_map: dict[str, float]) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def modeled_dynamic_congestion(segment: Any, *, hour: int) -> float:
    feature_map = segment_feature_map(segment, hour=hour)
    bucket_index = profile_bucket(hour)
    peak_load = (
        feature_map[f"congestion_profile_3h[{bucket_index}]"]
        * feature_map["peak_sensitivity"]
        * directional_peak_bias(
            feature_map["from_lat"],
            feature_map["from_lon"],
            feature_map["to_lat"],
            feature_map["to_lon"],
            hour,
        )
    )
    return clamp(feature_map["base_congestion_index"] * peak_load, 0.10, 0.96)


def modeled_segment_travel_time(segment: Any, *, hour: int, stop_go_penalty: float = 0.22) -> float:
    feature_map = segment_feature_map(segment, hour=hour)
    bucket_index = profile_bucket(hour)
    dynamic_congestion = modeled_dynamic_congestion(segment, hour=hour)
    current_speed = max(12.0, feature_map["free_flow_speed_kmh"] * (1.0 - 0.50 * dynamic_congestion))
    current_green_score = clamp(
        (feature_map["green_wave_score"] * 0.45) + (feature_map[f"green_profile_3h[{bucket_index}]"] * 0.55),
        0.20,
        0.98,
    )
    signal_delay = feature_map["avg_intersection_delay_s"] * (0.60 + dynamic_congestion * 0.92) * (
        1.08 - current_green_score * 0.34
    )

    curbside_delay = 0.0
    road_class = next(
        (
            road_class_name
            for road_class_name in ROAD_CLASSES
            if feature_map.get(f"road_class={road_class_name}", 0.0) > 0.5
        ),
        "unclassified",
    )
    if stop_go_penalty >= 0.22 and road_class not in {"motorway", "motorway_link"}:
        curb_activity = feature_map[f"curb_activity_profile_3h[{bucket_index}]"]
        curbside_delay = curb_activity * (3.5 + feature_map["signal_count"] * 1.4) * (0.8 + stop_go_penalty)

    return feature_map["length_km"] / current_speed * 60.0 + signal_delay / 60.0 + curbside_delay / 60.0


@dataclass(frozen=True, slots=True)
class RandomForestBundle:
    model: Any
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogisticBundle:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteMLPrediction:
    segment_count: int
    predicted_time_min: float | None
    predicted_time_delta_min: float | None
    congestion_label: str | None
    congestion_mix: dict[str, float]
    high_congestion_share: float | None
    average_green_corridor_probability: float | None


@dataclass(frozen=True, slots=True)
class MLModelStatus:
    travel_time_available: bool
    congestion_available: bool
    green_corridor_available: bool
    observed_correction_available: bool

    @property
    def any_available(self) -> bool:
        return (
            self.travel_time_available
            or self.congestion_available
            or self.green_corridor_available
            or self.observed_correction_available
        )


@dataclass(frozen=True, slots=True)
class MLSegmentScore:
    predicted_time_min: float | None
    congestion_label: str | None
    high_congestion_probability: float | None
    green_corridor_probability: float | None
    observed_correction_factor: float | None


@lru_cache(maxsize=1)
def load_travel_time_bundle() -> RandomForestBundle | None:
    if not TRAVEL_TIME_MODEL_PATH.exists():
        return None
    payload = joblib.load(TRAVEL_TIME_MODEL_PATH)
    return RandomForestBundle(
        model=payload["model"],
        feature_names=tuple(payload["feature_names"]),
        class_names=(),
    )


@lru_cache(maxsize=1)
def load_congestion_bundle() -> RandomForestBundle | None:
    if not CONGESTION_MODEL_PATH.exists():
        return None
    payload = joblib.load(CONGESTION_MODEL_PATH)
    class_names = tuple(payload.get("class_names", CONGESTION_CLASS_NAMES))
    return RandomForestBundle(
        model=payload["model"],
        feature_names=tuple(payload["feature_names"]),
        class_names=class_names,
    )


@lru_cache(maxsize=1)
def load_green_corridor_bundle() -> LogisticBundle | None:
    if not GREEN_CORRIDOR_MODEL_PATH.exists():
        return None
    payload = np.load(GREEN_CORRIDOR_MODEL_PATH, allow_pickle=True)
    return LogisticBundle(
        weights=np.asarray(payload["weights"], dtype=np.float64),
        bias=float(np.asarray(payload["bias"], dtype=np.float64)[0]),
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
        feature_names=tuple(str(value) for value in payload["feature_names"].tolist()),
    )


@lru_cache(maxsize=1)
def load_observed_correction_bundle() -> RandomForestBundle | None:
    if not OBSERVED_CORRECTION_MODEL_PATH.exists():
        return None
    payload = joblib.load(OBSERVED_CORRECTION_MODEL_PATH)
    return RandomForestBundle(
        model=payload["model"],
        feature_names=tuple(payload["feature_names"]),
        class_names=(),
    )


def ml_model_status() -> MLModelStatus:
    return MLModelStatus(
        travel_time_available=load_travel_time_bundle() is not None,
        congestion_available=load_congestion_bundle() is not None,
        green_corridor_available=load_green_corridor_bundle() is not None,
        observed_correction_available=load_observed_correction_bundle() is not None,
    )


@lru_cache(maxsize=24)
def batch_predict_network_scores(dataset_path: str, hour: int) -> dict[str, MLSegmentScore]:
    travel_time_bundle = load_travel_time_bundle()
    congestion_bundle = load_congestion_bundle()
    green_bundle = load_green_corridor_bundle()
    observed_correction_bundle = load_observed_correction_bundle()
    if (
        travel_time_bundle is None
        and congestion_bundle is None
        and green_bundle is None
        and observed_correction_bundle is None
    ):
        return {}

    scores: dict[str, MLSegmentScore] = {}
    chunk_segment_ids: list[str] = []
    travel_rows: list[list[float]] = []
    congestion_rows: list[list[float]] = []
    green_rows: list[list[float]] = []
    observed_rows: list[list[float]] = []
    chunk_size = 4096

    def flush_chunk() -> None:
        if not chunk_segment_ids:
            return

        predicted_times: np.ndarray | None = None
        high_probabilities: np.ndarray | None = None
        congestion_labels: list[str | None] = [None] * len(chunk_segment_ids)
        green_probabilities: np.ndarray | None = None
        observed_corrections: np.ndarray | None = None

        if travel_time_bundle is not None and travel_rows:
            travel_x = np.asarray(travel_rows, dtype=np.float64)
            predicted_times = np.maximum(travel_time_bundle.model.predict(travel_x), 0.01)

        if congestion_bundle is not None and congestion_rows:
            congestion_x = np.asarray(congestion_rows, dtype=np.float64)
            probability_matrix = np.asarray(congestion_bundle.model.predict_proba(congestion_x), dtype=np.float64)
            class_names = congestion_bundle.class_names or CONGESTION_CLASS_NAMES
            best_indices = probability_matrix.argmax(axis=1)
            congestion_labels[:] = [str(class_names[index]).title() for index in best_indices]
            high_index = next((index for index, class_name in enumerate(class_names) if class_name == "high"), None)
            if high_index is not None:
                high_probabilities = probability_matrix[:, high_index]

        if green_bundle is not None and green_rows:
            green_x = np.asarray(green_rows, dtype=np.float64)
            standardized = (green_x - green_bundle.mean) / green_bundle.std
            green_probabilities = sigmoid(standardized @ green_bundle.weights + green_bundle.bias)
        if observed_correction_bundle is not None and observed_rows:
            observed_x = np.asarray(observed_rows, dtype=np.float64)
            observed_corrections = np.clip(
                np.asarray(observed_correction_bundle.model.predict(observed_x), dtype=np.float64),
                0.78,
                1.42,
            )

        for index, segment_id in enumerate(chunk_segment_ids):
            scores[segment_id] = MLSegmentScore(
                predicted_time_min=float(predicted_times[index]) if predicted_times is not None else None,
                congestion_label=congestion_labels[index],
                high_congestion_probability=float(high_probabilities[index]) if high_probabilities is not None else None,
                green_corridor_probability=float(green_probabilities[index]) if green_probabilities is not None else None,
                observed_correction_factor=float(observed_corrections[index]) if observed_corrections is not None else None,
            )

        chunk_segment_ids.clear()
        travel_rows.clear()
        congestion_rows.clear()
        green_rows.clear()
        observed_rows.clear()

    with Path(dataset_path).open(newline="", encoding="utf-8") as handle:
        import csv

        reader = csv.DictReader(handle)
        for row in reader:
            feature_map = segment_feature_map(row, hour=hour)
            chunk_segment_ids.append(row["segment_id"])
            if travel_time_bundle is not None:
                travel_rows.append(feature_vector(travel_time_bundle.feature_names, feature_map))
            if congestion_bundle is not None:
                congestion_rows.append(feature_vector(congestion_bundle.feature_names, feature_map))
            if green_bundle is not None:
                green_rows.append(feature_vector(green_bundle.feature_names, feature_map))
            if observed_correction_bundle is not None:
                observed_rows.append(feature_vector(observed_correction_bundle.feature_names, feature_map))
            if len(chunk_segment_ids) >= chunk_size:
                flush_chunk()

    flush_chunk()
    return scores


def predict_route_ml_summary(route: Any, *, hour: int) -> RouteMLPrediction | None:
    segments = getattr(route, "segments", None)
    if not segments:
        return None

    feature_maps = [segment_feature_map(segment.edge, hour=hour) for segment in segments]

    predicted_time_min: float | None = None
    predicted_time_delta_min: float | None = None
    congestion_label: str | None = None
    congestion_mix: dict[str, float] = {}
    high_congestion_share: float | None = None
    average_green_corridor_probability: float | None = None

    travel_time_bundle = load_travel_time_bundle()
    if travel_time_bundle is not None:
        travel_x = np.array(
            [feature_vector(travel_time_bundle.feature_names, feature_map) for feature_map in feature_maps],
            dtype=np.float64,
        )
        segment_predictions = np.maximum(travel_time_bundle.model.predict(travel_x), 0.0)
        predicted_time_min = float(segment_predictions.sum())
        predicted_time_delta_min = predicted_time_min - float(getattr(route, "total_time_min", 0.0))

    congestion_bundle = load_congestion_bundle()
    if congestion_bundle is not None:
        congestion_x = np.array(
            [feature_vector(congestion_bundle.feature_names, feature_map) for feature_map in feature_maps],
            dtype=np.float64,
        )
        probability_matrix = np.asarray(congestion_bundle.model.predict_proba(congestion_x), dtype=np.float64)
        average_probabilities = probability_matrix.mean(axis=0)
        class_names = congestion_bundle.class_names or CONGESTION_CLASS_NAMES
        congestion_mix = {
            str(class_name): float(average_probabilities[index])
            for index, class_name in enumerate(class_names)
        }
        congestion_label = str(class_names[int(np.argmax(average_probabilities))]).title()
        high_index = next((index for index, class_name in enumerate(class_names) if class_name == "high"), None)
        if high_index is not None:
            high_congestion_share = float(probability_matrix[:, high_index].mean())

    green_bundle = load_green_corridor_bundle()
    if green_bundle is not None:
        green_x = np.array(
            [feature_vector(green_bundle.feature_names, feature_map) for feature_map in feature_maps],
            dtype=np.float64,
        )
        standardized = (green_x - green_bundle.mean) / green_bundle.std
        probabilities = sigmoid(standardized @ green_bundle.weights + green_bundle.bias)
        average_green_corridor_probability = float(probabilities.mean())

    return RouteMLPrediction(
        segment_count=len(segments),
        predicted_time_min=predicted_time_min,
        predicted_time_delta_min=predicted_time_delta_min,
        congestion_label=congestion_label,
        congestion_mix=congestion_mix,
        high_congestion_share=high_congestion_share,
        average_green_corridor_probability=average_green_corridor_probability,
    )
