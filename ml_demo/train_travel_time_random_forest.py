from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_models import DIRECTIONS, ROAD_CLASSES, feature_vector, modeled_segment_travel_time, segment_feature_map


RANDOM_SEED = 42
SAMPLE_SIZE = 24000
VEHICLE_STOP_GO_PENALTY = 0.22
TRAINING_HOURS = tuple(range(24))

FEATURE_NAMES = [
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
]


def build_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    all_rows: list[dict[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        all_rows.extend(reader)

    total_rows = len(all_rows)
    rng = random.Random(RANDOM_SEED)
    sampled_rows = rng.sample(all_rows, min(SAMPLE_SIZE, total_rows))
    sampled_examples = [(row, rng.choice(TRAINING_HOURS)) for row in sampled_rows]

    x_matrix: list[list[float]] = []
    y_vector: list[float] = []

    for row, hour in sampled_examples:
        feature_map = segment_feature_map(row, hour=hour)
        x_matrix.append(feature_vector(tuple(FEATURE_NAMES), feature_map))
        y_vector.append(modeled_segment_travel_time(row, hour=hour, stop_go_penalty=VEHICLE_STOP_GO_PENALTY))

    return np.array(x_matrix, dtype=np.float64), np.array(y_vector, dtype=np.float64), FEATURE_NAMES, total_rows


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def top_features(model: RandomForestRegressor, feature_names: list[str], top_n: int = 10) -> list[dict[str, float | str]]:
    pairs = list(zip(feature_names, model.feature_importances_.tolist()))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(score)} for name, score in pairs[:top_n]]


def main() -> None:
    dataset_path = PROJECT_ROOT / "sofia_route_network.csv"
    output_dir = Path(__file__).resolve().parent
    model_path = output_dir / "travel_time_random_forest.joblib"
    results_path = output_dir / "travel_time_random_forest_results.json"

    x_matrix, y_vector, feature_names, total_rows = build_dataset(dataset_path)
    train_x, test_x, train_y, test_y = train_test_split(
        x_matrix,
        y_vector,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    model = RandomForestRegressor(
        n_estimators=48,
        max_depth=12,
        min_samples_leaf=4,
        max_samples=0.55,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    metrics = {
        "mae_min": float(mean_absolute_error(test_y, predictions)),
        "rmse_min": rmse(test_y, predictions),
        "r2_score": float(r2_score(test_y, predictions)),
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "sample_size": int(len(x_matrix)),
            "hour_mode": "multi-hour",
            "training_hours": list(TRAINING_HOURS),
        },
        model_path,
    )

    results = {
        "dataset": {
            "path": str(dataset_path),
            "source_rows": int(total_rows),
            "sampled_rows": int(len(x_matrix)),
            "feature_count": int(x_matrix.shape[1]),
            "target": "Predicted travel_time_min across departure hours 0-23",
            "hour_mode": "multi-hour",
            "training_hours": list(TRAINING_HOURS),
        },
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 48,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_samples": 0.55,
            "random_seed": RANDOM_SEED,
        },
        "split": {
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
        },
        "metrics": metrics,
        "target_summary": {
            "mean_test_travel_time_min": float(np.mean(test_y)),
            "median_test_travel_time_min": float(np.median(test_y)),
        },
        "top_features": top_features(model, feature_names),
        "artifacts": {
            "model_path": str(model_path),
            "results_path": str(results_path),
        },
    }

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {results_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
