from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_models import (
    CONGESTION_CLASS_NAMES,
    DIRECTIONS,
    ROAD_CLASSES,
    feature_vector,
    modeled_dynamic_congestion,
    segment_feature_map,
)


RANDOM_SEED = 42
SAMPLE_SIZE = 24000
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


def build_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int, dict[str, float]]:
    all_rows: list[dict[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        all_rows.extend(reader)

    total_rows = len(all_rows)
    rng = random.Random(RANDOM_SEED)
    sampled_rows = rng.sample(all_rows, min(SAMPLE_SIZE, total_rows))
    sampled_examples = [(row, rng.choice(TRAINING_HOURS)) for row in sampled_rows]

    congestion_scores = np.array(
        [modeled_dynamic_congestion(row, hour=hour) for row, hour in sampled_examples],
        dtype=np.float64,
    )
    low_threshold = float(np.quantile(congestion_scores, 1.0 / 3.0))
    high_threshold = float(np.quantile(congestion_scores, 2.0 / 3.0))

    x_matrix: list[list[float]] = []
    y_vector: list[int] = []

    for (row, hour), score in zip(sampled_examples, congestion_scores, strict=True):
        feature_map = segment_feature_map(row, hour=hour)
        x_matrix.append(feature_vector(tuple(FEATURE_NAMES), feature_map))
        if score <= low_threshold:
            label = 0
        elif score <= high_threshold:
            label = 1
        else:
            label = 2
        y_vector.append(label)

    thresholds = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    return np.array(x_matrix, dtype=np.float64), np.array(y_vector, dtype=np.int64), FEATURE_NAMES, total_rows, thresholds


def top_features(model: RandomForestClassifier, feature_names: list[str], top_n: int = 10) -> list[dict[str, float | str]]:
    pairs = list(zip(feature_names, model.feature_importances_.tolist()))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(score)} for name, score in pairs[:top_n]]


def confusion_matrix_by_label(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        true_label: {
            predicted_label: int(matrix[true_index, predicted_index])
            for predicted_index, predicted_label in enumerate(CONGESTION_CLASS_NAMES)
        }
        for true_index, true_label in enumerate(CONGESTION_CLASS_NAMES)
    }


def main() -> None:
    dataset_path = PROJECT_ROOT / "sofia_route_network.csv"
    output_dir = Path(__file__).resolve().parent
    model_path = output_dir / "congestion_random_forest.joblib"
    results_path = output_dir / "congestion_random_forest_results.json"

    x_matrix, y_vector, feature_names, total_rows, thresholds = build_dataset(dataset_path)
    train_x, test_x, train_y, test_y = train_test_split(
        x_matrix,
        y_vector,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_vector,
    )

    model = RandomForestClassifier(
        n_estimators=64,
        max_depth=14,
        min_samples_leaf=3,
        max_samples=0.60,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision_macro": float(precision_score(test_y, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_y, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "sample_size": int(len(x_matrix)),
            "class_names": CONGESTION_CLASS_NAMES,
            "thresholds": thresholds,
            "hour_mode": "multi-hour",
            "training_hours": list(TRAINING_HOURS),
        },
        model_path,
    )

    label_distribution = {
        class_name: int((y_vector == index).sum())
        for index, class_name in enumerate(CONGESTION_CLASS_NAMES)
    }

    results = {
        "dataset": {
            "path": str(dataset_path),
            "source_rows": int(total_rows),
            "sampled_rows": int(len(x_matrix)),
            "feature_count": int(x_matrix.shape[1]),
            "target": "Three-class congestion band across departure hours 0-23",
            "thresholds": thresholds,
            "label_distribution": label_distribution,
            "hour_mode": "multi-hour",
            "training_hours": list(TRAINING_HOURS),
        },
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 64,
            "max_depth": 14,
            "min_samples_leaf": 3,
            "max_samples": 0.60,
            "class_weight": "balanced_subsample",
            "random_seed": RANDOM_SEED,
        },
        "split": {
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_by_label(test_y, predictions),
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
