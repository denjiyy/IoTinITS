from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = [
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
    "avg_intersection_delay_s",
    "grade_percent",
    "eco_factor",
]

ROAD_CLASSES = [
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
]

DIRECTIONS = ["northbound", "southbound", "eastbound", "westbound"]

PROFILE_FIELDS = {
    "congestion_profile_3h": 8,
    "green_profile_3h": 8,
    "curb_activity_profile_3h": 8,
    "weekday_volume_profile": 7,
}

TARGET_FIELD = "green_corridor"
RANDOM_SEED = 42
LEARNING_RATE = 0.08
EPOCHS = 200


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[list[float], float]] = []
    feature_names: list[str] = []

    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features: list[float] = []

            for field in NUMERIC_FIELDS:
                features.append(float(row[field]))
            if not feature_names:
                feature_names.extend(NUMERIC_FIELDS)

            road_class = row["road_class"]
            road_class_features = [1.0 if road_class == value else 0.0 for value in ROAD_CLASSES]
            features.extend(road_class_features)
            if len(feature_names) == len(NUMERIC_FIELDS):
                feature_names.extend(f"road_class={value}" for value in ROAD_CLASSES)

            direction = row["direction"]
            direction_features = [1.0 if direction == value else 0.0 for value in DIRECTIONS]
            features.extend(direction_features)
            if len(feature_names) == len(NUMERIC_FIELDS) + len(ROAD_CLASSES):
                feature_names.extend(f"direction={value}" for value in DIRECTIONS)

            for field, expected_size in PROFILE_FIELDS.items():
                values = [float(value) for value in row[field].split("|")]
                if len(values) != expected_size:
                    raise ValueError(f"Unexpected profile length for {field}: expected {expected_size}, got {len(values)}")
                features.extend(values)
                if len(feature_names) < len(features):
                    feature_names.extend(f"{field}[{index}]" for index in range(expected_size))

            target = float(row[TARGET_FIELD])
            rows.append((features, target))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)

    x_matrix = np.array([features for features, _ in rows], dtype=np.float64)
    y_vector = np.array([target for _, target in rows], dtype=np.float64)
    return x_matrix, y_vector, feature_names


def train_test_split(
    x_matrix: np.ndarray, y_vector: np.ndarray, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cut = int(len(x_matrix) * train_ratio)
    return x_matrix[:cut], x_matrix[cut:], y_vector[:cut], y_vector[cut:]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def train_logistic_regression(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(EPOCHS):
        logits = train_x @ weights + bias
        probabilities = sigmoid(logits)

        grad_weights = train_x.T @ (probabilities - train_y) / len(train_x)
        grad_bias = float((probabilities - train_y).mean())

        weights -= LEARNING_RATE * grad_weights
        bias -= LEARNING_RATE * grad_bias

        if epoch in {0, 49, 99, 149, EPOCHS - 1}:
            loss = -(
                train_y * np.log(probabilities + 1e-9)
                + (1.0 - train_y) * np.log(1.0 - probabilities + 1e-9)
            ).mean()
            history.append({"epoch": float(epoch), "loss": float(loss)})

    return weights, bias, history


def evaluate_model(test_x: np.ndarray, test_y: np.ndarray, weights: np.ndarray, bias: float) -> dict[str, float]:
    probabilities = sigmoid(test_x @ weights + bias)
    predictions = (probabilities >= 0.5).astype(np.float64)

    true_positive = float(((predictions == 1.0) & (test_y == 1.0)).sum())
    true_negative = float(((predictions == 0.0) & (test_y == 0.0)).sum())
    false_positive = float(((predictions == 1.0) & (test_y == 0.0)).sum())
    false_negative = float(((predictions == 0.0) & (test_y == 1.0)).sum())

    accuracy = float((predictions == test_y).mean())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1_score = 2.0 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def top_weight_features(feature_names: list[str], weights: np.ndarray, top_n: int = 8) -> dict[str, list[dict[str, float | str]]]:
    weighted = list(zip(feature_names, weights.tolist()))
    positives = sorted(weighted, key=lambda item: item[1], reverse=True)[:top_n]
    negatives = sorted(weighted, key=lambda item: item[1])[:top_n]
    return {
        "positive": [{"feature": name, "weight": float(weight)} for name, weight in positives],
        "negative": [{"feature": name, "weight": float(weight)} for name, weight in negatives],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "sofia_route_network.csv"
    output_dir = Path(__file__).resolve().parent
    model_path = output_dir / "green_corridor_logreg_model.npz"
    results_path = output_dir / "green_corridor_demo_results.json"

    x_matrix, y_vector, feature_names = load_dataset(dataset_path)
    train_x, test_x, train_y, test_y = train_test_split(x_matrix, y_vector)
    train_x, test_x, mean, std = standardize(train_x, test_x)
    weights, bias, history = train_logistic_regression(train_x, train_y)
    metrics = evaluate_model(test_x, test_y, weights, bias)
    feature_summary = top_weight_features(feature_names, weights)

    np.savez(
        model_path,
        weights=weights,
        bias=np.array([bias], dtype=np.float64),
        mean=mean,
        std=std,
        feature_names=np.array(feature_names, dtype=object),
    )

    results = {
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(x_matrix)),
            "feature_count": int(x_matrix.shape[1]),
            "train_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
            "positive_rate": float(y_vector.mean()),
            "task": "Tabular binary classification: predict whether a road segment belongs to a synchronized green corridor",
        },
        "model": {
            "type": "NumPy logistic regression",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
        },
        "metrics": metrics,
        "training_history": history,
        "top_features": feature_summary,
        "artifacts": {
            "model_path": str(model_path),
            "results_path": str(results_path),
        },
    }

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {results_path}")
    print(json.dumps(results["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
