from __future__ import annotations

import csv
import json
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

from ml_models import SEGMENT_MODEL_FEATURE_NAMES, feature_vector, segment_feature_map


RANDOM_SEED = 42
MIN_OBSERVATION_ROWS = 120


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _load_network_rows(dataset_path: Path) -> dict[str, dict[str, str]]:
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["segment_id"]: row for row in reader}


def build_dataset(
    dataset_path: Path,
    observation_path: Path,
    *,
    min_observation_rows: int = MIN_OBSERVATION_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    network_rows = _load_network_rows(dataset_path)
    x_matrix: list[list[float]] = []
    y_vector: list[float] = []
    sample_weight: list[float] = []
    route_signatures: set[str] = set()
    matched_rows = 0

    with observation_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segment_id = row.get("segment_id", "")
            network_row = network_rows.get(segment_id)
            if network_row is None:
                continue
            try:
                departure_hour = int(row["departure_hour_local"])
                observed_ratio = clamp(float(row["observed_to_modeled_ratio"]), 0.60, 1.90)
                matched_distance = max(float(row["matched_distance_km"]), 0.01)
            except (KeyError, ValueError):
                continue

            feature_map = segment_feature_map(network_row, hour=departure_hour)
            x_matrix.append(feature_vector(SEGMENT_MODEL_FEATURE_NAMES, feature_map))
            y_vector.append(observed_ratio)
            sample_weight.append(matched_distance)
            matched_rows += 1
            if row.get("route_signature"):
                route_signatures.add(row["route_signature"])

    if matched_rows < min_observation_rows:
        raise ValueError(
            f"Not enough observed rows to train a correction model. Need at least {min_observation_rows}, got {matched_rows}."
        )

    return (
        np.asarray(x_matrix, dtype=np.float64),
        np.asarray(y_vector, dtype=np.float64),
        np.asarray(sample_weight, dtype=np.float64),
        matched_rows,
        len(route_signatures),
    )


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))


def top_features(model: RandomForestRegressor, feature_names: tuple[str, ...], top_n: int = 10) -> list[dict[str, float | str]]:
    pairs = list(zip(feature_names, model.feature_importances_.tolist()))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(score)} for name, score in pairs[:top_n]]


def train_model(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[RandomForestRegressor, dict[str, float], dict[str, int]]:
    train_x, test_x, train_y, test_y, train_w, test_w = train_test_split(
        x_matrix,
        y_vector,
        sample_weight,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    model = RandomForestRegressor(
        n_estimators=64,
        max_depth=10,
        min_samples_leaf=2,
        max_samples=0.70,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train_y, sample_weight=train_w)
    predictions = np.clip(model.predict(test_x), 0.78, 1.42)

    metrics = {
        "mae_ratio": float(mean_absolute_error(test_y, predictions, sample_weight=test_w)),
        "rmse_ratio": weighted_rmse(test_y, predictions, test_w),
        "r2_score": float(r2_score(test_y, predictions, sample_weight=test_w)),
    }
    split = {
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
    }
    return model, metrics, split


def main() -> None:
    dataset_path = PROJECT_ROOT / "sofia_route_network.csv"
    observation_path = PROJECT_ROOT / "live_route_observations.csv"
    output_dir = Path(__file__).resolve().parent
    model_path = output_dir / "observed_time_correction_random_forest.joblib"
    results_path = output_dir / "observed_time_correction_random_forest_results.json"

    if not observation_path.exists():
        raise SystemExit(
            "No live_route_observations.csv file was found. Collect live TomTom routes first, then run this trainer."
        )

    x_matrix, y_vector, sample_weight, matched_rows, route_count = build_dataset(dataset_path, observation_path)
    model, metrics, split = train_model(x_matrix, y_vector, sample_weight)

    joblib.dump(
        {
            "model": model,
            "feature_names": list(SEGMENT_MODEL_FEATURE_NAMES),
            "observation_path": str(observation_path),
            "matched_rows": matched_rows,
            "route_count": route_count,
        },
        model_path,
    )

    results = {
        "dataset": {
            "path": str(dataset_path),
            "observation_path": str(observation_path),
            "matched_rows": matched_rows,
            "route_count": route_count,
            "feature_count": len(SEGMENT_MODEL_FEATURE_NAMES),
            "target": "Observed-to-modeled travel time ratio from live routing observations",
        },
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 64,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_samples": 0.70,
            "random_seed": RANDOM_SEED,
        },
        "split": split,
        "metrics": metrics,
        "target_summary": {
            "mean_ratio": float(np.mean(y_vector)),
            "median_ratio": float(np.median(y_vector)),
            "min_ratio": float(np.min(y_vector)),
            "max_ratio": float(np.max(y_vector)),
        },
        "top_features": top_features(model, SEGMENT_MODEL_FEATURE_NAMES),
        "artifacts": {
            "model_path": str(model_path),
            "results_path": str(results_path),
        },
    }

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved observed correction model to {model_path}")
    print(f"Saved metrics to {results_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
