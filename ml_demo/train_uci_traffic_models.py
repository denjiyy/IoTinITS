from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

DATASET_FILENAME = "metro_interstate_traffic_volume.csv.gz"
DATASET_SOURCE_URL = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
DATASET_DOI = "10.24432/C5X60B"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CITATION = (
    "Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B"
)

TARGET_FIELD = "traffic_volume"
CLASS_NAMES = ("low", "medium", "high")

CATEGORICAL_FIELDS = ("holiday", "weather_main", "weather_description")
NUMERIC_FIELDS = (
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "dayofweek",
    "month",
    "year",
    "day",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)
FEATURE_FIELDS = (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    return frame


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame.copy()
    feature_frame["date_time"] = pd.to_datetime(feature_frame["date_time"])
    feature_frame = feature_frame.sort_values("date_time").reset_index(drop=True)

    feature_frame["holiday"] = feature_frame["holiday"].fillna("None").astype(str)
    feature_frame["weather_main"] = feature_frame["weather_main"].fillna("Unknown").astype(str)
    feature_frame["weather_description"] = feature_frame["weather_description"].fillna("Unknown").astype(str)

    feature_frame["hour"] = feature_frame["date_time"].dt.hour
    feature_frame["dayofweek"] = feature_frame["date_time"].dt.dayofweek
    feature_frame["month"] = feature_frame["date_time"].dt.month
    feature_frame["year"] = feature_frame["date_time"].dt.year
    feature_frame["day"] = feature_frame["date_time"].dt.day
    feature_frame["is_weekend"] = (feature_frame["dayofweek"] >= 5).astype(int)

    feature_frame["hour_sin"] = np.sin(2.0 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["hour_cos"] = np.cos(2.0 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["month_sin"] = np.sin(2.0 * np.pi * feature_frame["month"] / 12.0)
    feature_frame["month_cos"] = np.cos(2.0 * np.pi * feature_frame["month"] / 12.0)
    return feature_frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = len(frame)
    train_end = int(total_rows * TRAIN_RATIO)
    valid_end = int(total_rows * (TRAIN_RATIO + VALID_RATIO))
    train_frame = frame.iloc[:train_end].copy()
    valid_frame = frame.iloc[train_end:valid_end].copy()
    test_frame = frame.iloc[valid_end:].copy()
    return train_frame, valid_frame, test_frame


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FIELDS),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL_FIELDS),
            ),
        ]
    )


def regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=2.0)),
        ]
    )


def regression_baseline_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", DummyRegressor(strategy="median")),
        ]
    )


def classification_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2500, solver="lbfgs", random_state=RANDOM_SEED)),
        ]
    )


def classification_baseline_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2_score": float(r2_score(y_true, predictions)),
    }


def traffic_band_labels(values: pd.Series, low_threshold: float, high_threshold: float) -> np.ndarray:
    raw = values.to_numpy(dtype=np.float64)
    return np.where(raw <= low_threshold, 0, np.where(raw <= high_threshold, 1, 2))


def classification_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }


def split_summary(frame: pd.DataFrame) -> dict[str, str | int]:
    return {
        "rows": int(len(frame)),
        "start": frame["date_time"].min().isoformat(),
        "end": frame["date_time"].max().isoformat(),
    }


def top_linear_features(feature_names: list[str], coefficients: np.ndarray, *, top_n: int = 10) -> list[dict[str, float | str]]:
    pairs = sorted(zip(feature_names, coefficients.tolist()), key=lambda pair: abs(pair[1]), reverse=True)
    return [{"feature": name, "coefficient": float(value)} for name, value in pairs[:top_n]]


def confusion_matrix_by_label(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1, 2])
    return {
        true_label: {
            predicted_label: int(matrix[true_index, predicted_index])
            for predicted_index, predicted_label in enumerate(CLASS_NAMES)
        }
        for true_index, true_label in enumerate(CLASS_NAMES)
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    dataset_path = output_dir / "data" / DATASET_FILENAME
    regression_model_path = output_dir / "uci_traffic_volume_ridge.joblib"
    classification_model_path = output_dir / "uci_traffic_congestion_logreg.joblib"
    results_path = output_dir / "uci_traffic_demo_results.json"

    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    raw_frame = load_dataset(dataset_path)
    feature_frame = build_feature_frame(raw_frame)
    train_frame, valid_frame, test_frame = chronological_split(feature_frame)

    regression_baseline = regression_baseline_pipeline()
    regression_model = regression_pipeline()

    regression_baseline.fit(train_frame[list(FEATURE_FIELDS)], train_frame[TARGET_FIELD])
    regression_model.fit(train_frame[list(FEATURE_FIELDS)], train_frame[TARGET_FIELD])

    regression_results: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, split_frame in (("validation", valid_frame), ("test", test_frame)):
        baseline_predictions = regression_baseline.predict(split_frame[list(FEATURE_FIELDS)])
        model_predictions = regression_model.predict(split_frame[list(FEATURE_FIELDS)])
        regression_results[split_name] = {
            "baseline_dummy_regressor": regression_metrics(split_frame[TARGET_FIELD], baseline_predictions),
            "ridge_regressor": regression_metrics(split_frame[TARGET_FIELD], model_predictions),
        }

    low_threshold = float(train_frame[TARGET_FIELD].quantile(1.0 / 3.0))
    high_threshold = float(train_frame[TARGET_FIELD].quantile(2.0 / 3.0))

    train_labels = traffic_band_labels(train_frame[TARGET_FIELD], low_threshold, high_threshold)
    valid_labels = traffic_band_labels(valid_frame[TARGET_FIELD], low_threshold, high_threshold)
    test_labels = traffic_band_labels(test_frame[TARGET_FIELD], low_threshold, high_threshold)

    classification_baseline = classification_baseline_pipeline()
    classification_model = classification_pipeline()

    classification_baseline.fit(train_frame[list(FEATURE_FIELDS)], train_labels)
    classification_model.fit(train_frame[list(FEATURE_FIELDS)], train_labels)

    classification_results: dict[str, dict[str, dict[str, float]]] = {}
    classification_confusion: dict[str, dict[str, int]] | None = None
    for split_name, split_frame, split_labels in (
        ("validation", valid_frame, valid_labels),
        ("test", test_frame, test_labels),
    ):
        baseline_predictions = classification_baseline.predict(split_frame[list(FEATURE_FIELDS)])
        model_predictions = classification_model.predict(split_frame[list(FEATURE_FIELDS)])
        classification_results[split_name] = {
            "baseline_dummy_classifier": classification_metrics(split_labels, baseline_predictions),
            "logistic_regression_classifier": classification_metrics(split_labels, model_predictions),
        }
        if split_name == "test":
            classification_confusion = confusion_matrix_by_label(split_labels, model_predictions)

    trained_preprocessor = regression_model.named_steps["preprocessor"]
    regression_feature_names = trained_preprocessor.get_feature_names_out().tolist()
    regression_coefficients = regression_model.named_steps["model"].coef_

    classification_preprocessor = classification_model.named_steps["preprocessor"]
    classification_feature_names = classification_preprocessor.get_feature_names_out().tolist()
    classification_coefficients = classification_model.named_steps["model"].coef_

    joblib.dump(
        {
            "pipeline": regression_model,
            "feature_fields": list(FEATURE_FIELDS),
            "target_field": TARGET_FIELD,
            "dataset_path": str(dataset_path),
        },
        regression_model_path,
    )
    joblib.dump(
        {
            "pipeline": classification_model,
            "feature_fields": list(FEATURE_FIELDS),
            "target_field": TARGET_FIELD,
            "class_names": list(CLASS_NAMES),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "dataset_path": str(dataset_path),
        },
        classification_model_path,
    )

    results = {
        "dataset": {
            "path": str(dataset_path),
            "source_url": DATASET_SOURCE_URL,
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "citation": DATASET_CITATION,
            "rows": int(len(feature_frame)),
            "original_columns": raw_frame.columns.tolist(),
            "feature_fields": list(FEATURE_FIELDS),
            "target": TARGET_FIELD,
            "task_family": "Tabular data",
            "description": "Hourly highway traffic volume with weather and holiday features for westbound I-94.",
        },
        "split": {
            "strategy": "Chronological 70/15/15 split to avoid temporal leakage",
            "train": split_summary(train_frame),
            "validation": split_summary(valid_frame),
            "test": split_summary(test_frame),
        },
        "regression_task": {
            "target": "traffic_volume",
            "baseline_model": "DummyRegressor(strategy='median')",
            "trained_model": "Ridge(alpha=2.0)",
            "validation_metrics": regression_results["validation"],
            "test_metrics": regression_results["test"],
            "top_coefficients": top_linear_features(regression_feature_names, regression_coefficients),
            "artifacts": {
                "model_path": str(regression_model_path),
            },
        },
        "classification_task": {
            "target": "traffic volume band (low / medium / high)",
            "class_names": list(CLASS_NAMES),
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "baseline_model": "DummyClassifier(strategy='most_frequent')",
            "trained_model": "LogisticRegression(max_iter=2500)",
            "validation_metrics": classification_results["validation"],
            "test_metrics": classification_results["test"],
            "confusion_matrix_test": classification_confusion,
            "top_coefficients_by_class": {
                class_name: top_linear_features(classification_feature_names, classification_coefficients[class_index], top_n=8)
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
            "artifacts": {
                "model_path": str(classification_model_path),
            },
        },
        "submission_summary": {
            "dataset": "UCI Metro Interstate Traffic Volume",
            "models": [
                "Ridge regression for hourly traffic volume prediction",
                "Logistic regression for low/medium/high traffic band classification",
            ],
            "result": {
                "ridge_test_r2": regression_results["test"]["ridge_regressor"]["r2_score"],
                "ridge_test_rmse": regression_results["test"]["ridge_regressor"]["rmse"],
                "logreg_test_accuracy": classification_results["test"]["logistic_regression_classifier"]["accuracy"],
                "logreg_test_f1_macro": classification_results["test"]["logistic_regression_classifier"]["f1_macro"],
            },
        },
    }

    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Saved regression pipeline to {regression_model_path}")
    print(f"Saved classification pipeline to {classification_model_path}")
    print(f"Saved results to {results_path}")
    print(json.dumps(results["submission_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
