from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ml_demo.train_observed_time_correction_random_forest import build_dataset, train_model


NETWORK_FIELDS = [
    "segment_id",
    "road_class",
    "direction",
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
    "green_corridor",
    "avg_intersection_delay_s",
    "grade_percent",
    "eco_factor",
    "congestion_profile_3h",
    "green_profile_3h",
    "curb_activity_profile_3h",
    "weekday_volume_profile",
]

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


def _network_row(segment_id: str, index: int) -> dict[str, str]:
    return {
        "segment_id": segment_id,
        "road_class": "primary" if index % 2 == 0 else "secondary",
        "direction": "eastbound" if index % 3 == 0 else "northbound",
        "from_lat": f"{42.70 + index * 0.0002:.6f}",
        "from_lon": f"{23.30 + index * 0.0002:.6f}",
        "to_lat": f"{42.7001 + index * 0.0002:.6f}",
        "to_lon": f"{23.3012 + index * 0.0002:.6f}",
        "lanes": "2",
        "length_km": f"{0.08 + index * 0.005:.3f}",
        "speed_limit_kmh": "50",
        "free_flow_speed_kmh": "42.0",
        "base_congestion_index": f"{0.24 + index * 0.01:.3f}",
        "peak_sensitivity": "1.08",
        "traffic_volume_aadt": f"{16000 + index * 800}",
        "heavy_vehicle_share": "0.08",
        "signal_count": "2",
        "green_wave_score": f"{0.58 + index * 0.01:.3f}",
        "green_corridor": "0",
        "avg_intersection_delay_s": f"{10.0 + index:.1f}",
        "grade_percent": "0.6",
        "eco_factor": "1.04",
        "congestion_profile_3h": "0.45|0.50|0.95|1.25|1.05|1.10|1.22|0.72",
        "green_profile_3h": "0.76|0.75|0.70|0.64|0.66|0.62|0.60|0.72",
        "curb_activity_profile_3h": "0.08|0.06|0.12|0.24|0.28|0.22|0.16|0.09",
        "weekday_volume_profile": "1.05|1.08|1.09|1.07|1.10|0.86|0.73",
    }


class ObservedCorrectionTrainingTests(unittest.TestCase):
    def test_build_dataset_and_train_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "network.csv"
            observation_path = temp_path / "observations.csv"

            with dataset_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=NETWORK_FIELDS)
                writer.writeheader()
                for index in range(12):
                    writer.writerow(_network_row(f"seg-{index}", index))

            with observation_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
                writer.writeheader()
                for index in range(12):
                    writer.writerow(
                        {
                            "route_signature": f"route-{index // 3}",
                            "recorded_at_utc": "2026-03-30T18:00:00Z",
                            "provider": "TomTom Routing API",
                            "scenario_name": "Traffic-aware fastest",
                            "route_type": "fastest",
                            "start_label": "Start",
                            "end_label": "End",
                            "vehicle_key": "Passenger Petrol",
                            "departure_hour_local": str(7 + (index % 4)),
                            "segment_id": f"seg-{index}",
                            "road_class": "primary" if index % 2 == 0 else "secondary",
                            "matched_distance_km": f"{0.07 + index * 0.004:.4f}",
                            "modeled_time_min": f"{1.40 + index * 0.05:.4f}",
                            "observed_time_min": f"{1.55 + index * 0.06:.4f}",
                            "observed_to_modeled_ratio": f"{0.96 + index * 0.015:.4f}",
                            "traffic_delay_min": "0.8",
                            "no_traffic_time_min": "3.2",
                        }
                    )

            x_matrix, y_vector, sample_weight, matched_rows, route_count = build_dataset(
                dataset_path,
                observation_path,
                min_observation_rows=4,
            )
            model, metrics, split = train_model(x_matrix, y_vector, sample_weight)

        self.assertEqual(matched_rows, 12)
        self.assertEqual(route_count, 4)
        self.assertEqual(x_matrix.shape[0], 12)
        self.assertEqual(len(y_vector), 12)
        self.assertEqual(len(sample_weight), 12)
        self.assertGreater(split["train_rows"], 0)
        self.assertGreater(split["test_rows"], 0)
        self.assertIn("mae_ratio", metrics)
        self.assertIn("rmse_ratio", metrics)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
