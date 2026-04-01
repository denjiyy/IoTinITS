from __future__ import annotations

import unittest

import pandas as pd

from ml_demo.train_uci_traffic_models import build_feature_frame, chronological_split, traffic_band_labels


def _sample_frame(rows: int = 20) -> pd.DataFrame:
    date_range = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "holiday": ["None"] * rows,
            "temp": [273.15 + (index % 10) for index in range(rows)],
            "rain_1h": [0.0] * rows,
            "snow_1h": [0.0] * rows,
            "clouds_all": [10 + index for index in range(rows)],
            "weather_main": ["Clear"] * rows,
            "weather_description": ["sky is clear"] * rows,
            "date_time": list(reversed(date_range)),
            "traffic_volume": [1000 + index * 100 for index in range(rows)],
        }
    )


class UciTrafficModelTests(unittest.TestCase):
    def test_build_feature_frame_sorts_and_derives_columns(self) -> None:
        raw = _sample_frame(6)
        engineered = build_feature_frame(raw)

        self.assertEqual(engineered["date_time"].iloc[0].isoformat(), "2024-01-01T00:00:00")
        self.assertEqual(engineered["date_time"].iloc[-1].isoformat(), "2024-01-01T05:00:00")
        self.assertIn("hour_sin", engineered.columns)
        self.assertIn("hour_cos", engineered.columns)
        self.assertIn("is_weekend", engineered.columns)

    def test_chronological_split_uses_expected_ratios(self) -> None:
        engineered = build_feature_frame(_sample_frame(20))
        train_frame, valid_frame, test_frame = chronological_split(engineered)

        self.assertEqual(len(train_frame), 14)
        self.assertEqual(len(valid_frame), 3)
        self.assertEqual(len(test_frame), 3)
        self.assertLess(train_frame["date_time"].max(), test_frame["date_time"].min())

    def test_traffic_band_labels_respect_thresholds(self) -> None:
        values = pd.Series([1200.0, 2157.0, 3000.0, 4555.0, 6000.0])
        labels = traffic_band_labels(values, low_threshold=2157.0, high_threshold=4555.0)
        self.assertEqual(labels.tolist(), [0, 0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
