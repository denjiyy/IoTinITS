from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from traffic_calibration import build_edge_matcher, load_calibration_bundle, record_live_route_observations


def _edge(segment_id: str, from_lat: float, from_lon: float, to_lat: float, to_lon: float, road_class: str = "primary"):
    return SimpleNamespace(
        segment_id=segment_id,
        from_lat=from_lat,
        from_lon=from_lon,
        to_lat=to_lat,
        to_lon=to_lon,
        road_class=road_class,
        direction="eastbound",
        lanes=2,
        length_km=0.12,
        speed_limit_kmh=50,
        free_flow_speed_kmh=42.0,
        base_congestion_index=0.34,
        peak_sensitivity=1.08,
        traffic_volume_aadt=18000,
        heavy_vehicle_share=0.08,
        signal_count=2,
        green_wave_score=0.62,
        green_corridor=False,
        avg_intersection_delay_s=12.0,
        grade_percent=0.5,
        eco_factor=1.05,
        congestion_profile_3h=(0.45, 0.50, 0.95, 1.25, 1.05, 1.10, 1.22, 0.72),
        green_profile_3h=(0.76, 0.75, 0.70, 0.64, 0.66, 0.62, 0.60, 0.72),
        curb_activity_profile_3h=(0.08, 0.06, 0.12, 0.24, 0.28, 0.22, 0.16, 0.09),
        weekday_volume_profile=(1.05, 1.08, 1.09, 1.07, 1.10, 0.86, 0.73),
    )


class TrafficCalibrationTests(unittest.TestCase):
    def test_recording_and_loading_calibration_bundle(self) -> None:
        edges = [
            _edge("seg-a", 42.7000, 23.3000, 42.7000, 23.3015),
            _edge("seg-b", 42.7000, 23.3015, 42.7000, 23.3030),
        ]
        matcher = build_edge_matcher(edges, cell_size_deg=0.002, max_match_distance_km=0.20)
        live_route = SimpleNamespace(
            provider="TomTom Routing API",
            name="Traffic-aware fastest",
            route_type="fastest",
            coordinates=[
                [23.3000, 42.7000],
                [23.3015, 42.7000],
                [23.3030, 42.7000],
            ],
            total_time_min=4.6,
            total_distance_km=0.24,
            traffic_delay_min=0.8,
            no_traffic_time_min=3.5,
            departure_time="2026-03-30T08:15:00+03:00",
            legs_count=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            observation_path = Path(temp_dir) / "live_route_observations.csv"
            first_write = record_live_route_observations(
                observation_path=observation_path,
                matcher=matcher,
                route=live_route,
                start_label="Start",
                end_label="End",
                vehicle_key="Passenger Petrol",
                min_match_ratio=0.30,
            )
            second_write = record_live_route_observations(
                observation_path=observation_path,
                matcher=matcher,
                route=live_route,
                start_label="Start",
                end_label="End",
                vehicle_key="Passenger Petrol",
                min_match_ratio=0.30,
            )

            bundle = load_calibration_bundle(
                observation_path,
                min_segment_samples=1,
                min_group_samples=1,
            )

        self.assertEqual(first_write, 2)
        self.assertEqual(second_write, 0)
        self.assertTrue(bundle.active)
        self.assertEqual(bundle.route_count, 1)
        self.assertEqual(bundle.observation_rows, 2)
        self.assertGreaterEqual(bundle.time_factor("seg-a", "primary", 8), 0.75)
        self.assertLessEqual(bundle.time_factor("seg-a", "primary", 8), 1.45)


if __name__ == "__main__":
    unittest.main()
