from __future__ import annotations

import unittest

from ml_models import batch_predict_network_scores, ml_model_status, predict_route_ml_summary
from route_engine import DATA_PATH, available_hubs, load_network, route_scenarios


class MLModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes, cls.adjacency, cls.edges = load_network()
        cls.hubs = available_hubs(cls.nodes)

    def test_ml_models_are_available(self) -> None:
        status = ml_model_status()
        self.assertTrue(status.travel_time_available)
        self.assertTrue(status.congestion_available)
        self.assertTrue(status.green_corridor_available)

    def test_batch_network_scores_cover_loaded_segments(self) -> None:
        scores = batch_predict_network_scores(str(DATA_PATH), 8)
        self.assertIn(self.edges[0].segment_id, scores)
        first_score = scores[self.edges[0].segment_id]
        self.assertIsNotNone(first_score.predicted_time_min)
        self.assertIsNotNone(first_score.high_congestion_probability)

    def test_route_ml_summary_returns_predictions_for_local_route(self) -> None:
        scenarios = route_scenarios(
            adjacency=self.adjacency,
            start_node=self.hubs["Lyulin Center"],
            end_node=self.hubs["Sofia Airport Terminal 2"],
            stops=[],
            hour=8,
            vehicle_key="Passenger Petrol",
            user_weights=(45, 35, 20),
            strict_green=False,
        )

        route = scenarios["Balanced"]
        self.assertIsNotNone(route)
        assert route is not None

        summary = predict_route_ml_summary(route, hour=8)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.segment_count, len(route.segments))
        self.assertIsNotNone(summary.predicted_time_min)
        self.assertGreater(summary.predicted_time_min or 0.0, 5.0)
        self.assertIn(summary.congestion_label, {"Low", "Medium", "High"})
        self.assertTrue(summary.congestion_mix)
        self.assertGreaterEqual(summary.average_green_corridor_probability or 0.0, 0.0)
        self.assertLessEqual(summary.average_green_corridor_probability or 0.0, 1.0)

    def test_route_scenarios_can_use_hybrid_ml_scoring(self) -> None:
        ml_scores = batch_predict_network_scores(str(DATA_PATH), 8)
        scenarios = route_scenarios(
            adjacency=self.adjacency,
            start_node=self.hubs["Lyulin Center"],
            end_node=self.hubs["Sofia Airport Terminal 2"],
            stops=[],
            hour=8,
            vehicle_key="Passenger Petrol",
            user_weights=(45, 35, 20),
            strict_green=False,
            ml_segment_scores=ml_scores,
        )

        balanced = scenarios["Balanced"]
        self.assertIsNotNone(balanced)
        assert balanced is not None
        self.assertTrue(balanced.uses_ml_scoring)
        self.assertTrue(any(segment.ml_predicted_time_min is not None for segment in balanced.segments))


if __name__ == "__main__":
    unittest.main()
