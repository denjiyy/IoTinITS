from __future__ import annotations

import unittest

from route_engine import available_hubs, load_network, nearest_node_id, route_scenarios, validate_network


class RouteEngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes, cls.adjacency, cls.edges = load_network()
        cls.hubs = available_hubs(cls.nodes)
        cls.summary = validate_network(cls.nodes, cls.adjacency, cls.edges, cls.hubs)

    def test_network_validation_summary_is_healthy(self) -> None:
        self.assertGreaterEqual(self.summary.hub_count, 10)
        self.assertGreater(self.summary.edge_count, 50000)
        self.assertGreaterEqual(self.summary.reachable_hub_count, 10)
        self.assertGreater(self.summary.green_corridor_count, 1000)
        self.assertEqual(len(self.edges[0].congestion_profile_3h), 8)
        self.assertEqual(len(self.edges[0].green_profile_3h), 8)
        self.assertEqual(len(self.edges[0].weekday_volume_profile), 7)

    def test_balanced_route_exists_between_core_hubs(self) -> None:
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

        balanced = scenarios["Balanced"]
        self.assertIsNotNone(balanced)
        assert balanced is not None
        self.assertGreater(balanced.total_distance_km, 5.0)
        self.assertLess(balanced.total_time_min, 90.0)
        self.assertGreater(balanced.total_emissions_g, 1000.0)

    def test_invalid_stop_nodes_raise_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            route_scenarios(
                adjacency=self.adjacency,
                start_node=self.hubs["Lyulin Center"],
                end_node=self.hubs["Sofia Airport Terminal 2"],
                stops=["NOT_A_REAL_NODE"],
                hour=8,
                vehicle_key="Passenger Petrol",
                user_weights=(45, 35, 20),
                strict_green=False,
            )

    def test_nearest_node_snaps_custom_coordinate_near_city_network(self) -> None:
        node_id, distance = nearest_node_id(self.nodes, 42.7173, 23.2672)
        self.assertIn(node_id, self.nodes)
        self.assertLess(distance, 0.5)


if __name__ == "__main__":
    unittest.main()
