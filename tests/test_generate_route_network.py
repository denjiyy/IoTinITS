from __future__ import annotations

import unittest

from generate_route_network import CSV_FIELDS, generate_rows_from_elements


class GenerateRouteNetworkTests(unittest.TestCase):
    def test_generate_rows_from_elements_handles_bidirectional_and_oneway_roads(self) -> None:
        elements = [
            {
                "id": 101,
                "tags": {"highway": "primary", "name:en": "Sample Blvd"},
                "geometry": [
                    {"lat": 42.7000, "lon": 23.3000},
                    {"lat": 42.7006, "lon": 23.3012},
                    {"lat": 42.7012, "lon": 23.3022},
                ],
            },
            {
                "id": 202,
                "tags": {"highway": "residential", "name:en": "One Way St", "oneway": "yes"},
                "geometry": [
                    {"lat": 42.6990, "lon": 23.3100},
                    {"lat": 42.6997, "lon": 23.3113},
                ],
            },
        ]

        rows = generate_rows_from_elements(elements)

        self.assertGreaterEqual(len(rows), 5)
        self.assertTrue(all(field in rows[0] for field in CSV_FIELDS))
        self.assertTrue(all(float(row["length_km"]) > 0 for row in rows))
        self.assertEqual(len(rows[0]["congestion_profile_3h"].split("|")), 8)
        self.assertEqual(len(rows[0]["green_profile_3h"].split("|")), 8)
        self.assertEqual(len(rows[0]["curb_activity_profile_3h"].split("|")), 8)
        self.assertEqual(len(rows[0]["weekday_volume_profile"].split("|")), 7)

        sample_rows = [row for row in rows if row["road_name"] == "Sample Blvd"]
        oneway_rows = [row for row in rows if row["road_name"] == "One Way St"]

        self.assertEqual(len(sample_rows), 4)
        self.assertEqual(len(oneway_rows), 1)
        self.assertTrue(all(row["direction"] in {"eastbound", "westbound", "northbound", "southbound"} for row in rows))


if __name__ == "__main__":
    unittest.main()
