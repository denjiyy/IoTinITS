from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app_config import AppConfig
from live_routing import LiveRoutingError, TomTomRoutingClient, resolve_backend


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _test_config(api_key: str = "test-key") -> AppConfig:
    return AppConfig(
        app_name="Urban Route Optimizer",
        data_path=Path("sofia_route_network.csv"),
        source_path=Path("sofia_osm_overpass.json"),
        observation_log_path=Path("live_route_observations.csv"),
        map_height=660,
        route_padding_deg=0.012,
        max_delivery_stops=2,
        log_level="INFO",
        routing_backend="auto",
        tomtom_api_key=api_key,
        tomtom_base_url="https://api.tomtom.com",
        tomtom_timeout_s=12.0,
        calibration_min_segment_samples=3,
        calibration_min_group_samples=8,
        observation_min_match_ratio=0.35,
    )


class LiveRoutingTests(unittest.TestCase):
    def test_tomtom_client_parses_geocode_candidates(self) -> None:
        payload = {
            "results": [
                {
                    "position": {"lat": 42.6887, "lon": 23.4080},
                    "address": {"freeformAddress": "Sofia Airport Terminal 2, Sofia"},
                    "score": 9.8,
                },
                {
                    "position": {"lat": 42.6977, "lon": 23.3225},
                    "address": {"freeformAddress": "Serdika Square, Sofia"},
                    "score": 8.6,
                },
            ]
        }

        captured: dict[str, str] = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return _FakeResponse(payload)

        client = TomTomRoutingClient(_test_config())
        with patch("live_routing.urlopen", side_effect=fake_urlopen):
            candidates = client.geocode("Sofia Airport", around_point=(42.69, 23.33), limit=5)

        self.assertIn("/search/2/geocode/Sofia%20Airport.json", captured["url"])
        self.assertIn("countrySet=BG", captured["url"])
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].label, "Sofia Airport Terminal 2, Sofia")
        self.assertAlmostEqual(candidates[0].lat, 42.6887, places=4)

    def test_tomtom_client_parses_guidance_and_optimized_stops(self) -> None:
        payload = {
            "routes": [
                {
                    "summary": {
                        "lengthInMeters": 17600,
                        "travelTimeInSeconds": 2250,
                        "trafficDelayInSeconds": 240,
                        "noTrafficTravelTimeInSeconds": 2010,
                        "historicTrafficTravelTimeInSeconds": 2190,
                        "liveTrafficIncidentsTravelTimeInSeconds": 2250,
                        "departureTime": "2026-03-29T19:52:00+02:00",
                        "arrivalTime": "2026-03-29T20:29:30+02:00",
                    },
                    "legs": [
                        {
                            "points": [
                                {"latitude": 42.7178, "longitude": 23.2680},
                                {"latitude": 42.7130, "longitude": 23.2850},
                                {"latitude": 42.7060, "longitude": 23.3218},
                                {"latitude": 42.6977, "longitude": 23.3225},
                            ]
                        },
                        {
                            "points": [
                                {"latitude": 42.6977, "longitude": 23.3225},
                                {"latitude": 42.6932, "longitude": 23.3348},
                                {"latitude": 42.6887, "longitude": 23.4080},
                            ]
                        },
                    ],
                    "guidance": {
                        "instructions": [
                            {
                                "message": "Head east on Dimitar Talev",
                                "maneuver": "DEPART",
                                "street": "Dimitar Talev",
                                "routeOffsetInMeters": 0,
                                "travelTimeInSeconds": 0,
                            },
                            {
                                "message": "Turn right onto Tsaritsa Yoanna Blvd",
                                "maneuver": "TURN_RIGHT",
                                "street": "Tsaritsa Yoanna Blvd",
                                "routeOffsetInMeters": 4300,
                                "travelTimeInSeconds": 480,
                            },
                            {
                                "message": "Arrive at your destination",
                                "maneuver": "ARRIVE",
                                "routeOffsetInMeters": 17600,
                                "travelTimeInSeconds": 2250,
                            },
                        ]
                    },
                }
            ],
            "optimizedWaypoints": [
                {"providedIndex": 0, "optimizedIndex": 1},
                {"providedIndex": 1, "optimizedIndex": 0},
            ],
        }

        captured: dict[str, str] = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = str(timeout)
            return _FakeResponse(payload)

        client = TomTomRoutingClient(_test_config())
        with patch("live_routing.urlopen", side_effect=fake_urlopen):
            result = client.calculate_route(
                scenario_name="Traffic-aware fastest",
                start_point=(42.7178, 23.2680),
                end_point=(42.6887, 23.4080),
                stop_points=[(42.6977, 23.3225), (42.7060, 23.3218)],
                stop_labels=["Serdika", "Lions Bridge"],
                vehicle_key="Passenger Petrol",
                route_type="fastest",
            )

        self.assertIn("routeType=fastest", captured["url"])
        self.assertIn("traffic=true", captured["url"])
        self.assertIn("computeBestOrder=true", captured["url"])
        self.assertEqual(result.ordered_stops, ["Lions Bridge", "Serdika"])
        self.assertEqual(result.provider, "TomTom Routing API")
        self.assertAlmostEqual(result.total_distance_km, 17.6, places=1)
        self.assertAlmostEqual(result.traffic_delay_min, 4.0, places=1)
        self.assertEqual(len(result.coordinates), 6)
        self.assertEqual(result.guidance_steps[1].maneuver, "TURN_RIGHT")
        self.assertGreater(result.total_emissions_g, 2000.0)

    def test_missing_api_key_raises_clear_error(self) -> None:
        client = TomTomRoutingClient(_test_config(api_key=""))
        with self.assertRaises(LiveRoutingError):
            client.calculate_route(
                scenario_name="Traffic-aware fastest",
                start_point=(42.7178, 23.2680),
                end_point=(42.6887, 23.4080),
                stop_points=[],
                stop_labels=[],
                vehicle_key="Passenger Petrol",
                route_type="fastest",
            )

    def test_resolve_backend_falls_back_to_local_without_key(self) -> None:
        self.assertEqual(resolve_backend(_test_config(api_key=""), "tomtom"), "local")
        self.assertEqual(resolve_backend(_test_config(api_key="test-key"), "tomtom"), "tomtom")


if __name__ == "__main__":
    unittest.main()
