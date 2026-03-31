# Urban Route Optimizer

A Streamlit routing app with two routing backends and two trip input modes:

- A validated local Sofia street graph built from OpenStreetMap, with modeled congestion, emissions, green-corridor scoring, and per-segment profile arrays for time-of-day traffic behavior.
- An optional TomTom live-routing backend for traffic-aware ETA, waypoint optimization, and turn-by-turn guidance.
- Hub presets for quick comparisons, or direct latitude/longitude entry for free-point routing.
- A coordinate-mode point picker map, plus TomTom-backed address lookup when a live API key is configured.

## What is in this repo

- `main.py`: Streamlit UI and runtime bootstrap.
- `route_engine.py`: routing, scoring, network validation, and trip scenario generation.
- `generate_route_network.py`: converts the local OSM extract into the routable CSV used by the app.
- `sofia_osm_overpass.json`: local Sofia road extract from Overpass.
- `sofia_route_network.csv`: generated routing dataset consumed by the app.
- `ml_demo/`: separate tabular ML demonstration project for the academic submission requirement.

## Academic Demo

For the course requirement about a small ML/FastAI-style demonstration project, see:

- [ml_demo/README.md](./ml_demo/README.md)
- [ml_demo/green_corridor_tabular_demo.ipynb](./ml_demo/green_corridor_tabular_demo.ipynb)
- [ml_demo/travel_time_random_forest_demo.ipynb](./ml_demo/travel_time_random_forest_demo.ipynb)
- [ml_demo/congestion_random_forest_demo.ipynb](./ml_demo/congestion_random_forest_demo.ipynb)

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

The app will open at `http://localhost:8501`.

## Enable live traffic routing

Set a TomTom API key before launching the app:

```bash
export TOMTOM_API_KEY="your-key-here"
python3 main.py
```

Optional environment variables:

- `IOTINITS_ROUTING_BACKEND`: `auto`, `local`, or `tomtom`
- `TOMTOM_BASE_URL`
- `TOMTOM_TIMEOUT_S`

## Regenerate the routing dataset

```bash
python3 generate_route_network.py
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## Pilot deployment notes

- Streamlit runtime defaults live in `.streamlit/config.toml`.
- The app expects `sofia_route_network.csv` and `sofia_osm_overpass.json` to exist next to the source files.
- Startup validates the road network before rendering the UI.
- Environment variables:
  - `IOTINITS_LOG_LEVEL`
  - `IOTINITS_MAP_HEIGHT`
  - `IOTINITS_ROUTE_PADDING_DEG`
  - `IOTINITS_MAX_DELIVERY_STOPS`
  - `IOTINITS_ROUTING_BACKEND`
  - `TOMTOM_API_KEY`
  - `TOMTOM_BASE_URL`
  - `TOMTOM_TIMEOUT_S`

## Docker

```bash
docker build -t urban-route-optimizer .
docker run --rm -p 8501:8501 urban-route-optimizer
```

## Important limitations

- The app now supports direct coordinate entry, but not map-click pin drops or continuous GPS navigation yet.
- The local backend remains modeled. It does not ingest live signal-controller feeds.
- Estimated CO2 is still computed locally, even when the live routing backend is active.
