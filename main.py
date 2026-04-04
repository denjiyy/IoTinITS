from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from app_config import load_app_config
from app_logging import get_logger, setup_logging

APP_PATH = Path(__file__).resolve()
REQUIREMENTS_PATH = APP_PATH.with_name("requirements.txt")
APP_CONFIG = load_app_config()


def missing_dependency_help(module_name: str) -> None:
    print(f"Missing Python package: {module_name}")
    print()
    print("Install the app dependencies into the same interpreter you are using to launch the app:")
    print(f"  {sys.executable} -m pip install -r {REQUIREMENTS_PATH.name}")
    print()
    print("Launch the UI with one of these commands:")
    print(f"  {sys.executable} -m streamlit run {APP_PATH.name}")
    print(f"  {sys.executable} {APP_PATH.name}")
    raise SystemExit(1)


try:
    import streamlit as st
    import streamlit.runtime
except ModuleNotFoundError as exc:
    missing_dependency_help(exc.name)


if __name__ == "__main__" and not streamlit.runtime.exists():
    try:
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(APP_PATH),
                    "--browser.gatherUsageStats",
                    "false",
                ]
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(0)


try:
    import pandas as pd
    import pydeck as pdk
except ModuleNotFoundError as exc:
    missing_dependency_help(exc.name)

from ml_models import (
    MLModelStatus,
    RouteMLPrediction,
    batch_predict_network_scores,
    ml_model_status,
    predict_route_ml_summary,
)
from traffic_calibration import (
    build_edge_matcher,
    calibration_status,
    load_calibration_bundle,
    record_live_route_observations,
)
from live_routing import (
    GeocodeCandidate,
    LiveRouteResult,
    LiveRoutingError,
    geocode_candidates,
    live_backend_status_message,
    live_route_scenarios,
    live_routing_available,
    resolve_backend,
)
from route_engine import (
    VEHICLE_PROFILES,
    available_hubs,
    grouped_steps,
    load_network,
    nearest_node_id,
    route_path_coordinates,
    route_scenarios,
    route_waypoint_names,
    validate_network,
)

setup_logging(APP_CONFIG.log_level)
LOGGER = get_logger(__name__)

st.set_page_config(
    page_title=APP_CONFIG.app_name,
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
        :root {
            --bg: #f5f5ef;
            --panel: rgba(255, 255, 255, 0.82);
            --panel-strong: rgba(255, 255, 255, 0.96);
            --ink: #102018;
            --muted: #56645d;
            --line: rgba(16, 32, 24, 0.10);
            --accent: #0f766e;
            --accent-2: #c9721d;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(201, 114, 29, 0.10), transparent 26%),
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
                linear-gradient(180deg, #f7f7f2 0%, #f1f2eb 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #15191d 0%, #1b2127 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #e8ece8;
        }
        .shell {
            display: grid;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .headline {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            padding-bottom: 0.9rem;
            border-bottom: 1px solid var(--line);
        }
        .headline h1 {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            line-height: 1;
            margin: 0;
            color: var(--ink);
        }
        .headline p {
            margin: 0.35rem 0 0 0;
            color: var(--muted);
            max-width: 780px;
        }
        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            font-weight: 700;
        }
        .backend-strip {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .backend-pill {
            border: 1px solid rgba(15, 118, 110, 0.18);
            background: rgba(15, 118, 110, 0.10);
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-size: 0.86rem;
            font-weight: 600;
            color: #0d5c56;
        }
        .backend-note {
            color: var(--muted);
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.85rem 0.95rem;
            background: var(--panel-strong);
            box-shadow: 0 12px 24px rgba(17, 24, 39, 0.04);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] label,
        [data-testid="stMetricLabel"] span {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            background: transparent !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] p,
        [data-testid="stMetricValue"] span {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            opacity: 1 !important;
            background: transparent !important;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stMetricDeltaDescription"],
        [data-testid="stMetricDeltaDescription"] * {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
            color: #e8ece8 !important;
            -webkit-text-fill-color: #e8ece8 !important;
        }
        div[data-testid="stMetric"] ::selection {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            background: rgba(15, 118, 110, 0.18) !important;
        }
        [data-baseweb="button-group"] button {
            background: rgba(16, 32, 24, 0.06) !important;
            border: 1px solid rgba(16, 32, 24, 0.08) !important;
            color: var(--ink) !important;
        }
        [data-baseweb="button-group"] button[aria-pressed="true"] {
            background: rgba(15, 118, 110, 0.14) !important;
            border-color: rgba(15, 118, 110, 0.26) !important;
            color: #0d5c56 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            background: var(--panel-strong);
        }
        .panel-note {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .section-sep {
            border-top: 1px solid var(--line);
            margin-top: 0.6rem;
            padding-top: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_network():
    nodes, adjacency, edges = load_network(APP_CONFIG.data_path)
    hubs = available_hubs(nodes)
    summary = validate_network(nodes, adjacency, edges, hubs)
    corridor_edges = [
        edge for edge in edges if edge.green_corridor and edge.road_class in {"motorway", "trunk", "primary", "secondary"}
    ]
    LOGGER.info(
        "Loaded network: nodes=%s edges=%s hubs=%s reachable_hubs=%s green_corridors=%s",
        summary.node_count,
        summary.edge_count,
        summary.hub_count,
        summary.reachable_hub_count,
        summary.green_corridor_count,
    )
    return nodes, adjacency, edges, corridor_edges, hubs, summary


@st.cache_resource(show_spinner=False)
def cached_edge_matcher():
    return build_edge_matcher(edges)


def observation_log_version_token() -> str:
    try:
        stat = APP_CONFIG.observation_log_path.stat()
    except FileNotFoundError:
        return "missing"
    return f"{int(stat.st_mtime_ns)}:{stat.st_size}"


@st.cache_data(show_spinner=False)
def cached_calibration_bundle(observation_path: str, version_token: str):
    del version_token
    return load_calibration_bundle(
        observation_path,
        min_segment_samples=APP_CONFIG.calibration_min_segment_samples,
        min_group_samples=APP_CONFIG.calibration_min_group_samples,
    )


@st.cache_data(show_spinner=False)
def cached_calibration_status(observation_path: str, version_token: str):
    del version_token
    return calibration_status(
        observation_path,
        min_segment_samples=APP_CONFIG.calibration_min_segment_samples,
        min_group_samples=APP_CONFIG.calibration_min_group_samples,
    )


@st.cache_data(show_spinner=False)
def cached_local_route_scenarios(
    start_node: str,
    end_node: str,
    stop_nodes: tuple[str, ...],
    hour: int,
    vehicle_key: str,
    user_weights: tuple[int, int, int],
    strict_green: bool,
    local_engine_mode: str,
    observation_log_version: str,
):
    ml_segment_scores = None
    calibration = None
    if local_engine_mode == "Hybrid ML-aware":
        ml_segment_scores = batch_predict_network_scores(str(APP_CONFIG.data_path), hour)
        calibration = cached_calibration_bundle(
            str(APP_CONFIG.observation_log_path),
            observation_log_version,
        )
    return route_scenarios(
        adjacency=adjacency,
        start_node=start_node,
        end_node=end_node,
        stops=list(stop_nodes),
        hour=hour,
        vehicle_key=vehicle_key,
        user_weights=user_weights,
        strict_green=strict_green,
        ml_segment_scores=ml_segment_scores,
        calibration=calibration,
    )


def is_live_route(route: Any) -> bool:
    return isinstance(route, LiveRouteResult)


def network_bounds(nodes: dict[str, Any]) -> tuple[float, float, float, float]:
    latitudes = [node.lat for node in nodes.values()]
    longitudes = [node.lon for node in nodes.values()]
    return min(latitudes), max(latitudes), min(longitudes), max(longitudes)


def route_coordinates(route: Any) -> list[list[float]]:
    if route is None:
        return []
    if is_live_route(route):
        return route.coordinates
    return route_path_coordinates(route)


def route_delay_minutes(route: Any) -> float:
    if route is None:
        return 0.0
    if is_live_route(route):
        return route.traffic_delay_min
    return route.total_signal_delay_min


def route_stop_labels(route: Any, start_label: str, end_label: str, hub_label_by_node: dict[str, str]) -> list[str]:
    if route is None:
        return [start_label, end_label]
    if is_live_route(route):
        return [start_label, *route.ordered_stops, end_label]
    return route_waypoint_names(route, hub_label_by_node, start_label, end_label)


def route_provider_label(route: Any, backend: str) -> str:
    if route is None:
        return "Unavailable"
    if is_live_route(route):
        return route.provider
    if getattr(route, "uses_ml_scoring", False) and getattr(route, "uses_observed_target_model", False):
        return "Local Sofia graph + ML + observed-target correction"
    if getattr(route, "uses_ml_scoring", False) and getattr(route, "uses_empirical_calibration", False):
        return "Local Sofia graph + ML + empirical calibration"
    if getattr(route, "uses_ml_scoring", False):
        return "Local Sofia graph + ML scoring"
    if getattr(route, "uses_empirical_calibration", False):
        return "Local Sofia graph + empirical calibration"
    return "Local Sofia street graph"


def route_layer_dataframe(route: Any, color: list[int], width: int) -> pd.DataFrame:
    coords = route_coordinates(route)
    if not coords:
        return pd.DataFrame(columns=["path", "color", "width"])

    return pd.DataFrame(
        [
            {
                "path": coords,
                "route_name": route.name,
                "distance_km": round(route.total_distance_km, 1),
                "time_min": round(route.total_time_min, 1),
                "delay_min": round(route_delay_minutes(route), 1),
                "provider": route_provider_label(route, "tomtom" if is_live_route(route) else "local"),
                "color": color,
                "width": width,
            }
        ]
    )


def corridor_layer_dataframe(corridor_edges, route: Any) -> pd.DataFrame:
    if route is None or is_live_route(route):
        return pd.DataFrame(columns=["path", "color", "width"])

    coords = route_coordinates(route)
    if not coords:
        return pd.DataFrame(columns=["path", "color", "width"])

    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    min_lon, max_lon = min(lons) - APP_CONFIG.route_padding_deg, max(lons) + APP_CONFIG.route_padding_deg
    min_lat, max_lat = min(lats) - APP_CONFIG.route_padding_deg, max(lats) + APP_CONFIG.route_padding_deg

    records = []
    for edge in corridor_edges:
        if not (
            min_lon <= edge.from_lon <= max_lon
            and min_lon <= edge.to_lon <= max_lon
            and min_lat <= edge.from_lat <= max_lat
            and min_lat <= edge.to_lat <= max_lat
        ):
            continue
        records.append(
            {
                "path": [[edge.from_lon, edge.from_lat], [edge.to_lon, edge.to_lat]],
                "color": [15, 118, 110, 74],
                "width": 2,
            }
        )
    return pd.DataFrame(records)


def waypoint_dataframe(route: Any, start_label: str, end_label: str, hub_points: dict[str, tuple[float, float]], hub_label_by_node) -> pd.DataFrame:
    points = []
    start_lat, start_lon = hub_points[start_label]
    end_lat, end_lon = hub_points[end_label]

    points.append({"name": start_label, "lon": start_lon, "lat": start_lat, "kind": "Start", "color": [15, 118, 110]})

    labels = route_stop_labels(route, start_label, end_label, hub_label_by_node)
    for label in labels[1:-1]:
        stop_lat, stop_lon = hub_points[label]
        points.append(
            {
                "name": label,
                "lon": stop_lon,
                "lat": stop_lat,
                "kind": "Delivery stop",
                "color": [201, 114, 29],
            }
        )

    points.append(
        {"name": end_label, "lon": end_lon, "lat": end_lat, "kind": "Destination", "color": [41, 72, 255]}
    )
    return pd.DataFrame(points)


def map_view_state(route: Any, start_label: str, end_label: str, hub_points: dict[str, tuple[float, float]]) -> pdk.ViewState:
    coords = route_coordinates(route)
    if not coords:
        start_lat, start_lon = hub_points[start_label]
        end_lat, end_lon = hub_points[end_label]
        avg_lat = (start_lat + end_lat) / 2
        avg_lon = (start_lon + end_lon) / 2
        return pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=11.4, pitch=18)

    avg_lon = sum(point[0] for point in coords) / len(coords)
    avg_lat = sum(point[1] for point in coords) / len(coords)
    return pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=11.8, pitch=22)


@st.cache_data(show_spinner=False)
def picker_node_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node_id": node.node_id, "name": node.name, "lat": node.lat, "lon": node.lon}
            for node in nodes.values()
        ]
    )


def picker_view_state(point_lookup: dict[str, tuple[float, float]], start_label: str, end_label: str) -> pdk.ViewState:
    start_lat, start_lon = point_lookup[start_label]
    end_lat, end_lon = point_lookup[end_label]
    return pdk.ViewState(
        latitude=(start_lat + end_lat) / 2,
        longitude=(start_lon + end_lon) / 2,
        zoom=11.2,
        pitch=0,
    )


def selected_picker_object(selection_state: Any) -> dict[str, Any] | None:
    if selection_state is None:
        return None
    selection = getattr(selection_state, "selection", None)
    if selection is None:
        return None

    objects = getattr(selection, "objects", None)
    if objects is None and isinstance(selection, dict):
        objects = selection.get("objects", {})
    if not isinstance(objects, dict):
        return None

    for items in objects.values():
        if isinstance(items, list) and items:
            candidate = items[0]
            if isinstance(candidate, dict):
                return candidate
    return None


def geocode_option_label(candidate: GeocodeCandidate) -> str:
    return f"{candidate.label} ({candidate.lat:.5f}, {candidate.lon:.5f})"


def live_step_rows(route: LiveRouteResult) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for step in route.guidance_steps:
        rows.append(
            {
                "maneuver": step.maneuver.replace("_", " ").title(),
                "instruction": step.instruction,
                "street": step.street or "Continue",
                "distance_km": step.distance_km,
                "time_min": step.time_min,
            }
        )
    return rows


def format_congestion_mix(summary: RouteMLPrediction | None) -> str:
    if summary is None or not summary.congestion_mix:
        return "Unavailable"
    ordered = []
    for label in ("low", "medium", "high"):
        if label in summary.congestion_mix:
            ordered.append(f"{label.title()} {summary.congestion_mix[label] * 100:.0f}%")
    return " / ".join(ordered) if ordered else "Unavailable"


def render_ml_forecast(
    summary: RouteMLPrediction | None,
    status: MLModelStatus,
    *,
    ml_guides_optimizer: bool,
    empirical_calibration_active: bool,
    observed_target_model_active: bool,
) -> None:
    if not status.any_available:
        return

    st.caption("ML route forecast")
    ml_metric1, ml_metric2, ml_metric3, ml_metric4 = st.columns(4)

    if summary is None or summary.predicted_time_min is None:
        ml_metric1.metric("ML ETA", "Unavailable")
    else:
        ml_metric1.metric(
            "ML ETA",
            f"{summary.predicted_time_min:.1f} min",
            delta=f"{summary.predicted_time_delta_min:+.1f} min" if summary.predicted_time_delta_min is not None else None,
        )

    ml_metric2.metric("Predicted traffic", summary.congestion_label if summary and summary.congestion_label else "Unavailable")

    if summary is None or summary.high_congestion_share is None:
        ml_metric3.metric("High traffic share", "Unavailable")
    else:
        ml_metric3.metric("High traffic share", f"{summary.high_congestion_share * 100:.0f}%")

    if summary is None or summary.average_green_corridor_probability is None:
        ml_metric4.metric("Corridor confidence", "Unavailable")
    else:
        ml_metric4.metric("Corridor confidence", f"{summary.average_green_corridor_probability:.2f}")

    st.caption(
        "These predictions come from the trained tabular models in ml_demo. "
        + (
            "They are also feeding the local optimizer in Hybrid ML-aware mode."
            if ml_guides_optimizer
            else "The path geometry still comes from the validated Sofia street-graph optimizer."
        )
        + (
            " Empirical calibration from saved live TomTom observations is active."
            if empirical_calibration_active
            else ""
        )
        + (
            " A retrained observed-target correction model is also active."
            if observed_target_model_active
            else ""
        )
    )


def format_comparison_row(name: str, route: Any, backend: str) -> dict[str, str]:
    if route is None:
        if backend == "tomtom":
            return {
                "Scenario": name,
                "Distance (km)": "Unavailable",
                "ETA (min)": "Unavailable",
                "Traffic delay (min)": "Unavailable",
                "CO2 (kg)": "Unavailable",
                "Provider": "Unavailable",
            }
        return {
            "Scenario": name,
            "Distance (km)": "Unavailable",
            "Time (min)": "Unavailable",
            "CO2 (kg)": "Unavailable",
            "Avg green score": "Unavailable",
            "Signal delay (min)": "Unavailable",
        }

    if backend == "tomtom":
        return {
            "Scenario": name,
            "Distance (km)": f"{route.total_distance_km:.1f}",
            "ETA (min)": f"{route.total_time_min:.1f}",
            "Traffic delay (min)": f"{route.traffic_delay_min:.1f}",
            "CO2 (kg)": f"{route.total_emissions_g / 1000:.2f}",
            "Provider": route.provider,
        }

    return {
        "Scenario": name,
        "Distance (km)": f"{route.total_distance_km:.1f}",
        "Time (min)": f"{route.total_time_min:.1f}",
        "CO2 (kg)": f"{route.total_emissions_g / 1000:.2f}",
        "Avg green score": f"{route.average_green_score:.2f}",
        "Signal delay (min)": f"{route.total_signal_delay_min:.1f}",
    }


def snap_warning_message(
    trip_input_mode: str,
    backend: str,
    start_snap_distance_km: float,
    end_snap_distance_km: float,
) -> str:
    if trip_input_mode != "Coordinates" or backend != "local":
        return ""
    max_snap_distance_km = max(start_snap_distance_km, end_snap_distance_km)
    if max_snap_distance_km <= 0.05:
        return ""
    return (
        f"Custom points were snapped to the nearest routable road nodes. "
        f"Start offset: {start_snap_distance_km:.2f} km, destination offset: {end_snap_distance_km:.2f} km."
    )


try:
    nodes, adjacency, edges, corridor_edges, hub_options, network_summary = cached_network()
except Exception as exc:
    LOGGER.exception("Failed to load the validated road network.")
    st.error(
        "The road network failed validation and the app cannot start safely. "
        "Check the local dataset files and regenerate the network."
    )
    st.exception(exc)
    st.stop()

hub_names = list(hub_options.keys())
hub_label_by_node = {node_id: label for label, node_id in hub_options.items()}
hub_points = {
    label: (nodes[node_id].lat, nodes[node_id].lon)
    for label, node_id in hub_options.items()
}

if not hub_names:
    st.error("No valid hubs are available in the current road network.")
    st.stop()

edge_matcher = cached_edge_matcher()
current_observation_log_version = observation_log_version_token()
current_calibration_status = cached_calibration_status(
    str(APP_CONFIG.observation_log_path),
    current_observation_log_version,
)


@st.cache_data(show_spinner=False, ttl=60)
def cached_live_route_scenarios(
    start_label: str,
    start_point: tuple[float, float],
    end_label: str,
    end_point: tuple[float, float],
    stop_labels: tuple[str, ...],
    vehicle_key: str,
):
    return live_route_scenarios(
        config=APP_CONFIG,
        start_label=start_label,
        start_point=start_point,
        end_label=end_label,
        end_point=end_point,
        stop_labels=list(stop_labels),
        stop_points=[hub_points[label] for label in stop_labels],
        vehicle_key=vehicle_key,
    )


@st.cache_data(show_spinner=False, ttl=600)
def cached_geocode_candidates(query: str, around_point: tuple[float, float] | None):
    return geocode_candidates(config=APP_CONFIG, query=query, around_point=around_point, limit=5)


default_start = "Lyulin Center" if "Lyulin Center" in hub_options else hub_names[0]
default_end = "Sofia Airport Terminal 2" if "Sofia Airport Terminal 2" in hub_options else hub_names[-1]
backend_index_lookup = {"auto": 0, "tomtom": 1, "local": 2}
provider_available = live_routing_available(APP_CONFIG)
ml_status = ml_model_status()
default_local_engine_mode = "Hybrid ML-aware" if ml_status.any_available else "Physics only"
network_min_lat, network_max_lat, network_min_lon, network_max_lon = network_bounds(nodes)
default_start_lat, default_start_lon = hub_points[default_start]
default_end_lat, default_end_lon = hub_points[default_end]

st.session_state.setdefault("custom_start_lat", float(default_start_lat))
st.session_state.setdefault("custom_start_lon", float(default_start_lon))
st.session_state.setdefault("custom_end_lat", float(default_end_lat))
st.session_state.setdefault("custom_end_lon", float(default_end_lon))
st.session_state.setdefault("active_geocode_query", "")

with st.sidebar:
    st.header("Trip setup")
    trip_input_mode = st.radio(
        "Trip input",
        ["Hub presets", "Coordinates"],
        horizontal=True,
        help="Use curated Sofia hubs for quick comparisons or enter exact coordinates for free-point routing.",
    )
    backend_choice = st.selectbox(
        "Routing backend",
        ["Auto", "TomTom live traffic", "Local Sofia model"],
        index=backend_index_lookup.get(APP_CONFIG.routing_backend, 0),
        help="Auto uses TomTom live routing when TOMTOM_API_KEY is configured, otherwise it falls back to the local Sofia model.",
    )
    if not provider_available:
        st.caption("Set `TOMTOM_API_KEY` to enable live traffic, incident-aware ETA, and provider turn guidance.")

    if trip_input_mode == "Hub presets":
        start_display_label = st.selectbox("Start hub", hub_names, index=hub_names.index(default_start))
        end_display_label = st.selectbox("Destination hub", hub_names, index=hub_names.index(default_end))
        start_point = hub_points[start_display_label]
        end_point = hub_points[end_display_label]
    else:
        start_display_label = "Custom start"
        end_display_label = "Custom destination"
        start_cols = st.columns(2)
        start_lat = start_cols[0].number_input(
            "Start latitude",
            min_value=float(network_min_lat - 0.02),
            max_value=float(network_max_lat + 0.02),
            format="%.6f",
            key="custom_start_lat",
        )
        start_lon = start_cols[1].number_input(
            "Start longitude",
            min_value=float(network_min_lon - 0.02),
            max_value=float(network_max_lon + 0.02),
            format="%.6f",
            key="custom_start_lon",
        )
        end_cols = st.columns(2)
        end_lat = end_cols[0].number_input(
            "Destination latitude",
            min_value=float(network_min_lat - 0.02),
            max_value=float(network_max_lat + 0.02),
            format="%.6f",
            key="custom_end_lat",
        )
        end_lon = end_cols[1].number_input(
            "Destination longitude",
            min_value=float(network_min_lon - 0.02),
            max_value=float(network_max_lon + 0.02),
            format="%.6f",
            key="custom_end_lon",
        )
        start_point = (float(start_lat), float(start_lon))
        end_point = (float(end_lat), float(end_lon))

        if provider_available:
            st.header("Address lookup")
            geocode_target = st.radio(
                "Apply search result to",
                ["Start", "Destination"],
                horizontal=True,
                key="geocode_target",
            )
            geocode_query_input = st.text_input(
                "Find address or place",
                placeholder="Example: Sofia Airport Terminal 2",
                key="geocode_query_input",
            )
            if st.button("Search addresses", width="stretch"):
                st.session_state["active_geocode_query"] = geocode_query_input.strip()

            active_geocode_query = st.session_state.get("active_geocode_query", "").strip()
            if active_geocode_query:
                try:
                    around_point = start_point if geocode_target == "Start" else end_point
                    geocode_results = cached_geocode_candidates(active_geocode_query, around_point)
                except LiveRoutingError as exc:
                    st.warning(str(exc))
                    geocode_results = []

                if geocode_results:
                    selected_candidate_index = st.selectbox(
                        "Address matches",
                        options=list(range(len(geocode_results))),
                        format_func=lambda index: geocode_option_label(geocode_results[index]),
                        key="geocode_match_index",
                    )
                    if st.button("Use searched address", width="stretch"):
                        candidate = geocode_results[selected_candidate_index]
                        if geocode_target == "Start":
                            st.session_state["custom_start_lat"] = candidate.lat
                            st.session_state["custom_start_lon"] = candidate.lon
                        else:
                            st.session_state["custom_end_lat"] = candidate.lat
                            st.session_state["custom_end_lon"] = candidate.lon
                        st.rerun()
                else:
                    st.caption("No address matches cached yet for this query.")

    stop_labels = st.multiselect(
        "Optional delivery stops",
        [name for name in hub_names if name not in {start_display_label, end_display_label}],
        max_selections=APP_CONFIG.max_delivery_stops,
        help="TomTom can optimize waypoint order for live routing. The local model uses a greedy quick-delivery heuristic.",
    )

    requested_backend = {
        "Auto": "auto",
        "TomTom live traffic": "tomtom",
        "Local Sofia model": "local",
    }[backend_choice]
    effective_backend = resolve_backend(APP_CONFIG, requested_backend)

    st.header("Traffic model")
    hour = st.slider(
        "Departure hour",
        min_value=0,
        max_value=23,
        value=8,
        disabled=effective_backend == "tomtom",
        help="Used by the local Sofia model. Live routing uses departure now so the provider can return actual traffic-aware ETA.",
    )
    local_engine_mode = st.radio(
        "Local engine",
        ["Hybrid ML-aware", "Physics only"],
        horizontal=True,
        index=0 if default_local_engine_mode == "Hybrid ML-aware" else 1,
        disabled=effective_backend == "tomtom" or not ml_status.any_available,
        help="Hybrid ML-aware mode batches the trained tabular models across the Sofia network and blends them into the local route cost.",
    )
    if effective_backend == "local" and not ml_status.any_available:
        st.caption("ML artifacts are missing, so the local engine stays on the physics model only.")
    vehicle_key = st.selectbox("Vehicle profile", list(VEHICLE_PROFILES.keys()), index=1)
    strict_green = st.checkbox(
        "Require synchronized green corridors",
        value=False,
        disabled=effective_backend == "tomtom",
        help="This remains a local-model feature. Real all-green routing would need direct signal-controller integration.",
    )
    if effective_backend == "tomtom":
        st.caption("Live routing keeps real-time traffic and turn restrictions. The green-corridor hard filter stays disabled.")

    st.header("Objective weights")
    time_weight = st.slider(
        "Travel time",
        0,
        100,
        45,
        5,
        disabled=effective_backend == "tomtom",
        help="These weights only affect the local Sofia model.",
    )
    emissions_weight = st.slider(
        "Emissions",
        0,
        100,
        35,
        5,
        disabled=effective_backend == "tomtom",
        help="These weights only affect the local Sofia model.",
    )
    green_weight = st.slider(
        "Green-light friendliness",
        0,
        100,
        20,
        5,
        disabled=effective_backend == "tomtom",
        help="These weights only affect the local Sofia model.",
    )

if start_point == end_point:
    st.error("Choose different start and destination points.")
    st.stop()

point_lookup = dict(hub_points)
point_lookup[start_display_label] = start_point
point_lookup[end_display_label] = end_point

start_snap_distance_km = 0.0
end_snap_distance_km = 0.0
if trip_input_mode == "Hub presets":
    start_node = hub_options[start_display_label]
    end_node = hub_options[end_display_label]
else:
    start_node, start_snap_distance_km = nearest_node_id(nodes, start_point[0], start_point[1])
    end_node, end_snap_distance_km = nearest_node_id(nodes, end_point[0], end_point[1])
stop_nodes = [hub_options[label] for label in stop_labels]

backend_warning = ""
snap_warning = ""
new_observation_rows = 0
if trip_input_mode == "Coordinates" and effective_backend == "local":
    max_snap_distance_km = max(start_snap_distance_km, end_snap_distance_km)
    if max_snap_distance_km > 1.5:
        st.error("One of the custom points is too far from the loaded Sofia road network. Move the point closer to the city streets.")
        st.stop()

if effective_backend == "tomtom":
    try:
        scenarios = cached_live_route_scenarios(
            start_label=start_display_label,
            start_point=start_point,
            end_label=end_display_label,
            end_point=end_point,
            stop_labels=tuple(stop_labels),
            vehicle_key=vehicle_key,
        )
    except LiveRoutingError as exc:
        backend_warning = f"{exc} Falling back to the local Sofia model for this session."
        LOGGER.warning("Live routing unavailable, falling back to local model: %s", exc)
        effective_backend = "local"
        scenarios = cached_local_route_scenarios(
            start_node=start_node,
            end_node=end_node,
            stop_nodes=tuple(stop_nodes),
            hour=hour,
            vehicle_key=vehicle_key,
            user_weights=(time_weight, emissions_weight, green_weight),
            strict_green=strict_green,
            local_engine_mode=local_engine_mode,
            observation_log_version=current_observation_log_version,
        )
else:
    scenarios = cached_local_route_scenarios(
        start_node=start_node,
        end_node=end_node,
        stop_nodes=tuple(stop_nodes),
        hour=hour,
        vehicle_key=vehicle_key,
        user_weights=(time_weight, emissions_weight, green_weight),
        strict_green=strict_green,
        local_engine_mode=local_engine_mode,
        observation_log_version=current_observation_log_version,
    )

if effective_backend == "tomtom":
    for live_route in scenarios.values():
        try:
            new_observation_rows += record_live_route_observations(
                observation_path=APP_CONFIG.observation_log_path,
                matcher=edge_matcher,
                route=live_route,
                start_label=start_display_label,
                end_label=end_display_label,
                vehicle_key=vehicle_key,
                min_match_ratio=APP_CONFIG.observation_min_match_ratio,
            )
        except Exception as exc:
            LOGGER.warning("Failed to persist live observation rows: %s", exc)
    if new_observation_rows:
        current_observation_log_version = observation_log_version_token()
        current_calibration_status = cached_calibration_status(
            str(APP_CONFIG.observation_log_path),
            current_observation_log_version,
        )

snap_warning = snap_warning_message(trip_input_mode, effective_backend, start_snap_distance_km, end_snap_distance_km)

if backend_warning:
    st.warning(backend_warning)
if snap_warning:
    st.info(snap_warning)
if new_observation_rows:
    st.success(f"Saved {new_observation_rows} new live observation rows for local-route calibration.")

st.markdown(
    f"""
    <div class="shell">
        <div class="headline">
            <div>
                <div class="eyebrow">{"Live Traffic Routing" if effective_backend == "tomtom" else "Validated Sofia Geometry"}</div>
                <h1>{APP_CONFIG.app_name}</h1>
                <p>
                    {"TomTom live traffic, turn restrictions, and turn-by-turn guidance are active. The app still keeps the local Sofia graph as a safe fallback for offline use and testing."
                    if effective_backend == "tomtom"
                    else "Routing follows a validated Sofia street graph built from OpenStreetMap. Congestion, emissions, and synchronized-green scoring are modeled locally on top of the real street geometry."}
                </p>
            </div>
        </div>
        <div class="backend-strip">
            <span class="backend-pill">{"TomTom live provider" if effective_backend == "tomtom" else "Local Sofia model"}</span>
            <span class="backend-note">{live_backend_status_message(APP_CONFIG, effective_backend)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if trip_input_mode == "Coordinates":
    picker_left, picker_right = st.columns([1.0, 1.7], gap="large")
    with picker_left:
        st.subheader("Point Picker")
        st.caption("Click a road node on the map, then assign it to the trip start or destination.")
        selected_picker = None
    with picker_right:
        picker_layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=picker_node_dataframe(),
                id="picker_nodes",
                get_position="[lon, lat]",
                get_fill_color=[73, 80, 87, 84],
                get_radius=38,
                radius_min_pixels=2,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(
                    [
                        {
                            "label": start_display_label,
                            "lon": start_point[1],
                            "lat": start_point[0],
                            "color": [15, 118, 110],
                            "radius": 220,
                        },
                        {
                            "label": end_display_label,
                            "lon": end_point[1],
                            "lat": end_point[0],
                            "color": [41, 72, 255],
                            "radius": 220,
                        },
                    ]
                ),
                id="picker_targets",
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                radius_min_pixels=6,
                pickable=False,
            ),
        ]
        picker_state = st.pydeck_chart(
            pdk.Deck(
                map_provider="carto",
                map_style="light",
                initial_view_state=picker_view_state(point_lookup, start_display_label, end_display_label),
                tooltip={
                    "html": "<b>{name}</b><br/>Node: {node_id}<br/>Lat: {lat}<br/>Lon: {lon}",
                    "style": {"backgroundColor": "rgba(17, 24, 39, 0.92)", "color": "white"},
                },
                layers=picker_layers,
            ),
            width="stretch",
            height=300,
            on_select="rerun",
            selection_mode="single-object",
            key="coordinate_picker_map",
        )
        selected_picker = selected_picker_object(picker_state)
        st.caption("Gray points are routable road nodes. Green and blue markers show the current start and destination.")

    with picker_left:
        st.write(f"Start: **{start_point[0]:.6f}, {start_point[1]:.6f}**")
        st.write(f"Destination: **{end_point[0]:.6f}, {end_point[1]:.6f}**")
        if selected_picker is None:
            st.info("Select a road node from the picker map to apply it.")
        else:
            st.write(f"Selected node: **{selected_picker.get('name') or selected_picker['node_id']}**")
            st.write(f"Latitude: **{float(selected_picker['lat']):.6f}**")
            st.write(f"Longitude: **{float(selected_picker['lon']):.6f}**")
            assign_col1, assign_col2 = st.columns(2)
            if assign_col1.button("Use as start", width="stretch"):
                st.session_state["custom_start_lat"] = float(selected_picker["lat"])
                st.session_state["custom_start_lon"] = float(selected_picker["lon"])
                st.rerun()
            if assign_col2.button("Use as destination", width="stretch"):
                st.session_state["custom_end_lat"] = float(selected_picker["lat"])
                st.session_state["custom_end_lon"] = float(selected_picker["lon"])
                st.rerun()
        if st.button("Reset custom points", width="stretch"):
            st.session_state["custom_start_lat"] = float(default_start_lat)
            st.session_state["custom_start_lon"] = float(default_start_lon)
            st.session_state["custom_end_lat"] = float(default_end_lat)
            st.session_state["custom_end_lon"] = float(default_end_lon)
            st.rerun()

scenario_names = list(scenarios.keys())
selected_scenario_name = st.segmented_control(
    "Scenario view",
    scenario_names,
    default=scenario_names[0],
    selection_mode="single",
    width="stretch",
)
selected_route = scenarios[selected_scenario_name]
route_ml_summary = None
if effective_backend == "local" and selected_route is not None and ml_status.any_available:
    route_ml_summary = predict_route_ml_summary(selected_route, hour=hour)

if selected_route is None and effective_backend == "local" and strict_green:
    st.warning(
        "No route met the strict synchronized-corridor filter for this trip. Turn the hard filter off to keep the real-road geometry while using a softer green preference."
    )

metric1, metric2, metric3, metric4 = st.columns(4)
if selected_route is None:
    metric1.metric("Displayed route", "Unavailable")
    metric2.metric("Estimated CO2", "Unavailable")
    metric3.metric("Traffic delay" if effective_backend == "tomtom" else "Avg green score", "Unavailable")
    metric4.metric("Navigation" if effective_backend == "tomtom" else "Delivery plan", "Unavailable")
else:
    metric1.metric("Displayed route", f"{selected_route.total_time_min:.1f} min")
    metric2.metric("Estimated CO2", f"{selected_route.total_emissions_g / 1000:.2f} kg")
    if effective_backend == "tomtom" and is_live_route(selected_route):
        metric3.metric("Traffic delay", f"{selected_route.traffic_delay_min:.1f} min")
        metric4.metric("Navigation", f"{len(selected_route.guidance_steps)} steps")
    else:
        metric3.metric("Avg green score", f"{selected_route.average_green_score:.2f}")
        metric4.metric("Delivery plan", "Direct" if not selected_route.ordered_stops else f"{len(selected_route.ordered_stops)} stop(s)")

if effective_backend == "local":
    render_ml_forecast(
        route_ml_summary,
        ml_status,
        ml_guides_optimizer=selected_route is not None and getattr(selected_route, "uses_ml_scoring", False),
        empirical_calibration_active=selected_route is not None
        and getattr(selected_route, "uses_empirical_calibration", False),
        observed_target_model_active=selected_route is not None
        and getattr(selected_route, "uses_observed_target_model", False),
    )

map_col, panel_col = st.columns([2.05, 1.0], gap="large")

with map_col:
    st.subheader("Route Map")
    layers = []
    corridor_df = corridor_layer_dataframe(corridor_edges, selected_route)
    if not corridor_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=corridor_df,
                get_path="path",
                get_color="color",
                get_width="width",
                width_scale=2,
                pickable=False,
            )
        )
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=route_layer_dataframe(selected_route, [217, 114, 29, 225], 8),
            get_path="path",
            get_color="color",
            get_width="width",
            width_scale=2,
            pickable=True,
        )
    )
    layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=waypoint_dataframe(selected_route, start_display_label, end_display_label, point_lookup, hub_label_by_node),
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=170,
                radius_min_pixels=7,
                pickable=True,
        )
    )
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=map_view_state(selected_route, start_display_label, end_display_label, point_lookup),
        tooltip={
            "html": "<b>{route_name}</b><br/>Distance: {distance_km} km<br/>ETA: {time_min} min<br/>Delay: {delay_min} min<br/>{provider}",
            "style": {"backgroundColor": "rgba(17, 24, 39, 0.92)", "color": "white"},
        },
        layers=layers,
    )
    st.pydeck_chart(deck, width="stretch", height=APP_CONFIG.map_height)
    if effective_backend == "tomtom":
        st.caption("Orange shows the TomTom live route geometry. Delivery stops are optimized by the provider when live routing is active.")
    else:
        st.caption("Orange shows the selected local route. The faint green overlay marks nearby modeled green corridors on top of the real Sofia street map.")

with panel_col:
    st.subheader("Route Summary")
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    if selected_route is None:
        st.info("Choose a different trip or relax the strict green requirement.")
    else:
        st.write(f"Distance: **{selected_route.total_distance_km:.1f} km**")
        if effective_backend == "tomtom" and is_live_route(selected_route):
            st.write(f"Traffic delay: **{selected_route.traffic_delay_min:.1f} min**")
            st.write(f"Provider: **{selected_route.provider}**")
            st.write(f"Legs returned: **{selected_route.legs_count}**")
            if selected_route.departure_time:
                st.write(f"Departure: **{selected_route.departure_time}**")
            if selected_route.arrival_time:
                st.write(f"Arrival: **{selected_route.arrival_time}**")
        else:
            st.write(f"Signal delay: **{selected_route.total_signal_delay_min:.1f} min**")
            st.write(f"Segments used: **{len(selected_route.segments)}**")
            st.write(
                "Local engine: **"
                + ("Hybrid ML-aware**" if getattr(selected_route, "uses_ml_scoring", False) else "Physics only**")
            )
            st.write(
                "Empirical calibration: **"
                + ("Active**" if getattr(selected_route, "uses_empirical_calibration", False) else "Not active**")
            )
            st.write(
                "Observed target model: **"
                + ("Active**" if getattr(selected_route, "uses_observed_target_model", False) else "Not active**")
            )
            if route_ml_summary is not None:
                if route_ml_summary.predicted_time_min is not None:
                    st.write(f"ML ETA: **{route_ml_summary.predicted_time_min:.1f} min**")
                if route_ml_summary.congestion_label is not None:
                    st.write(f"ML traffic mix: **{format_congestion_mix(route_ml_summary)}**")
                if route_ml_summary.average_green_corridor_probability is not None:
                    st.write(
                        "ML corridor confidence: "
                        f"**{route_ml_summary.average_green_corridor_probability:.2f}**"
                    )

        st.write("Itinerary:")
        for label in route_stop_labels(selected_route, start_display_label, end_display_label, hub_label_by_node):
            st.write(f"- {label}")

        if effective_backend == "tomtom" and is_live_route(selected_route):
            st.dataframe(pd.DataFrame(live_step_rows(selected_route)), width="stretch", hide_index=True, height=360)
        else:
            st.dataframe(pd.DataFrame(grouped_steps(selected_route)), width="stretch", hide_index=True, height=360)

lower_left, lower_right = st.columns([1.35, 1.0], gap="large")

with lower_left:
    st.subheader("Scenario Comparison")
    comparison_df = pd.DataFrame(
        [format_comparison_row(name, route, effective_backend) for name, route in scenarios.items()]
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)

with lower_right:
    st.subheader("Data Notes")
    if effective_backend == "tomtom":
        st.markdown(
            """
            <div class="panel-note">
                Live mode uses TomTom for traffic-aware navigation geometry, optimized waypoint order, and text guidance.
                Estimated CO2 is still computed locally from the provider distance and delay because this app does not yet
                integrate a fleet-grade emissions model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Fallback dataset: **{Path(APP_CONFIG.data_path).name}**")
        st.write(f"Selectable Sofia hubs: **{network_summary.hub_count}**")
        st.write(f"Provider configured: **{'Yes' if provider_available else 'No'}**")
        st.write(f"Logged live routes: **{current_calibration_status.route_count}**")
    else:
        st.markdown(
            """
            <div class="panel-note">
                The local mode uses a validated Sofia street graph and adds modeled congestion, emissions,
                synchronized-green scoring, tabular ML predictions, and empirical correction factors on top.
                It remains the offline-safe fallback when a live provider is not configured or reachable.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"Street segments loaded: **{len(edges):,}**")
        st.write(f"Selectable hubs: **{network_summary.hub_count}**")
        st.write(f"Reachable hubs in main graph: **{network_summary.reachable_hub_count}**")
        st.write(f"Dataset: **{Path(APP_CONFIG.data_path).name}**")
        st.write(f"Local engine mode: **{local_engine_mode}**")
        st.write(f"ML forecast ready: **{'Yes' if ml_status.any_available else 'No'}**")
        st.write(f"Observed target model ready: **{'Yes' if ml_status.observed_correction_available else 'No'}**")
        st.write(f"Observation rows: **{current_calibration_status.observation_rows}**")
        st.write(f"Logged live routes: **{current_calibration_status.route_count}**")
        st.write(f"Calibrated segments: **{current_calibration_status.calibrated_segments}**")
        st.write(f"Calibrated road/hour groups: **{current_calibration_status.calibrated_groups}**")
