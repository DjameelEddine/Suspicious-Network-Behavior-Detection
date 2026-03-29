from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import APP_CONFIG, resolve_dataset_path, validate_artifacts
from src.data_simulator import DataSimulator
from src.event_store import EventStore
from src.explainability import ExplainabilityService
from src.metrics import MetricsTracker
from src.predictor import PredictionService
from src.preprocessing import Preprocessor
from src.utils import redact_internal_path


st.set_page_config(
    page_title="Real-Time Explainable IDS Dashboard",
    page_icon="shield",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_preprocessor() -> Preprocessor:
    return Preprocessor.from_artifacts(
        scaler_path=str(APP_CONFIG.artifact_paths.scaler),
        feature_list_path=str(APP_CONFIG.artifact_paths.feature_list),
    )


@st.cache_resource(show_spinner=False)
def get_predictor() -> PredictionService:
    return PredictionService.from_artifacts(APP_CONFIG)


@st.cache_resource(show_spinner=False)
def get_explainability(feature_names: Tuple[str, ...]) -> ExplainabilityService:
    predictor = get_predictor()
    return ExplainabilityService(
        model=predictor.binary_model,
        feature_names=list(feature_names),
    )


def initialize_session_state(dataset_path: Path, batch_size: int, rows_per_second: float, shuffle: bool) -> None:
    if "event_store" not in st.session_state:
        st.session_state.event_store = EventStore(max_events=APP_CONFIG.runtime.max_events_kept)

    if "metrics" not in st.session_state:
        st.session_state.metrics = MetricsTracker()

    simulator: DataSimulator | None = st.session_state.get("simulator")
    should_recreate = (
        simulator is None
        or simulator.dataset_path != dataset_path
        or simulator.shuffle != shuffle
    )

    if should_recreate:
        st.session_state.simulator = DataSimulator(
            dataset_path=dataset_path,
            batch_size=batch_size,
            rows_per_second=rows_per_second,
            shuffle=shuffle,
        )
    else:
        simulator.set_batch_size(batch_size)
        simulator.set_speed(rows_per_second)

    st.session_state.auto_refresh = st.session_state.get(
        "auto_refresh", APP_CONFIG.runtime.auto_refresh_default
    )
    st.session_state.last_error = st.session_state.get("last_error", "")


def process_next_batch() -> Tuple[int, int]:
    simulator: DataSimulator = st.session_state.simulator
    preprocessor: Preprocessor = st.session_state.preprocessor
    predictor: PredictionService = st.session_state.predictor
    explainability: ExplainabilityService = st.session_state.explainability
    event_store: EventStore = st.session_state.event_store
    metrics: MetricsTracker = st.session_state.metrics

    try:
        batch = simulator.next_batch()
        if batch.empty:
            return (0, 0)

        preprocessing_output = preprocessor.transform_batch(batch)

        if preprocessing_output.invalid_events:
            metrics.record_invalid(len(preprocessing_output.invalid_events))
            event_store.add_events(preprocessing_output.invalid_events)

        if preprocessing_output.transformed.empty:
            return (0, len(preprocessing_output.invalid_events))

        events = predictor.predict_batch(
            transformed=preprocessing_output.transformed,
            raw_valid_rows=preprocessing_output.raw_valid_rows,
        )
        events = explainability.enrich_events(events, top_k=3)

        for event in events:
            metrics.record_prediction(
                latency_ms=float(event.get("latency_ms", 0.0)),
                is_attack=str(event.get("binary_prediction", "")).lower() == "attack",
            )

        event_store.add_events(events)
        return (len(events), len(preprocessing_output.invalid_events))

    except Exception as exc:
        metrics.record_error()
        st.session_state.last_error = redact_internal_path(str(exc), APP_CONFIG.root_dir)
        return (0, 0)


def risk_row_style(row: pd.Series) -> List[str]:
    risk = str(row.get("risk_level", "")).lower()
    color = ""
    if risk == "high":
        color = "background-color: #ffe3e3"
    elif risk == "medium":
        color = "background-color: #fff7db"
    return [color] * len(row)


def render_header(snapshot: Dict[str, float], simulator: DataSimulator) -> None:
    st.title("Real-Time Explainable Intrusion Detection Dashboard")

    top_cols = st.columns(4)
    top_cols[0].metric("Runtime Status", str(snapshot.get("runtime_status", "unknown")).title())
    top_cols[1].metric("Processed Flows", int(snapshot.get("processed_flows", 0)))
    top_cols[2].metric("Throughput (flows/s)", float(snapshot.get("throughput_fps", 0.0)))
    top_cols[3].metric("System Health", str(snapshot.get("health", "Unknown")))

    status_cols = st.columns(4)
    status_cols[0].metric("Simulator State", simulator.status.title())
    status_cols[1].metric("Current Position", simulator.cursor)
    status_cols[2].metric("Dataset Size", simulator.total_rows)
    status_cols[3].metric("Progress", f"{simulator.progress_ratio() * 100.0:.2f}%")


def render_live_table(event_store: EventStore) -> pd.DataFrame:
    st.subheader("Live Prediction Table")

    display_columns = [
        "flow_id",
        "timestamp_utc",
        "binary_prediction",
        "attack_type",
        "confidence",
        "risk_level",
        "risk_score",
        "latency_ms",
        "status",
    ]

    events_df = event_store.as_dataframe(limit=APP_CONFIG.runtime.max_display_rows)
    if events_df.empty:
        st.info("No flow events available yet. Start simulation to populate live predictions.")
        return events_df

    for column in display_columns:
        if column not in events_df.columns:
            events_df[column] = np.nan

    table_df = events_df[display_columns].copy()
    table_df = table_df.sort_values(by="timestamp_utc", ascending=False)

    styled = table_df.style.apply(risk_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, height=360)
    return events_df


def render_summary(snapshot: Dict[str, float], events_df: pd.DataFrame) -> None:
    st.subheader("Traffic Summary")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Benign", int(snapshot.get("benign_count", 0)))
    summary_cols[1].metric("Malicious", int(snapshot.get("attack_count", 0)))
    summary_cols[2].metric("Invalid", int(snapshot.get("invalid_count", 0)))
    summary_cols[3].metric("Errors", int(snapshot.get("error_count", 0)))

    chart_cols = st.columns(2)

    if events_df.empty:
        chart_cols[0].info("Attack distribution will appear once attack events are detected.")
        chart_cols[1].info("Confidence distribution will appear once predictions are processed.")
        return

    attacks = events_df[events_df["binary_prediction"] == "Attack"]
    if attacks.empty:
        chart_cols[0].info("No attacks predicted yet.")
    else:
        attack_distribution = attacks["attack_type"].value_counts()
        chart_cols[0].bar_chart(attack_distribution)

    confidence_series = pd.to_numeric(events_df["confidence"], errors="coerce").dropna()
    if confidence_series.empty:
        chart_cols[1].info("Confidence scores are not available yet.")
    else:
        hist, bin_edges = np.histogram(confidence_series, bins=10, range=(0.0, 1.0))
        hist_df = pd.DataFrame(
            {
                "bin": [f"{bin_edges[idx]:.1f}-{bin_edges[idx + 1]:.1f}" for idx in range(len(hist))],
                "count": hist,
            }
        ).set_index("bin")
        chart_cols[1].bar_chart(hist_df)


def render_detailed_panel(events_df: pd.DataFrame, explainability: ExplainabilityService) -> None:
    st.subheader("Detailed Inspection Panel")

    if events_df.empty:
        st.info("Select a processed flow after simulation starts.")
        return

    flow_ids = events_df["flow_id"].astype(str).tolist()
    selected_flow_id = st.selectbox("Select flow ID", options=flow_ids, index=0)
    selected_event = st.session_state.event_store.get_by_flow_id(selected_flow_id)

    if not selected_event:
        st.warning("Selected flow is no longer in memory.")
        return

    left_col, right_col = st.columns(2)

    left_col.markdown("### Prediction")
    left_col.write(
        {
            "flow_id": selected_event.get("flow_id"),
            "timestamp_utc": selected_event.get("timestamp_utc"),
            "binary_prediction": selected_event.get("binary_prediction"),
            "attack_type": selected_event.get("attack_type"),
            "confidence": selected_event.get("confidence"),
            "risk_level": selected_event.get("risk_level"),
            "risk_score": selected_event.get("risk_score"),
            "latency_ms": selected_event.get("latency_ms"),
            "status": selected_event.get("status"),
            "actual_label": selected_event.get("actual_label"),
        }
    )

    right_col.markdown("### SHAP / Contribution Summary")
    right_col.write(str(selected_event.get("shap_summary", "No explanation available.")))

    positive = selected_event.get("top_positive_features", [])
    negative = selected_event.get("top_negative_features", [])

    positive_df = pd.DataFrame(positive, columns=["feature", "contribution"])
    negative_df = pd.DataFrame(negative, columns=["feature", "contribution"])

    feature_cols = st.columns(2)
    feature_cols[0].markdown("#### Top Positive Contributors")
    if positive_df.empty:
        feature_cols[0].info("No positive contributors available.")
    else:
        feature_cols[0].dataframe(positive_df, use_container_width=True)

    feature_cols[1].markdown("#### Top Negative Contributors")
    if negative_df.empty:
        feature_cols[1].info("No negative contributors available.")
    else:
        feature_cols[1].dataframe(negative_df, use_container_width=True)

    st.markdown("#### Global Feature Importance")
    st.caption(explainability.status_message)
    global_importance = explainability.global_importance(top_n=15)
    st.bar_chart(global_importance.set_index("feature"))


def render_metrics(snapshot: Dict[str, float]) -> None:
    st.subheader("Metrics Panel")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Avg Latency (ms)", float(snapshot.get("avg_latency_ms", 0.0)))
    metric_cols[1].metric("Throughput", float(snapshot.get("throughput_fps", 0.0)))
    metric_cols[2].metric("Total Processed", int(snapshot.get("processed_flows", 0)))
    metric_cols[3].metric("Invalid Flows", int(snapshot.get("invalid_count", 0)))
    metric_cols[4].metric("Error Rate", float(snapshot.get("error_rate", 0.0)))


def render_export_controls() -> None:
    st.subheader("Logging and Export")

    export_cols = st.columns(2)

    if export_cols[0].button("Export prediction log as CSV", use_container_width=True):
        filename = f"predictions_{int(time.time())}.csv"
        path = APP_CONFIG.exports_dir / filename
        st.session_state.event_store.export_csv(path)
        st.success(f"CSV export completed: {filename}")

    if export_cols[1].button("Export prediction log as JSON", use_container_width=True):
        filename = f"predictions_{int(time.time())}.json"
        path = APP_CONFIG.exports_dir / filename
        st.session_state.event_store.export_json(path)
        st.success(f"JSON export completed: {filename}")


def main() -> None:
    artifact_errors = validate_artifacts(APP_CONFIG)

    if artifact_errors:
        st.error("Model artifacts are missing or incomplete. Build artifacts before running the dashboard.")
        st.write("Required artifacts:")
        st.write("- models/binary_model.pkl")
        st.write("- models/multiclass_model.pkl")
        st.write("- models/scaler.pkl")
        st.write("- models/feature_list.pkl")
        st.write("Suggested command:")
        st.code("python scripts/build_artifacts.py --dataset clean_traffic.csv", language="bash")
        for issue in artifact_errors:
            st.warning(issue)
        st.stop()

    dataset_override = st.sidebar.text_input("Dataset path (optional)", value="")
    dataset_path = resolve_dataset_path(dataset_override if dataset_override else None)

    if dataset_path is None:
        st.error("No dataset file was found. Place clean_traffic.csv or traffic.csv in the project root.")
        st.stop()

    rows_per_second = st.sidebar.slider(
        "Rows per second",
        min_value=0.5,
        max_value=100.0,
        value=float(APP_CONFIG.runtime.default_rows_per_second),
        step=0.5,
    )
    batch_size = st.sidebar.slider(
        "Batch size",
        min_value=1,
        max_value=200,
        value=int(APP_CONFIG.runtime.default_batch_size),
        step=1,
    )
    shuffle = st.sidebar.checkbox("Shuffle flows", value=False)
    auto_refresh = st.sidebar.checkbox(
        "Auto refresh",
        value=st.session_state.get("auto_refresh", APP_CONFIG.runtime.auto_refresh_default),
    )

    initialize_session_state(dataset_path, batch_size, rows_per_second, shuffle)

    st.session_state.auto_refresh = auto_refresh
    st.session_state.preprocessor = get_preprocessor()
    st.session_state.predictor = get_predictor()
    st.session_state.explainability = get_explainability(
        tuple(st.session_state.preprocessor.feature_list)
    )

    simulator: DataSimulator = st.session_state.simulator
    metrics: MetricsTracker = st.session_state.metrics

    simulator.set_batch_size(batch_size)
    simulator.set_speed(rows_per_second)

    control_cols = st.columns(6)
    if control_cols[0].button("Start", use_container_width=True):
        simulator.start()
        metrics.set_runtime_status("running")

    if control_cols[1].button("Pause", use_container_width=True):
        simulator.pause()
        metrics.set_runtime_status("paused")

    if control_cols[2].button("Resume", use_container_width=True):
        simulator.resume()
        metrics.set_runtime_status("running")

    if control_cols[3].button("Stop", use_container_width=True):
        simulator.stop(reset_cursor=False)
        metrics.set_runtime_status("stopped")

    if control_cols[4].button("Reset", use_container_width=True):
        simulator.stop(reset_cursor=True)
        st.session_state.event_store.clear()
        metrics.reset()

    if control_cols[5].button("Process Next Batch", use_container_width=True):
        simulator.resume() if simulator.status == "paused" else None
        if simulator.status == "stopped":
            simulator.start()
            metrics.set_runtime_status("running")
        process_next_batch()

    if simulator.status == "running":
        metrics.set_runtime_status("running")
    elif simulator.status == "paused":
        metrics.set_runtime_status("paused")
    elif simulator.status == "completed":
        metrics.set_runtime_status("completed")
    else:
        metrics.set_runtime_status("stopped")

    if st.session_state.last_error:
        st.warning(f"Last processing issue: {st.session_state.last_error}")

    snapshot = metrics.snapshot()
    render_header(snapshot, simulator)

    events_df = render_live_table(st.session_state.event_store)
    render_summary(snapshot, events_df)
    render_detailed_panel(events_df, st.session_state.explainability)
    render_metrics(snapshot)
    render_export_controls()

    if auto_refresh and simulator.status == "running":
        process_next_batch()
        delay = min(2.0, max(0.1, simulator.seconds_between_batches()))
        time.sleep(delay)
        st.rerun()


if __name__ == "__main__":
    main()
