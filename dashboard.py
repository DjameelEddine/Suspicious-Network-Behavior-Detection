"""
dashboard.py
============
Real-time Network Intrusion Detection System Dashboard
Two-layer detection:  Layer 1 → Binary (BENIGN / ATTACK)
                      Layer 2 → Multi-class (attack type)

Usage
-----
  streamlit run dashboard.py

Requirements
------------
  pip install streamlit pandas numpy joblib plotly scikit-learn xgboost
"""

import time
import json
import os
import threading
import queue
from collections import deque, Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Real-Time Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODELS_DIR   = "./saved_models/models"
DEFAULT_SIM_CSV      = "./testing_data/simulation_data.csv"
DEFAULT_INTERVAL     = 5          # seconds between rows
MAX_HISTORY          = 500        # rows kept in session history
ROLLING_WINDOW       = 50         # rows for rolling attack rate chart

ATTACK_COLORS = {
    "BENIGN":                    "#2ecc71",
    "DoS Hulk":                  "#e74c3c",
    "PortScan":                  "#e67e22",
    "DDoS":                      "#c0392b",
    "DoS GoldenEye":             "#9b59b6",
    "FTP-Patator":               "#1abc9c",
    "SSH-Patator":               "#16a085",
    "DoS slowloris":             "#d35400",
    "DoS Slowhttptest":          "#e91e63",
    "Bot":                       "#673ab7",
    "Web Attack - Brute Force":  "#ff5722",
    "Web Attack - XSS":          "#795548",
    "Infiltration":              "#607d8b",
    "Web Attack - Sql Injection":"#f44336",
    "Heartbleed":                "#b71c1c",
    "ATTACK":                    "#e74c3c",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json_safe(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def label_color(label: str) -> str:
    return ATTACK_COLORS.get(label, "#95a5a6")


def badge(label: str) -> str:
    color = label_color(label)
    text_color = "#ffffff"
    return (
        f'<span style="background:{color};color:{text_color};'
        f'padding:3px 10px;border-radius:12px;font-weight:600;'
        f'font-size:0.85rem;">{label}</span>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "running":         False,
        "row_index":       0,
        "history":         [],          # list of dicts
        "binary_model":    None,
        "multi_model":     None,
        "sim_data":        None,
        "feature_cols":    None,
        "models_loaded":   False,
        "data_loaded":     False,
        "total_processed": 0,
        "attack_count":    0,
        "benign_count":    0,
        "class_counter":   Counter(),
        "alert_queue":     deque(maxlen=10),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Model / data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(models_dir: str, bin_name: str, multi_name: str):
    bin_path   = os.path.join(models_dir, "binary",  bin_name)
    multi_path = os.path.join(models_dir, "multi",   multi_name)
    errors = []
    for p in [bin_path, multi_path]:
        if not os.path.exists(p):
            errors.append(f"File not found: {p}")
    if errors:
        return None, None, errors
    binary = joblib.load(bin_path)
    multi  = joblib.load(multi_path)
    return binary, multi, []


def load_simulation_data(csv_path: str):
    if not os.path.exists(csv_path):
        return None, f"File not found: {csv_path}"
    df = pd.read_csv(csv_path)
    label_col = "Label" if "Label" in df.columns else None
    if label_col is None:
        return None, "CSV must contain a 'Label' column."
    feature_cols = [c for c in df.columns if c not in ("Label", "Timestamp")]
    return df, feature_cols, None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────────────────────

def predict_row(row_features: np.ndarray):
    """Run the two-layer prediction pipeline.
    
    Both layers always run and make predictions independently.
    Final prediction is based on which layer is more confident:
    - Layer 1 BENIGN confidence vs Layer 2 attack confidence
    - Trust whichever has the higher confidence score
    """
    binary_model = st.session_state.binary_model
    multi_model  = st.session_state.multi_model

    # Layer 1 – binary (BENIGN=0, ATTACK=1)
    bin_pred  = int(binary_model.predict(row_features)[0])
    try:
        bin_proba_attack = float(binary_model.predict_proba(row_features)[0][1])
    except AttributeError:
        bin_proba_attack = float(bin_pred)
    
    bin_proba_benign = 1.0 - bin_proba_attack

    # Layer 2 – multi-class (always run independently)
    multi_pred = multi_model.predict(row_features)[0]
    try:
        proba_arr  = multi_model.predict_proba(row_features)[0]
        multi_conf = float(np.max(proba_arr))
    except AttributeError:
        multi_conf = 1.0

    # Choose final prediction based on highest confidence
    # Layer 1 BENIGN confidence vs Layer 2 attack confidence
    if bin_proba_benign > multi_conf:
        # Layer 1 is more confident this is BENIGN
        return "BENIGN", bin_proba_benign, str(multi_pred), multi_conf
    else:
        # Layer 2 is more confident about an attack type
        return "ATTACK", bin_proba_attack, str(multi_pred), multi_conf


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/firewall.png", width=64)
    st.title("🛡️ IDS Dashboard")
    st.markdown("---")

    # ── Paths ──────────────────────────────────────────────────────
    st.subheader("📁 Paths")
    models_dir = st.text_input("Models directory", value=DEFAULT_MODELS_DIR)
    sim_csv    = st.text_input("Simulation CSV",   value=DEFAULT_SIM_CSV)

    # ── Model selection ────────────────────────────────────────────
    st.subheader("🤖 Model Selection (Best are selected by default)")

    # Discover available binary models
    bin_dir = os.path.join(models_dir, "binary")
    if os.path.isdir(bin_dir):
        bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".joblib")])
        # Move xgboost to front as default
        if "xgboost_binary.joblib" in bin_files:
            bin_files.remove("xgboost_binary.joblib")
            bin_files.insert(0, "xgboost_binary.joblib")
    else:
        bin_files = ["xgboost_binary.joblib"]

    multi_dir = os.path.join(models_dir, "multi")
    if os.path.isdir(multi_dir):
        multi_files = sorted([f for f in os.listdir(multi_dir) if f.endswith(".joblib")])
        # Move knn to front as default
        if "knn_multi.joblib" in multi_files:
            multi_files.remove("knn_multi.joblib")
            multi_files.insert(0, "knn_multi.joblib")
    else:
        multi_files = ["knn_multi.joblib"]

    selected_bin   = st.selectbox("Layer 1 – Binary model",    bin_files   if bin_files   else ["<none>"])
    selected_multi = st.selectbox("Layer 2 – Multi-class model", multi_files if multi_files else ["<none>"])

    if st.button("🔄 Load Models & Data", use_container_width=True):
        with st.spinner("Loading …"):
            binary, multi, errs = load_models(models_dir, selected_bin, selected_multi)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.session_state.binary_model  = binary
                st.session_state.multi_model   = multi
                st.session_state.models_loaded = True

            result = load_simulation_data(sim_csv)
            if len(result) == 3:
                df, feature_cols, err = result
            else:
                df, err = result
                feature_cols = None

            if err:
                st.error(err)
            else:
                st.session_state.sim_data     = df
                st.session_state.feature_cols = feature_cols
                st.session_state.data_loaded  = True
                st.session_state.row_index    = 0

        if st.session_state.models_loaded and st.session_state.data_loaded:
            st.success(f"✅ Ready — {len(st.session_state.sim_data):,} rows loaded")

    st.markdown("---")

    # ── Simulation controls ────────────────────────────────────────
    st.subheader("⚙️ Simulation")
    interval = st.slider("Interval (seconds)", min_value=1, max_value=30,
                          value=DEFAULT_INTERVAL, step=1)
    show_benign = st.checkbox("Show BENIGN rows in log", value=False)

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", use_container_width=True,
                               disabled=not (st.session_state.models_loaded and
                                             st.session_state.data_loaded))
    with col2:
        stop_btn  = st.button("⏸ Pause", use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    reset_btn = st.button("🔁 Reset Simulation", use_container_width=True)
    if reset_btn:
        st.session_state.running         = False
        st.session_state.row_index       = 0
        st.session_state.history         = []
        st.session_state.total_processed = 0
        st.session_state.attack_count    = 0
        st.session_state.benign_count    = 0
        st.session_state.class_counter   = Counter()
        st.session_state.alert_queue     = deque(maxlen=10)
        st.rerun()

    st.markdown("---")
    st.caption(f"Row {st.session_state.row_index} / "
               f"{len(st.session_state.sim_data) if st.session_state.sim_data is not None else '–'}")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛡️ Real-Time Network Intrusion Detection System")
st.markdown(
    "Two-layer ML pipeline: **Layer 1** binary (BENIGN/ATTACK) → "
    "**Layer 2** multi-class attack identification."
)

# ── Status banner ─────────────────────────────────────────────────────────────
status_placeholder = st.empty()
if not st.session_state.models_loaded or not st.session_state.data_loaded:
    status_placeholder.warning(
        "⚠️  Load models and simulation data using the sidebar before starting."
    )
elif st.session_state.running:
    status_placeholder.success("🟢  Simulation RUNNING")
else:
    status_placeholder.info("🟡  Simulation PAUSED")

# ── KPI row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi_total   = kpi1.empty()
kpi_attacks = kpi2.empty()
kpi_benign  = kpi3.empty()
kpi_rate    = kpi4.empty()

# ── Charts row ────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.subheader("📈 Rolling Attack Rate (last 50 rows)")
    rolling_chart = st.empty()

with chart_col2:
    st.subheader(" Class Distribution")
    pie_chart = st.empty()

# ── Alert feed ────────────────────────────────────────────────────────────────
st.subheader("🚨 Alert Feed")
alert_placeholder = st.empty()

# ── Detection log ─────────────────────────────────────────────────────────────
st.subheader("📋 Detection Log")
log_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: render charts & KPIs
# ─────────────────────────────────────────────────────────────────────────────

def render_kpis():
    total   = st.session_state.total_processed
    attacks = st.session_state.attack_count
    benign  = st.session_state.benign_count
    rate    = (attacks / total * 100) if total > 0 else 0.0

    kpi_total.metric("Total Processed", f"{total:,}")
    kpi_attacks.metric("Attacks Detected", f"{attacks:,}",
                        delta=None if total == 0 else f"{rate:.1f}%")
    kpi_benign.metric("Benign Flows",  f"{benign:,}")
    kpi_rate.metric("Attack Rate", f"{rate:.1f}%")


def render_rolling_chart():
    history = st.session_state.history
    if len(history) < 2:
        rolling_chart.info("Waiting for data …")
        return

    df_h  = pd.DataFrame(history[-ROLLING_WINDOW:]).reset_index(drop=True)
    df_h["attack_flag"] = (df_h["binary_pred"] == "ATTACK").astype(int)
    # Calculate running average of attack rate (not expanding from beginning)
    df_h["rolling_rate"] = (df_h["attack_flag"].sum() / (df_h.index + 1)) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_h.index, y=df_h["rolling_rate"],
        mode="lines", name="Attack rate %",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis_title="Row #",
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    rolling_chart.plotly_chart(fig, use_container_width=True)


def render_pie():
    counter = st.session_state.class_counter
    if not counter:
        pie_chart.info("Waiting for data …")
        return

    labels = list(counter.keys())
    values = list(counter.values())
    colors = [label_color(l) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        textfont_size=10,
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    pie_chart.plotly_chart(fig, use_container_width=True)


def render_alerts():
    alerts = list(st.session_state.alert_queue)
    if not alerts:
        alert_placeholder.info("No alerts yet.")
        return
    html = ""
    for a in reversed(alerts[-5:]):
        # Color based on correctness
        if a["correct_multi"]:
            bg = "#1a4d2e"  # Dark green - correct
        else:
            bg = "#3d0000"  # Dark red - incorrect
        
        html += (
            f'<div style="background:{bg};border-radius:8px;'
            f'padding:8px 14px;margin-bottom:6px;">'
            f'<strong>Row {a["row_idx"]}</strong> &nbsp;'
            f'{badge(a["binary_pred"])} &nbsp; → &nbsp;'
            f'{badge(a["multi_pred"])} &nbsp; '
            f'<span style="color:#aaa;font-size:0.8rem;">'
            f'conf {a["multi_conf"]:.2%} | true: {a["true_label"]}</span>'
            f'</div>'
        )
    alert_placeholder.markdown(html, unsafe_allow_html=True)


def render_log():
    history = st.session_state.history
    rows_to_show = [
        r for r in history[-30:]
        if show_benign or r["binary_pred"] == "ATTACK"
    ]
    if not rows_to_show:
        log_placeholder.info("No entries to display.")
        return

    df_log = pd.DataFrame(rows_to_show)[
        ["row_idx", "true_label", "binary_pred", "binary_conf",
         "multi_pred", "multi_conf", "correct_binary", "correct_multi"]
    ].rename(columns={
        "row_idx":        "Row",
        "true_label":     "True Label",
        "binary_pred":    "L1 Prediction",
        "binary_conf":    "L1 Conf",
        "multi_pred":     "L2 Prediction",
        "multi_conf":     "L2 Conf",
        "correct_binary": "L1 ✓",
        "correct_multi":  "L2 ✓",
    })
    df_log["L1 Conf"] = df_log["L1 Conf"].apply(lambda x: f"{x:.2%}")
    df_log["L2 Conf"] = df_log["L2 Conf"].apply(lambda x: f"{x:.2%}")

    # Highlight based on prediction correctness
    def highlight(row):
        if row["L2 ✓"]:
            # Correct prediction - light green
            return ["background-color: #1a4d2e"] * len(row)
        else:
            # Incorrect prediction - red
            return ["background-color: #3d0000"] * len(row)

    log_placeholder.dataframe(
        df_log.iloc[::-1].style.apply(highlight, axis=1),
        use_container_width=True,
        height=400,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def process_one_row():
    """Process the next row from simulation data and update session state."""
    df           = st.session_state.sim_data
    feature_cols = st.session_state.feature_cols
    idx          = st.session_state.row_index

    if idx >= len(df):
        st.session_state.running = False
        return False

    row       = df.iloc[idx]
    X_row     = row[feature_cols].values.reshape(1, -1).astype(float)
    true_label = row["Label"]

    binary_pred, binary_conf, multi_pred, multi_conf = predict_row(X_row)

    # Determine correctness
    true_binary = "BENIGN" if true_label == "BENIGN" else "ATTACK"
    correct_bin   = binary_pred == true_binary
    correct_multi = (multi_pred == true_label) if binary_pred == "ATTACK" else (binary_pred == true_binary)

    record = {
        "row_idx":        idx,
        "true_label":     true_label,
        "binary_pred":    binary_pred,
        "binary_conf":    binary_conf,
        "multi_pred":     multi_pred,
        "multi_conf":     multi_conf,
        "correct_binary": correct_bin,
        "correct_multi":  correct_multi,
    }

    # Update state
    st.session_state.history.append(record)
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history.pop(0)

    st.session_state.total_processed += 1
    if binary_pred == "ATTACK":
        st.session_state.attack_count += 1
        st.session_state.alert_queue.append(record)
    else:
        st.session_state.benign_count += 1

    st.session_state.class_counter[multi_pred] += 1
    st.session_state.row_index = idx + 1
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Render current state (always, even when paused)
# ─────────────────────────────────────────────────────────────────────────────
render_kpis()
render_rolling_chart()
render_pie()
render_alerts()
render_log()

# ─────────────────────────────────────────────────────────────────────────────
# Simulation tick
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    ok = process_one_row()
    if ok:
        time.sleep(interval)
        st.rerun()
    else:
        st.session_state.running = False
        status_placeholder.success("✅  Simulation complete — all rows processed.")
        st.balloons()
