"""
Streamlit MLOps dashboard for the electricity price forecasting project.

A single-file, multi-page app (st.sidebar.radio navigation) that works both:
  - online  — talking to the FastAPI server (live predictions, pipeline triggers)
  - offline — reading evaluation artifacts directly from disk

Run with:
    streamlit run src/dashboard.py --server.port=8501
"""
import os
import sys
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dependency, but be safe
    requests = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent
PROJECT_ROOT = _SRC.parent if _SRC.name == "src" else _SRC

sys.path.insert(0, str(_SRC))

_ARTIFACTS = PROJECT_ROOT / "artifacts"
_SRC_ARTIFACTS = _SRC / "artifacts"
_HETERO = _SRC / "artifacts_hetero"

XGB_METRICS_PATH = _ARTIFACTS / "xgboost_metrics.json"
HOMO_METRICS_PATH = _SRC_ARTIFACTS / "homo_gnn_metrics.json"
GAT_METRICS_PATH = _HETERO / "gat_metrics_clean.json"
HETERO_METRICS_PATH = _HETERO / "hetero_metrics_clean.json"
ST_HETERO_METRICS_PATH = _HETERO / "st_hetero_metrics.json"
DAY_AHEAD_PATH = _HETERO / "day_ahead_results.json"
ABLATION_PATH = _HETERO / "st_ablation_results.json"
ROBUSTNESS_PATH = _HETERO / "st_robustness_results.json"
INTERP_PATH = _HETERO / "st_interpretability.json"
LAST_RUN_PATH = _HETERO / "last_pipeline_run.json"

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

HETERO_COLOR = "#2563eb"
ST_COLOR = "#7c3aed"
NEUTRAL_COLORS = ["#9ca3af", "#6b7280", "#4b5563", HETERO_COLOR, ST_COLOR]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json_safe(path):
    """Load a JSON file, returning a dict (or list) or None on any error."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def api_get(endpoint):
    """GET against the FastAPI server. Returns parsed JSON or None on failure."""
    if requests is None:
        return None
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def api_post(endpoint, json_payload=None, params=None):
    """POST against the FastAPI server. Returns parsed JSON or None on failure."""
    if requests is None:
        return None
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, json=json_payload or {}, params=params, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        # Surface useful error detail back to the caller
        try:
            return {"_error": resp.status_code, "detail": resp.json().get("detail")}
        except Exception:
            return {"_error": resp.status_code, "detail": resp.text}
    except Exception:
        return None


def api_is_up():
    """Return True if GET /health succeeds."""
    return api_get("/health") is not None


def api_banner():
    """Render a small connectivity banner in the sidebar."""
    if api_is_up():
        st.sidebar.success("🟢 API connected")
        return True
    st.sidebar.info("🔵 Offline mode (reading artifacts directly)")
    return False


def _fmt(x, nd=2):
    try:
        return round(float(x), nd)
    except Exception:
        return x


# ---------------------------------------------------------------------------
# Page 1 — Overview & Leaderboard
# ---------------------------------------------------------------------------
def page_overview():
    st.title("⚡ Electricity Price Forecasting — MLOps Dashboard")
    st.markdown(
        "**Research question:** *How can a heterogeneous graph be effectively constructed "
        "and integrated into GNN models to improve accuracy, interpretability, and "
        "robustness of multi-area day-ahead electricity price forecasting in the Nordic "
        "power market?*"
    )

    xgb = load_json_safe(XGB_METRICS_PATH)
    homo = load_json_safe(HOMO_METRICS_PATH)
    gat = load_json_safe(GAT_METRICS_PATH)
    hetero = load_json_safe(HETERO_METRICS_PATH)
    st_m = load_json_safe(ST_HETERO_METRICS_PATH)

    if not any([xgb, homo, gat, hetero, st_m]):
        st.warning("Run the training scripts (make train-all) to generate model metrics.")
        return

    models = [
        ("XGBoost (tabular baseline)", xgb),
        ("Homogeneous GNN (GraphSAGE)", homo),
        ("GAT", gat),
        ("HeteroSAGE", hetero),
        ("ST-HeteroSAGE (ours ★)", st_m),
    ]

    rows = []
    for name, m in models:
        if not m:
            rows.append({"Model": name, "MAE (DKK)": None, "RMSE": None, "R²": None, "SMAPE (%)": None})
            continue
        rows.append({
            "Model": name,
            "MAE (DKK)": _fmt(m.get("mae")),
            "RMSE": _fmt(m.get("rmse")),
            "R²": _fmt(m.get("r2"), 4),
            "SMAPE (%)": _fmt(m.get("smape")),
        })
    df = pd.DataFrame(rows)

    st.subheader("🏆 Model Leaderboard (DK1+DK2 test set, chronological 80/10/10 split)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- metric row -------------------------------------------------------
    best = st_m or hetero
    if best:
        c1, c2, c3 = st.columns(3)
        label = "ST-HeteroSAGE" if st_m else "HeteroSAGE"
        c1.metric(f"{label} MAE", f"{_fmt(best.get('mae'))} DKK")
        c2.metric(f"{label} R²", f"{_fmt(best.get('r2'), 4)}")
        if xgb and xgb.get("mae"):
            impr = (xgb["mae"] - best.get("mae", 0)) / xgb["mae"] * 100.0
            c3.metric("MAE improvement vs XGBoost", f"{impr:+.1f}%")
        else:
            c3.metric("MAE improvement vs XGBoost", "n/a")

    # ---- charts -----------------------------------------------------------
    chart_df = df.dropna(subset=["MAE (DKK)"])
    if not chart_df.empty:
        def _bar_color(name):
            if "ST-HeteroSAGE" in name:
                return ST_COLOR
            if "HeteroSAGE" in name:
                return HETERO_COLOR
            return "#9ca3af"

        colors = [_bar_color(n) for n in chart_df["Model"]]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_mae = go.Figure(
                go.Bar(
                    x=chart_df["Model"],
                    y=chart_df["MAE (DKK)"],
                    marker_color=colors,
                    text=chart_df["MAE (DKK)"],
                    textposition="outside",
                )
            )
            fig_mae.update_layout(
                title="MAE across models (lower is better)",
                yaxis_title="MAE (DKK)",
                xaxis_tickangle=-20,
                height=440,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_b:
            r2_df = chart_df.dropna(subset=["R²"])
            fig_r2 = go.Figure(
                go.Bar(
                    x=r2_df["Model"],
                    y=r2_df["R²"],
                    marker_color=[_bar_color(n) for n in r2_df["Model"]],
                    text=r2_df["R²"],
                    textposition="outside",
                )
            )
            fig_r2.update_layout(
                title="R² across models (higher is better)",
                yaxis_title="R²",
                xaxis_tickangle=-20,
                height=440,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Verdict")
    st.markdown(
        "**ST-HeteroSAGE** (spatio-temporal heterogeneous GNN with CausalTCN) achieves the "
        "best results across all metrics: **MAE 151 DKK, R² 0.696** — a **26.5% MAE reduction** "
        "versus XGBoost and a 6.7% improvement over the homogeneous GraphSAGE baseline. "
        "The ablation study shows that CausalTCN accounts for the largest single gain, while "
        "the heterogeneous market bridge (market nodes + interconnect edges) provides a further "
        "meaningful improvement over a purely spatial or purely temporal approach."
    )


# ---------------------------------------------------------------------------
# Page 2 — Live Prediction
# ---------------------------------------------------------------------------
def page_predict(api_up):
    st.title("🔮 Live Prediction")
    st.markdown(
        "Single-step price forecast from the served HeteroSAGE model. Inputs map "
        "directly onto the FastAPI `PredictRequest` schema."
    )

    if not api_up:
        st.info(
            "The prediction endpoint needs the FastAPI server running. Start it with "
            "`make serve` (or `docker compose up api`), then reload this page. "
            f"The dashboard is targeting `API_BASE={API_BASE}`."
        )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            zone = st.selectbox("Zone", ["DK1", "DK2"])
            lag_24h = st.number_input("lag_24h (DKK)", value=500.0, step=10.0)
            lag_48h = st.number_input("lag_48h (DKK)", value=510.0, step=10.0)
            lag_168h = st.number_input("lag_168h (DKK)", value=480.0, step=10.0)
        with c2:
            rolling_24h_mean = st.number_input("rolling_24h_mean (DKK)", value=495.0, step=10.0)
            rolling_24h_std = st.number_input("rolling_24h_std (DKK)", value=60.0, step=5.0)
            temperature_c = st.number_input("temperature_c (°C)", value=8.0, step=0.5)
            wind_speed_ms = st.number_input("wind_speed_ms (m/s)", value=6.0, step=0.5)
        with c3:
            cloud_cover_pct = st.number_input("cloud_cover_pct (%)", value=50.0, min_value=0.0, max_value=100.0, step=5.0)
            humidity_pct = st.number_input("humidity_pct (%)", value=70.0, min_value=0.0, max_value=100.0, step=5.0)
            hour_of_day = st.slider("hour_of_day", 0, 23, 18)
            day_of_week = st.slider("day_of_week (0=Mon)", 0, 6, 0)

        submitted = st.form_submit_button("Predict price", type="primary", disabled=not api_up)

    if submitted:
        payload = {
            "zone": zone,
            "lag_24h": lag_24h,
            "lag_48h": lag_48h,
            "lag_168h": lag_168h,
            "rolling_24h_mean": rolling_24h_mean,
            "rolling_24h_std": rolling_24h_std,
            "temperature_c": temperature_c,
            "wind_speed_ms": wind_speed_ms,
            "cloud_cover_pct": cloud_cover_pct,
            "humidity_pct": humidity_pct,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        }
        result = api_post("/predict", payload)
        if result is None:
            st.error("Could not reach the API. Is the server running (`make serve`)?")
        elif "_error" in result:
            st.error(f"API returned {result['_error']}: {result.get('detail')}")
        else:
            st.metric(
                f"Predicted price — {result.get('zone', zone)}",
                f"{result.get('predicted_price_dkk', float('nan')):.2f} DKK",
            )


# ---------------------------------------------------------------------------
# Page 3 — Forecast Analysis
# ---------------------------------------------------------------------------
def page_forecast():
    st.title("📈 Forecast Analysis")
    da = load_json_safe(DAY_AHEAD_PATH)
    if not da:
        st.warning("Run day_ahead_forecast.py to generate day_ahead_results.json.")
        return

    # ---- per-zone metrics -------------------------------------------------
    st.subheader("Per-zone day-ahead metrics — HeteroSAGE")
    pz = da.get("per_zone_metrics", {})
    rows = []
    for zone, m in pz.items():
        rows.append({
            "Zone": zone,
            "MAE (DKK)": _fmt(m.get("mae")),
            "RMSE": _fmt(m.get("rmse")),
            "R²": _fmt(m.get("r2"), 4),
            "SMAPE (%)": _fmt(m.get("smape")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Note: **HYDRO** and **DE** are zero-filled market-context nodes (no real price "
        "target), so their near-zero MAE / R²=0 are expected and should be ignored. The "
        "real evaluation is on **DK1** and **DK2**."
    )

    zone = st.selectbox("Zone for profile charts", ["DK1", "DK2"])
    profiles = da.get("price_profiles", {}).get(zone, {})

    # ---- daily profile ----------------------------------------------------
    st.subheader(f"Daily price profile — {zone} (by hour of day)")
    by_hour = profiles.get("by_hour", {})
    if by_hour:
        hours = sorted(by_hour.keys(), key=lambda h: int(h))
        actual = [by_hour[h]["actual"] for h in hours]
        pred = [by_hour[h]["predicted"] for h in hours]
        xs = [int(h) for h in hours]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=actual, name="Actual", mode="lines+markers", line=dict(color="#111827")))
        fig.add_trace(go.Scatter(x=xs, y=pred, name="Predicted", mode="lines+markers", line=dict(color=HETERO_COLOR)))
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Price (DKK)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No by-hour profile available for this zone.")

    # ---- weekly profile ---------------------------------------------------
    st.subheader(f"Weekly price profile — {zone} (by day of week)")
    by_dow = profiles.get("by_day_of_week", {})
    if by_dow:
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days = sorted(by_dow.keys(), key=lambda d: int(d))
        labels = [dow_labels[int(d)] if int(d) < 7 else d for d in days]
        actual = [by_dow[d]["actual"] for d in days]
        pred = [by_dow[d]["predicted"] for d in days]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=actual, name="Actual", marker_color="#9ca3af"))
        fig.add_trace(go.Bar(x=labels, y=pred, name="Predicted", marker_color=HETERO_COLOR))
        fig.update_layout(barmode="group", xaxis_title="Day of week", yaxis_title="Price (DKK)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No by-day-of-week profile available for this zone.")

    # ---- horizon degradation ---------------------------------------------
    st.subheader("Horizon degradation — DK1 (recursive 24-step forecast)")
    dk1_h = da.get("horizon_mae", {}).get("DK1", [])
    if dk1_h:
        steps = list(range(1, len(dk1_h) + 1))
        fig = go.Figure(
            go.Scatter(x=steps, y=dk1_h, mode="lines+markers", line=dict(color="#dc2626"))
        )
        fig.update_layout(
            xaxis_title="Forecast horizon (hours ahead)",
            yaxis_title="MAE (DKK, log scale)",
            yaxis_type="log",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The y-axis is log-scaled: recursive forecasting feeds each prediction back "
            "in as the next step's lag feature, so errors **compound** and MAE blows up "
            "by orders of magnitude over a 24-hour horizon."
        )
    else:
        st.warning("No DK1 horizon_mae array available.")


# ---------------------------------------------------------------------------
# Page 4 — Interpretability
# ---------------------------------------------------------------------------
def page_interpretability():
    st.title("🔍 Interpretability")

    interp = load_json_safe(INTERP_PATH)
    abl = load_json_safe(ABLATION_PATH)

    # ---- feature importance (ST format: list of {feature, importance}) ---
    st.subheader("Feature importance (gradient saliency — ST-HeteroSAGE)")
    fi_raw = (interp or {}).get("feature_importance")
    if not fi_raw:
        st.warning("Run st_interpretability.py to generate st_interpretability.json.")
    else:
        # ST format: list[{feature, importance}] — already sorted descending
        if isinstance(fi_raw, list):
            items = sorted(fi_raw, key=lambda d: d["importance"])
            names = [d["feature"] for d in items]
            vals  = [d["importance"] for d in items]
            max_v = max(vals) if vals else 1.0
            vals_norm = [v / max_v for v in vals]
        else:
            # fallback: old zone-keyed dict format
            zone = st.selectbox("Zone", list(fi_raw.keys()))
            scores = fi_raw.get(zone, {})
            items_kv = sorted(scores.items(), key=lambda kv: kv[1])
            names = [k for k, _ in items_kv]
            vals_norm = [v for _, v in items_kv]
        fig = go.Figure(go.Bar(x=vals_norm, y=names, orientation="h", marker_color=ST_COLOR))
        fig.update_layout(
            title="ST-HeteroSAGE feature importance (normalised gradient saliency)",
            xaxis_title="Saliency (normalised, top feature = 1.0)",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Gas price and day-ahead lag (t-24h) dominate; renewable generation and "
            "weekly seasonality follow. No t-1 leakage — all lags ≥ 24 h."
        )

    # ---- temporal error profile ------------------------------------------
    err = (interp or {}).get("error_by_time", {})
    if err:
        col_h, col_d = st.columns(2)
        with col_h:
            st.subheader("MAE by hour of day")
            by_h = err.get("by_hour", {})
            if by_h:
                hours = sorted(by_h.keys(), key=int)
                fig = go.Figure(go.Bar(
                    x=[int(h) for h in hours],
                    y=[by_h[h] for h in hours],
                    marker_color=ST_COLOR,
                ))
                fig.update_layout(xaxis_title="Hour", yaxis_title="MAE (DKK)", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col_d:
            st.subheader("MAE by day of week")
            by_d = err.get("by_dow", {})
            if by_d:
                labels_dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                days = sorted(by_d.keys(), key=int)
                fig = go.Figure(go.Bar(
                    x=[labels_dow[int(d)] if int(d) < 7 else d for d in days],
                    y=[by_d[d] for d in days],
                    marker_color=ST_COLOR,
                ))
                fig.update_layout(xaxis_title="Day", yaxis_title="MAE (DKK)", height=340)
                st.plotly_chart(fig, use_container_width=True)

    # ---- ablation (ST format: {A_full, B_no_tcn, C_no_spatial, ...}) ----
    st.subheader("Ablation study — overall MAE by ST-HeteroSAGE component removed")
    if not abl:
        st.warning("Run st_ablation.py to generate st_ablation_results.json.")
        return

    st_labels = {
        "A_full":      "A. Full ST-HeteroSAGE",
        "B_no_tcn":    "B. No CausalTCN",
        "C_no_spatial":"C. No spatial edges",
        "D_no_cooccurs":"D. No co-occurs edges",
        "E_no_market": "E. No market nodes",
        "F_no_hydro":  "F. No HYDRO zone",
    }
    full_mae = abl.get("A_full", {}).get("mae", 0)
    rows = []
    for key, label in st_labels.items():
        if key not in abl:
            continue
        m = abl[key]
        delta = _fmt(m.get("mae", 0) - full_mae)
        rows.append({
            "Variant": label,
            "key": key,
            "MAE (DKK)": _fmt(m.get("mae")),
            "Δ vs full": delta,
            "R²": _fmt(m.get("r2"), 4),
        })
    if not rows:
        st.warning("ST ablation file present but missing expected variants.")
        return

    abl_df = pd.DataFrame(rows)
    colors = [ST_COLOR if r == "A_full" else "#f59e0b" for r in abl_df["key"]]
    fig = go.Figure(
        go.Bar(
            x=abl_df["Variant"],
            y=abl_df["MAE (DKK)"],
            marker_color=colors,
            text=[f"Δ {d:+.1f}" if isinstance(d, (int, float)) else "" for d in abl_df["Δ vs full"]],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="MAE (DKK)", xaxis_tickangle=-20, height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(abl_df.drop(columns=["key"]), use_container_width=True, hide_index=True)

    st.markdown(
        "**Which ST-HeteroSAGE components matter?**\n\n"
        "- **CausalTCN (B)** is the single biggest contributor: removing it increases MAE "
        "by **+106 DKK (+70%)** — dilated temporal convolutions capture the intra-day and "
        "weekly periodicity that GNN message-passing alone cannot.\n"
        "- **Spatial edges (C)** matter more than in the simpler HeteroSAGE: removing them "
        "adds **+158 DKK** because the TCN operates zone-independently and the GNN layer is "
        "the only path for cross-zone information.\n"
        "- **co_occurs_with edges (D)** and **market nodes (E)** each contribute ~10–25 DKK — "
        "meaningful but secondary to the temporal component."
    )


# ---------------------------------------------------------------------------
# Page 5 — Robustness
# ---------------------------------------------------------------------------
def page_robustness():
    st.title("🛡️ Robustness")
    rob = load_json_safe(ROBUSTNESS_PATH)
    if not rob:
        st.warning("Run st_robustness.py to generate st_robustness_results.json.")
        return

    # ST format: baseline = {mae, r2}; scenarios = {mae, r2, delta_mae, delta_pct}
    base_info = rob.get("baseline", {})
    # Accept both flat and DK1-nested formats
    base_mae = base_info.get("mae") or (base_info.get("DK1") or {}).get("mae")
    if base_mae is not None:
        st.metric("Baseline MAE (clean inputs)", f"{base_mae:.2f} DKK")

    families = [
        ("gaussian_noise", "Gaussian noise injection"),
        ("feature_dropout", "Feature dropout"),
        ("price_spike", "Price spike"),
    ]

    for key, label in families:
        group = rob.get(key, {})
        if not group:
            continue
        st.subheader(label)
        scenarios = list(group.keys())

        def _get_mae(s_dict):
            if "mae" in s_dict:
                return s_dict["mae"]
            return (s_dict.get("DK1") or {}).get("mae", 0)

        def _get_delta(s_dict):
            if "delta_pct" in s_dict:
                return s_dict["delta_pct"]
            return (s_dict.get("DK1") or {}).get("mae_delta_pct", 0)

        maes   = [_get_mae(group[s])   for s in scenarios]
        deltas = [_get_delta(group[s]) for s in scenarios]

        fig = go.Figure(
            go.Bar(
                x=scenarios,
                y=maes,
                marker_color=ST_COLOR,
                text=[f"{d:+.2f}%" for d in deltas],
                textposition="outside",
            )
        )
        if base_mae is not None:
            fig.add_hline(
                y=base_mae,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text="baseline",
                annotation_position="top left",
            )
        fig.update_layout(yaxis_title="MAE (DKK)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Takeaway:**\n\n"
        "- **Gaussian noise** causes only a modest MAE increase even at 30% noise (+6.5%), "
        "because the CausalTCN and graph structure distribute the signal across multiple "
        "lag paths — no single input is critical.\n"
        "- **Feature dropout** degrades performance more sharply (up to +30% at 30% dropout), "
        "since randomly zeroing features removes temporal context the TCN relies on.\n"
        "- **Price spike multipliers** have minimal impact: the spike is applied to ground-truth "
        "targets only, not to input lags, so the model's predictions are unchanged."
    )


# ---------------------------------------------------------------------------
# Page 6 — MLOps & Monitoring
# ---------------------------------------------------------------------------
def page_mlops(api_up):
    st.title("⚙️ MLOps & Monitoring")

    # ---- pipeline status --------------------------------------------------
    st.subheader("Pipeline status")
    status = api_get("/pipeline/status") if api_up else None
    source = "API"
    if status is None:
        status = load_json_safe(LAST_RUN_PATH)
        source = "last_pipeline_run.json"

    if not status:
        st.warning("No pipeline run found yet. Trigger one below or run `make pipeline`.")
    else:
        st.caption(f"Source: {source}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", str(status.get("status", "unknown")))
        dur = status.get("duration_seconds")
        c2.metric("Duration", f"{dur:.1f}s" if isinstance(dur, (int, float)) else "—")
        c3.metric("Improved", "✅" if status.get("improved") else "—")

        if status.get("timestamp"):
            st.write(f"**Timestamp:** {status['timestamp']}")

        stages = status.get("stages_completed", []) or []
        if stages:
            st.write("**Stages completed:**")
            st.write("  ".join(
                f"{'⏭️' if ':skipped' in str(s) else '✅'} {str(s).replace(':skipped', '')}"
                for s in stages
            ))

        errors = status.get("errors") or ([status["error"]] if status.get("error") else [])
        if errors:
            st.error("Errors:\n\n" + "\n".join(f"- {e}" for e in errors))

    # ---- trigger ----------------------------------------------------------
    st.subheader("Trigger a pipeline run")
    force_rebuild = st.checkbox("Force full rebuild (fresh scaler + from-scratch retrain)", value=False)
    if api_up:
        if st.button("🚀 Trigger Pipeline Run", type="primary"):
            resp = api_post("/pipeline/run", params={"force_rebuild": force_rebuild})
            if resp is None:
                st.error("Could not reach the API.")
            elif "_error" in resp:
                st.error(f"API returned {resp['_error']}: {resp.get('detail')}")
            else:
                st.success(f"{resp.get('status', 'ok')}: {resp.get('message', '')}")
    else:
        st.button("🚀 Trigger Pipeline Run", disabled=True)
        st.caption("Triggering the pipeline requires the API server (`make serve`).")

    # ---- monitoring report -----------------------------------------------
    st.subheader("Monitoring report")
    report = api_get("/monitoring/report") if api_up else None
    mon_source = "API"
    if report is None:
        try:
            from monitoring import get_monitoring_report
            report = get_monitoring_report()
            mon_source = "monitoring.get_monitoring_report() (direct import)"
        except Exception as exc:
            report = None
            st.warning(f"Monitoring report unavailable: {exc}")

    if report:
        st.caption(f"Source: {mon_source}")
        c1, c2, c3 = st.columns(3)
        dk1 = report.get("rolling_mae_dk1", {}) or {}
        dk2 = report.get("rolling_mae_dk2", {}) or {}
        c1.metric(
            "Rolling MAE — DK1",
            f"{dk1['mae']:.2f} DKK" if dk1.get("mae") is not None else "no data",
            help=f"{dk1.get('n_samples', 0)} samples / {dk1.get('window_hours', '?')}h window",
        )
        c2.metric(
            "Rolling MAE — DK2",
            f"{dk2['mae']:.2f} DKK" if dk2.get("mae") is not None else "no data",
            help=f"{dk2.get('n_samples', 0)} samples / {dk2.get('window_hours', '?')}h window",
        )
        c3.metric("Predictions (last 24h)", report.get("predictions_last_24h", 0))

        drift = report.get("drift_status", {}) or {}
        d_status = drift.get("status", "unknown")
        if d_status == "drift_detected":
            st.error(f"Drift detected: {drift.get('drifted_features', [])}")
        else:
            note = drift.get("note", "")
            st.success(f"Drift status: {d_status}" + (f" — {note}" if note else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Electricity Forecasting MLOps",
        page_icon="⚡",
        layout="wide",
    )

    st.sidebar.title("⚡ Electricity Forecasting")
    api_up = api_banner()

    if st.sidebar.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview & Leaderboard",
            "🔮 Live Prediction",
            "📈 Forecast Analysis",
            "🔍 Interpretability",
            "🛡️ Robustness",
            "⚙️ MLOps & Monitoring",
        ],
    )

    st.sidebar.caption(f"API base: `{API_BASE}`")

    if page == "📊 Overview & Leaderboard":
        page_overview()
    elif page == "🔮 Live Prediction":
        page_predict(api_up)
    elif page == "📈 Forecast Analysis":
        page_forecast()
    elif page == "🔍 Interpretability":
        page_interpretability()
    elif page == "🛡️ Robustness":
        page_robustness()
    elif page == "⚙️ MLOps & Monitoring":
        page_mlops(api_up)


if __name__ == "__main__":
    main()
