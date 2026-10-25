"""
PredictIQ — Streamlit Dashboard
Deep-dark engineering UI with glow effects, grid background, and custom design tokens.
Reads from PostgreSQL (latest pipeline run) and can trigger new DAG runs.
"""
from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar always open — hide collapse button with multiple selector strategies
st.markdown("""
<style>
/* Hide the collapse/expand arrow — try every known selector */
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
[class*="collapsedControl"],
button[kind="header"],
.st-emotion-cache-1egp75f,
section[data-testid="stSidebar"] > div > div > div > button,
section[data-testid="stSidebar"] button[title="Close sidebar"],
section[data-testid="stSidebar"] button[aria-label="Close sidebar"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
/* Lock sidebar open */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
    transform: none !important;
    left: 0 !important;
}
/* Hide the >> expand button on the main area */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}
</style>
<script>
// JS fallback: keep removing the collapse button if it appears
const hideBtn = () => {
    const btns = document.querySelectorAll(
        '[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], ' +
        'section[data-testid="stSidebar"] button'
    );
    btns.forEach(b => { b.style.display = 'none'; });
};
setInterval(hideBtn, 500);
</script>
""", unsafe_allow_html=True)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design tokens ──────────────────────────────────────────────────────── */
:root {
  --bg:           #03050a;
  --panel:        #090e1a;
  --panel-strong: #0d1526;
  --ink:          #f0f8ff;
  --muted:        #7ea4c8;
  --line:         #1e3254;
  --soft-line:    #142240;
  --green:        #00f5a0;
  --blue:         #3d9fff;
  --red:          #ff3b4e;
  --amber:        #ffc020;
  --cyan:         #00e5ff;
  --shadow:       0 20px 60px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.06);
  --glow-green:   0 0 32px rgba(0,245,160,0.35);
  --glow-blue:    0 0 32px rgba(61,159,255,0.35);
  --glow-red:     0 0 32px rgba(255,59,78,0.38);
  --glow-cyan:    0 0 28px rgba(0,229,255,0.30);
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
#MainMenu { display: none; }
footer { display: none; }


/* ── Full-app background with grid ─────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(circle at 18% 10%, rgba(0,245,160,0.13), transparent 28%),
        radial-gradient(circle at 88% 4%,  rgba(61,159,255,0.18), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(0,229,255,0.07), transparent 35%),
        linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px),
        #03050a;
    background-size: auto, auto, auto, 34px 34px, 34px 34px, auto;
    color: var(--ink);
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,17,30,0.99), rgba(3,5,10,0.99));
    border-right: 1px solid #1e3254;
    box-shadow: inset -1px 0 0 rgba(61,159,255,0.25);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] .stRadio label { color: #ccdcf0 !important; font-weight: 700; }

/* ── Metric / panel cards ───────────────────────────────────────────────── */
.metric-card {
    border: 1px solid var(--line);
    border-top: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    background: linear-gradient(160deg, rgba(13,21,40,0.97), rgba(5,9,18,0.98));
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 34px;
    font-weight: 900;
    line-height: 1;
}
.metric-critical { color: var(--red);   text-shadow: 0 0 28px rgba(255,59,78,0.55); }
.metric-warning  { color: var(--amber); text-shadow: 0 0 24px rgba(255,192,32,0.48); }
.metric-healthy  { color: var(--green); text-shadow: 0 0 28px rgba(0,245,160,0.50); }
.metric-neutral  { color: var(--cyan);  text-shadow: 0 0 24px rgba(0,229,255,0.48); }

/* ── Section titles ─────────────────────────────────────────────────────── */
.section-title {
    font-size: 11px;
    font-weight: 900;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 22px 0 10px 0;
    border-bottom: 1px solid var(--soft-line);
    padding-bottom: 7px;
    text-shadow: var(--glow-cyan);
}

/* ── Status badges ──────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    min-height: 22px;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.badge-critical {
    border: 1px solid rgba(255,59,78,0.70);
    background: rgba(255,59,78,0.18);
    color: var(--red);
    box-shadow: 0 0 12px rgba(255,59,78,0.25);
}
.badge-monitor {
    border: 1px solid rgba(255,192,32,0.60);
    background: rgba(255,192,32,0.14);
    color: var(--amber);
    box-shadow: 0 0 12px rgba(255,192,32,0.20);
}
.badge-healthy {
    border: 1px solid rgba(0,245,160,0.55);
    background: rgba(0,245,160,0.12);
    color: var(--green);
    box-shadow: 0 0 12px rgba(0,245,160,0.22);
}

/* ── Streamlit native overrides ─────────────────────────────────────────── */
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00b37a, #0099cc);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 900;
    padding: 8px 18px;
    box-shadow: var(--glow-green);
    letter-spacing: 0.04em;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
    transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 44px rgba(0,245,160,0.55);
}

/* Text input */
[data-testid="stTextInput"] input {
    background: #040810;
    border: 1px solid #1e3254;
    border-radius: 5px;
    color: var(--cyan);
    padding: 8px 10px;
    font-weight: 700;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(61,159,255,0.22);
}
[data-testid="stTextInput"] label { color: var(--cyan) !important; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; text-shadow: var(--glow-cyan); }

/* Select / multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #060b13;
    border-color: #263a5b;
    color: var(--ink);
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label { color: var(--muted) !important; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }

/* Radio */
[data-testid="stRadio"] > div { gap: 4px; }
[data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 9px 11px;
    font-weight: 700;
    color: #b9c7db;
    transition: all 120ms;
}
[data-testid="stRadio"] label:hover {
    border-color: rgba(23,216,139,0.38);
    background: rgba(23,216,139,0.07);
    color: var(--green);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
}

/* Spinner */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* Divider */
hr { border-color: var(--soft-line); }

/* Success / error */
[data-testid="stAlert"] {
    background: rgba(23,216,139,0.08);
    border: 1px solid rgba(23,216,139,0.30);
    border-radius: 8px;
    color: var(--green);
}
</style>
""", unsafe_allow_html=True)

# ── Design tokens for Python ──────────────────────────────────────────────────
C_GREEN  = "#00f5a0"
C_BLUE   = "#3d9fff"
C_RED    = "#ff3b4e"
C_AMBER  = "#ffc020"
C_CYAN   = "#00e5ff"
C_MUTED  = "#7ea4c8"
C_INK    = "#f0f8ff"
C_LINE   = "#1e3254"
C_BG     = "#03050a"
C_PANEL  = "#090e1a"

STATUS_COLOR = {
    "CRITICAL": C_RED,
    "MONITOR":  C_AMBER,
    "HEALTHY":  C_GREEN,
}
STATUS_BADGE = {
    "CRITICAL": "badge-critical",
    "MONITOR":  "badge-monitor",
    "HEALTHY":  "badge-healthy",
}

# ── Component display names ────────────────────────────────────────────────────
COMP_NAMES = {
    "comp1": "Fan",
    "comp2": "Core Engine",
    "comp3": "HP Turbine",
    "comp4": "LP Turbine",
}
COMP_KEYS = ["comp1", "comp2", "comp3", "comp4"]

# ── Plain-English status descriptions ─────────────────────────────────────────
STATUS_DESC = {
    "CRITICAL": "Failure likely within 48 hours — immediate action required",
    "MONITOR":  "Degradation detected — schedule inspection within 3 days",
    "HEALTHY":  "Operating normally — no action needed",
}

# ── Priority descriptions ──────────────────────────────────────────────────────
PRIORITY_DESC = {
    1: "Highest — fix before all other machines",
    2: "High — address within 24 hours",
    3: "Medium — schedule this week",
    4: "Low — routine monitoring",
}

# ── Info expander helper ──────────────────────────────────────────────────────
def info_box(summary: str, detail: str) -> None:
    """Render a clickable ℹ️ expander with a full explanation inside."""
    with st.expander(f"ℹ️  {summary}"):
        st.markdown(f"""
        <div style="font-size:13px; color:{C_INK}; line-height:1.8; padding:4px 0;">
            {detail}
        </div>
        """, unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_PANEL,
    font_color=C_INK,
    font_size=12,
)

# ── DB config ─────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "predictiq_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_latest_run() -> tuple[pd.DataFrame, pd.DataFrame, datetime | None]:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            cur = conn.cursor()
            cur.execute("SELECT MAX(pipeline_run_at) FROM classifier_results")
            latest_run_at = cur.fetchone()[0]
            if not latest_run_at:
                return pd.DataFrame(), pd.DataFrame(), None

            clf = pd.read_sql("""
                SELECT machine_id, overall_status, priority,
                       comp1_prob, comp1_alert, comp1_status,
                       comp2_prob, comp2_alert, comp2_status,
                       comp3_prob, comp3_alert, comp3_status,
                       comp4_prob, comp4_alert, comp4_status,
                       pipeline_run_at
                FROM classifier_results
                WHERE pipeline_run_at = %s
                ORDER BY machine_id
            """, conn, params=(latest_run_at,))

            rul = pd.read_sql("""
                SELECT machine_id, component, predicted_rul_hours,
                       day_bucket, day_label, interpretation,
                       pipeline_run_at
                FROM rul_results
                WHERE pipeline_run_at = %s
                ORDER BY machine_id, component
            """, conn, params=(latest_run_at,))

            return clf, rul, latest_run_at
        finally:
            conn.close()
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), None


def trigger_dag(as_of: str) -> bool:
    try:
        import httpx, time
        resp = httpx.post(
            "http://webserver:8080/api/v1/dags/predictiq_pipeline/dagRuns",
            json={"conf": {"as_of": as_of}},
            auth=("admin", "admin"),
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            return False
        run_id = resp.json()["dag_run_id"]
        for _ in range(60):  # wait up to 5 minutes
            time.sleep(5)
            status_resp = httpx.get(
                f"http://webserver:8080/api/v1/dags/predictiq_pipeline/dagRuns/{run_id}",
                auth=("admin", "admin"),
                timeout=10,
            )
            state = status_resp.json().get("state", "")
            if state == "success":
                return True
            if state in ("failed", "upstream_failed"):
                return False
        return False
    except Exception:
        return False


def random_as_of() -> str:
    start = pd.Timestamp("2015-01-15")
    end   = pd.Timestamp("2015-11-15")
    delta = int((end - start).total_seconds() / 3600)
    ts    = start + pd.Timedelta(hours=random.randint(0, delta))
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def dark_table(df: pd.DataFrame, color_fns: dict | None = None) -> None:
    """Render a fully dark-themed HTML table matching the dashboard palette."""
    STATUS_BG  = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
    STATUS_CLR = {"CRITICAL":C_RED,"MONITOR":C_AMBER,"HEALTHY":C_GREEN}

    th_style = (
        f"padding:9px 14px;text-align:left;font-size:11px;font-weight:800;"
        f"text-transform:uppercase;letter-spacing:0.10em;color:{C_MUTED};"
        f"border-bottom:1px solid {C_LINE};background:{C_PANEL};white-space:nowrap;"
    )
    td_base = f"padding:9px 14px;font-size:13px;border-bottom:1px solid {C_LINE};"

    headers = "".join(f"<th style='{th_style}'>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            extra = ""
            if color_fns and col in color_fns:
                extra = color_fns[col](val)
            cells += f"<td style='{td_base}{extra}'>{val}</td>"
        rows_html += f"<tr style='transition:background 120ms;'>{cells}</tr>"

    html = f"""
    <div style="border:1px solid {C_LINE};border-radius:8px;overflow:hidden;
                background:{C_PANEL};box-shadow:0 8px 32px rgba(0,0,0,0.4);">
      <table style="width:100%;border-collapse:collapse;font-family:inherit;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plotly_bar(values, labels, colors, yaxis_title="", height=280):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
        showlegend=False,
        yaxis_title=yaxis_title,
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED, linecolor=C_LINE),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED, linecolor=C_LINE),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:6px 0 18px 0; border-bottom:1px solid {C_LINE}; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px; height:36px; border-radius:8px;
                        background:linear-gradient(135deg,{C_GREEN},{C_BLUE});
                        display:grid; place-items:center;
                        color:#031018; font-weight:900; font-size:16px;
                        box-shadow:0 0 24px rgba(23,216,139,0.24);">⚡</div>
            <div>
                <div style="font-size:16px; font-weight:850; color:{C_INK};">PredictIQ</div>
                <div style="font-size:11px; color:{C_MUTED};">Predictive Maintenance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Overview", "Machine Detail", "Alerts"],
        label_visibility="collapsed",
    )

    st.markdown(f'<div class="section-title">Run Pipeline</div>', unsafe_allow_html=True)

    if "as_of" not in st.session_state:
        st.session_state.as_of = random_as_of()

    AS_OF_MIN = pd.Timestamp("2015-01-15 00:00:00")
    AS_OF_MAX = pd.Timestamp("2015-11-15 23:59:59")
    AS_OF_FMT = "%Y-%m-%d %H:%M:%S"

    def validate_as_of(raw: str):
        """Returns (parsed_ts, error_message). One of the two will be None."""
        raw = raw.strip()
        if not raw:
            return None, "Date cannot be empty"
        try:
            ts = pd.Timestamp(raw)
        except Exception:
            return None, f"Invalid format — use YYYY-MM-DD HH:MM:SS"
        if ts < AS_OF_MIN:
            return None, f"Too early — minimum is {AS_OF_MIN.strftime(AS_OF_FMT)}"
        if ts > AS_OF_MAX:
            return None, f"Too late — maximum is {AS_OF_MAX.strftime(AS_OF_FMT)}"
        return ts, None

    as_of_input = st.text_input(
        "AS_OF DATETIME",
        value=st.session_state.as_of,
        placeholder="YYYY-MM-DD HH:MM:SS",
    )

    # Live validation feedback
    parsed_ts, val_error = validate_as_of(as_of_input)
    if val_error:
        st.markdown(f"""
        <div style="background:rgba(255,59,78,0.12); border:1px solid rgba(255,59,78,0.45);
                    border-radius:6px; padding:8px 10px; margin:4px 0 8px 0;
                    font-size:12px; color:{C_RED}; font-weight:700;">
            ⚠ {val_error}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(0,245,160,0.08); border:1px solid rgba(0,245,160,0.35);
                    border-radius:6px; padding:8px 10px; margin:4px 0 8px 0;
                    font-size:12px; color:{C_GREEN}; font-weight:700;">
            ✓ Valid — {parsed_ts.strftime(AS_OF_FMT)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:10px; color:{C_MUTED}; margin-bottom:8px;">Range: 2015-01-15 → 2015-11-15</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random"):
            st.session_state.as_of = random_as_of()
            st.rerun()
    with col2:
        run_clicked = st.button("▶ Run DAG", disabled=(val_error is not None))

    if run_clicked and parsed_ts is not None:
        with st.spinner(f"Running pipeline for {parsed_ts.strftime(AS_OF_FMT)}…"):
            ok = trigger_dag(parsed_ts.strftime(AS_OF_FMT))
        if ok:
            st.success("Pipeline complete!")
            load_latest_run.clear()
            st.rerun()
        else:
            st.error("DAG failed — check Airflow logs")

    st.markdown("---")
    st.markdown(f'<div style="font-size:11px; color:{C_MUTED}; margin-bottom:6px;">Auto-refresh every 30s</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        load_latest_run.clear()
        st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────

clf, rul, run_at = load_latest_run()

if clf.empty or rul.empty:
    st.markdown(f"""
    <div style="display:grid; min-height:60vh; place-items:center; align-content:center;
                gap:12px; text-align:center;">
        <div style="font-size:52px;">⚡</div>
        <div style="font-size:22px; font-weight:800; color:{C_INK};">No pipeline runs yet</div>
        <div style="font-size:14px; color:{C_MUTED};">
            Pick an as_of datetime in the sidebar and click ▶ Run DAG
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Derived ───────────────────────────────────────────────────────────────────

status_counts = clf["overall_status"].value_counts().to_dict()
n_critical = status_counts.get("CRITICAL", 0)
n_monitor  = status_counts.get("MONITOR",  0)
n_healthy  = status_counts.get("HEALTHY",  0)
n_total    = len(clf)

rul_min = rul.groupby("machine_id")["predicted_rul_hours"].min().reset_index()
rul_min.columns = ["machine_id", "min_rul_hours"]
clf_rich = clf.merge(rul_min, on="machine_id", how="left")

priority_order = {"CRITICAL": 0, "MONITOR": 1, "HEALTHY": 2}
clf_rich["_prio"] = clf_rich["overall_status"].map(priority_order)
clf_rich_sorted = clf_rich.sort_values(["_prio", "min_rul_hours"]).drop(columns="_prio")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    run_at_str = run_at.strftime("%Y-%m-%d %H:%M UTC") if run_at else "—"

    st.markdown(f"""
    <div style="display:flex; align-items:flex-start; justify-content:space-between;
                margin-bottom:4px;">
        <div>
            <h1 style="margin:0 0 4px; font-size:25px; color:#fff;
                       text-shadow:0 0 22px rgba(47,140,255,0.28);">Fleet Overview</h1>
            <p style="margin:0; color:{C_MUTED}; font-size:13px;">
                Real-time health across {n_total} machines
            </p>
        </div>
        <div style="border:1px solid {C_LINE}; border-radius:8px;
                    background:rgba(10,19,32,0.78); padding:8px 14px; text-align:right;">
            <div style="font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:2px;">Last Run</div>
            <div style="font-size:13px; color:{C_CYAN}; font-weight:700;">{run_at_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How to read this dashboard
    with st.expander("ℹ️  How to read this dashboard"):
        st.markdown(f"""
        <div style="font-size:13px; color:{C_INK}; line-height:1.8;">
        <b style="color:{C_CYAN};">What is this?</b><br>
        PredictIQ monitors a fleet of turbofan engines and predicts which components are likely to fail —
        before they actually fail. This gives engineers time to schedule maintenance and avoid unplanned breakdowns.<br><br>

        <b style="color:{C_CYAN};">What do the statuses mean?</b><br>
        🔴 <b style="color:{C_RED};">CRITICAL</b> — {STATUS_DESC["CRITICAL"]}<br>
        🟡 <b style="color:{C_AMBER};">MONITOR</b> — {STATUS_DESC["MONITOR"]}<br>
        🟢 <b style="color:{C_GREEN};">HEALTHY</b> — {STATUS_DESC["HEALTHY"]}<br><br>

        <b style="color:{C_CYAN};">What is RUL?</b><br>
        RUL stands for <b>Remaining Useful Life</b> — how many hours the component is predicted to last
        before it needs maintenance. A low RUL means act soon.<br><br>

        <b style="color:{C_CYAN};">What are the components?</b><br>
        Each engine has 4 monitored parts: <b>Fan</b>, <b>Core Engine</b>, <b>HP Turbine</b> (High-Pressure), and <b>LP Turbine</b> (Low-Pressure).
        </div>
        """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpi_descs = [
        ("Total Machines", n_total,   "metric-neutral",  "Total number of engines being monitored."),
        ("🔴 Critical",    n_critical,"metric-critical", STATUS_DESC["CRITICAL"]),
        ("🟡 Monitor",     n_monitor, "metric-warning",  STATUS_DESC["MONITOR"]),
        ("🟢 Healthy",     n_healthy, "metric-healthy",  STATUS_DESC["HEALTHY"]),
    ]
    for col, (label, val, cls, desc) in zip([c1,c2,c3,c4], kpi_descs):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{val}</div>
                <div style="font-size:11px; color:{C_MUTED}; margin-top:8px; line-height:1.4;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-title">Fleet Status Split</div>', unsafe_allow_html=True)
        info_box(
            "What does this chart show?",
            f"""This pie chart shows the <b>current health split</b> across all {n_total} machines in the fleet.<br><br>
            Each machine is classified into one of three categories based on its sensor readings and failure predictions:<br><br>
            🔴 <b style="color:{C_RED};">CRITICAL</b> — The model has detected that one or more components on this machine are highly likely to fail very soon (within ~48 hours). These machines need <b>immediate attention</b>.<br><br>
            🟡 <b style="color:{C_AMBER};">MONITOR</b> — The model has detected early signs of degradation. The machine is not in immediate danger but should be <b>inspected within the next 3 days</b> to prevent escalation.<br><br>
            🟢 <b style="color:{C_GREEN};">HEALTHY</b> — All components on this machine are operating normally. <b>No action required</b> at this time.<br><br>
            <b>How to use this:</b> If the red slice is growing over multiple runs, your fleet health is deteriorating and you may need to increase maintenance frequency."""
        )
        fig_pie = go.Figure(go.Pie(
            labels=["CRITICAL","MONITOR","HEALTHY"],
            values=[n_critical, n_monitor, n_healthy],
            marker_colors=[C_RED, C_AMBER, C_GREEN],
            hole=0.58,
            textinfo="percent",
            textfont=dict(size=13, color=C_INK),
        ))
        fig_pie.update_layout(
            **PLOTLY_THEME,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=-0.08),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Average Failure Probability by Component</div>', unsafe_allow_html=True)
        info_box(
            "What does this chart show?",
            f"""This bar chart shows the <b>average failure probability per component</b>, calculated across all {n_total} machines.<br><br>
            Each machine has 4 monitored components: <b>Fan, Core Engine, HP Turbine,</b> and <b>LP Turbine</b>. For each component, our ML model produces a failure probability between 0% and 100%.<br><br>
            This chart averages that probability across all machines to show you <b>which type of component is most at risk fleet-wide</b>.<br><br>
            <b>Colour guide:</b><br>
            🔴 <b style="color:{C_RED};">Red bar</b> — Above 50%: this component type is failing frequently across the fleet<br>
            🟡 <b style="color:{C_AMBER};">Amber bar</b> — Between 30–50%: early warning, worth monitoring closely<br>
            🟢 <b style="color:{C_GREEN};">Green bar</b> — Below 30%: this component type is generally healthy<br><br>
            <b>How to use this:</b> If one component type consistently shows high failure rates, it may indicate a systemic issue — a batch of faulty parts, a common operating condition causing wear, or a maintenance gap for that component type."""
        )
        comp_probs = {COMP_NAMES[c]: clf[f"{c}_prob"].mean() for c in COMP_KEYS}
        bar_colors = [C_RED if v>0.5 else C_AMBER if v>0.3 else C_GREEN for v in comp_probs.values()]
        fig_bar = plotly_bar(
            [v*100 for v in comp_probs.values()],
            list(comp_probs.keys()),
            bar_colors,
            yaxis_title="Avg Failure Probability (%)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top 10 table
    st.markdown('<div class="section-title">Top 10 Machines Needing Attention</div>', unsafe_allow_html=True)
    info_box(
        "How is this table sorted and what does each column mean?",
        f"""This table lists the <b>10 machines that need attention most urgently</b>, sorted so the most critical machines appear at the top.<br><br>
        <b>Column explanations:</b><br><br>
        📌 <b>Machine</b> — The unique ID of the engine (e.g. M032 = Machine 32)<br><br>
        🚦 <b>Status</b> — The overall health of this machine. CRITICAL means act now, MONITOR means watch closely, HEALTHY means all good.<br><br>
        🔢 <b>Priority (1=urgent)</b> — A ranking from 1 to 4. Priority 1 means this machine should be fixed <i>before all others</i>. Priority 4 means it can wait.<br><br>
        ⏱ <b>Min RUL (hours)</b> — The shortest Remaining Useful Life across all 4 components on this machine. This is how many hours you have before the most degraded component needs maintenance. <span style="color:{C_RED};">Red = under 48h</span>, <span style="color:{C_AMBER};">Amber = under 72h</span>.<br><br>
        📊 <b>Fan / Core Engine / HP Turbine / LP Turbine Fail %</b> — The failure probability (0–100%) for each individual component. 100% means the model is certain this component will fail very soon. 0% means it's healthy.<br><br>
        <b>How to use this:</b> Work down this list from top to bottom. Schedule maintenance for Priority 1 machines immediately, then work your way down."""
    )
    top10 = clf_rich_sorted.head(10)[[
        "machine_id","overall_status","priority","min_rul_hours",
        "comp1_prob","comp2_prob","comp3_prob","comp4_prob",
    ]].copy()
    top10["min_rul_hours"] = top10["min_rul_hours"].round(1)
    for c in COMP_KEYS:
        top10[c + "_prob"] = (top10[c + "_prob"]*100).round(1)
    top10.columns = [
        "Machine", "Status", "Priority (1=urgent)",
        "Min RUL (hours)",
        f"Fan Fail %", f"Core Engine Fail %", f"HP Turbine Fail %", f"LP Turbine Fail %"
    ]

    STATUS_BG = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
    def _top10_fns(col):
        if col == "Status":
            return lambda v: (f"background:{STATUS_BG.get(v,'')};color:{STATUS_COLOR.get(v,C_INK)};"
                              f"font-weight:800;")
        if col == "Min RUL (hours)":
            return lambda v: (f"color:{C_RED};font-weight:700;" if isinstance(v,(int,float)) and v<48
                              else f"color:{C_AMBER};font-weight:700;" if isinstance(v,(int,float)) and v<72
                              else f"color:{C_INK};")
        return None
    dark_table(top10, {c: _top10_fns(c) for c in top10.columns if _top10_fns(c)})

    # RUL histogram
    st.markdown('<div class="section-title">Remaining Useful Life — All Engines</div>', unsafe_allow_html=True)
    info_box(
        "What does this histogram show?",
        f"""This chart shows the <b>distribution of Remaining Useful Life (RUL)</b> across every component of every machine in the fleet — that's {n_total} machines × 4 components = {n_total*4} data points.<br><br>
        <b>What is RUL?</b><br>
        RUL (Remaining Useful Life) is the number of hours a component is predicted to last before it needs maintenance or replacement. It is calculated by our ML model based on historical sensor patterns.<br><br>
        <b>How to read this chart:</b><br>
        — The <b>horizontal axis</b> shows hours of remaining life<br>
        — The <b>vertical axis</b> shows how many components have that RUL<br>
        — <span style="color:{C_RED};">🔴 Red bars (URGENT)</span> — these components need attention very soon<br>
        — <span style="color:{C_AMBER};">🟡 Amber bars (MONITOR)</span> — these are in early-warning territory<br>
        — <span style="color:{C_GREEN};">🟢 Green bars (HEALTHY)</span> — these components have plenty of life left<br><br>
        <b>How to use this:</b> A healthy fleet should have most bars on the right (high RUL). If you see a large cluster of red bars on the left, a wave of maintenance work is approaching soon."""
    )
    fig_hist = px.histogram(
        rul, x="predicted_rul_hours", nbins=40,
        color="interpretation",
        color_discrete_map={"HEALTHY": C_GREEN, "MONITOR": C_AMBER, "URGENT": C_RED},
        labels={"predicted_rul_hours":"Remaining Useful Life (hours)", "interpretation": "Outlook"},
    )
    fig_hist.update_layout(
        **PLOTLY_THEME, height=280,
        margin=dict(l=20,r=20,t=20,b=20), bargap=0.05,
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED, title="Number of Components"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MACHINE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Machine Detail":
    machine_ids = sorted(clf["machine_id"].unique().tolist())

    col_h, col_s = st.columns([3,1])
    with col_h:
        st.markdown(f'<h1 style="margin:0 0 4px; font-size:25px; color:#fff; text-shadow:0 0 22px rgba(47,140,255,0.28);">Machine Detail</h1>', unsafe_allow_html=True)
    with col_s:
        selected_id = st.selectbox("Machine", machine_ids, label_visibility="collapsed")

    machine_clf = clf[clf["machine_id"] == selected_id].iloc[0]
    machine_rul = rul[rul["machine_id"] == selected_id].sort_values("component")

    status  = machine_clf["overall_status"]
    color   = STATUS_COLOR.get(status, C_INK)
    glow    = f"0 0 28px {color}44"

    prio_int = int(machine_clf['priority'])
    st.markdown(f"""
    <div class="metric-card" style="border-color:{color}55; box-shadow:{glow}; margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div style="font-size:34px; font-weight:900; color:{color};
                        text-shadow:0 0 22px {color}66;">M{selected_id:03d}</div>
            <div>
                <div style="font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                            letter-spacing:0.1em;">Overall Status</div>
                <div style="font-size:20px; font-weight:800; color:{color};">{status}</div>
                <div style="font-size:11px; color:{C_MUTED}; margin-top:3px;">{STATUS_DESC.get(status,"")}</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                            letter-spacing:0.1em;">Repair Priority</div>
                <div style="font-size:20px; font-weight:800; color:{C_CYAN};">#{prio_int}</div>
                <div style="font-size:11px; color:{C_MUTED}; margin-top:3px;">{PRIORITY_DESC.get(prio_int,"")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Component cards
    st.markdown('<div class="section-title">Component Health Breakdown</div>', unsafe_allow_html=True)
    info_box(
        "How do I read these component cards?",
        f"""Each card represents one of the <b>4 monitored components</b> on this engine. Here's what each number means:<br><br>
        🔴 <b>⚠ ALERT badge</b> — This component has crossed the failure threshold. It needs to be inspected or replaced as soon as possible.<br><br>
        📊 <b>Failure Probability (%)</b> — This is the ML model's confidence that this component will fail in the near future, expressed as a percentage.<br>
        &nbsp;&nbsp;&nbsp;• 0–30% = Healthy, no concern<br>
        &nbsp;&nbsp;&nbsp;• 30–50% = Early warning, monitor closely<br>
        &nbsp;&nbsp;&nbsp;• 50–100% = Alert — schedule maintenance<br>
        &nbsp;&nbsp;&nbsp;• 100% = The model is certain failure is imminent<br><br>
        ⏱ <b>Remaining Useful Life (hours)</b> — How many hours this component is predicted to last before it requires maintenance, based on its current degradation pattern. This is calculated by a separate regression model trained on historical failure data.<br><br>
        📅 <b>Day estimate · Outlook</b> — A human-readable translation of the RUL hours into days (e.g. "3–7 days"), and the overall outlook (URGENT / MONITOR / HEALTHY).<br><br>
        <b>What to do:</b> Any component showing an alert badge or above 50% failure probability should be scheduled for maintenance. Start with the component showing the lowest RUL hours."""
    )
    cols = st.columns(4)
    for i, (_, crow) in enumerate(machine_rul.iterrows()):
        comp      = crow["component"]
        comp_name = COMP_NAMES.get(comp, comp)
        rul_h     = crow["predicted_rul_hours"]
        interp    = crow["interpretation"]
        day_label = crow["day_label"]
        prob      = machine_clf[f"{comp}_prob"]
        cstatus   = machine_clf[f"{comp}_status"]
        alert     = machine_clf[f"{comp}_alert"]
        c         = STATUS_COLOR.get(cstatus, C_INK)
        alert_tag = '<span style="font-size:10px; background:rgba(255,59,78,0.2); color:#ff3b4e; border-radius:4px; padding:1px 6px; margin-left:4px;">⚠ ALERT</span>' if alert else ""

        with cols[i]:
            card_html = (
                f'<div class="metric-card" style="border-color:{c}55; box-shadow:0 0 20px {c}22;">'
                f'<div style="font-size:13px; font-weight:850; color:{c}; letter-spacing:0.04em; margin-bottom:4px;">{comp_name}{alert_tag}</div>'
                f'<div style="font-size:10px; color:{C_MUTED}; margin-bottom:10px;">{cstatus}</div>'
                f'<div style="font-size:10px; color:{C_MUTED}; margin-bottom:2px;">Failure Probability</div>'
                f'<div style="font-size:24px; font-weight:800; color:{c}; text-shadow:0 0 16px {c}55; margin-bottom:2px;">{prob*100:.1f}%</div>'
                f'<div style="font-size:10px; color:{C_MUTED}; margin-bottom:10px;">Chance this part fails soon</div>'
                f'<div style="font-size:10px; color:{C_MUTED}; margin-bottom:2px;">Remaining Useful Life</div>'
                f'<div style="font-size:20px; font-weight:700; color:{C_INK}; margin-bottom:2px;">{rul_h:.1f} hours</div>'
                f'<div style="font-size:10px; color:{C_MUTED};">&asymp; {day_label} &middot; {interp}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Remaining Useful Life by Component</div>', unsafe_allow_html=True)
        info_box(
            "What does this chart show?",
            f"""This bar chart shows the <b>Remaining Useful Life (RUL) for each of the 4 components</b> on this specific machine, in hours.<br><br>
            <b>The dotted blue line at 72 hours</b> represents the 3-day maintenance threshold — any component whose bar falls below this line should be scheduled for inspection or replacement within 3 days.<br><br>
            <b>Bar colours:</b><br>
            🔴 <span style="color:{C_RED};">Red</span> — URGENT: below the threshold, act immediately<br>
            🟡 <span style="color:{C_AMBER};">Amber</span> — MONITOR: approaching the threshold, plan maintenance<br>
            🟢 <span style="color:{C_GREEN};">Green</span> — HEALTHY: plenty of life remaining<br><br>
            <b>How to use this:</b> The shortest bar tells you which component to replace first and sets the deadline for this machine's next maintenance visit."""
        )
        interps    = machine_rul["interpretation"].tolist()
        comp_labels = [COMP_NAMES.get(c, c) for c in machine_rul["component"].tolist()]
        bar_colors = [C_RED if i=="URGENT" else C_AMBER if i=="MONITOR" else C_GREEN for i in interps]
        fig_rul = plotly_bar(
            machine_rul["predicted_rul_hours"].tolist(),
            comp_labels,
            bar_colors, yaxis_title="Hours Remaining",
        )
        fig_rul.add_hline(y=72, line_dash="dot", line_color=C_BLUE,
                          annotation_text="3-day threshold", annotation_font_color=C_BLUE)
        st.plotly_chart(fig_rul, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Failure Probability by Component</div>', unsafe_allow_html=True)
        info_box(
            "What does this chart show?",
            f"""This bar chart shows the <b>failure probability for each component</b> on this machine, as predicted by the ML classifier model.<br><br>
            <b>What is failure probability?</b><br>
            The model analyses the machine's recent sensor readings (voltage, rotation, pressure, vibration) and compares them to historical patterns seen before past failures. It outputs a probability between 0% and 100% for each component.<br><br>
            <b>The threshold lines:</b><br>
            🔴 <b style="color:{C_RED};">Red dotted line at 50%</b> — This is the <b>alert threshold</b>. Any component above this line has triggered an active alert and needs maintenance.<br>
            🟡 <b style="color:{C_AMBER};">Amber dotted line at 30%</b> — This is the <b>monitor threshold</b>. Components between 30–50% are showing early degradation signs and should be watched closely.<br><br>
            <b>How to use this:</b> Components above the red line need action now. Components between the two lines should be re-checked at the next scheduled inspection."""
        )
        prob_vals  = [machine_clf[f"{c}_prob"]*100 for c in COMP_KEYS]
        prob_color = [C_RED if v>50 else C_AMBER if v>30 else C_GREEN for v in prob_vals]
        fig_prob   = plotly_bar(prob_vals, [COMP_NAMES[c] for c in COMP_KEYS],
                                prob_color, yaxis_title="Failure Probability (%)")
        fig_prob.add_hline(y=50, line_dash="dot", line_color=C_RED,
                           annotation_text="Alert threshold (50%)", annotation_font_color=C_RED)
        fig_prob.add_hline(y=30, line_dash="dot", line_color=C_AMBER,
                           annotation_text="Monitor threshold (30%)", annotation_font_color=C_AMBER)
        st.plotly_chart(fig_prob, use_container_width=True)

    # Fleet box comparison
    st.markdown('<div class="section-title">How This Machine Compares to the Fleet</div>', unsafe_allow_html=True)
    info_box(
        "What does this chart show?",
        f"""This chart puts the selected machine <b>in context of the entire fleet</b> — so you can see whether this machine is unusually bad or just average.<br><br>
        <b>The blue box (box plot)</b> represents the spread of Remaining Useful Life across <i>all</i> {n_total} engines in the fleet:<br>
        &nbsp;&nbsp;&nbsp;• The <b>middle line</b> in the box = the median RUL (half the fleet is above this, half below)<br>
        &nbsp;&nbsp;&nbsp;• The <b>box edges</b> = the middle 50% of all machines<br>
        &nbsp;&nbsp;&nbsp;• The <b>whiskers</b> = the full range, excluding outliers<br><br>
        <b>The coloured diamond markers</b> show where THIS machine's individual components sit within that fleet range:<br>
        🔴 <span style="color:{C_RED};">Red diamond</span> — This component is in URGENT territory<br>
        🟡 <span style="color:{C_AMBER};">Amber diamond</span> — This component is being monitored<br>
        🟢 <span style="color:{C_GREEN};">Green diamond</span> — This component is healthy<br><br>
        <b>How to use this:</b> If a diamond sits <b>below the bottom of the box</b>, this machine's component is performing significantly worse than the rest of the fleet and warrants priority attention."""
    )
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=rul["predicted_rul_hours"], name="All Engines",
        marker_color=C_BLUE, line_color=C_BLUE,
        fillcolor="rgba(47,140,255,0.08)",
    ))
    for _, r in machine_rul.iterrows():
        c = C_RED if r["interpretation"]=="URGENT" else C_AMBER if r["interpretation"]=="MONITOR" else C_GREEN
        fig_box.add_trace(go.Scatter(
            x=["All Engines"], y=[r["predicted_rul_hours"]],
            mode="markers", name=COMP_NAMES.get(r["component"], r["component"]),
            marker=dict(size=13, color=c, symbol="diamond",
                        line=dict(width=2, color=C_BG)),
        ))
    fig_box.update_layout(
        **PLOTLY_THEME, height=300,
        margin=dict(l=20,r=20,t=20,b=20),
        yaxis=dict(title="Remaining Useful Life (hours)", gridcolor=C_LINE, tickfont_color=C_MUTED),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Alerts":
    st.markdown(f"""
    <h1 style="margin:0 0 4px; font-size:25px; color:#fff;
               text-shadow:0 0 22px rgba(47,140,255,0.28);">Alerts</h1>
    <p style="margin:0 0 8px; color:{C_MUTED}; font-size:13px;">
        Engines with components that have crossed failure thresholds — prioritised by urgency
    </p>
    """, unsafe_allow_html=True)

    info_box(
        "What is this page and how do I use it?",
        f"""This page is your <b>action list</b> — it shows only the machines and components that have crossed failure thresholds and need real-world intervention.<br><br>
        <b>When is an alert raised?</b><br>
        An alert is automatically triggered when the ML model's failure probability for a component exceeds <b>50%</b>. This threshold was chosen to balance sensitivity (catching failures early) with specificity (not raising too many false alarms).<br><br>
        <b>How is the list sorted?</b><br>
        Alerts are sorted by <b>Repair Priority first</b> (1 = most urgent), then by <b>Remaining Life</b> (shortest first). This means the top row is always the single most urgent component in your entire fleet.<br><br>
        <b>What action should I take?</b><br>
        1️⃣ Start at the top of the table — that's your highest priority<br>
        2️⃣ Check the <b>Remaining Life (hours)</b> column to understand your deadline<br>
        3️⃣ Use the <b>Machine Detail</b> page (sidebar) to drill deeper into that specific machine<br>
        4️⃣ Schedule maintenance or replacement for that component before the RUL runs out<br><br>
        <b>Filters:</b> Use the filters below to focus on specific machine statuses or component types. By default, only CRITICAL and MONITOR machines are shown."""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(255,77,94,0.40); box-shadow:var(--glow-red);">
            <div class="metric-label">🔴 Critical Engines</div>
            <div class="metric-value metric-critical">{n_critical}</div>
            <div style="font-size:11px; color:{C_MUTED}; margin-top:8px;">{STATUS_DESC["CRITICAL"]}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(255,176,32,0.35);">
            <div class="metric-label">🟡 Monitor Engines</div>
            <div class="metric-value metric-warning">{n_monitor}</div>
            <div style="font-size:11px; color:{C_MUTED}; margin-top:8px;">{STATUS_DESC["MONITOR"]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Filter Alerts</div>', unsafe_allow_html=True)
    cf1, cf2 = st.columns(2)
    with cf1:
        status_filter = st.multiselect(
            "Engine Status", ["CRITICAL","MONITOR","HEALTHY"],
            default=["CRITICAL","MONITOR"],
            help="Filter by the overall engine status"
        )
    with cf2:
        comp_display_filter = st.multiselect(
            "Component", list(COMP_NAMES.values()),
            default=list(COMP_NAMES.values()),
            help="Filter by which component triggered the alert"
        )
    # Map display names back to keys
    comp_filter = [k for k, v in COMP_NAMES.items() if v in comp_display_filter]

    # Build alerts
    alert_rows = []
    for _, row in clf_rich_sorted.iterrows():
        if row["overall_status"] not in status_filter:
            continue
        for comp in COMP_KEYS:
            if comp not in comp_filter or not row[f"{comp}_alert"]:
                continue
            cr = rul[(rul["machine_id"]==row["machine_id"]) & (rul["component"]==comp)]
            rul_h  = cr["predicted_rul_hours"].values[0] if len(cr) else None
            interp = cr["interpretation"].values[0] if len(cr) else "—"
            alert_rows.append({
                "Machine":           f"M{int(row['machine_id']):03d}",
                "Engine Status":     row["overall_status"],
                "Component":         COMP_NAMES[comp],
                "Failure Probability": f"{row[f'{comp}_prob']*100:.1f}%",
                "Component Status":  row[f"{comp}_status"],
                "Remaining Life (h)": round(rul_h,1) if rul_h is not None else "—",
                "Outlook":           interp,
                "Repair Priority":   int(row["priority"]),
            })

    if not alert_rows:
        st.markdown(f"""
        <div style="text-align:center; padding:40px; color:{C_GREEN};
                    font-size:16px; font-weight:700;">
            ✓ No alerts for the selected filters
        </div>
        """, unsafe_allow_html=True)
    else:
        alerts_df = pd.DataFrame(alert_rows).sort_values(["Repair Priority","Remaining Life (h)"])

        STATUS_BG2 = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
        def _alert_fns(col):
            if col in ("Engine Status","Component Status"):
                return lambda v: (f"background:{STATUS_BG2.get(v,'')};color:{STATUS_COLOR.get(v,C_INK)};font-weight:800;")
            if col == "Remaining Life (h)":
                return lambda v: (f"color:{C_RED};font-weight:700;" if isinstance(v,(int,float)) and v<48
                                  else f"color:{C_AMBER};font-weight:700;" if isinstance(v,(int,float)) and v<72
                                  else f"color:{C_INK};")
            if col == "Failure Probability":
                return lambda v: f"color:{C_RED};font-weight:700;"
            return None
        dark_table(alerts_df, {c: _alert_fns(c) for c in alerts_df.columns if _alert_fns(c)})
        st.markdown(f'<div style="font-size:11px; color:{C_MUTED}; margin-top:6px;">'
                    f'{len(alert_rows)} alert(s) across {len(set(r["Machine"] for r in alert_rows))} engine(s)</div>',
                    unsafe_allow_html=True)

    # Risk scatter
    st.markdown('<div class="section-title">Fleet Risk Map — All Engines</div>', unsafe_allow_html=True)
    info_box(
        "How do I read this scatter plot?",
        f"""This chart plots <b>every machine in the fleet</b> as a single dot, giving you a bird's-eye view of overall fleet risk in one glance.<br><br>
        <b>What each axis means:</b><br>
        ↔ <b>Horizontal axis (X) — Shortest Remaining Life (hours)</b>: How many hours the most degraded component on this machine has left. Machines on the <b>left</b> are running out of time.<br>
        ↕ <b>Vertical axis (Y) — Highest Failure Probability (%)</b>: The worst failure probability across all 4 components on this machine. Machines at the <b>top</b> have a high chance of imminent failure.<br><br>
        <b>The four quadrants:</b><br>
        🔴 <b>Top-left</b> — High failure probability AND low remaining life: <b>most urgent, act immediately</b><br>
        🟡 <b>Top-right</b> — High failure probability but more time remaining: <b>plan maintenance soon</b><br>
        🟡 <b>Bottom-left</b> — Low remaining life but lower failure probability: <b>watch closely</b><br>
        🟢 <b>Bottom-right</b> — Low failure probability AND plenty of life remaining: <b>safe zone</b><br><br>
        <b>Dot colours:</b> 🔴 CRITICAL &nbsp;|&nbsp; 🟡 MONITOR &nbsp;|&nbsp; 🟢 HEALTHY<br><br>
        <b>How to use this:</b> Hover over any dot to see the machine ID and priority. Any dot in the top-left quadrant (above the 50% line and left of the 72h line) should already appear in the alert table above."""
    )
    sdf = clf_rich.copy()
    sdf["max_prob"] = sdf[["comp1_prob","comp2_prob","comp3_prob","comp4_prob"]].max(axis=1)*100

    fig_sc = px.scatter(
        sdf, x="min_rul_hours", y="max_prob",
        color="overall_status",
        color_discrete_map=STATUS_COLOR,
        hover_data={"machine_id":True,"priority":True},
        labels={
            "min_rul_hours": "Shortest Remaining Life (hours)",
            "max_prob": "Highest Failure Probability (%)",
            "overall_status": "Status"
        },
    )
    fig_sc.update_traces(marker=dict(size=9, opacity=0.88, line=dict(width=0.5, color=C_BG)))
    fig_sc.update_layout(
        **PLOTLY_THEME, height=380,
        margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
    )
    fig_sc.add_vline(x=72, line_dash="dot", line_color=C_MUTED, opacity=0.45,
                     annotation_text="72h threshold", annotation_font_color=C_MUTED)
    fig_sc.add_hline(y=50, line_dash="dot", line_color=C_MUTED, opacity=0.45,
                     annotation_text="50% alert threshold", annotation_font_color=C_MUTED)
    st.plotly_chart(fig_sc, use_container_width=True)