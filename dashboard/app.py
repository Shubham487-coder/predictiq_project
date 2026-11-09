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
.badge-warning {
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
@keyframes pulse-ff3b4e {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,59,78,0); }
    50% { box-shadow: 0 0 24px 6px rgba(255,59,78,0.35); }
}
@keyframes pulse-amber {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,192,32,0); }
    50% { box-shadow: 0 0 16px 4px rgba(255,192,32,0.30); }
}
.fault-row {
    background: rgba(23,216,139,0.04);
    border: 1px solid rgba(23,216,139,0.12);
    box-shadow: inset 0 0 40px rgba(23,216,139,0.04), 0 0 20px rgba(23,216,139,0.06);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    min-height: 120px;
}
.fault-row-header {
    font-size: 15px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8faac8;
    min-width: 160px;
    padding-right: 24px;
    flex-shrink: 0;
}
.machine-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.machine-tag {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 10px;
    border-radius: 8px;
    min-width: 56px;
    cursor: default;
    transition: transform 0.15s;
}
.machine-tag:hover { transform: translateY(-2px); }
.machine-tag.critical {
    background: rgba(255,59,78,0.10);
    border: 1px solid rgba(255,59,78,0.55);
    animation: pulse-ff3b4e 2s infinite;
}
.machine-tag.warning {
    background: rgba(255,192,32,0.08);
    border: 1px solid rgba(255,192,32,0.45);
    animation: pulse-amber 3s infinite;
}
.machine-tag.healthy {
    background: rgba(0,245,160,0.06);
    border: 1px solid rgba(0,245,160,0.30);
    animation: none;
}
.machine-tag .tag-id {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.machine-tag.critical .tag-id { color: #ff3b4e; }
.machine-tag.warning  .tag-id { color: #ffc020; }
.allclear {
    font-size: 12px;
    color: #3a7a5a;
    font-weight: 700;
    padding: 6px 0;
}
.action-group-header {
    font-size: 13px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    padding: 16px 0 8px 0;
    margin-bottom: 6px;
    text-shadow: 0 0 20px currentColor;
}
.action-row {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 18px 20px 14px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.action-row.critical {
    background: rgba(255,59,78,0.07);
    border-left: 4px solid #ff3b4e;
    box-shadow: 0 0 30px rgba(255,59,78,0.12), inset 0 0 40px rgba(255,59,78,0.04);
    animation: pulse-ff3b4e 2s infinite;
}
.action-row.warning {
    background: rgba(255,192,32,0.06);
    border-left: 4px solid #ffc020;
    box-shadow: 0 0 24px rgba(255,192,32,0.10), inset 0 0 40px rgba(255,192,32,0.03);
    animation: pulse-amber 3s infinite;
}
.action-top {
    display: flex;
    align-items: center;
    gap: 14px;
}
.action-machine {
    font-size: 22px;
    font-weight: 900;
    color: #fff;
    min-width: 72px;
    letter-spacing: 0.04em;
}
.action-badge {
    font-size: 10px;
    font-weight: 900;
    padding: 4px 10px;
    border-radius: 4px;
    min-width: 76px;
    text-align: center;
    letter-spacing: 0.08em;
}
.action-badge.critical { background:rgba(255,59,78,0.20); color:#ff3b4e; border:1px solid rgba(255,59,78,0.40); }
.action-badge.warning  { background:rgba(255,192,32,0.15); color:#ffc020; border:1px solid rgba(255,192,32,0.35); }
.action-comps {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    flex: 1;
}
.action-comp-tag.critical {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    background: rgba(255,59,78,0.15);
    color: #ff3b4e;
    border: 1px solid rgba(255,59,78,0.30);
}
.action-comp-tag.warning {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    background: rgba(255,192,32,0.12);
    color: #ffc020;
    border: 1px solid rgba(255,192,32,0.28);
}
.action-btns {
    display: flex;
    gap: 8px;
    margin-left: auto;
    flex-shrink: 0;
}
.action-btn {
    font-size: 11px;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 5px;
    cursor: pointer;
    border: 1px solid;
}
.action-btn.schedule {
    background: rgba(47,140,255,0.12);
    border-color: rgba(47,140,255,0.40);
    color: #2f8cff;
}
.action-btn.alert {
    background: rgba(255,192,32,0.08);
    border-color: rgba(255,192,32,0.35);
    color: #ffc020;
}
.heartbeat-line {
    width: 100%;
    height: 32px;
    overflow: visible;
}
@keyframes ecg-critical {
    0%   { stroke-dashoffset: 300; }
    100% { stroke-dashoffset: 0; }
}
@keyframes ecg-warning {
    0%   { stroke-dashoffset: 300; }
    100% { stroke-dashoffset: 0; }
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
    "WARNING":  C_AMBER,
    "HEALTHY":  C_GREEN,
}
STATUS_BADGE = {
    "CRITICAL": "badge-critical",
    "WARNING":  "badge-monitor",
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
    "WARNING":  "Degradation detected — schedule inspection within 3 days",
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
    STATUS_BG  = {"CRITICAL":"rgba(255,59,78,0.14)","WARNING":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
    STATUS_CLR = {"CRITICAL":C_RED,"WARNING":C_AMBER,"HEALTHY":C_GREEN}

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

@st.fragment
def render_sidebar():
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
            st.rerun(scope="fragment")
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

with st.sidebar:
    st.session_state.page = st.radio(
        "nav",
        ["Overview", "RUL Explorer", "Alerts"],
        label_visibility="collapsed",
    )
    render_sidebar()
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
n_warning  = status_counts.get("WARNING",  0)
n_healthy  = status_counts.get("HEALTHY",  0)
n_total    = len(clf)

rul_min = rul.groupby("machine_id")["predicted_rul_hours"].min().reset_index()
rul_min.columns = ["machine_id", "min_rul_hours"]
clf_rich = clf.merge(rul_min, on="machine_id", how="left")

priority_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2}
clf_rich["_prio"] = clf_rich["overall_status"].map(priority_order)
clf_rich_sorted = clf_rich.sort_values(["_prio", "min_rul_hours"]).drop(columns="_prio")

COMP_SVG = {
    "comp1": """<svg viewBox="0 0 40 40" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="fan-g" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c1}"/>
          <stop offset="100%" stop-color="{c2}"/>
        </radialGradient>
      </defs>
      <circle cx="20" cy="20" r="18" fill="url(#fan-g)" opacity="0.15"/>
      <circle cx="20" cy="20" r="4" fill="{c1}" opacity="0.9"/>
      <path d="M20,16 C22,10 28,8 26,14 C24,18 22,19 20,20 Z" fill="{c1}" opacity="0.85"/>
      <path d="M24,22 C30,20 32,26 26,26 C22,26 20,24 20,20 Z" fill="{c1}" opacity="0.70"/>
      <path d="M18,24 C16,30 10,30 12,24 C14,20 17,19 20,20 Z" fill="{c1}" opacity="0.55"/>
      <path d="M16,18 C10,16 10,10 16,13 C19,15 20,18 20,20 Z" fill="{c1}" opacity="0.70"/>
      <circle cx="20" cy="20" r="3" fill="{c1}"/>
    </svg>""",
    "comp2": """<svg viewBox="0 0 40 40" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="eng-g" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{c1}"/>
          <stop offset="100%" stop-color="{c2}"/>
        </linearGradient>
      </defs>
      <rect x="6" y="14" width="28" height="12" rx="6" fill="url(#eng-g)" opacity="0.20"/>
      <rect x="6" y="14" width="28" height="12" rx="6" fill="none" stroke="{c1}" stroke-width="1.2" opacity="0.7"/>
      <ellipse cx="34" cy="20" rx="4" ry="6" fill="{c1}" opacity="0.25"/>
      <ellipse cx="34" cy="20" rx="4" ry="6" fill="none" stroke="{c1}" stroke-width="1" opacity="0.6"/>
      <ellipse cx="6" cy="20" rx="3" ry="5" fill="{c1}" opacity="0.15"/>
      <line x1="12" y1="17" x2="12" y2="23" stroke="{c1}" stroke-width="1" opacity="0.5"/>
      <line x1="18" y1="15" x2="18" y2="25" stroke="{c1}" stroke-width="1" opacity="0.5"/>
      <line x1="24" y1="15" x2="24" y2="25" stroke="{c1}" stroke-width="1" opacity="0.5"/>
      <line x1="30" y1="17" x2="30" y2="23" stroke="{c1}" stroke-width="1" opacity="0.5"/>
    </svg>""",
    "comp3": """<svg viewBox="0 0 40 40" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="hpt-g" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c1}" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="{c2}" stop-opacity="0.3"/>
        </radialGradient>
      </defs>
      <circle cx="20" cy="20" r="16" fill="url(#hpt-g)" opacity="0.15"/>
      <circle cx="20" cy="20" r="16" fill="none" stroke="{c1}" stroke-width="1" opacity="0.4"/>
      <circle cx="20" cy="20" r="5" fill="{c1}" opacity="0.8"/>
      <circle cx="20" cy="20" r="5" fill="none" stroke="{c1}" stroke-width="0.5" opacity="0.5"/>
      <path d="M20,15 L22,8 L20,10 L18,8 Z" fill="{c1}" opacity="0.8"/>
      <path d="M25,20 L32,18 L30,20 L32,22 Z" fill="{c1}" opacity="0.8"/>
      <path d="M20,25 L22,32 L20,30 L18,32 Z" fill="{c1}" opacity="0.8"/>
      <path d="M15,20 L8,18 L10,20 L8,22 Z" fill="{c1}" opacity="0.8"/>
      <path d="M23,13 L28,8 L27,11 L30,10 Z" fill="{c1}" opacity="0.6"/>
      <path d="M27,23 L32,28 L29,27 L30,30 Z" fill="{c1}" opacity="0.6"/>
      <path d="M17,27 L12,32 L13,29 L10,30 Z" fill="{c1}" opacity="0.6"/>
      <path d="M13,17 L8,12 L11,13 L10,10 Z" fill="{c1}" opacity="0.6"/>
    </svg>""",
    "comp4": """<svg viewBox="0 0 40 40" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="lpt-g" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{c1}" stop-opacity="0.8"/>
          <stop offset="100%" stop-color="{c2}" stop-opacity="0.2"/>
        </radialGradient>
      </defs>
      <circle cx="20" cy="20" r="18" fill="url(#lpt-g)" opacity="0.12"/>
      <circle cx="20" cy="20" r="18" fill="none" stroke="{c1}" stroke-width="0.8" opacity="0.35"/>
      <circle cx="20" cy="20" r="10" fill="none" stroke="{c1}" stroke-width="0.8" opacity="0.25"/>
      <circle cx="20" cy="20" r="4" fill="{c1}" opacity="0.7"/>
      <path d="M20,16 L21,6 L20,9 L19,6 Z" fill="{c1}" opacity="0.75"/>
      <path d="M24,20 L34,19 L31,20 L34,21 Z" fill="{c1}" opacity="0.75"/>
      <path d="M20,24 L21,34 L20,31 L19,34 Z" fill="{c1}" opacity="0.75"/>
      <path d="M16,20 L6,19 L9,20 L6,21 Z" fill="{c1}" opacity="0.75"/>
      <path d="M22,14 L26,5 L25,9 L28,7 Z" fill="{c1}" opacity="0.55"/>
      <path d="M26,22 L35,26 L31,24 L33,27 Z" fill="{c1}" opacity="0.55"/>
      <path d="M18,26 L14,35 L15,31 L12,33 Z" fill="{c1}" opacity="0.55"/>
      <path d="M14,18 L5,14 L9,16 L7,13 Z" fill="{c1}" opacity="0.55"/>
    </svg>""",
}

COMP_LABEL = {
    "comp1": "Fan",
    "comp2": "Core Engine",
    "comp3": "HP Turbine",
    "comp4": "LP Turbine",
}


def make_tag(machine_id, severity, comp_key):
    if severity == "HEALTHY":
        c1, c2, css_class = "#17d88b", "#0a5a3a", "healthy"
        label_html = ""
    elif severity == "CRITICAL":
        c1, c2, css_class = "#ff3b4e", "#7a0010", "critical"
        label_html = f'<div class="tag-id">M{str(machine_id).zfill(3)}</div>'
    else:
        c1, c2, css_class = "#ffc020", "#7a5500", "warning"
        label_html = f'<div class="tag-id">M{str(machine_id).zfill(3)}</div>'
    svg = COMP_SVG[comp_key].replace("{c1}", c1).replace("{c2}", c2)
    return f"""<div class="machine-tag {css_class}">
        {svg}
        {label_html}
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

page = st.session_state.get("page", "Overview")

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
    with st.expander("ℹ️  How to read these cards"):
        st.markdown(f"""
        <div style="font-size:13px; color:{C_INK}; line-height:1.8;">
        Each card shows how many of the 100 engines currently fall into that health category,
        based on the latest pipeline run.<br><br>
        🔴 <b style="color:{C_RED};">CRITICAL</b> — One or more components on this engine are
        predicted to fail very soon. Fix these first.<br><br>
        🟡 <b style="color:{C_AMBER};">WARNING</b> — Early signs of degradation detected.
        Not urgent, but schedule an inspection this week.<br><br>
        🟢 <b style="color:{C_GREEN};">HEALTHY</b> — All components are operating normally.
        No action needed.<br><br>
        The bar under each number shows that count as a share of all 100 engines.
        A wider red bar means more machines need immediate attention.
        </div>
        """, unsafe_allow_html=True)

    # KPI row — 3 cards only
    c1, c2, c3 = st.columns(3)

    def stat_card(col, label, val, color, desc, pulse=False):
        pct = round((val / n_total) * 100, 1) if n_total else 0
        pulse_css = f"animation: pulse-{color.replace('#','')} 2s infinite;" if pulse and val > 0 else ""
        with col:
            st.markdown(f"""
            <div class="metric-card" style="position:relative; overflow:hidden; {pulse_css}">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}; font-size:42px; font-weight:900;
                     margin:6px 0 2px; line-height:1;">{val}</div>
                <div style="font-size:11px; color:{C_MUTED}; margin-bottom:10px;">of {n_total} machines</div>
                <div style="background:{C_LINE}; border-radius:4px; height:5px; width:100%; margin-bottom:8px;">
                    <div style="background:{color}; border-radius:4px; height:5px;
                         width:{pct}%; opacity:0.85; transition:width 0.6s ease;"></div>
                </div>
                <div style="font-size:11px; color:{C_MUTED}; line-height:1.4;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    stat_card(c1, "🔴 CRITICAL", n_critical, C_RED,    STATUS_DESC["CRITICAL"], pulse=True)
    stat_card(c2, "🟡 WARNING",  n_warning,  C_AMBER,  STATUS_DESC["WARNING"])
    stat_card(c3, "🟢 HEALTHY",  n_healthy,  C_GREEN,  STATUS_DESC["HEALTHY"])

    # Charts row


    st.markdown('<div class="section-title">Component Fault Panel</div>', unsafe_allow_html=True)
    info_box(
        "What does this panel show?",
        f"""This panel shows which machines are currently flagging each component as degraded.<br><br>
        Each tag represents one machine — <b style="color:{C_RED};">🔴 red tags</b> need immediate action,
        <b style="color:{C_AMBER};">🟡 amber tags</b> should be inspected soon.<br><br>
        A machine appearing in multiple rows has more than one component affected — prioritise those first.<br><br>
        Machines running normally are not shown. If a row shows <span style="color:#3a7a5a;">✓ All clear</span>,
        no machine in the fleet is flagging that component right now."""
    )




    rows_html = ""
    for comp_key in COMP_KEYS:
        label = COMP_NAMES[comp_key]
        flagging = clf[clf[f"{comp_key}_status"] != "HEALTHY"][["machine_id", f"{comp_key}_status"]].copy()
        flagging = flagging.sort_values(f"{comp_key}_status", key=lambda s: s.map({"CRITICAL": 0, "WARNING": 1}))
        n_flag = len(flagging)
        count_str = f"{n_flag} flagging" if n_flag > 0 else "All clear"
        count_color = C_RED if any(
            flagging[f"{comp_key}_status"] == "CRITICAL") else C_AMBER if n_flag > 0 else "#3a7a5a"

        if n_flag == 0:
            tags_html = make_tag(0, "HEALTHY", comp_key)
        else:
            tags_html = '<div class="machine-tags">'
            for _, row in flagging.iterrows():
                tags_html += make_tag(row["machine_id"], row[f"{comp_key}_status"], comp_key)
            tags_html += '</div>'

        rows_html += f"""
                        <div class="fault-row">
                            <div class="fault-row-header">{label}</div>
                            <div style="flex:1; display:flex; align-items:center;">
                                {tags_html}
                            </div>
                            <div style="font-size:16px; font-weight:800; color:{count_color};
                                        margin-left:16px; flex-shrink:0; min-width:80px; text-align:right;">
                                {count_str}
                            </div>
                        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)

    # Top 10 table
    st.markdown('<div class="section-title">Maintenance Action Panel</div>', unsafe_allow_html=True)
    info_box(
        "What does this panel show?",
        f"""This panel lists only the machines that need attention right now — healthy machines are not shown.<br><br>
        🔴 <b style="color:{C_RED};">CRITICAL</b> — A component on this machine is predicted to fail very soon.
        Act immediately — do not wait for the next scheduled inspection.<br><br>
        🟡 <b style="color:{C_AMBER};">WARNING</b> — Early signs of degradation detected. Schedule an inspection
        within 3 days to prevent escalation to CRITICAL.<br><br>
        The <b>Flagging Components</b> column shows which specific components need attention on each machine.
        Use the action buttons to schedule maintenance or set an alert."""
    )

    action_machines = clf_rich_sorted[clf_rich_sorted["overall_status"] != "HEALTHY"].copy()

    if action_machines.empty:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:14px; padding:24px 20px;
                    border-radius:10px; border:1px solid rgba(23,216,139,0.25);
                    background:rgba(23,216,139,0.05); margin-top:8px;">
            <div style="font-size:32px;">✅</div>
            <div>
                <div style="font-size:15px; font-weight:800; color:{C_GREEN};">
                    All machines operating normally
                </div>
                <div style="font-size:12px; color:{C_MUTED}; margin-top:4px;">
                    No maintenance action required at this time
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for severity, group_label, group_color in [
            ("CRITICAL", "🔴 Requires Immediate Action", C_RED),
            ("WARNING",  "🟡 Schedule Within 3 Days",   C_AMBER),
        ]:
            group = action_machines[action_machines["overall_status"] == severity]
            if group.empty:
                continue
            st.markdown(f'<div class="action-group-header" style="color:{group_color};">{group_label}</div>',
                        unsafe_allow_html=True)
            rows = ""
            for _, row in group.iterrows():
                mid = f"M{str(int(row['machine_id'])).zfill(3)}"
                css = severity.lower()
                stroke = "#ff3b4e" if severity == "CRITICAL" else "#ffc020"
                speed = "1.2s" if severity == "CRITICAL" else "2.2s"
                flagging_comps = [COMP_NAMES[c] for c in COMP_KEYS if row.get(f"{c}_status") != "HEALTHY"]
                comp_tags = "".join([f'<span class="action-comp-tag {css}">{c}</span>' for c in flagging_comps])
                if severity == "CRITICAL":
                    ecg_path = "M0,16 L18,16 L24,2 L30,30 L36,16 L44,16 L48,8 L52,24 L56,16 L300,16"
                else:
                    ecg_path = "M0,16 L30,16 L38,6 L44,26 L50,16 L300,16"
                rows += f"""
                            <div class="action-row {css}">
                                <div class="action-top">
                                    <div class="action-machine">{mid}</div>
                                    <div class="action-badge {css}">{severity}</div>
                                    <div class="action-comps">{comp_tags}</div>
                                    <div class="action-btns">
                                        <div class="action-btn schedule">📅 Schedule</div>
                                        <div class="action-btn alert">🔔 Set Alert</div>
                                    </div>
                                </div>
                                <svg class="heartbeat-line" viewBox="0 0 300 32" preserveAspectRatio="none">
                                    <path d="{ecg_path}" fill="none" stroke="{stroke}" stroke-width="1.5"
                                          stroke-dasharray="300" stroke-dashoffset="300" opacity="0.6"
                                          style="animation: ecg-{css} {speed} linear infinite;"/>
                                </svg>
                            </div>"""
            st.markdown(rows, unsafe_allow_html=True)

    # RUL histogram



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MACHINE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "RUL Explorer":

    RUL_LABEL = {
        "CRITICAL": "REPLACE NOW",
        "URGENT":   "INSPECT SOON",
        "MONITOR":  "MONITOR",
        "HEALTHY":  "RUNNING OK",
    }
    RUL_COLOR = {
        "CRITICAL": C_RED,
        "URGENT":   C_AMBER,
        "MONITOR":  C_CYAN,
        "HEALTHY":  C_GREEN,
    }
    RUL_ORDER = ["CRITICAL", "URGENT", "MONITOR", "HEALTHY"]

    st.markdown(
        f'<h1 style="margin:0 0 4px; font-size:25px; color:#fff; '
        f'text-shadow:0 0 22px rgba(47,140,255,0.28);">RUL Explorer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:13px; color:{C_MUTED}; margin-bottom:20px;">'
        f'Remaining Useful Life — how long before each component needs replacement</div>',
        unsafe_allow_html=True,
    )

    # ── Component filter ─────────────────────────────────────────────────────
    COMP_FILTER_KEYS  = ["all", "comp1", "comp2", "comp3", "comp4"]
    COMP_FILTER_NAMES = {"all": "All Components", "comp1": "Fan",
                         "comp2": "Core Engine", "comp3": "HP Turbine", "comp4": "LP Turbine"}
    COMP_FILTER_COLOR = {"all": "#17d88b", "comp1": "#2f8cff",
                         "comp2": "#ff6b35", "comp3": "#c97aff", "comp4": "#17d88b"}

    if "rul_filter" not in st.session_state:
        st.session_state.rul_filter = "all"

    # ── Component filter cards ────────────────────────────────────────────────
    cards_html = '<div style="display:flex; gap:12px; margin-bottom:8px;">'
    for key in COMP_FILTER_KEYS:
        color = COMP_FILTER_COLOR[key]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        is_active = st.session_state.rul_filter == key
        glow   = f"0 0 18px rgba({r},{g},{b},0.65)" if is_active else f"0 0 10px rgba({r},{g},{b},0.28)"
        border = f"2px solid rgba({r},{g},{b},1.0)" if is_active else f"1px solid rgba({r},{g},{b},0.45)"
        bg     = f"rgba({r},{g},{b},0.18)" if is_active else f"rgba({r},{g},{b},0.07)"
        label_color = "#fff" if is_active else f"rgba({r},{g},{b},0.85)"
        fw     = "800" if is_active else "500"
        if key == "all":
            icon_html = '<div style="font-size:24px; margin-bottom:6px;">🔧</div>'
        else:
            svg = COMP_SVG[key].replace("{c1}", color).replace("{c2}", color)
            icon_html = f'<div style="margin-bottom:6px;">{svg}</div>'
        cards_html += (
            f'<div style="flex:1; border:{border}; background:{bg}; box-shadow:{glow}; '
            f'border-radius:12px; height:88px; display:flex; flex-direction:column; '
            f'align-items:center; justify-content:center;">'
            f'{icon_html}'
            f'<div style="font-size:11px; font-weight:{fw}; color:{label_color}; text-align:center;">'
            f'{COMP_FILTER_NAMES[key]}</div></div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("""<style>
    div.rul-btn-row { margin-top:-96px; opacity:0; }
    div.rul-btn-row button { height:88px !important; }
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="rul-btn-row">', unsafe_allow_html=True)
    filter_cols = st.columns(5)
    for i, key in enumerate(COMP_FILTER_KEYS):
        with filter_cols[i]:
            if st.button(COMP_FILTER_NAMES[key], key=f"rul_btn_{key}", use_container_width=True):
                st.session_state.rul_filter = key
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── Filter data ───────────────────────────────────────────────────────────
    rul_filtered = rul.copy()
    if st.session_state.rul_filter != "all":
        rul_filtered = rul_filtered[rul_filtered["component"] == st.session_state.rul_filter]

    # ── Summary line ──────────────────────────────────────────────────────────
    n_replace  = (rul_filtered["interpretation"] == "CRITICAL").sum()
    n_inspect  = (rul_filtered["interpretation"] == "URGENT").sum()
    n_total_f  = len(rul_filtered)
    summary_parts = []
    if n_replace:
        summary_parts.append(f'<span style="color:{C_RED}; font-weight:700;">{n_replace} REPLACE NOW</span>')
    if n_inspect:
        summary_parts.append(f'<span style="color:{C_AMBER}; font-weight:700;">{n_inspect} INSPECT SOON</span>')
    if not summary_parts:
        summary_html = f'<span style="color:{C_GREEN}; font-weight:700;">All {n_total_f} components running OK</span>'
    else:
        summary_html = " · ".join(summary_parts) + f' <span style="color:{C_MUTED};">out of {n_total_f} components</span>'
    st.markdown(
        f'<div style="font-size:13px; margin-bottom:12px;">{summary_html}</div>',
        unsafe_allow_html=True,
    )

    info_box(
        "How to read this chart?",
        f"""This histogram shows the <b>Remaining Useful Life (RUL)</b> distribution across components in the fleet.<br><br>
        Each bar = a group of components with similar hours remaining before maintenance is needed.<br><br>
        <span style="color:{C_RED};">■ REPLACE NOW</span> — under 24 hours — act immediately<br>
        <span style="color:{C_AMBER};">■ INSPECT SOON</span> — 24–48 hours — schedule within 2 days<br>
        <span style="color:{C_CYAN};">■ MONITOR</span> — 48–72 hours — watch closely<br>
        <span style="color:{C_GREEN};">■ RUNNING OK</span> — over 72 hours — no action needed<br><br>
        Use the component buttons above to filter by Fan, Core Engine, HP Turbine or LP Turbine."""
    )

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig = px.histogram(
        rul_filtered,
        x="predicted_rul_hours",
        nbins=30,
        color="interpretation",
        color_discrete_map={k: RUL_COLOR[k] for k in RUL_ORDER},
        category_orders={"interpretation": RUL_ORDER},
        labels={"predicted_rul_hours": "Remaining Useful Life (hours)", "interpretation": "Status"},
    )
    for trace in fig.data:
        trace.name = RUL_LABEL.get(trace.name, trace.name)

    for hrs, label, clr in [(24, "24h — Replace Now", C_RED),
                             (48, "48h — Inspect Soon", C_AMBER),
                             (72, "72h — Monitor", C_CYAN)]:
        fig.add_vline(x=hrs, line_dash="dot", line_color=clr, line_width=1.5, opacity=0.6,
                      annotation_text=label, annotation_font_color=clr,
                      annotation_font_size=10, annotation_position="top")

    fig.update_layout(
        **PLOTLY_THEME, height=360,
        margin=dict(l=20, r=20, t=30, b=40),
        bargap=0.04,
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED,
                   title_font_color=C_MUTED, range=[0, 102]),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED,
                   title="Number of Components", title_font_color=C_MUTED),
    )
    st.plotly_chart(fig, use_container_width=True)


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
        <b>Filters:</b> Use the filters below to focus on specific machine statuses or component types. By default, only CRITICAL and WARNING machines are shown."""
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
            <div class="metric-label">🟡 Warning Engines</div>
            <div class="metric-value metric-warning">{n_warning}</div>
            <div style="font-size:11px; color:{C_MUTED}; margin-top:8px;">{STATUS_DESC["WARNING"]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Filter Alerts</div>', unsafe_allow_html=True)
    cf1, cf2 = st.columns(2)
    with cf1:
        status_filter = st.multiselect(
            "Engine Status", ["CRITICAL","WARNING","HEALTHY"],
            default=["CRITICAL","WARNING"],
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

        STATUS_BG2 = {"CRITICAL":"rgba(255,59,78,0.14)","WARNING":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
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
        <b>Dot colours:</b> 🔴 CRITICAL &nbsp;|&nbsp; 🟡 WARNING &nbsp;|&nbsp; 🟢 HEALTHY<br><br>
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