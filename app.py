import base64
import html

import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

import actions
import bundle_engine as be
import ingest
import companies as co
import priority_engine as pe
import product_insights as pi


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ADPULSE AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# Company-specific paths (dataset + models) are chosen further down,
# once the active company is known.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_csv(path, mtime):
    """Cached by path AND modified time, so re-processed files are re-read."""

    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


def read_csv(path):

    return load_csv(str(path), path.stat().st_mtime if path.exists() else 0)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    if df.empty:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def find_success_rate_column(df):

    if df.empty:
        return None

    exact_candidates = [
        "success_rate_pct",
        "success_rate",
        "action_success_rate",
        "historical_action_success_rate",
        "historical_success_rate",
        "success_percentage",
        "success_percent"
    ]

    column = find_column(
        df,
        exact_candidates
    )

    if column is not None:
        return column

    for column in df.columns:

        name = str(column).lower()

        if (
            "success" in name
            and (
                "rate" in name
                or "percent" in name
                or "%" in name
            )
        ):

            return column

    return None


def find_action_column(df):

    if df.empty:
        return None

    candidates = [
        "action",
        "recommended_action",
        "current_action",
        "autonomous_action",
        "policy_action"
    ]

    column = find_column(
        df,
        candidates
    )

    if column is not None:
        return column

    for column in df.columns:

        name = str(column).lower()

        if "action" in name:

            return column

    return None


def pretty(value):
    """PROTECT_INVENTORY -> Protect Inventory"""

    text = str(value).replace("_", " ").strip()

    return text.title() if text.isupper() else text


def esc(value):

    return html.escape(str(value))


def safe_float(value, default=0.0):

    try:
        number = float(value)

    except (TypeError, ValueError):
        return default

    return default if pd.isna(number) else number


# ============================================================
# DESIGN TOKENS
# ============================================================

ORANGE = "#ff7a3d"
ORANGE_SOFT = "#ffb48a"
CREAM = "#f6efe7"
TEAL = "#7dd3c0"
SKY = "#9cc5f5"
ROSE = "#f7a1ad"
AMBER = "#f5cf72"
STONE = "#a8a29e"

ACTION_COLORS = {
    "PROTECT_INVENTORY": ORANGE,
    "RETARGET": ROSE,
    "FIX_PAYMENT": TEAL,
    "INCREASE_BUDGET": "#9be39b",
    "DECREASE_BUDGET": AMBER,
    "MAINTAIN": CREAM,
    "CHANGE_CREATIVE": SKY,
}

ACTION_ICONS = {
    "PROTECT_INVENTORY": "📦",
    "RETARGET": "🎯",
    "FIX_PAYMENT": "💳",
    "INCREASE_BUDGET": "📈",
    "DECREASE_BUDGET": "📉",
    "MAINTAIN": "⚖️",
    "CHANGE_CREATIVE": "🎨",
}

QUALITY_COLORS = {
    "STRONG": ORANGE,
    "GOOD": ORANGE_SOFT,
    "WEAK": "#d6d3d1",
    "POOR": "#8a817c",
}

FALLBACK_PALETTE = [ORANGE, TEAL, SKY, ROSE, AMBER, CREAM]


def action_color(action):

    return ACTION_COLORS.get(str(action).upper(), STONE)


TEXT = "#ffffff"
MUTED = "rgba(255,255,255,0.62)"
GRID = "rgba(255,255,255,0.06)"


def glass_chart(chart, height=300):
    """Quiet chart styling: faint horizontal guides only, few ticks, no clutter."""

    return (
        chart
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor=GRID,
            domain=False,
            ticks=False,
            labelFont="Inter",
            titleFont="Inter",
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            labelPadding=8,
            tickCount=5,
            labelOverlap=True
        )
        .configure_axisX(grid=False)
        .configure_axisBand(grid=False)
        .configure_legend(
            labelColor=TEXT,
            titleColor=MUTED,
            labelFont="Inter",
            titleFont="Inter",
            orient="top",
            direction="horizontal",
            offset=14,
            symbolType="circle",
            labelFontSize=11
        )
    )


def show_chart(chart, height=300):

    st.altair_chart(
        glass_chart(chart, height),
        width="stretch",
        theme=None
    )


# ============================================================
# BACKGROUND
# Drop any photo at assets/background.jpg (or .png / .webp) to
# use it behind the glass, exactly like the reference design.
# ============================================================

@st.cache_data
def background_css():

    for name in ["background.jpg", "background.jpeg", "background.png", "background.webp"]:

        path = ASSETS_DIR / name

        if path.exists():

            mime = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else path.suffix[1:].lower()
            data = base64.b64encode(path.read_bytes()).decode()

            return (
                "linear-gradient(180deg, rgba(0,0,0,.25), rgba(0,0,0,.45)), "
                f"url('data:image/{mime};base64,{data}') center / cover no-repeat fixed"
            )

    # Default: a warm, softly-lit interior scene built from gradients
    return (
        "radial-gradient(ellipse 18% 22% at 7% 88%, rgba(255,186,96,.75), transparent 70%),"
        "radial-gradient(ellipse 35% 40% at 10% 85%, rgba(255,120,40,.45), transparent 70%),"
        "radial-gradient(ellipse 30% 45% at 92% 72%, rgba(70,120,80,.60), transparent 70%),"
        "radial-gradient(ellipse 22% 35% at 97% 25%, rgba(90,140,95,.45), transparent 70%),"
        "radial-gradient(ellipse 45% 30% at 55% 0%, rgba(255,238,215,.30), transparent 70%),"
        "radial-gradient(ellipse 12% 12% at 30% 20%, rgba(255,214,170,.18), transparent 70%),"
        "radial-gradient(ellipse 10% 10% at 75% 15%, rgba(255,214,170,.15), transparent 70%),"
        "linear-gradient(180deg, #4a4039 0%, #2f2925 45%, #1c1816 100%) fixed"
    )


# ============================================================
# GLASSMORPHISM STYLES
# ============================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --panel: rgba(38, 34, 32, 0.38);
    --card: rgba(255, 255, 255, 0.08);
    --card-hover: rgba(255, 255, 255, 0.13);
    --line: rgba(255, 255, 255, 0.16);
    --text: #ffffff;
    --muted: rgba(255, 255, 255, 0.62);
    --orange: {ORANGE};
    --orange-soft: {ORANGE_SOFT};
    --teal: {TEAL};
}}

html, body, .stApp, [class*="css"], button, input {{
    font-family: 'Inter', sans-serif !important;
}}

.stApp {{
    color: var(--text);
    background: {background_css()};
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{ background: transparent; }}
footer, [data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ display: none !important; }}

/* ---------- The big frosted panel ---------- */
.block-container, .stMainBlockContainer {{
    position: relative;
    z-index: 0;
    max-width: 1280px;
    margin: 2.2rem auto 2.5rem;
    padding: 2rem 2.2rem 2.4rem !important;
}}
.block-container::before, .stMainBlockContainer::before {{
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    border-radius: 34px;
    background: var(--panel);
    border: 1px solid rgba(255,255,255,.22);
    backdrop-filter: blur(28px) saturate(130%);
    -webkit-backdrop-filter: blur(28px) saturate(130%);
    box-shadow: 0 30px 80px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.18);
}}
@media (min-width: 900px) {{
    .block-container, .stMainBlockContainer {{ margin-left: max(110px, calc((100% - 1280px) / 2)); }}
}}

/* ---------- Glass cards ---------- */
div[class*="st-key-glass"], .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.1rem 1.25rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 10px 30px rgba(0,0,0,.18);
    transition: background .25s ease, transform .25s ease, border-color .25s ease;
}}
div[class*="st-key-glass"]:hover, .card.hover:hover {{
    background: var(--card-hover);
    border-color: rgba(255,255,255,.26);
}}
.card.hover:hover {{ transform: translateY(-2px); }}

.fade-in {{ animation: fadeUp .45s ease both; }}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ---------- Top bar ---------- */
.topbar {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1.3rem;
}}
.tb-title {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -.01em; color: var(--text); }}
.tb-sub {{ color: var(--muted); font-size: .88rem; margin-top: .15rem; }}
.tb-right {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.chip {{
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .42rem .9rem; border-radius: 999px;
    background: rgba(255,255,255,.08); border: 1px solid var(--line);
    font-size: .8rem; color: var(--text); white-space: nowrap;
}}
.chip.warn {{ color: #ffd9b8; border-color: rgba(255,122,61,.4); background: rgba(255,122,61,.12); }}

/* ---------- Card text ---------- */
.c-title {{ font-size: 1.02rem; font-weight: 600; color: var(--text); }}
.c-sub {{ color: var(--muted); font-size: .8rem; margin-top: .1rem; margin-bottom: .6rem; }}
.c-row {{ display: flex; justify-content: space-between; align-items: center; gap: .6rem; }}
.muted {{ color: var(--muted); }}

/* KPI strip (like the balance row) */
.kpi-strip {{
    display: grid; gap: .9rem;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
}}
.kpi {{ display: flex; align-items: center; gap: .8rem; padding: .85rem 1rem; }}
.kpi-icon {{
    width: 42px; height: 42px; flex-shrink: 0; border-radius: 50%;
    display: grid; place-items: center; font-size: 1.05rem;
    border: 1px solid var(--line); background: rgba(255,255,255,.08);
}}
.kpi-label {{ color: var(--muted); font-size: .74rem; }}
.kpi-value {{ font-size: 1.25rem; font-weight: 600; color: var(--text); line-height: 1.25; }}
.kpi-note {{ font-size: .72rem; color: var(--orange-soft); }}

/* Big number tiles (Goals) */
.goal-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }}
.goal {{ padding: .9rem 1rem; }}
.goal-v {{ font-size: 1.9rem; font-weight: 600; color: var(--text); line-height: 1.1; }}
.goal-l {{ color: var(--muted); font-size: .75rem; margin-top: .45rem; }}
.bar {{ height: 6px; border-radius: 999px; background: rgba(255,255,255,.14); overflow: hidden; margin-top: .55rem; }}
.bar > span {{ display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--orange), var(--orange-soft)); }}

/* Profile card */
.avatar {{
    width: 64px; height: 64px; border-radius: 50%; margin: .2rem auto .55rem;
    display: grid; place-items: center; font-size: 1.7rem;
    background: radial-gradient(circle at 30% 30%, #ffb48a, #ff7a3d 60%, #c2410c);
    box-shadow: 0 0 0 4px rgba(255,255,255,.14), 0 10px 26px rgba(255,122,61,.4);
}}
.p-name {{ text-align: center; font-weight: 600; font-size: 1.02rem; }}
.p-sub {{ text-align: center; color: var(--muted); font-size: .78rem; }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #4ade80; margin-right: .35rem; box-shadow: 0 0 8px #4ade80; }}

/* Policy "bank card" */
.pcard {{
    margin-top: .9rem; padding: 1.1rem 1.2rem; border-radius: 20px; position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(20,20,22,.78), rgba(48,44,42,.6));
    border: 1px solid rgba(255,255,255,.14);
    box-shadow: 0 14px 30px rgba(0,0,0,.35);
}}
.pcard::after {{
    content: ""; position: absolute; width: 180px; height: 180px; right: -60px; top: -70px;
    background: radial-gradient(circle, rgba(255,122,61,.55), transparent 65%);
}}
.pcard .lbl {{ font-size: .66rem; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }}
.pcard .big {{ font-size: 1.35rem; font-weight: 600; margin: .3rem 0 .9rem; }}
.pcard .row {{ display: flex; justify-content: space-between; align-items: flex-end; }}
.pcard .num {{ font-size: 1rem; font-weight: 600; letter-spacing: .04em; }}
.chipline {{
    width: 34px; height: 24px; border-radius: 6px;
    background: linear-gradient(135deg, #f5cf72, #c8962e);
}}

/* Pills / badges */
.badge {{
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .22rem .65rem; border-radius: 999px; font-size: .74rem; font-weight: 500;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    white-space: nowrap;
}}

/* List rows (transactions style) */
.lrow {{
    display: flex; align-items: center; gap: .75rem; padding: .6rem .2rem;
    border-bottom: 1px solid rgba(255,255,255,.08);
}}
.lrow:last-child {{ border-bottom: none; }}
.l-icon {{
    width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
    display: grid; place-items: center; background: rgba(255,255,255,.1); border: 1px solid var(--line);
}}
.l-main {{ flex: 1; min-width: 0; }}
.l-name {{ font-size: .88rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.l-sub {{ font-size: .72rem; color: var(--muted); }}
.l-val {{ font-weight: 600; font-size: .9rem; text-align: right; }}

/* Insight tiles */
.grid-3 {{ display: grid; gap: .9rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
.insight {{ padding: 1.1rem 1.2rem; height: 100%; }}
.insight .ic {{
    width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center;
    background: rgba(255,122,61,.18); border: 1px solid rgba(255,122,61,.35); margin-bottom: .6rem;
}}
.insight h4 {{ margin: 0 0 .3rem; font-size: .98rem; font-weight: 600; color: var(--text); }}
.insight p {{ color: var(--muted); font-size: .85rem; margin: 0; line-height: 1.55; }}
.insight p b {{ color: var(--text); font-weight: 600; }}

/* Reasoning chain */
.chain {{ display: grid; gap: .8rem; grid-template-columns: repeat(2, 1fr); }}
@media (max-width: 560px) {{ .chain {{ grid-template-columns: 1fr; }} }}
.step {{ padding: 1rem 1.1rem; position: relative; }}
.step .no {{ font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
.step .val {{ font-size: 1.25rem; font-weight: 600; margin: .3rem 0; }}
.step .note {{ color: var(--muted); font-size: .8rem; line-height: 1.45; }}

/* Glass table */
.gt-wrap {{ overflow: auto; border-radius: 16px; border: 1px solid rgba(255,255,255,.1); }}
table.gt {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
table.gt th {{
    position: sticky; top: 0; z-index: 1; text-align: left; font-weight: 500;
    color: var(--muted); font-size: .74rem; letter-spacing: .03em;
    padding: .7rem .9rem; background: rgba(40,36,34,.85); backdrop-filter: blur(10px);
}}
table.gt td {{ padding: .62rem .9rem; border-top: 1px solid rgba(255,255,255,.07); color: var(--text); white-space: nowrap; }}
table.gt tr:hover td {{ background: rgba(255,255,255,.05); }}
.cell-bar {{ display: flex; align-items: center; gap: .55rem; min-width: 130px; }}
.cell-bar .bar {{ flex: 1; margin: 0; }}
.mono {{ font-variant-numeric: tabular-nums; }}

/* Timeline */
.tl {{ position: relative; padding-left: 1.5rem; }}
.tl::before {{
    content: ""; position: absolute; left: .45rem; top: .5rem; bottom: .5rem; width: 2px;
    background: linear-gradient(180deg, var(--orange), rgba(255,255,255,.15));
}}
.tl-item {{ position: relative; padding: .8rem 1.1rem; margin-bottom: .65rem; }}
.tl-item::before {{
    content: ""; position: absolute; left: -1.33rem; top: 1.15rem; width: 11px; height: 11px; border-radius: 50%;
    background: var(--orange); box-shadow: 0 0 0 4px rgba(255,122,61,.22);
}}
.tl-file {{ font-family: ui-monospace, monospace; font-size: .74rem; color: var(--orange-soft); }}

/* ---------- Left icon rail ---------- */
.st-key-rail {{
    position: fixed !important;
    left: 26px; top: 50%; transform: translateY(-50%);
    z-index: 1000;
    width: 66px !important;
    padding: 12px 9px;
    gap: 7px !important;
    align-items: center;
    border-radius: 40px;
    background: rgba(40, 36, 34, 0.42);
    border: 1px solid rgba(255,255,255,.2);
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 20px 50px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.15);
}}
.st-key-rail > div {{ width: auto !important; }}
.st-key-rail button {{
    width: 40px !important; height: 40px; min-height: 40px; padding: 0 !important;
    border-radius: 50% !important; justify-content: center;
    background: rgba(255,255,255,.07) !important; border: 1px solid rgba(255,255,255,.14) !important;
    color: rgba(255,255,255,.85) !important;
}}
.st-key-rail button [data-testid="stMarkdownContainer"] {{ display: none; }}
.st-key-rail button:hover {{ background: rgba(255,255,255,.18) !important; transform: scale(1.06); }}
.st-key-rail button[kind="primary"] {{
    background: var(--orange) !important; border-color: transparent !important; color: #fff !important;
    box-shadow: 0 6px 20px rgba(255,122,61,.55);
}}
@media (max-width: 899px) {{
    .st-key-rail {{
        top: auto; bottom: 14px; left: 50%; transform: translateX(-50%);
        width: auto !important; flex-direction: row !important; padding: 8px 10px; gap: 6px !important;
    }}
    .st-key-rail button {{ width: 40px !important; height: 40px; min-height: 40px; }}
    .block-container, .stMainBlockContainer {{ margin: 1rem .5rem 5.5rem; padding: 1.3rem 1rem 2rem !important; }}
    .goal-grid {{ grid-template-columns: 1fr; }}
}}

/* ---------- Native widgets ---------- */
.stButton > button {{
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.08);
    color: var(--text);
    font-weight: 500;
    transition: all .2s ease;
}}
.stButton > button:hover {{ background: rgba(255,255,255,.16); border-color: rgba(255,255,255,.3); color: #fff; }}
.stButton > button[kind="primary"] {{
    background: var(--orange); border-color: transparent; color: #fff;
    box-shadow: 0 6px 18px rgba(255,122,61,.4);
}}
.stButton > button[kind="primary"]:hover {{ background: #ff8b55; }}

/* quick action buttons inside the profile card */
.st-key-quick button {{
    flex-direction: column; gap: .2rem; padding: .55rem .2rem !important; border-radius: 16px !important;
    font-size: .7rem; min-height: 0;
}}
.st-key-quick button p {{ font-size: .7rem !important; }}

div[data-baseweb="select"] > div, [data-testid="stTextInput"] input {{
    background: rgba(255,255,255,.08) !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
}}
[data-testid="stTextInput"] > div > div {{ border-radius: 999px !important; background: transparent; border: none; }}
[data-testid="stWidgetLabel"] p {{ color: var(--muted) !important; font-size: .78rem !important; }}

[data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {{
    border-radius: 999px !important;
    background: rgba(255,255,255,.06) !important;
    border: 1px solid var(--line) !important;
    color: var(--muted) !important;
}}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{
    background: rgba(255,122,61,.85) !important; color: #fff !important; border-color: transparent !important;
}}

[data-testid="stExpander"] details {{
    background: rgba(255,255,255,.05); border: 1px solid var(--line); border-radius: 18px;
}}

/* ---------- Product cards ---------- */
.pc-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: .5rem; }}
.pc-name {{ font-weight: 600; font-size: .98rem; color: var(--text); }}
.pc-meta {{ color: var(--muted); font-size: .74rem; margin-top: .1rem; }}
.pc-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin: .85rem 0 .7rem; }}
.pc-stat {{ background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: .45rem .55rem; }}
.pc-stat .v {{ font-weight: 600; font-size: .92rem; }}
.pc-stat .l {{ color: var(--muted); font-size: .66rem; }}
.pc-badges {{ display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .75rem; }}

/* ---------- Suggestion card ---------- */
.rec {{ padding: 1.1rem 1.2rem; border-left: 3px solid var(--accent); }}
.rec-label {{ font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }}
.rec-title {{ font-size: 1.08rem; font-weight: 600; margin: .45rem 0 .7rem; color: var(--text); line-height: 1.35; }}
.rec h5 {{ font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: .7rem 0 .35rem; font-weight: 600; }}
.rec ul {{ margin: 0; padding-left: 1.1rem; }}
.rec li {{ color: rgba(255,255,255,.82); font-size: .84rem; line-height: 1.55; margin-bottom: .2rem; }}
.rec ol {{ margin: 0; padding-left: 1.2rem; }}
.rec ol li {{ color: var(--text); }}
.rec ol li::marker {{ color: var(--accent); font-weight: 700; }}

.mini-kpis {{ display: grid; gap: .6rem; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); margin-bottom: 1rem; }}
.mini {{ padding: .7rem .85rem; }}
.mini .l {{ color: var(--muted); font-size: .7rem; }}
.mini .v {{ font-weight: 600; font-size: 1.1rem; }}

.combo {{ display: flex; align-items: center; gap: .7rem; padding: .75rem .9rem; margin-bottom: .55rem; }}
.combo .plus {{ color: var(--orange); font-weight: 700; font-size: 1.1rem; }}

.board {{ display: grid; gap: .8rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.board .card {{ padding: .9rem 1rem; }}
.board .v {{ font-size: 1.6rem; font-weight: 600; }}
.board .l {{ color: var(--muted); font-size: .75rem; }}

/* ---------- AI Priorities / signal cards ---------- */
.level {{ display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; margin: .2rem 0 .8rem; }}
.level .n {{
    font-size: .66rem; font-weight: 700; letter-spacing: .14em; color: var(--orange);
    border: 1px solid rgba(255,122,61,.45); border-radius: 999px; padding: .15rem .55rem;
}}
.level .t {{ font-size: 1.15rem; font-weight: 600; }}
.level .q {{ color: var(--muted); font-size: .85rem; }}
.sig {{ padding: 1rem 1.1rem; border-top: 3px solid var(--accent); height: 100%; }}
.pri-cat {{ font-size: .66rem; letter-spacing: .14em; color: var(--muted); font-weight: 600; }}
.pri-title {{ font-size: .95rem; font-weight: 700; letter-spacing: .02em; text-transform: uppercase; margin: .45rem 0 .1rem; }}
.pri-product {{ font-size: 1.02rem; font-weight: 500; color: var(--orange-soft); margin-bottom: .1rem; }}
.ev-row {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .55rem 0 .6rem; }}
.ev {{
    font-size: .74rem; padding: .22rem .55rem; border-radius: 8px;
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.1); color: rgba(255,255,255,.88);
}}
.sig-line {{ font-size: .82rem; color: rgba(255,255,255,.85); margin-bottom: .3rem; line-height: 1.45; }}
.sig-line b {{
    display: inline-block; min-width: 3.6rem; font-size: .66rem; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted); font-weight: 600;
}}
.pri-action {{ font-size: .86rem; line-height: 1.45; margin-top: .2rem; }}
.pri-action b {{ color: var(--orange-soft); font-weight: 600; margin-right: .3rem; }}
.pri-foot {{
    display: flex; justify-content: space-between; align-items: center; gap: .8rem;
    margin-top: .7rem; padding-top: .6rem; border-top: 1px solid rgba(255,255,255,.08);
}}
.pri-foot .impact {{ font-size: .8rem; font-weight: 600; color: var(--text); }}
.pri-foot .conf {{ display: flex; align-items: center; gap: .4rem; font-size: .74rem; color: var(--muted); white-space: nowrap; }}
.pri-foot .conf .bar {{ width: 54px; margin: 0; }}
.pri-more {{ font-size: .74rem; color: var(--muted); margin-top: .45rem; }}
.sig-chips {{ display: flex; flex-wrap: wrap; gap: .3rem; min-height: 1.6rem; margin-bottom: .7rem; }}
.sig-chip {{
    font-size: .68rem; font-weight: 600; letter-spacing: .06em; padding: .18rem .5rem; border-radius: 6px;
    color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
}}
.healthy {{ font-size: .74rem; color: {TEAL}; }}
div[class*="st-key-glasstile_pr_HIGH"] {{ border-top: 3px solid {ROSE} !important; }}
div[class*="st-key-glasstile_pr_MEDIUM"] {{ border-top: 3px solid {AMBER} !important; }}
div[class*="st-key-glasstile_pr_LOW"] {{ border-top: 3px solid {SKY} !important; }}
div[class*="st-key-glasstile_pr_"] .stButton > button {{ font-size: .8rem; min-height: 0; padding: .3rem .8rem; }}

/* ---------- Recommendation center / bundle cards ---------- */
.rc-head {{ display: flex; justify-content: space-between; align-items: center; gap: .8rem; flex-wrap: wrap; }}
.rc-head .left {{ display: flex; align-items: center; gap: .7rem; flex-wrap: wrap; }}
.rc-title {{ font-weight: 700; letter-spacing: .03em; text-transform: uppercase; font-size: .95rem; }}
.rc-conf {{ color: var(--muted); font-size: .82rem; white-space: nowrap; }}
.rc-conf b {{ color: var(--text); font-size: 1.05rem; margin-left: .2rem; }}
.rc-sub {{ margin: .55rem 0 .1rem; font-size: .95rem; }}
.rc-sub b {{ font-weight: 600; }}
.rc-sub span {{ color: var(--muted); }}
.rc-rows {{ display: grid; grid-template-columns: 7rem 1fr; gap: .35rem .9rem; font-size: .87rem; margin-top: .4rem; }}
.rc-rows .k {{ color: var(--muted); }}
.rc-rows .v {{ color: rgba(255,255,255,.88); line-height: 1.5; }}
.rc-rows .v.strong {{ color: var(--text); font-weight: 600; }}
.rc-rows .v.offer {{ color: var(--orange-soft); font-weight: 600; }}
.caution {{
    display: inline-block; margin: .7rem .4rem 0 0; font-size: .78rem; padding: .32rem .7rem; border-radius: 10px;
    color: #ffd9b8; background: rgba(255,122,61,.10); border: 1px solid rgba(255,122,61,.3);
}}
.cat-pill {{
    display: inline-flex; align-items: center; gap: .35rem; font-size: .7rem; font-weight: 700; letter-spacing: .08em;
    padding: .25rem .65rem; border-radius: 999px; color: var(--accent);
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
}}
.cat-pill::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }}
div[class*="st-key-glassrc_HIGH"] {{ border-left: 3px solid {ROSE} !important; }}
div[class*="st-key-glassrc_MEDIUM"] {{ border-left: 3px solid {AMBER} !important; }}
div[class*="st-key-glassrc_LOW"] {{ border-left: 3px solid {SKY} !important; }}
div[class*="st-key-glassbd_"] {{ border-left: 3px solid {ORANGE} !important; }}
div[class*="st-key-glassrc_"] .stButton > button, div[class*="st-key-glassbd_"] .stButton > button,
div[class*="st-key-famv_"] .stButton > button {{ font-size: .78rem; min-height: 0; padding: .28rem .85rem; }}
[data-testid="stPills"] button, [data-testid="stButtonGroup"] button[kind*="pills"] {{ border-radius: 999px !important; }}
.fam-sec {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 1rem 0 .55rem; }}

/* ---------- Dialog ---------- */
body:has([data-testid="stDialog"]) .st-key-rail {{ opacity: 0; pointer-events: none; }}
[data-testid="stDialog"] {{ background: rgba(10, 8, 7, 0.45); backdrop-filter: blur(6px); }}
[data-testid="stDialog"] section[role="dialog"] {{
    background: rgba(44, 38, 35, 0.82) !important;
    backdrop-filter: blur(30px) saturate(130%);
    -webkit-backdrop-filter: blur(30px) saturate(130%);
    border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 28px !important;
    box-shadow: 0 30px 90px rgba(0,0,0,.6) !important;
    color: var(--text);
}}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {{
    gap: .3rem; flex-wrap: wrap; border: 1px solid var(--line) !important;
    background: rgba(255,255,255,.05);
    border-radius: 22px; padding: .3rem; width: fit-content; max-width: 100%;
    margin-bottom: .9rem; box-shadow: none !important;
}}
[data-testid="stTab"] {{
    border-radius: 999px !important; padding: .4rem .95rem !important; height: auto !important;
    background: transparent; color: var(--muted) !important; transition: all .2s ease;
}}
[data-testid="stTab"]:hover {{ background: rgba(255,255,255,.08); color: var(--text) !important; }}
[data-testid="stTab"] p {{ font-size: .82rem !important; }}
[data-testid="stTab"][aria-selected="true"] {{
    background: var(--orange) !important; color: #fff !important; box-shadow: 0 4px 14px rgba(255,122,61,.4);
}}
[data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none !important; }}

.take {{
    display: flex; gap: .55rem; align-items: flex-start;
    font-size: .92rem; line-height: 1.45; color: var(--text);
    padding: .6rem .85rem; margin: .1rem 0 .8rem; border-radius: 12px;
    background: rgba(255,122,61,.09); border: 1px solid rgba(255,122,61,.22);
}}
.take::before {{ content: "💡"; }}
.take b {{ color: var(--orange-soft); font-weight: 600; }}

/* ---------- Decision explainer ---------- */
.verdict {{ padding: 1.4rem 1.6rem; display: grid; grid-template-columns: 1fr auto; gap: 1.5rem; align-items: center; }}
.verdict .who {{ font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
.verdict .act {{ font-size: 1.75rem; font-weight: 600; margin: .3rem 0 .6rem; }}
.verdict p {{ margin: 0; font-size: .95rem; line-height: 1.65; color: rgba(255,255,255,.88); max-width: 760px; }}
.verdict p b {{ color: var(--text); }}
.ring {{ text-align: center; min-width: 120px; }}
.ring .v {{ font-size: 2.1rem; font-weight: 700; color: var(--orange-soft); line-height: 1; }}
.ring .l {{ font-size: .72rem; color: var(--muted); margin-top: .35rem; }}
.ring .bar {{ width: 110px; margin: .6rem auto 0; }}
.why {{ display: grid; grid-template-columns: 2.4rem 1fr; gap: .2rem 1rem; padding: 1.05rem 0; border-top: 1px solid rgba(255,255,255,.08); }}
.why:first-child {{ border-top: none; padding-top: .3rem; }}
.why .n {{
    width: 2.1rem; height: 2.1rem; border-radius: 50%; display: grid; place-items: center;
    font-weight: 700; font-size: .85rem; color: var(--orange-soft);
    background: rgba(255,122,61,.12); border: 1px solid rgba(255,122,61,.35);
}}
.why .k {{ font-size: .66rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
.why .t {{ font-size: 1.05rem; font-weight: 600; margin: .15rem 0 .35rem; }}
.why p {{ margin: 0 0 .5rem; font-size: .9rem; line-height: 1.6; color: rgba(255,255,255,.84); max-width: 860px; }}
.why p b {{ color: var(--text); }}
.checks {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .2rem; }}
.todo {{ margin: 0; padding-left: 1.2rem; }}
.todo li {{ font-size: .92rem; line-height: 1.55; margin-bottom: .55rem; color: rgba(255,255,255,.88); }}
.todo li::marker {{ color: var(--orange); font-weight: 700; }}
.todo li span {{ color: var(--muted); font-size: .82rem; }}
@media (max-width: 760px) {{ .verdict {{ grid-template-columns: 1fr; }} }}

/* ---------- "How it's calculated" hover widget ---------- */
.calc {{
    position: relative; cursor: help; outline: none;
    border-bottom: 1px dashed rgba(255,255,255,.38);
}}
.calc:hover, .calc:focus {{ border-bottom-color: var(--orange); }}
.calc-pop {{
    position: absolute; left: 0; top: calc(100% + 10px); z-index: 10000;
    display: block; width: max-content; min-width: 220px; max-width: 340px;
    padding: .75rem .85rem .7rem; border-radius: 14px;
    background: rgba(26, 22, 20, .97); border: 1px solid rgba(255,122,61,.4);
    box-shadow: 0 18px 40px rgba(0,0,0,.55);
    font-size: .78rem; font-weight: 400; line-height: 1.45; color: var(--text);
    text-align: left; text-transform: none; letter-spacing: normal; white-space: normal;
    visibility: hidden; opacity: 0; transform: translateY(-4px);
    transition: opacity .15s ease, transform .15s ease, visibility .15s;
    pointer-events: none;
}}
.calc-pop.right {{ left: auto; right: 0; }}
.calc:hover .calc-pop, .calc:focus .calc-pop {{ visibility: visible; opacity: 1; transform: translateY(0); }}
.calc-pop .h {{
    display: block; font-size: .64rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--orange-soft); margin-bottom: .4rem;
}}
.calc-pop .f {{
    display: block; margin: .25rem 0; padding: .32rem .5rem; border-radius: 8px;
    background: rgba(255,255,255,.06); font-variant-numeric: tabular-nums;
}}
.calc-pop .r {{ display: block; margin-top: .35rem; font-weight: 600; color: var(--orange-soft); }}
.calc-pop .s {{ display: block; margin-top: .45rem; font-size: .68rem; color: var(--muted); }}
/* lift the hovered block above its neighbours so the popover is never covered */
[data-testid="stElementContainer"]:has(.calc:hover), [data-testid="stElementContainer"]:has(.calc:focus),
div[class*="st-key-"]:has(.calc:hover), div[class*="st-key-"]:has(.calc:focus),
.card:has(.calc:hover), .card:has(.calc:focus) {{ position: relative; z-index: 1000 !important; }}
.kpi {{ overflow: visible !important; }}
.kpi::before {{ border-radius: 20px 0 0 20px; }}

.footer-note {{ text-align: center; color: var(--muted); font-size: .76rem; margin-top: 1.6rem; }}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# UI BUILDING BLOCKS
# ============================================================

def html_block(markup):

    st.markdown(markup, unsafe_allow_html=True)


def card_head(title, sub="", right=""):

    sub_html = f'<div class="c-sub">{sub}</div>' if sub else '<div style="height:.5rem"></div>'

    html_block(
        f'<div class="c-row"><div class="c-title">{title}</div>{right}</div>{sub_html}'
    )


def calc(value, lines, source=None, result=None, align="left", title="How it's calculated"):
    """Wrap a displayed number in a hover popover showing its working.

    lines  — formula steps with the real numbers plugged in
    result — optional final line (e.g. "= 92%")
    source — where the inputs come from (file / column)
    """

    if not lines:
        return str(value)

    steps = "".join(f'<span class="f">{line}</span>' for line in lines)
    end = f'<span class="r">{result}</span>' if result else ""
    src = f'<span class="s">Source: {source}</span>' if source else ""
    side = " right" if align == "right" else ""

    return (
        f'<span class="calc" tabindex="0">{value}'
        f'<span class="calc-pop{side}"><span class="h">{title}</span>{steps}{end}{src}</span></span>'
    )


def takeaway(text):
    """One plain-English sentence telling the viewer what the chart means."""

    html_block(f'<div class="take"><span>{text}</span></div>')


def badge(text, accent, icon=""):

    return (
        f'<span class="badge" style="--accent:{accent}">'
        f'{icon} {esc(text)}</span>'
    )


def bar(pct):

    pct = max(0.0, min(100.0, pct))

    return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'


def bar_cell(pct, label=None):

    label = f"{pct:.1f}%" if label is None else label

    return f'<div class="cell-bar">{bar(pct)}<span class="mono">{label}</span></div>'


def kpi(icon, label, value, note="", how=None):
    """how = (lines, source[, result]) -> hover popover with the working."""

    note_html = f'<div class="kpi-note">{note}</div>' if note else ""

    if how:
        value = calc(value, *how)

    return f"""
<div class="card hover kpi fade-in">
  <div class="kpi-icon">{icon}</div>
  <div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{note_html}</div>
</div>
"""


def kpi_strip(cards):

    html_block('<div class="kpi-strip">' + "".join(cards) + "</div>")


def insight(icon, title, body):

    return f"""
<div class="card hover insight fade-in">
  <div class="ic">{icon}</div>
  <h4>{title}</h4>
  <p>{body}</p>
</div>
"""


def glass_table(headers, rows, max_height=440):

    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )

    html_block(
        f'<div class="gt-wrap" style="max-height:{max_height}px">'
        f'<table class="gt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def spacer(rem=1.0):

    html_block(f'<div style="height:{rem}rem"></div>')


# ============================================================
# COMPANY / DATASET SELECTION
# Every company has its own dataset and its own AI outputs.
# ============================================================

STAGE_STATES = {"done": ("Done", TEAL), "stale": ("Out of date", AMBER), "pending": ("Not run", STONE),
                "unavailable": ("Needs complete data", STONE)}

KEY_COLUMNS = [
    "date", "sku", "product", "category", "ad_type", "ad_spend_inr", "revenue_inr", "roas",
    "sales_volume", "stock_on_hand", "days_of_inventory_remaining", "recommended_action",
]

# Widget keys whose options depend on the active company's data
COMPANY_SCOPED_KEYS = [
    "rc_category", "rc_product", "rc_search", "diag_cause", "decide_action", "decide_search",
    "trend_products", "rev_spend_product", "product_search", "affiliate_chart_last", "inventory_chart_last",
]


@st.cache_data
def company_profile(path, mtime):

    return co.profile(path)


def profile_of(company):

    path = company["dataset"]

    return company_profile(str(path), path.stat().st_mtime) if path.exists() else None


def switch_company(slug):

    st.session_state.company = slug
    st.session_state.company_select = slug
    st.session_state.dataset_view = slug
    st.query_params["company"] = slug

    for key in ["open_sku", "open_family", "open_tab"] + COMPANY_SCOPED_KEYS:
        st.session_state.pop(key, None)


def on_company_select():

    switch_company(st.session_state.company_select)


def company_switcher(active):

    names = {c["slug"]: c["name"] for c in co.list_companies()}

    if st.session_state.get("company_select") not in names:
        st.session_state.company_select = active["slug"]

    st.selectbox(
        "Company", list(names), key="company_select", on_change=on_company_select,
        format_func=lambda slug: f"🏢  {names[slug]}", label_visibility="collapsed",
        help="Switch the whole dashboard to another company's dataset",
    )


def company_card(company, active_slug, prefix):
    """Card with dataset facts, processing status and switch/view buttons."""

    prof = profile_of(company)
    processed = co.is_processed(company)
    is_active = company["slug"] == active_slug

    status = badge("AI outputs ready", TEAL, "✓") if processed else badge("Needs processing", AMBER, "⏳")
    simulated = badge("Simulated", STONE) if company["simulated"] else badge("Uploaded", SKY)
    active = badge("Active", ORANGE, "●") if is_active else ""

    facts = (
        f'{prof["rows"]:,} rows · {prof["skus"]} SKUs · {prof["products"]} products · {prof["days"]} days'
        if prof else "Dataset file missing"
    )
    dates = f'{prof["start"]} → {prof["end"]}' if prof else ""

    with st.container(key=f"glasstile_co_{prefix}_{company['slug']}"):

        html_block(
            f"""
<div class="c-row"><div class="c-title">🏢 {esc(company["name"])}</div>{active}</div>
<div class="c-sub" style="min-height:2.4rem">{esc(company["description"])}</div>
<div style="font-size:.85rem;margin-bottom:.2rem">{facts}</div>
<div class="c-sub">{dates}</div>
<div class="pc-badges">{status} {simulated}</div>
"""
        )

        with st.container(horizontal=True, gap="small", key=f"cob_{prefix}_{company['slug']}"):

            st.button(
                "Active" if is_active else ("Open dashboard" if processed else "Select"),
                key=f"co_switch_{prefix}_{company['slug']}",
                icon=":material/check:" if is_active else ":material/swap_horiz:",
                on_click=switch_company, args=(company["slug"],),
                disabled=is_active, type="primary" if not is_active and processed else "secondary",
            )

            if prefix == "ds":
                st.button(
                    "View dataset", key=f"co_view_{company['slug']}", icon=":material/table_view:",
                    on_click=lambda slug=company["slug"]: st.session_state.update(dataset_view=slug),
                )


def run_pipeline_ui(company, retrain=False):
    """Run the ML pipeline for one company with a live log. Returns True on success."""

    lines = []

    with st.status(f"Processing {company['name']}'s dataset…", expanded=True) as status:

        current = st.empty()
        log = st.empty()

        try:
            for stage, line in co.run_pipeline(company, retrain=retrain):
                current.markdown(f"**{stage}**")
                lines.append(line)
                log.code("\n".join(lines[-18:]), language=None)

        except RuntimeError as error:
            status.update(label=f"Processing failed — {str(error).splitlines()[0]}", state="error")
            st.session_state[f"log_{company['slug']}"] = "\n".join(lines)
            return False

        status.update(label=f"{company['name']} processed — every AI output is up to date", state="complete", expanded=False)

    st.session_state[f"log_{company['slug']}"] = "\n".join(lines)
    st.cache_data.clear()

    return True


def pipeline_panel(company, key):
    """Each pipeline stage, the files it wrote for this company, and a run button."""

    stages = co.stage_status(company)
    done = sum(s["state"] == "done" for s in stages)

    card_head(
        "AI processing pipeline",
        f"{done} of {len(stages)} stages up to date for {esc(company['name'])} · outputs in "
        f"<span class='mono'>{esc(company['models'].relative_to(co.ROOT))}</span>",
    )

    rows = []

    for i, stage in enumerate(stages, start=1):

        label, color = STAGE_STATES[stage["state"]]
        chips = []

        for f in stage["files"]:
            when = f["modified"].strftime("%d %b %Y %H:%M") if f["modified"] else "not created"
            rows_text = f" · {f['rows']:,} rows" if f["rows"] is not None else ""
            mark = "✓" if f["exists"] else "·"
            chips.append(f'<span class="ev" title="{esc(when)}">{mark} {esc(f["name"])}{rows_text}</span>')

        files = " ".join(chips)
        updated = max((f["modified"] for f in stage["files"] if f["modified"]), default=None)

        rows.append(
            f"""
<div class="lrow" style="align-items:flex-start">
  <div class="l-icon" style="font-size:.8rem;font-weight:700">{i}</div>
  <div class="l-main">
    <div class="c-row"><span class="l-name">{esc(stage["label"])} <span class="muted mono" style="font-size:.72rem">· {esc(stage["script"])}</span></span>{badge(label, color)}</div>
    <div class="ev-row" style="margin:.35rem 0 .1rem">{files}</div>
    <div class="l-sub">{"Updated " + updated.strftime("%d %b %Y, %H:%M") if updated else "Not run yet"}</div>
  </div>
</div>
"""
        )

    html_block("".join(rows))

    retrain = False

    with st.container(horizontal=True, gap="small", vertical_alignment="center", key=f"pipe_ctl_{key}"):

        run = st.button(
            "Run AI pipeline" if done < len(stages) else "Re-run AI pipeline",
            key=f"run_{key}", icon=":material/play_arrow:", type="primary",
        )

        if not company["builtin"]:
            retrain = st.checkbox("Retrain model", key=f"retrain_{key}",
                                  help="Rebuild the ROAS model from this company's data before running the engines")

    if company["builtin"]:
        html_block('<div class="c-sub" style="margin-top:.4rem">Glowroots keeps its original trained model — '
                   're-running only refreshes the engines.</div>')

    if run and run_pipeline_ui(company, retrain=retrain):
        st.rerun()

    previous = st.session_state.get(f"log_{company['slug']}")

    if previous:
        with st.expander("Last processing log"):
            st.code(previous[-6000:], language=None)


def dataset_preview(company, key):
    """Look inside the raw dataset and the processed outputs."""

    raw_tab, out_tab = st.tabs(["📄 Raw dataset", "🧠 Processed outputs"])

    with raw_tab:

        if not company["dataset"].exists():
            st.info("No dataset file.")
        else:
            raw = pd.read_csv(company["dataset"])

            c1, c2 = st.columns([2, 1], gap="small")

            with c1:
                products = ["All products"] + sorted(raw["product"].dropna().unique().tolist()) if "product" in raw else ["All products"]
                chosen = st.selectbox("Product", products, key=f"prev_product_{key}", label_visibility="collapsed")

            with c2:
                all_columns = st.toggle(f"All {raw.shape[1]} columns", key=f"prev_all_{key}")

            if chosen != "All products":
                raw = raw[raw["product"] == chosen]

            columns = raw.columns.tolist() if all_columns else [c for c in KEY_COLUMNS if c in raw.columns]

            st.dataframe(raw[columns], height=300, hide_index=True, width="stretch")
            html_block(f'<div class="c-sub">{len(raw):,} rows · {esc(company["dataset"].relative_to(co.ROOT))}</div>')

    with out_tab:

        outputs = sorted(p.name for p in company["models"].glob("*.csv")) if company["models"].exists() else []

        if not outputs:
            st.info("Nothing processed yet — run the AI pipeline above.")
        else:
            default = outputs.index("final_ai_decision_report.csv") if "final_ai_decision_report.csv" in outputs else 0
            name = st.selectbox("Output file", outputs, index=default, key=f"prev_out_{key}", label_visibility="collapsed")
            out = pd.read_csv(company["models"] / name)
            st.dataframe(out.head(300), height=300, hide_index=True, width="stretch")
            html_block(f'<div class="c-sub">{len(out):,} rows · {out.shape[1]} columns · written by the pipeline for {esc(company["name"])}</div>')


def company_grid(active_slug, prefix):

    companies_list = co.list_companies()

    for start in range(0, len(companies_list), 3):

        cols = st.columns(3, gap="medium")

        for col, company in zip(cols, companies_list[start:start + 3]):

            with col:
                company_card(company, active_slug, prefix)


def render_chooser():
    """First screen of a session: pick which company to analyse."""

    html_block(
        """
<div class="topbar fade-in" style="margin-bottom:.4rem">
  <div>
    <div class="tb-title">⚡ ADPULSE AI</div>
    <div class="tb-sub">Choose a company — every view, prediction and recommendation is built from that company's own dataset</div>
  </div>
</div>
"""
    )

    takeaway("Each company has its <b>own dataset</b> and its <b>own AI outputs</b>. "
             "You can switch at any time from the menu at the top right.")

    company_grid(None, "pick")

    spacer(1)

    st.button(
        "Analyse your own CSVs — drag & drop", key="pick_upload", icon=":material/upload_file:", type="primary",
        on_click=lambda: (switch_company(co.DEFAULT_SLUG), st.session_state.update(nav="Upload")),
    )


def render_unprocessed(company):
    """A company is selected but its dataset hasn't been through the pipeline yet."""

    left, right = st.columns([3, 1.1], gap="medium")

    with left:
        html_block(
            f'<div class="topbar fade-in"><div><div class="tb-title">{esc(company["name"])}</div>'
            f'<div class="tb-sub">This dataset hasn\'t been processed yet</div></div></div>'
        )

    with right:
        company_switcher(company)

    takeaway(f"Run the AI pipeline once to train {esc(company['name'])}'s model and generate its decisions — "
             "the dashboard opens as soon as it finishes.")

    with st.container(key="glass_unprocessed_pipeline"):
        pipeline_panel(company, "unprocessed")

    spacer(1)

    with st.container(key="glass_unprocessed_preview"):
        card_head("What's in this dataset")
        dataset_preview(company, "unprocessed")


def page_datasets():
    """Panel to browse every company's dataset and its processing status."""

    takeaway(
        f"<b>{len(co.list_companies())} companies</b> available. Each dataset is processed separately — "
        "the pipeline below shows exactly which AI outputs were produced for which company."
    )

    company_grid(ACTIVE["slug"], "ds")

    view_slug = st.session_state.get("dataset_view") or ACTIVE["slug"]
    viewed = co.get_company(view_slug) or ACTIVE

    spacer(1.2)

    html_block(
        f'<div class="level"><span class="t">{esc(viewed["name"])}</span>'
        f'<span class="q">{"active company" if viewed["slug"] == ACTIVE["slug"] else "viewing — not active"}</span></div>'
    )

    prof = profile_of(viewed)

    if prof:
        kpi_strip([
            kpi("🧾", "Rows", f"{prof['rows']:,}", f"{prof['columns']} columns"),
            kpi("📦", "Products", f"{prof['products']}", f"{prof['skus']} SKUs"),
            kpi("📅", "Date range", f"{prof['days']} days", f"{prof['start']} → {prof['end']}"),
            kpi("🗂️", "Categories", f"{len(prof['categories'])}", ", ".join(prof["categories"][:4])),
        ])

        spacer(0.6)

        if prof["problems"]:
            st.warning("Schema check: " + " · ".join(prof["problems"]))
        else:
            html_block(f'<div class="healthy" style="margin-bottom:.6rem">✓ Schema check passed — all '
                       f'{len(co.REQUIRED_COLUMNS)} required columns present</div>')

    left, right = st.columns([1, 1.25], gap="medium")

    with left:
        with st.container(key=f"glass_pipeline_{viewed['slug']}"):
            pipeline_panel(viewed, f"ds_{viewed['slug']}")

    with right:
        with st.container(key=f"glass_preview_{viewed['slug']}"):
            card_head("Inside the dataset", "Raw input on the left tab · AI outputs on the right")
            dataset_preview(viewed, f"ds_{viewed['slug']}")

    spacer(1)

    sources = co.manifest(viewed)

    if sources:
        spacer(1)
        with st.container(key=f"glass_sources_{viewed['slug']}"):
            card_head("Uploaded files", "Each file was detected, mapped and reconciled into the dataset above")
            glass_table(
                ["File", "Detected source", "Mapped columns", "Added"],
                [[esc(e["original"]), esc(e["source"]), f'{sum(1 for m in e["mapping"] if m["field"])} of {len(e["mapping"])}',
                  esc(e["added"].replace("T", " "))] for e in sources],
                260,
            )
            html_block(capability_list({k: tuple(v) for k, v in (co.capabilities_of(viewed) or {}).items()}))

    spacer(1)
    st.button("Upload CSV files", key="ds_upload", icon=":material/upload_file:", on_click=go_to, args=("Upload",))


# Which company is active? (?company=<slug> deep-links straight in)

if "company" not in st.session_state:
    st.session_state.company = st.query_params.get("company")

ACTIVE = co.get_company(st.session_state.company) if st.session_state.company else None

if ACTIVE is None:
    render_chooser()
    st.stop()

MODELS_DIR = ACTIVE["models"]
DATASET = ACTIVE["dataset"]

FINAL_REPORT = MODELS_DIR / "final_ai_decision_report.csv"
ACTION_PERFORMANCE = MODELS_DIR / "action_learning_performance.csv"
LEARNED_DECISIONS = MODELS_DIR / "latest_learned_decisions.csv"
ROOT_CAUSE = MODELS_DIR / "latest_root_cause_decisions.csv"
DECISION_RESULTS = MODELS_DIR / "decision_engine_results.csv"
ROAS_PREDICTIONS = MODELS_DIR / "roas_predictions.csv"
MODEL_EVALUATION = MODELS_DIR / "model_evaluation.csv"
FEATURE_IMPORTANCE = MODELS_DIR / "roas_feature_importance.csv"

final_df = read_csv(FINAL_REPORT)
action_df = read_csv(ACTION_PERFORMANCE)
learned_df = read_csv(LEARNED_DECISIONS)
root_df = read_csv(ROOT_CAUSE)
decision_df = read_csv(DECISION_RESULTS)
prediction_df = read_csv(ROAS_PREDICTIONS)
evaluation_df = read_csv(MODEL_EVALUATION)
feature_df = read_csv(FEATURE_IMPORTANCE)

# No dataset at all -> show the pipeline / upload screen
if not DATASET.exists():
    render_unprocessed(ACTIVE)
    st.stop()

# What the active company's data supports
ML_READY = not final_df.empty                       # full ML decision pipeline outputs exist
PRED_READY = not prediction_df.empty and not evaluation_df.empty
CAPS = co.capabilities_of(ACTIVE)                   # None for built-in, complete datasets


# ============================================================
# NUMERIC CONVERSION
# ============================================================


numeric_columns = [
    "predicted_next_day_roas",
    "predicted_next_day_profit_inr",
    "opportunity_score",
    "learned_decision_confidence",
    "historical_action_success_rate",
    "next_day_roas",
    "roas",
    "profit_inr",
    "revenue_inr",
    "ad_spend_inr",
    "success_rate_pct",
    "success_rate",
    "learning_score",
    "decision_count",
    "decisions",
    "success_count",
    "successful_decisions",
    "average_next_day_roas",
    "average_profit_change",
    "actual_roas",
    "baseline_prediction",
    "random_forest_prediction",
    "importance"
]


for df in [
    final_df,
    action_df,
    learned_df,
    root_df,
    decision_df,
    prediction_df,
    evaluation_df,
    feature_df
]:

    if df.empty:
        continue

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )



# ============================================================
# COLUMN DISCOVERY
# ============================================================

action_filter_column = find_column(
    final_df,
    [
        "current_action",
        "recommended_action",
        "autonomous_action"
    ]
)

root_filter_column = find_column(
    final_df,
    [
        "root_cause",
        "primary_root_cause"
    ]
)

confidence_column = find_column(
    final_df,
    [
        "learned_decision_confidence",
        "decision_confidence_pct",
        "decision_confidence",
        "confidence_pct"
    ]
)

roas_column = find_column(
    final_df,
    [
        "predicted_next_day_roas",
        "predicted_roas"
    ]
)

profit_column = find_column(
    final_df,
    [
        "predicted_next_day_profit_inr",
        "predicted_profit_inr",
        "predicted_profit"
    ]
)

sku_column = find_column(final_df, ["sku"])
product_column = find_column(final_df, ["product"])
history_column = find_column(final_df, ["historical_action_success_rate"])


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_skus = len(final_df)

high_confidence = 0

if confidence_column is not None:

    high_confidence = int(
        (final_df[confidence_column] >= 70).sum()
    )

high_share = high_confidence / total_skus * 100 if total_skus else 0.0

avg_roas = 0.0

if roas_column is not None:

    avg_roas = safe_float(final_df[roas_column].mean())

total_profit = 0.0

if profit_column is not None:

    total_profit = safe_float(final_df[profit_column].sum())

inventory_risk = 0
payment_risk = 0
funnel_risk = 0

if root_filter_column is not None:

    causes = final_df[root_filter_column].astype(str)

    inventory_risk = int(causes.str.contains("INVENTORY", case=False).sum())
    payment_risk = int(causes.str.contains("PAYMENT", case=False).sum())
    funnel_risk = int(causes.str.contains("FUNNEL", case=False).sum())

risk_alerts = 0

if root_filter_column is not None:

    risk_alerts = int(
        final_df[root_filter_column]
        .astype(str)
        .str.contains(
            "INVENTORY|PAYMENT|FUNNEL",
            case=False,
            regex=True
        )
        .sum()
    )


# Policy learning summary

policy_action_column = find_action_column(action_df)
policy_success_column = find_success_rate_column(action_df)
policy_count_column = find_column(action_df, ["decision_count", "decisions", "count"])
policy_success_count_column = find_column(action_df, ["success_count", "successful_decisions", "successes"])
policy_learning_column = find_column(action_df, ["learning_score", "learning"])
policy_strength_column = find_column(action_df, ["policy_strength", "action_quality", "strength"])
policy_avg_roas_column = find_column(action_df, ["average_next_day_roas"])

policy_temp = pd.DataFrame()

if (
    policy_action_column is not None
    and policy_success_column is not None
):

    policy_temp = action_df.copy()

    policy_temp[policy_success_column] = pd.to_numeric(
        policy_temp[policy_success_column],
        errors="coerce"
    )

    policy_temp = policy_temp.dropna(subset=[policy_success_column])

    # If stored as decimal instead of percentage
    if (
        not policy_temp.empty
        and policy_temp[policy_success_column].max() <= 1
    ):

        policy_temp[policy_success_column] *= 100

strongest_row = None
strongest_policy = "N/A"
strongest_rate = 0.0
weakest_policy = "N/A"
weakest_rate = 0.0

if not policy_temp.empty:

    strongest_row = policy_temp.loc[policy_temp[policy_success_column].idxmax()]
    worst_row = policy_temp.loc[policy_temp[policy_success_column].idxmin()]

    strongest_policy = str(strongest_row[policy_action_column])
    strongest_rate = float(strongest_row[policy_success_column])
    weakest_policy = str(worst_row[policy_action_column])
    weakest_rate = float(worst_row[policy_success_column])

total_decisions = int(action_df[policy_count_column].sum()) if policy_count_column else 0
total_successes = int(action_df[policy_success_count_column].sum()) if policy_success_count_column else 0
overall_success = total_successes / total_decisions * 100 if total_decisions else 0.0


# Model evaluation summary

def evaluation_value(metric, column_candidates):

    if evaluation_df.empty:
        return None

    metric_column = find_column(evaluation_df, ["metric"])
    value_column = find_column(evaluation_df, column_candidates)

    if metric_column is None or value_column is None:
        return None

    match = evaluation_df[
        evaluation_df[metric_column].astype(str).str.upper() == metric
    ]

    if match.empty:
        return None

    return safe_float(match.iloc[0][value_column], None)


rmse_gain = evaluation_value("RMSE_IMPROVEMENT_PERCENT", ["random_forest", "model"])


# ============================================================
# NAVIGATION
# ============================================================

PAGES = ["Overview", "Products", "Recommendations", "Bundling", "Analytics", "Decide", "Explain", "Actions", "Learn", "Pipeline", "Upload", "Datasets"]

PAGE_ICONS = {
    "Overview": ":material/space_dashboard:",
    "Products": ":material/inventory_2:",
    "Recommendations": ":material/auto_awesome:",
    "Bundling": ":material/layers:",
    "Analytics": ":material/monitoring:",
    "Decide": ":material/smart_toy:",
    "Explain": ":material/manage_search:",
    "Learn": ":material/autorenew:",
    "Pipeline": ":material/account_tree:",
    "Datasets": ":material/database:",
    "Actions": ":material/task_alt:",
    "Upload": ":material/upload_file:",
}

PAGE_TITLES = {
    "Overview": ("ADPULSE AI", "Data → Detect → Diagnose → Decide → Act → Learn"),
    "Products": ("Products", "AI Priorities → Product Intelligence → Product Insights"),
    "Recommendations": ("AI Recommendation Center", "Every suggestion in one place · pick a product to see everything we recommend for it"),
    "Bundling": ("Smart Product Bundling", "Pair a popular product with a slow one to lift order value and move weaker stock"),
    "Analytics": ("Performance & Forecast", "What happened, what the model expects next, stock and sales — in four tabs"),
    "Decide": ("Diagnose & Decide", "Why each product struggles, and the action the AI chose"),
    "Explain": ("Decision Explainer", "Follow the AI's reasoning for any single SKU"),
    "Learn": ("Policy Learning", "Step 4 · How past outcomes reshape decision confidence"),
    "Pipeline": ("AI Pipeline", "Nine stages from raw data to self-improving decisions"),
    "Datasets": ("Datasets", "Every company's data and what the AI produced from it"),
    "Actions": ("Act & Learn", "Approve a decision → simulated execution → measured outcome → the AI learns"),
    "Upload": ("Upload data", "Drop CSV exports — they run through the same decision engine"),
}

if "nav" not in st.session_state:
    st.session_state.nav = "Overview"


def go_to(page):

    st.session_state.nav = page
    st.session_state.open_sku = None
    st.session_state.open_family = None


# Pages that were merged keep working from old links / open sessions
MERGED = {"Predict": "Analytics", "Diagnose": "Decide"}

if st.session_state.nav not in PAGES:
    st.session_state.nav = MERGED.get(st.session_state.nav, "Overview")

page = st.session_state.nav


# Floating left icon rail

with st.container(key="rail"):

    for name in PAGES:

        st.button(
            name,
            key=f"rail_{name}",
            icon=PAGE_ICONS[name],
            help=name,
            on_click=go_to,
            args=(name,),
            type="primary" if name == page else "secondary"
        )


# Top bar: page title + company switcher

title, subtitle = PAGE_TITLES[page]

bar_left, bar_right = st.columns([3, 1.1], gap="medium", vertical_alignment="center")

with bar_left:
    html_block(
        f"""
<div class="topbar fade-in" style="margin-bottom:0">
  <div>
    <div class="tb-title">{title}</div>
    <div class="tb-sub">{subtitle}</div>
  </div>
</div>
"""
    )

with bar_right:
    company_switcher(ACTIVE)
    html_block(
        f'<div class="c-sub" style="text-align:right;margin:.3rem 0 0">'
        f'{"⚠ Simulated data" if ACTIVE["simulated"] else "Uploaded dataset"} · '
        f'{read_csv(DATASET).shape[0]:,} rows</div>'
    )

spacer(1)


# ============================================================
# SHARED CHARTS
# ============================================================

date_col = find_column(prediction_df, ["date"])
actual_col = find_column(prediction_df, ["actual_roas", "next_day_roas"])
baseline_pred_col = find_column(prediction_df, ["baseline_prediction"])
rf_pred_col = find_column(prediction_df, ["random_forest_prediction", "prediction"])


def roas_trend_chart(show):
    """Actual vs forecast ROAS per day. `show`: 'Random Forest', 'Baseline' or 'Both'."""

    if not (date_col and actual_col):
        return None

    series = {actual_col: "Actual"}

    if rf_pred_col and show in ("Random Forest", "Both"):
        series[rf_pred_col] = "AI forecast"

    if baseline_pred_col and show in ("Baseline", "Both"):
        series[baseline_pred_col] = "Simple baseline"

    daily = (
        prediction_df
        .groupby(date_col)[list(series)]
        .mean()
        .reset_index()
        .rename(columns=series)
    )

    long = daily.melt(date_col, var_name="Series", value_name="ROAS")

    series_colors = {"Actual": CREAM, "AI forecast": ORANGE, "Simple baseline": STONE}
    domain = [name for name in series_colors if name in series.values()]
    colors = [series_colors[name] for name in domain]

    return (
        alt.Chart(long)
        .mark_line(interpolate="monotone", strokeWidth=3, point=alt.OverlayMarkDef(size=45, filled=True))
        .encode(
            x=alt.X(f"{date_col}:T", title=None, axis=alt.Axis(format="%d %b", tickCount="day")),
            y=alt.Y("ROAS:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=domain, range=colors)),
            strokeDash=alt.condition(
                alt.datum.Series == "Simple baseline", alt.value([5, 4]), alt.value([0])
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date"),
                "Series:N",
                alt.Tooltip("ROAS:Q", format=".2f")
            ]
        )
    )


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview():

    data = raw_df
    revenue = data["revenue_inr"].sum(min_count=1)
    spend = data["ad_spend_inr"].sum(min_count=1)
    profit = data["profit_inr"].sum(min_count=1)
    blended = revenue / spend if pd.notna(spend) and spend else None
    days = data["date"].nunique()
    src = rel(DATASET)

    # 1 · Four numbers that matter
    kpi_strip([
        kpi("💰", "Revenue", fmt(revenue, ",.0f", "₹"), f"last {days} days", how=(
            [f"Σ daily revenue over {data['sku'].nunique()} products × {days} days"], f"{src} · revenue_inr")),
        kpi("📣", "Ad spend", fmt(spend, ",.0f", "₹"), "all platforms", how=(
            [f"Σ daily ad spend over {days} days"], f"{src} · ad_spend_inr")),
        kpi("📈", "Return per ₹1 of ads", fmt(blended, ".2f", "₹"), "blended ROAS", how=(
            [f"Revenue {fmt(revenue, ',.0f', '₹')} ÷ ad spend {fmt(spend, ',.0f', '₹')}"], src)),
        kpi("🧮", "Contribution profit", fmt(profit, ",.0f", "₹"), "after ad spend", how=(
            ["Σ (revenue × margin − ad spend) per product-day"], f"{src} · profit_inr", None, "right")),
    ])

    spacer(1.2)

    # 2 · What needs attention (the decision engine's output dominates the page)
    top = pe.select_priorities(recommendations, limit=5) if not recommendations.empty else []
    affected = recommendations["sku"].nunique() if not recommendations.empty else 0

    html_block(
        f'<div class="level"><span class="t">I found {len(top)} things that matter</span>'
        f'<span class="q">{len(recommendations)} signals across {affected} of {data["sku"].nunique()} products · ranked by estimated ₹ impact × confidence × urgency</span></div>'
    )

    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        with st.container(key="glass_attention"):

            if not top:
                html_block('<div class="healthy">✓ Nothing needs attention right now</div>')

            rows = "".join(
                f"""
<div class="lrow" style="align-items:flex-start;padding:.75rem .2rem">
  <div class="l-icon" style="font-size:.8rem;font-weight:700">{i}</div>
  <div class="l-main">
    <div class="c-row"><span class="l-name" style="white-space:normal">{esc(rec["title"])} · {esc(rec["label"])}</span>{rec_tier(rec)}</div>
    <div class="l-sub" style="margin:.15rem 0">{esc(rec["headline"])}</div>
    <div style="font-size:.84rem"><b style="color:var(--orange-soft)">Action</b> {esc(rec["action"])}</div>
    <div class="l-sub" style="margin-top:.2rem">{rec_impact(rec)} · {rec_confidence(rec, f"{rec['confidence']}% confidence", align="left")}</div>
  </div>
</div>
"""
                for i, rec in enumerate(top, start=1)
            )

            html_block(rows)

            st.button("Review & approve on AI Priorities", key="ov_priorities", icon=":material/arrow_forward:",
                      icon_position="right", on_click=go_to, args=("Products",), type="primary", width="stretch")

    with right:
        with st.container(key="glass_changes"):

            card_head("What changed", "Unusual moves and week-on-week shifts")

            changes = []

            for _, a in anomalies.head(3).iterrows():
                changes.append((f"{a['product']} · {a['metric']} {'up' if a['direction'] == 'up' else 'down'} {abs(a['change']):.0%}",
                                f"Last 3 days vs its usual level (robust z = {a['z']:.1f})"))

            shifts = recommendations[recommendations["kind"].isin(
                ["roas_decline", "cvr_decline", "demand_surge", "demand_drop", "platform_shift"])] if not recommendations.empty else pd.DataFrame()

            for _, r in shifts.head(5 - len(changes)).iterrows():
                changes.append((f"{r['label']} · {r['headline']}", r["cause"]))

            if changes:
                html_block("".join(
                    f'<div class="lrow"><div class="l-main"><div class="l-name" style="white-space:normal">{esc(t)}</div>'
                    f'<div class="l-sub">{esc(d)}</div></div></div>' for t, d in changes
                ))
            else:
                html_block('<div class="c-sub">No unusual changes detected in the latest data.</div>')

    spacer(1.2)

    # 3 · Forecast (secondary)
    if PRED_READY:

        mae = evaluation_value("MAE", ["random_forest", "model"]) or 0
        mae_base = evaluation_value("MAE", ["baseline"]) or 0

        with st.container(key="glass_statistic"):
            card_head("Forecast vs reality", "Average ROAS per day · held-out test days")
            takeaway(
                f"The AI forecast (orange) follows real ROAS (white); its typical miss is "
                + calc(f"<b>±{mae:.1f}</b>", [f"Average |actual − forecast| over {len(prediction_df)} test SKU-days"], rel(MODEL_EVALUATION))
                + f" vs ±{mae_base:.1f} for a simple guess."
            )
            chart = roas_trend_chart("Random Forest")
            if chart is not None:
                show_chart(chart, 220)

        spacer(1.2)

    # 4 · The closed loop, live
    log = actions.load(ACTIVE)
    approved = sum(e["decision"] == "approved" for e in log)
    measured = sum(e["outcome"] in ("worked", "did not work") for e in log)
    sources = co.manifest(ACTIVE)

    loop = [
        ("Data", "Upload", f"{len(data):,} rows" + (f" · {len(sources)} files" if sources else " · built-in")),
        ("Detect", "Analytics", f"{len(anomalies)} anomalies · {len(shifts)} shifts"),
        ("Diagnose", "Decide" if ML_READY else "Recommendations", f"{len(recommendations)} issues in {affected} products"),
        ("Decide", "Products", f"{int((recommendations['tier'] == 'HIGH').sum()) if not recommendations.empty else 0} high-priority actions"),
        ("Act", "Actions", f"{approved} approved · simulated"),
        ("Learn", "Actions", f"{measured} outcomes measured" + (" + past decisions" if ML_READY else "")),
    ]

    html_block('<div class="level"><span class="t">How ADPULSE works</span><span class="q">The loop, with this company\'s live numbers</span></div>')

    cols = st.columns(len(loop), gap="small")

    for i, (col, (name, target, result)) in enumerate(zip(cols, loop), start=1):
        with col:
            with st.container(key=f"glasstile_loop_{name}"):
                html_block(f'<div class="pri-cat">{i} · {name.upper()}</div>'
                           f'<div style="font-size:.84rem;margin:.35rem 0 .5rem;min-height:2.4rem">{esc(result)}</div>')
                st.button("Open", key=f"loop_{name}", on_click=go_to, args=(target,), width="stretch")


# ============================================================
# PAGE: PREDICT
# ============================================================

FEATURE_NAMES = {
    "roas": "Today's ROAS",
    "ad_spend_inr": "Ad spend",
    "margin_pct": "Profit margin",
    "profit_inr": "Today's profit",
    "sales_growth_pct": "Sales growth",
    "day_of_month": "Day of month",
    "stock_on_hand": "Stock on hand",
    "revenue_inr": "Revenue",
    "click_rate_pct": "Click rate",
    "days_of_inventory_remaining": "Days of stock left",
    "engagement_rate_pct": "Engagement rate",
}


def feature_label(raw):

    name = str(raw).split("__", 1)[-1]

    return FEATURE_NAMES.get(name, name.replace("_inr", "").replace("_pct", "").replace("_", " ").capitalize())


def forecast_view():

    mae = evaluation_value("MAE", ["random_forest", "model"])
    mae_base = evaluation_value("MAE", ["baseline"])
    r2 = evaluation_value("R2", ["random_forest", "model"])
    r2_base = evaluation_value("R2", ["baseline"])
    gain = (mae_base - mae) / mae_base * 100 if mae and mae_base else 0

    spacer(1.1)

    with st.container(key="glass_trend"):

        head_left, head_right = st.columns([3, 1])

        with head_left:
            card_head("Forecast vs reality", "Average ROAS per day across all products")

        with head_right:
            with_baseline = st.toggle("Compare with simple guess", key="predict_baseline")

        takeaway(
            f"The AI forecast (orange) tracks real ROAS (white) closely — its typical miss is "
            f"<b>±{(mae or 0):.1f}</b> vs <b>±{(mae_base or 0):.1f}</b> for a simple guess."
        )

        chart = roas_trend_chart("Both" if with_baseline else "Random Forest")

        if chart is not None:
            show_chart(chart, 300)
        else:
            st.info("ROAS prediction chart data is not available.")

    spacer(1.1)

    with st.container(key="glass_features"):

        feature_col = find_column(feature_df, ["feature"])
        importance_col = find_column(feature_df, ["importance"])

        if feature_df.empty or not (feature_col and importance_col):
            st.info("Feature importance output is not available.")
            return

        ranked = feature_df[[feature_col, importance_col]].dropna().sort_values(importance_col, ascending=False)
        top = ranked.head(5).copy()
        top["Signal"] = top[feature_col].map(feature_label)
        rest = ranked[importance_col].iloc[5:].sum()

        if rest > 0:
            top = pd.concat([top, pd.DataFrame({"Signal": ["Everything else"], importance_col: [rest]})])

        top["label"] = top[importance_col].map(lambda v: f"{v:.0%}")

        card_head("What drives the forecast", "Share of the model's decision made by each signal")
        takeaway(
            f"<b>{esc(top['Signal'].iloc[0])}</b> drives {top[importance_col].iloc[0]:.0%} of the forecast — "
            "products that performed well today usually do well tomorrow."
        )

        base = alt.Chart(top).encode(
            y=alt.Y("Signal:N", sort=None, title=None),
            x=alt.X(f"{importance_col}:Q", axis=None),
        )

        bars = base.mark_bar(cornerRadiusEnd=8, height={"band": 0.6}).encode(
            color=alt.condition(
                alt.datum.Signal == "Everything else", alt.value("rgba(255,255,255,.25)"), alt.value(ORANGE)
            ),
            tooltip=["Signal:N", alt.Tooltip(f"{importance_col}:Q", title="Share", format=".1%")]
        )

        text = base.mark_text(align="left", dx=6, color=TEXT, fontSize=12).encode(text="label:N")

        show_chart(bars + text, 230)

        with st.expander("Technical details (MAE · RMSE · R²)"):

            if not evaluation_df.empty:

                glass_table(
                    list(evaluation_df.columns),
                    [
                        [
                            f'<span class="mono">{v:.3f}</span>' if isinstance(v, float) else esc(v)
                            for v in record
                        ]
                        for record in evaluation_df.itertuples(index=False)
                    ]
                )


# ============================================================
# DECISION TABLE (shared)
# ============================================================

def render_decision_table(df, max_height=440):
    """Product · problem · AI action · confidence · expected return."""

    rows = []

    for _, r in df.iterrows():

        action = r[action_filter_column] if action_filter_column else ""

        rows.append([
            (f'{esc(r[product_column]) if product_column else ""}'
             f'<div class="l-sub">{esc(r[sku_column]) if sku_column else ""}</div>'),
            esc(pretty(r[root_filter_column])) if root_filter_column else "",
            badge(pretty(action), action_color(action), ACTION_ICONS.get(str(action).upper(), "")),
            bar_cell(safe_float(r[confidence_column]), f"{safe_float(r[confidence_column]):.0f}%") if confidence_column else "",
            f'<span class="mono">{safe_float(r[roas_column]):.1f}×</span>' if roas_column else "",
        ])

    glass_table(["Product", "Problem", "AI action", "Confidence", "Expected ROAS"], rows, max_height)


# ============================================================
# PAGE: DIAGNOSE & DECIDE
# ============================================================

CAUSE_TEXT = {
    "INVENTORY_SHORTAGE": ("📦", "Selling faster than stock can keep up"),
    "INVENTORY_RISK": ("⏳", "Stock is about to run out"),
    "FUNNEL_DROP_OFF": ("🛒", "Shoppers click but don't buy"),
    "PAYMENT_FRICTION": ("💳", "Payments fail at checkout"),
}


def page_decide():
    """Why products struggle (diagnosis) and what the AI decided (action) — one page."""

    if root_filter_column is None or action_filter_column is None:
        st.info("Root-cause and decision outputs are not available.")
        return

    causes = final_df[root_filter_column].astype(str)
    counts = causes.value_counts()
    stock_bound = int(causes.str.contains("INVENTORY", case=False).sum())

    takeaway(
        f"<b>{stock_bound} of {total_skus}</b> products are held back by <b>stock</b>, not by ads — "
        "the fix is restocking, not spending more."
    )

    # 1 · Problem -> decision, one card per diagnosed cause
    cards = []

    for cause, n in counts.items():

        icon, meaning = CAUSE_TEXT.get(cause.upper(), ("🔎", pretty(cause)))
        chosen = final_df.loc[causes == cause, action_filter_column]
        action = chosen.mode().iloc[0] if len(chosen) else ""

        cards.append(
            f"""
<div class="card insight fade-in">
  <div class="c-row"><span class="pri-cat">{esc(pretty(cause)).upper()}</span><span style="font-size:1.5rem;font-weight:600">{n}</span></div>
  <h4 style="margin-top:.4rem">{icon} {esc(meaning)}</h4>
  {bar(n / total_skus * 100)}
  <p style="margin-top:.6rem">AI decision → {badge(pretty(action), action_color(action), ACTION_ICONS.get(str(action).upper(), ""))}</p>
</div>
"""
        )

    html_block('<div class="grid-3">' + "".join(cards) + "</div>")

    spacer(1.1)

    # 2 · Every product: problem, decision, confidence — filterable
    with st.container(key="glass_decide"):

        card_head("Every decision", "Filter by problem or action · best expected return first")

        problem = st.pills(
            "Problem", ["All"] + counts.index.tolist(), default="All", key="diag_cause",
            format_func=lambda c: c if c == "All" else f"{pretty(c)} · {counts[c]}",
        ) or "All"

        action_counts = final_df[action_filter_column].astype(str).value_counts()

        f1, f2 = st.columns([2.2, 1], gap="small")

        with f1:
            chosen_action = st.pills(
                "AI action", ["All"] + action_counts.index.tolist(), default="All", key="decide_action",
                format_func=lambda a: a if a == "All" else f"{pretty(a)} · {action_counts[a]}",
            ) or "All"

        with f2:
            search = st.text_input("Search", placeholder="Product or SKU", key="decide_search")

        view = final_df.copy()

        if problem != "All":
            view = view[view[root_filter_column].astype(str) == problem]

        if chosen_action != "All":
            view = view[view[action_filter_column].astype(str) == chosen_action]

        if search:
            mask = pd.Series(False, index=view.index)
            for column in [sku_column, product_column]:
                if column is not None:
                    mask |= view[column].astype(str).str.contains(search, case=False, regex=False)
            view = view[mask]

        if roas_column is not None:
            view = view.sort_values(roas_column, ascending=False)

        html_block(f'<div class="c-sub" style="margin:.4rem 0 .5rem">{len(view)} of {total_skus} products</div>')

        if view.empty:
            st.info("No products match these filters.")
        else:
            render_decision_table(view)


# ============================================================
# PAGE: EXPLAIN
# ============================================================

ACTION_PLAIN = {
    "PROTECT_INVENTORY": "slow down advertising so ads don't sell stock that is about to run out, and restock",
    "DECREASE_BUDGET": "cut ad spend, because each rupee of ads is earning too little",
    "INCREASE_BUDGET": "spend more on ads — they are profitable and there is stock to sell",
    "RETARGET": "show ads to people who already showed interest, instead of new audiences",
    "FIX_PAYMENT": "fix the payment problem first — shoppers want to pay but payments fail",
    "CHANGE_CREATIVE": "replace the ad creative — people see the ad but don't act on it",
    "MAINTAIN": "keep everything as it is — performance is within the normal range",
}

LEARN_PLAIN = {
    "INCREASE_CONFIDENCE": "This action has a strong track record (75%+ success), so the AI now trusts it more.",
    "REVIEW_ACTION_POLICY": "This action has a weak track record (40% success or less), so the AI flags it for review.",
    "MAINTAIN_PROTECTIVE_POLICY": "Results are moderate, but protecting stock is a safety action, so the AI keeps it in place.",
    "MAINTAIN_POLICY": "Results are moderate, so the AI keeps using this action while it keeps watching outcomes.",
}

HEALTH_CHECKS = [
    ("inventory_risk", "Stock", {
        "HEALTHY": ("healthy", TEAL), "LOW": ("low risk", TEAL), "MEDIUM": ("getting low", AMBER),
        "HIGH": ("running low", ROSE), "CRITICAL": ("out / about to run out", ROSE)}),
    ("payment_status", "Payments", {
        "NORMAL": ("normal", TEAL), "MEDIUM": ("some failures", AMBER),
        "HIGH": ("many failures", ROSE), "CRITICAL": ("failing badly", ROSE)}),
    ("funnel_status", "Checkout", {
        "NORMAL": ("normal", TEAL), "HEALTHY_CONVERSION": ("converting well", TEAL),
        "CHECKOUT_CONVERSION": ("shoppers drop off", AMBER)}),
    ("creative_status", "Ad creative", {
        "NORMAL": ("working", TEAL), "CREATIVE_TO_CONVERSION_GAP": ("clicks but few buys", AMBER),
        "WEAK_CREATIVE": ("weak", ROSE)}),
    ("sales_status", "Sales", {"STABLE": ("stable", TEAL), "STRONG_GROWTH": ("growing fast", TEAL)}),
    ("profitability_status", "Profit", {
        "HIGHLY_PROFITABLE": ("highly profitable", TEAL), "PROFITABLE": ("profitable", TEAL),
        "LOSS_RISK": ("at risk of loss", ROSE)}),
]

CAUSE_TEXT.update({
    "HIGH_PERFORMER": ("🚀", "Strong performer with room to grow"),
    "NORMAL_PERFORMANCE": ("✅", "Nothing major is wrong"),
})


def cause_detail(cause, summary, learned):
    """One plain sentence with the numbers behind the diagnosis."""

    if summary is None:
        return ""

    if "INVENTORY" in cause:
        # same sales pace the days-of-stock figure is based on (latest day), so the maths adds up
        pace = summary["stock"] / summary["days_left"] if summary["days_left"] > 0 else summary["velocity"]
        return (
            f"It has <b>{summary['stock']:,.0f} units</b> left and is selling about <b>{pace:.0f} a day</b>, "
            f"so stock lasts roughly <b>{summary['days_left']:.1f} days</b>. It was out of stock on "
            f"{max(summary['oos_days'], summary['stockout_flag_days'])} of the last {int(summary['days'])} days."
        )

    if "PAYMENT" in cause:
        return (
            f"<b>{summary['payment_fail_rate']:.1%}</b> of payment attempts failed "
            f"({int(summary['payment_failed']):,} of {int(summary['payment_initiated']):,}) over {int(summary['days'])} days."
        )

    if cause == "FUNNEL_DROP_OFF":
        return (
            f"{int(summary['add_to_cart']):,} shoppers added it to their cart, but only "
            f"<b>{int(summary['conversions']):,} bought</b> — <b>{summary['cart_abandon']:.0%}</b> of carts were abandoned."
        )

    if cause == "HIGH_PERFORMER":
        return (
            f"Ads return <b>{summary['roas_7d']:.1f}×</b> (last 7 days), it has {summary['days_left']:.0f} days of stock "
            f"and made ₹{summary['profit']:,.0f} profit in {int(summary['days'])} days."
        )

    return "All the health checks below are within their normal ranges."


def page_explain():

    if sku_column is None:
        st.warning("No SKU column found in the report.")
        return

    labels = (
        dict(zip(product_summary["sku"], product_summary["label"])) if not product_summary.empty else {}
    )
    options = final_df[sku_column].astype(str).tolist()

    with st.container(key="glass_picker"):
        selected = st.selectbox(
            "Pick a product to see why the AI decided what it did", options,
            format_func=lambda sku: f"{labels.get(sku, sku)}  ·  {sku}",
        )

    row = final_df[final_df[sku_column].astype(str) == selected].iloc[0]
    learned = learned_df[learned_df["sku"] == selected].iloc[0] if not learned_df.empty and "sku" in learned_df else None
    summary = (
        product_summary[product_summary["sku"] == selected].iloc[0]
        if not product_summary.empty and (product_summary["sku"] == selected).any() else None
    )

    name = labels.get(selected, row[product_column] if product_column else selected)
    action = str(row[action_filter_column]) if action_filter_column else "MAINTAIN"
    cause = str(row[root_filter_column]) if root_filter_column else "NORMAL_PERFORMANCE"
    confidence = safe_float(row[confidence_column]) if confidence_column else 0.0
    roas = safe_float(row[roas_column]) if roas_column else 0.0
    profit = safe_float(row[profit_column]) if profit_column else 0.0
    history = safe_float(row[history_column], 50.0) if history_column else 50.0
    base_conf = safe_float(learned["decision_confidence_pct"], None) if learned is not None else None
    cause_icon, cause_meaning = CAUSE_TEXT.get(cause.upper(), ("🔎", pretty(cause)))
    explanation = str(learned["root_cause_explanation"]) if learned is not None else ""
    adjustment = str(row.get("learning_adjustment", "")) if "learning_adjustment" in row else ""

    # Track record of this action (from the feedback engine)
    track = action_df[action_df[policy_action_column].astype(str) == action] if policy_action_column else pd.DataFrame()
    cases = int(safe_float(track[policy_count_column].iloc[0])) if not track.empty and policy_count_column else 0
    wins = int(safe_float(track[policy_success_count_column].iloc[0])) if not track.empty and policy_success_count_column else 0

    # Typical budget change logged for this action in the dataset
    budget_pct = None
    if not raw_df.empty and "recommended_action" in raw_df:
        logged = raw_df.loc[raw_df["recommended_action"] == action, "budget_change_pct"]
        budget_pct = float(logged.median()) if len(logged) else None
    spend_today = safe_float(learned["ad_spend_inr"], None) if learned is not None else None

    trend_7d = summary["roas_7d"] if summary is not None else None

    spacer(1)

    # ------------------------------------------------------------
    # 1 · The verdict, in plain words
    # ------------------------------------------------------------
    conf_calc = calc(
        f"{confidence:.0f}%",
        [f"0.6 × engine confidence {base_conf:.0f}% + 0.4 × past success {history:.1f}%"
         if base_conf is not None else f"Past success rate {history:.1f}%"],
        rel(LEARNED_DECISIONS), f"= {confidence:.1f}%", align="right", title="How confidence is worked out",
    )

    html_block(
        f"""
<div class="card verdict fade-in">
  <div>
    <div class="who">AI decision for {esc(name)} · {esc(selected)}</div>
    <div class="act">{ACTION_ICONS.get(action.upper(), "🤖")} {esc(pretty(action))}</div>
    <p>
      The AI recommends that you <b>{ACTION_PLAIN.get(action.upper(), esc(pretty(action)).lower())}</b>.
      Tomorrow it expects every ₹1 of ads to bring back <b>₹{roas:.2f}</b> (about ₹{profit:,.0f} profit),
      but the main thing holding this product back is: <b>{cause_meaning.lower()}</b>.
    </p>
  </div>
  <div class="ring">
    <div class="v">{conf_calc}</div>
    <div class="l">confident</div>
    {bar(confidence)}
  </div>
</div>
"""
    )

    spacer(1)

    # ------------------------------------------------------------
    # 2 · How the AI got there — four plain steps
    # ------------------------------------------------------------
    if trend_7d:
        change = (roas - trend_7d) / trend_7d
        direction = (
            f"higher than its recent average of {trend_7d:.1f}×" if change > 0.05
            else f"lower than its recent average of {trend_7d:.1f}×" if change < -0.05
            else f"close to its recent average of {trend_7d:.1f}×"
        )
    else:
        direction = ""

    mae = evaluation_value("MAE", ["random_forest", "model"])
    predict_chip = calc(
        f"Forecast ROAS {roas:.2f}×",
        ["Random Forest model using today's ROAS, ad spend, margin, stock, funnel and creative signals",
         f"Typical forecast error ±{mae:.1f} ROAS on the last 6 test days" if mae else "Accuracy: see the Predict page"],
        f"{rel(FINAL_REPORT)} · predicted_next_day_roas",
    )

    checks = []
    if learned is not None:
        for column, label, meanings in HEALTH_CHECKS:
            if column in learned:
                text, color = meanings.get(str(learned[column]), (pretty(learned[column]).lower(), STONE))
                checks.append(badge(f"{label}: {text}", color))

    problems = [c for c in checks if ROSE in c or AMBER in c]
    others = len(problems) - 1 if problems else 0

    order_note = (
        f" {others} other warning{'s' if others != 1 else ''} were also found; the engine always deals with "
        "stock first, then payments, then checkout, then the ad creative."
        if others > 0 else ""
    )

    budget_line = ""
    if budget_pct is not None:
        if budget_pct == 0:
            budget_line = "In the logged history, this action keeps the ad budget the same."
        else:
            per_day = f" — about ₹{abs(spend_today * budget_pct / 100):,.0f} a day {'less' if budget_pct < 0 else 'more'} for this product (today ₹{spend_today:,.0f})" if spend_today else ""
            budget_line = f"In the logged history, this action changes the daily ad budget by <b>{budget_pct:+.0f}%</b>{per_day}."

    record_line = (
        f"Past <b>{esc(pretty(action))}</b> decisions worked <b>{history:.1f}%</b> of the time"
        + (f" ({wins} of {cases})" if cases else "") + "."
    )
    blend_line = (
        f" Confidence blends the engine's own read of this product ({base_conf:.0f}%) with that track record: "
        f"0.6 × {base_conf:.0f} + 0.4 × {history:.1f} = <b>{confidence:.1f}%</b>."
        if base_conf is not None else ""
    )

    steps = [
        ("Predict", f"Tomorrow: ₹{roas:.2f} back for every ₹1 of ads",
         f"The forecasting model predicts <b>{roas:.2f}× ROAS</b> and about <b>₹{profit:,.0f}</b> profit for tomorrow"
         + (f", {direction}." if direction else "."),
         f'<div class="checks"><span class="ev">{predict_chip}</span></div>'),
        ("Diagnose", f"{cause_icon} {cause_meaning}",
         (f"{esc(explanation)} " if explanation else "") + cause_detail(cause, summary, learned) + order_note,
         f'<div class="checks">{"".join(checks)}</div>' if checks else ""),
        ("Decide", f"{ACTION_ICONS.get(action.upper(), '🤖')} {pretty(action)}",
         f"To address this, the AI chose to {ACTION_PLAIN.get(action.upper(), pretty(action).lower())}. {budget_line}",
         ""),
        ("Learn", f"{confidence:.1f}% confident",
         record_line + blend_line + (f" {LEARN_PLAIN[adjustment]}" if adjustment in LEARN_PLAIN else ""),
         ""),
    ]

    rows = "".join(
        f"""
<div class="why">
  <div class="n">{i}</div>
  <div>
    <div class="k">Step {i} · {key}</div>
    <div class="t">{title}</div>
    <p>{text}</p>
    {extra}
  </div>
</div>
"""
        for i, (key, title, text, extra) in enumerate(steps, start=1)
    )

    with st.container(key="glass_why"):
        card_head("How the AI reached this decision", "Four steps, in the order the engine runs them")
        html_block(rows)

    spacer(1)

    # ------------------------------------------------------------
    # 3 · What to do next
    # ------------------------------------------------------------
    found = sku_recs(selected) if not recommendations.empty else pd.DataFrame()

    with st.container(key="glass_todo"):

        card_head("What to do next", "The most valuable actions for this product, highest impact first")

        if found.empty:
            items = f"<li>{esc(ACTION_PLAIN.get(action.upper(), pretty(action)).capitalize())}.</li>"
        else:
            items = "".join(
                f"<li><b>{esc(rec['title'])}</b> — {esc(rec['action'])}<br><span>{rec_impact(rec)}</span></li>"
                for _, rec in found.sort_values("score", ascending=False).head(3).iterrows()
            )

        html_block(f'<ol class="todo">{items}</ol>')

        st.button(
            "Open full product insights", key="explain_open", icon=":material/open_in_new:",
            on_click=open_product, args=(selected,),
        )


# ============================================================
# PAGE: LEARN
# ============================================================

MIN_CASES = 10   # fewer past decisions than this = too few to judge


def page_learn():

    if action_df.empty or policy_temp.empty:
        st.info("Policy learning output is not available.")
        return

    chart_df = policy_temp.copy()
    chart_df["Action"] = chart_df[policy_action_column].map(pretty)
    chart_df["cases"] = chart_df[policy_count_column].astype(int) if policy_count_column else MIN_CASES
    chart_df["wins"] = chart_df[policy_success_count_column].astype(int) if policy_success_count_column else 0
    chart_df["reliable"] = chart_df["cases"] >= MIN_CASES
    chart_df["label"] = chart_df.apply(
        lambda r: f"{r[policy_success_column]:.0f}%  ·  {r['wins']} of {r['cases']}"
        + ("" if r["reliable"] else "  ·  too few to judge"),
        axis=1
    )

    reliable = chart_df[chart_df["reliable"]].sort_values(policy_success_column, ascending=False)
    best = reliable.iloc[0] if not reliable.empty else chart_df.iloc[0]
    worst = reliable.iloc[-1] if len(reliable) > 1 else None

    kpi_strip([
        kpi("🧾", "Past decisions checked", f"{total_decisions:,}", "did the action pay off?", how=(
            [" + ".join(f"{esc(a)} {n}" for a, n in zip(chart_df["Action"], chart_df["cases"]))],
            rel(ACTION_PERFORMANCE), f"= {total_decisions:,} decisions")),
        kpi("✅", "Worked overall", f"{overall_success:.0f}%", f"{total_successes:,} of {total_decisions:,}", how=(
            [f"{total_successes:,} successful ÷ {total_decisions:,} decisions",
             "Success = next-day ROAS went up after the action"],
            rel(ACTION_PERFORMANCE), f"= {overall_success:.1f}%")),
        kpi("🏆", "Most reliable action", esc(best["Action"]), f"worked {best[policy_success_column]:.0f}% of {best['cases']} times", how=(
            [f"{best['wins']} successful ÷ {best['cases']} decisions = {best[policy_success_column]:.1f}%",
             f"Highest rate among actions with ≥ {MIN_CASES} past decisions"],
            rel(ACTION_PERFORMANCE), None, "right")),
    ])

    spacer(1.1)

    with st.container(key="glass_policychart"):

        card_head("How often each action worked", f"Based on past outcomes · dashed line = average ({overall_success:.0f}%)")

        message = f"<b>{esc(best['Action'])}</b> pays off most reliably ({best[policy_success_column]:.0f}%)"

        if worst is not None:
            message += f"; <b>{esc(worst['Action'])}</b> is weakest ({worst[policy_success_column]:.0f}%), so the AI trusts it less next time."

        takeaway(message)

        order = chart_df.sort_values(["reliable", policy_success_column], ascending=False)["Action"].tolist()

        base = alt.Chart(chart_df).encode(
            y=alt.Y("Action:N", sort=order, title=None),
            x=alt.X(f"{policy_success_column}:Q", axis=None, scale=alt.Scale(domain=[0, 115])),
        )

        bars = base.mark_bar(cornerRadiusEnd=8, height={"band": 0.6}).encode(
            color=alt.condition(alt.datum.reliable, alt.value(ORANGE), alt.value("rgba(255,255,255,.22)")),
            tooltip=["Action:N", alt.Tooltip(f"{policy_success_column}:Q", title="Worked %", format=".1f"),
                     alt.Tooltip("cases:Q", title="Past decisions")]
        )

        text = base.mark_text(align="left", dx=6, color=TEXT, fontSize=11).encode(text="label:N")

        average = alt.Chart(pd.DataFrame({"avg": [overall_success]})).mark_rule(
            strokeDash=[5, 4], color="rgba(255,255,255,.5)"
        ).encode(x="avg:Q")

        show_chart(bars + text + average, 300)

        with st.expander("Full learning table"):

            glass_table(
                ["Action", "Past decisions", "Worked", "Success rate", "Learned strength"],
                [
                    [
                        esc(r["Action"]),
                        f'<span class="mono">{r["cases"]}</span>',
                        f'<span class="mono">{r["wins"]}</span>',
                        bar_cell(r[policy_success_column], f"{r[policy_success_column]:.0f}%"),
                        badge(str(r[policy_strength_column]), QUALITY_COLORS.get(str(r[policy_strength_column]).upper(), STONE))
                        if policy_strength_column else "",
                    ]
                    for _, r in chart_df.sort_values(policy_success_column, ascending=False).iterrows()
                ]
            )


# ============================================================
# PAGE: PIPELINE
# ============================================================

def page_pipeline():

    stages_ml = {st_["script"]: st_["state"] for st_ in co.stage_status(ACTIVE)}
    log = actions.load(ACTIVE)
    sources = co.manifest(ACTIVE)

    def ml(script):
        return {"done": "Done", "stale": "Out of date", "pending": "Not run", "unavailable": "Needs complete data"}[stages_ml.get(script, "pending")]

    flow = [
        ("Multi-source data", ", ".join(sorted({e["source"] for e in sources})) if sources else "Built-in unified dataset",
         "Done", "companies.py · data/"),
        ("Reconciliation", "Column mapping + join on date and SKU / product / campaign" if sources else "Not needed — already unified",
         "Done", "ingest.py"),
        ("Signal generation", "ROAS, contribution, CPM, CVR, velocity, days of stock, funnel rates",
         "Done", "product_insights.py"),
        ("Anomaly detection", f"Robust z-score per product — {len(anomalies)} found now", "Done", "priority_engine.detect_anomalies"),
        ("Root-cause diagnosis", "Stock → payments → checkout → creative", ml("root_cause_engine.py"), "root_cause_engine.py"),
        ("Prediction", "Random Forest, next-day ROAS", ml("train_model.py"), "train_model.py · train_roas.py"),
        ("Opportunity scoring", "Est. ₹ impact × confidence × urgency × ease", "Done", "priority_engine.py"),
        ("Decision engine", f"{len(recommendations)} evidenced recommendations, business constraints applied", "Done", "priority_engine.py · decision_engine.py"),
        ("Approval", f"{sum(e['decision'] == 'approved' for e in log)} approved by a human", "Live", "actions.py"),
        ("Execution", "Simulated — no live ad-platform API is called", "Simulated", "actions.py"),
        ("Outcome measurement", f"{sum(e['outcome'] in ('worked', 'did not work') for e in log)} measured from newer data", "Live", "actions.py"),
        ("Learning", "Measured outcomes adjust confidence" + (" · plus historical feedback engine" if ML_READY else ""),
         "Live", "actions.py · feedback_engine.py"),
    ]

    colors = {"Done": TEAL, "Live": TEAL, "Simulated": AMBER, "Out of date": AMBER, "Not run": STONE, "Needs complete data": STONE}

    takeaway("Every recommendation on screen is the output of this pipeline. Steps marked <b>Simulated</b> "
             "are not connected to live ad platforms in this prototype.")

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        items = "".join(
            f"""
<div class="card tl-item fade-in">
  <div class="c-row">
    <div class="c-title" style="font-size:.95rem">{i}. {name}</div>
    {badge(state, colors.get(state, STONE))}
  </div>
  <div class="c-sub" style="margin:.2rem 0 .15rem">{esc(desc)}</div>
  <div class="tl-file">{esc(code)}</div>
</div>
"""
            for i, (name, desc, state, code) in enumerate(flow, start=1)
        )
        html_block(f'<div class="tl">{items}</div>')

    with right:
        with st.container(key="glass_pipe_ml"):
            pipeline_panel(ACTIVE, "pipeline_page")

        spacer(0.9)

        with st.container(key="glass_pipe_caps"):
            card_head("What this data supports")
            html_block(capability_list(CAPS))


# ============================================================
# PRODUCT INTELLIGENCE DATA
# ============================================================

def engine_version():
    """Changes whenever a rules file is edited, so cached results are rebuilt."""

    return tuple(
        (BASE_DIR / name).stat().st_mtime
        for name in ("product_insights.py", "priority_engine.py", "bundle_engine.py")
    )


@st.cache_data
def load_product_data(path, final_report, mtime, version, platforms, learning, log_version):
    """The single decision engine — used identically for built-in and uploaded data."""

    df = pi.load_dataset(path)
    summary = pi.build_summary(df, final_report)

    recs = pe.detect_recommendations(df, summary, platforms, learning)

    if not recs.empty:
        recs["product"] = recs["sku"].map(summary.set_index("sku")["product"])

    return df, summary, recs, be.smart_bundles(df), pe.detect_anomalies(df)


PLATFORMS = co.load_platforms(ACTIVE)

# Closed loop: measure approved actions against any newer data, then learn from the outcomes
if (ACTIVE["workspace"] / "actions.json").exists():
    actions.measure(ACTIVE, pd.read_csv(DATASET))

LEARNING = actions.learning(ACTIVE)

raw_df, product_summary, recommendations, bundles, anomalies = load_product_data(
    DATASET, final_df, DATASET.stat().st_mtime, engine_version(), PLATFORMS, LEARNING, actions.version(ACTIVE)
)

TONES = {"good": TEAL, "warn": AMBER, "bad": ROSE, "info": SKY}

TIER_COLORS = {"HIGH": ROSE, "MEDIUM": AMBER, "LOW": SKY}

DIALOG_TABS = [
    ("📣", "Marketing"), ("📦", "Inventory"), ("🎨", "Ad creative"), ("🛒", "Abandoned carts"),
    ("🤝", "Combos"), ("🔗", "Affiliate"), ("📈", "Sales trend"),
]

VERDICT_COLORS = {
    "Increase": TEAL, "Decrease": ROSE, "Maintain": SKY,
    "Restock": ROSE, "Clearance": AMBER, "Healthy": TEAL,
    "Popular": ORANGE, "Steady": CREAM, "Slow": STONE,
}

STATUS_COLORS = {
    "Out of stock": ROSE, "Critical": "#fb923c", "Low": AMBER,
    "Healthy": TEAL, "Overstock": SKY,
}


def open_product(sku, tab=None):

    st.session_state.open_family = None
    st.session_state.open_sku = sku
    st.session_state.open_tab = tab


def open_family(product):

    st.session_state.open_sku = None
    st.session_state.open_family = product


def close_product():

    st.session_state.open_sku = None
    st.session_state.open_tab = None
    st.session_state.open_family = None


def sku_recs(sku):

    return pe.product_recommendations(recommendations, sku)


def tier_badge(tier):

    return badge(tier, TIER_COLORS.get(tier, STONE))


def rel(path):
    """Path shown in 'Source:' lines, relative to the project."""

    try:
        return esc(Path(path).relative_to(BASE_DIR).as_posix())
    except ValueError:
        return esc(str(path))


def fmt(value, spec, prefix="", suffix=""):
    """Format a number, or show — when the data doesn't contain it."""

    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return "—"

    return f"{prefix}{value:{spec}}{suffix}"


MEASURED_IMPACTS = ("lost in the last", "of stock tied up")


def impact_text(rec):
    """Every forward-looking ₹ figure is labelled as an estimate."""

    label = str(rec["impact_label"])

    if label.startswith("Est.") or any(m in label for m in MEASURED_IMPACTS):
        return label

    return "Est. " + label.lstrip("~")


def approve(rec):

    actions.record(ACTIVE, rec, raw_df, "approved")
    st.session_state["toast"] = f"Approved: {rec['title']} · {rec['label']} — logged as a simulated execution"


def decision_button(rec, key):
    """Human approval: logs the decision; execution is simulated (no ad platform is changed)."""

    decided = actions.status_for(ACTIVE).get((rec["sku"], rec["kind"]))

    if decided == "approved":
        st.button("Approved · simulated", key=f"ap_{key}", icon=":material/check_circle:", disabled=True)
    else:
        st.button(
            "Approve", key=f"ap_{key}", icon=":material/check:", on_click=approve, args=(dict(rec),),
            help="Logs this decision and simulates execution. No live ad platform is changed.",
        )


def capability_list(caps):

    if not caps:
        return '<span class="healthy">✓ Complete dataset — every analysis available</span>'

    return "".join(
        f'<div class="lrow" style="padding:.35rem 0"><div class="l-main"><span class="l-name">'
        f'{"✓" if ok else "✗"} {esc(name)}</span>'
        + ("" if ok else f'<div class="l-sub">{esc(why)}</div>')
        + "</div></div>"
        for name, (ok, why) in caps.items()
    )


def render_needs(page):
    """Honest empty state when this company's data can't support a page."""

    needs_ml = True

    takeaway(
        f"<b>{esc(PAGE_TITLES[page][0])}</b> uses the full ML decision pipeline "
        + ("(root-cause engine + feedback learning on past decisions), which needs a complete dataset including a history of decisions and outcomes."
           if needs_ml else "forecast model, which needs ad spend and revenue for at least 10 days.")
        + f" <b>{esc(ACTIVE['name'])}</b> doesn't have that yet — nothing is shown rather than a guess."
    )

    left, right = st.columns([1, 1], gap="medium")

    with left:
        with st.container(key="glass_needs_caps"):
            card_head("What can be analysed", "From the uploaded files")
            html_block(capability_list(CAPS))

    with right:
        with st.container(key="glass_needs_go"):
            card_head("Available for this data", "The decision engine still runs on everything uploaded")
            for target in ["Products", "Recommendations", "Actions", "Analytics"]:
                st.button(target, key=f"needs_{page}_{target}", icon=PAGE_ICONS[target],
                          on_click=go_to, args=(target,), width="stretch")
            if co.runnable_stages(ACTIVE):
                spacer(0.4)
                pipeline_panel(ACTIVE, "needs")


def rec_impact(rec, align="left"):

    return calc(
        esc(impact_text(rec)), [esc(line) for line in rec.get("impact_calc", [])],
        source=f"{rel(DATASET)} (last 7–14 days) · rules in priority_engine.py",
        title="How the ₹ impact is worked out", align=align,
    )


def rec_confidence(rec, text, align="right"):

    return calc(
        text, [esc(line) for line in rec.get("conf_calc", [])],
        result=f"= {rec['confidence']}% confidence",
        title="How confidence is scored", align=align,
    )


def rec_tier(rec, align="right"):

    return calc(
        tier_badge(rec["tier"]), [esc(line) for line in rec.get("score_calc", [])],
        title="How priority is ranked", align=align,
    )


def signal_card(rec, show_product=False):
    """One recommendation: category, title, evidence, cause, action, impact, confidence."""

    accent = TIER_COLORS.get(rec["tier"], STONE)
    evidence = "".join(f'<span class="ev">{esc(e)}</span>' for e in rec["evidence"])
    product = f'<div class="pri-product">{esc(rec["label"])}</div>' if show_product else ""

    return f"""
<div class="card sig fade-in" style="--accent:{accent}">
  <div class="c-row"><span class="pri-cat">{esc(rec["category"])}</span>{rec_tier(rec)}</div>
  <div class="pri-title">{esc(rec["title"])}</div>
  {product}
  <div class="ev-row">{evidence}</div>
  <div class="sig-line"><b>Cause</b>{esc(rec["cause"])}</div>
  <div class="sig-line"><b>Action</b>{esc(rec["action"])}</div>
  <div class="pri-foot">
    <span class="impact">{rec_impact(rec)}</span>
    <span class="conf">{rec_confidence(rec, f"{rec['confidence']}%")}<span class="bar"><span style="width:{rec["confidence"]}%"></span></span></span>
  </div>
</div>
"""


def rec_card(section, advice):

    accent = TONES.get(advice["tone"], SKY)
    reasons = "".join(f"<li>{esc(r)}</li>" for r in advice["reasons"])
    actions = "".join(f"<li>{esc(a)}</li>" for a in advice["actions"])

    html_block(
        f"""
<div class="card rec fade-in" style="--accent:{accent}">
  <div class="c-row"><span class="rec-label">{section}</span>{badge(advice["verdict"], accent)}</div>
  <div class="rec-title">{esc(advice["headline"])}</div>
  <h5>Why</h5><ul>{reasons}</ul>
  <h5>Suggested actions</h5><ol>{actions}</ol>
</div>
"""
    )


def mini_kpis(items):

    html_block(
        '<div class="mini-kpis">'
        + "".join(
            f'<div class="card mini"><div class="l">{item[0]}</div>'
            f'<div class="v">{calc(item[1], *item[2]) if len(item) > 2 and item[2] else item[1]}</div></div>'
            for item in items
        )
        + "</div>"
    )


def clickable_chart(chart, key, height=320):
    """Altair chart whose marks (carrying a `sku` field) open the product dialog on click."""

    pick = alt.selection_point(fields=["sku"], name="pick", on="click")

    event = st.altair_chart(
        glass_chart(chart.add_params(pick), height),
        width="stretch",
        theme=None,
        on_select="rerun",
        selection_mode="pick",
        key=key
    )

    picked = (event.selection.get("pick") if event and event.selection else None) or []
    sku = picked[0].get("sku") if picked else None

    if sku and st.session_state.get(f"{key}_last") != sku:
        open_product(sku)

    st.session_state[f"{key}_last"] = sku


# ------------------------------------------------------------
# Per-product charts (inside the dialog)
# ------------------------------------------------------------

def chart_revenue_vs_spend(data, height=260):

    daily = data.groupby("date")[["revenue_inr", "ad_spend_inr"]].sum().reset_index()
    daily["ROAS"] = daily["revenue_inr"] / daily["ad_spend_inr"].where(daily["ad_spend_inr"] > 0)

    tooltip = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("revenue_inr:Q", title="Revenue ₹", format=",.0f"),
        alt.Tooltip("ad_spend_inr:Q", title="Ad spend ₹", format=",.0f"),
        alt.Tooltip("ROAS:Q", format=".2f")
    ]

    base = alt.Chart(daily).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")))

    # Separate axes: revenue is many times larger than spend, so a shared
    # axis would flatten the spend line.
    revenue = base.mark_line(interpolate="monotone", strokeWidth=3, color=CREAM).encode(
        y=alt.Y("revenue_inr:Q", title=None, axis=alt.Axis(format="~s", labelColor=CREAM)),
        tooltip=tooltip
    )

    spend = base.mark_line(interpolate="monotone", strokeWidth=3, color=ORANGE).encode(
        y=alt.Y("ad_spend_inr:Q", title=None,
                axis=alt.Axis(format="~s", labelColor=ORANGE, orient="right", grid=False)),
        tooltip=tooltip
    )

    html_block(
        f'<div class="c-sub" style="display:flex;gap:1.2rem;margin:0">'
        f'<span><span style="color:{CREAM}">●</span> Revenue (left scale)</span>'
        f'<span><span style="color:{ORANGE}">●</span> Ad spend (right scale)</span></div>'
    )

    show_chart(alt.layer(revenue, spend).resolve_scale(y="independent"), height)


def chart_inventory(data):

    base = alt.Chart(data).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")))

    stock = base.mark_area(
        interpolate="step-after", opacity=0.55,
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="rgba(255,122,61,0.05)", offset=0),
                   alt.GradientStop(color="rgba(255,122,61,0.7)", offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        y=alt.Y("stock_on_hand:Q", title="Stock on hand (units)"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("stock_on_hand:Q", title="Stock"),
                 alt.Tooltip("days_of_inventory_remaining:Q", title="Days left", format=".1f")]
    )

    days = base.mark_line(color=CREAM, strokeWidth=2.5, interpolate="monotone").encode(
        y=alt.Y("days_of_inventory_remaining:Q", title="Days of inventory left")
    )

    oos = (
        alt.Chart(data[data["stock_on_hand"] <= 0])
        .mark_tick(color=ROSE, thickness=3, size=14)
        .encode(x="date:T", y=alt.value(0), tooltip=[alt.Tooltip("date:T", title="Out of stock on")])
    )

    show_chart(alt.layer(stock, days, oos).resolve_scale(y="independent"), 260)


def chart_creative(table, data):

    bars = (
        alt.Chart(table)
        .mark_bar(cornerRadiusEnd=8, height={"band": 0.6})
        .encode(
            x=alt.X("roas:Q", title="ROAS by ad format"),
            y=alt.Y("ad_type:N", sort="-x", title=None),
            color=alt.Color("format:N", title=None, scale=alt.Scale(domain=["Video", "Poster"], range=[ORANGE, CREAM])),
            tooltip=[
                alt.Tooltip("ad_type:N", title="Ad type"), "format:N",
                alt.Tooltip("roas:Q", title="ROAS", format=".2f"),
                alt.Tooltip("ctr:Q", title="CTR %", format=".2f"),
                alt.Tooltip("purchase_rate:Q", title="Purchase %", format=".2f"),
                alt.Tooltip("runs:Q", title="Days run")
            ]
        )
    )

    show_chart(bars, 190)

    video = data[data["ad_type"].isin(pi.VIDEO_TYPES)]

    if not video.empty and video["video_3sec_views"].sum() > 0:

        stages = [
            ("3 sec", "video_3sec_views"), ("25%", "video_25pct_watched"), ("50%", "video_50pct_watched"),
            ("75%", "video_75pct_watched"), ("100%", "video_100pct_watched"),
        ]

        start = video["video_3sec_views"].sum()

        retention = pd.DataFrame(
            [{"Stage": s, "order": i, "Retained": video[c].sum() / start} for i, (s, c) in enumerate(stages)]
        )

        html_block('<div class="c-sub" style="margin-top:.6rem">Video retention — share of viewers still watching</div>')

        curve = (
            alt.Chart(retention)
            .mark_area(
                interpolate="monotone", line={"color": ORANGE, "strokeWidth": 2.5},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color="rgba(255,122,61,0.05)", offset=0),
                           alt.GradientStop(color="rgba(255,122,61,0.45)", offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            )
            .encode(
                x=alt.X("Stage:N", sort=[s for s, _ in stages], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Retained:Q", title=None, axis=alt.Axis(format="%")),
                tooltip=["Stage:N", alt.Tooltip("Retained:Q", format=".0%")]
            )
        )

        show_chart(curve, 150)


def chart_funnel(row):

    steps = [
        ("Product clicks", row["product_clicks"]), ("Add to cart", row["add_to_cart"]),
        ("Checkout started", row["checkout_started"]), ("Payment initiated", row["payment_initiated"]),
        ("Purchased", row["conversions"]),
    ]

    funnel = pd.DataFrame(steps, columns=["Stage", "Users"]).dropna()   # only the steps present in the data
    funnel["order"] = range(len(funnel))
    funnel["of_prev"] = funnel["Users"] / funnel["Users"].shift(1)
    funnel["label"] = [
        f"{int(u):,}" + ("" if pd.isna(p) else f"  ·  {p:.0%} kept")
        for u, p in zip(funnel["Users"], funnel["of_prev"])
    ]

    base = alt.Chart(funnel).encode(
        y=alt.Y("Stage:N", sort=funnel["Stage"].tolist(), title=None),
        x=alt.X("Users:Q", title=None, axis=None),
    )

    bars = base.mark_bar(cornerRadiusEnd=8, height={"band": 0.62}).encode(
        color=alt.Color("order:O", legend=None, scale=alt.Scale(range=["#7a4a33", ORANGE])),
        tooltip=["Stage:N", alt.Tooltip("Users:Q", format=","), alt.Tooltip("of_prev:Q", title="Kept from previous", format=".0%")]
    )

    text = base.mark_text(align="left", dx=6, color=TEXT, fontSize=11).encode(text="label:N")

    show_chart(bars + text, 230)


def chart_affiliate_daily(data):

    days = data[data["affiliate_link_interactions"] > 0]

    if days.empty:
        html_block('<div class="c-sub">No affiliate activity recorded for this product.</div>')
        return

    chart = (
        alt.Chart(days)
        .mark_circle(size=110, opacity=0.85, color=ORANGE, stroke="rgba(255,255,255,.6)", strokeWidth=1)
        .encode(
            x=alt.X("affiliate_link_interactions:Q", title="Affiliate link interactions (per day)"),
            y=alt.Y("revenue_inr:Q", title="Revenue (₹)", axis=alt.Axis(format="~s")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("affiliate_link_interactions:Q", title="Interactions"),
                alt.Tooltip("revenue_inr:Q", title="Revenue ₹", format=",.0f"),
                alt.Tooltip("conversions:Q", title="Conversions"),
                alt.Tooltip("ad_type:N", title="Ad type")
            ]
        )
    )

    trend = chart.transform_regression("affiliate_link_interactions", "revenue_inr").mark_line(
        color=CREAM, strokeDash=[5, 4], strokeWidth=2
    )

    show_chart(chart + trend, 260)


def chart_sales_trend(data):

    daily = data[["date", "sales_volume", "revenue_inr"]].copy()
    daily["7-day average"] = daily["sales_volume"].rolling(7, min_periods=1).mean()

    long = daily.rename(columns={"sales_volume": "Units sold"}).melt(
        ["date", "revenue_inr"], ["Units sold", "7-day average"], var_name="Series", value_name="Units"
    )

    chart = (
        alt.Chart(long)
        .mark_line(interpolate="monotone", strokeWidth=3)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("Units:Q", title="Units per day"),
            color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=["Units sold", "7-day average"], range=[ORANGE, CREAM])),
            strokeDash=alt.condition(alt.datum.Series == "7-day average", alt.value([5, 4]), alt.value([0])),
            tooltip=[alt.Tooltip("date:T", title="Date"), "Series:N", alt.Tooltip("Units:Q", format=".1f"),
                     alt.Tooltip("revenue_inr:Q", title="Revenue ₹", format=",.0f")]
        )
    )

    show_chart(chart, 260)


# ------------------------------------------------------------
# Product dialog
# ------------------------------------------------------------

NO_DATA_VERDICTS = {"No ad data", "Unknown", "No creative data", "No funnel data", "No affiliate data", "No sales data"}


def product_dialog(sku):

    report = pi.product_report(sku, raw_df, product_summary)
    row = report["row"]
    data = raw_df[raw_df["sku"] == sku].sort_values("date")
    status = report["inventory"]["status"]

    ml_action = row.get("current_action") if "current_action" in row else None
    ml_badge = (
        badge(f"ML: {pretty(ml_action)}", action_color(ml_action), ACTION_ICONS.get(str(ml_action).upper(), ""))
        if isinstance(ml_action, str) else ""
    )

    html_block(
        f"""
<div class="c-row" style="align-items:flex-start;margin-bottom:.9rem">
  <div>
    <div class="level" style="margin:0 0 .3rem"><span class="n">LEVEL 3</span><span class="q">Product Insights · why it's happening and what to do</span></div>
    <div class="tb-title" style="font-size:1.45rem">{esc(row["label"])}</div>
    <div class="tb-sub">{esc(row["category"])} · {esc(sku)} · {int(row["days"])} days of data</div>
  </div>
  <div class="pc-badges" style="justify-content:flex-end;margin:0">
    {badge(row["popularity"], VERDICT_COLORS.get(row["popularity"], STONE))}
    {badge(status, STATUS_COLORS.get(status, STONE))}
    {ml_badge}
  </div>
</div>
"""
    )

    src = rel(DATASET)

    mini_kpis([
        ("7-day ROAS", fmt(row['roas_7d'], ".2f", suffix="×"), (
            ["Average of daily ROAS (revenue ÷ ad spend) over the last 7 days"], f"{src} · roas")),
        ("Revenue (28d)", fmt(row['revenue'], ",.0f", "₹"), (
            [f"Σ daily revenue over {int(row['days'])} days", f"{int(row['units']):,} units sold"], f"{src} · revenue_inr")),
        ("Profit (28d)", fmt(row['profit'], ",.0f", "₹"), (
            [f"Σ daily profit over {int(row['days'])} days",
             f"≈ revenue ₹{row['revenue']:,.0f} × margin − ad spend ₹{row['ad_spend']:,.0f}"], f"{src} · profit_inr")),
        ("Units / day", fmt(row['velocity'], ".1f"), (
            [f"Average daily sales velocity over {int(row['days'])} days"], f"{src} · inventory_velocity_units_per_day")),
        ("Stock on hand", fmt(row['stock'], ",.0f"), (
            ["Closing stock on the latest day in the dataset"], f"{src} · stock_on_hand")),
        ("Days of stock", fmt(row['days_left'], ".1f"), (
            [f"Stock {row['stock']:,.0f} ÷ velocity on the latest day"], f"{src} · days_of_inventory_remaining", None, "right")),
    ])

    # What the signal engine detected for THIS product
    found = sku_recs(sku)

    if found.empty:
        html_block(
            '<div class="card" style="padding:.8rem 1rem;margin-bottom:1rem">'
            '<span class="healthy">✓ No significant issues detected</span>'
            '<span class="muted" style="font-size:.8rem"> — detail tabs below show routine guidance.</span></div>'
        )
    else:
        html_block(
            f'<div class="level"><span class="t" style="font-size:1rem">Detected for this product</span>'
            f'<span class="q">{len(found)} signal{"s" if len(found) > 1 else ""} · ● marks the tab with the evidence</span></div>'
            '<div class="grid-3" style="margin-bottom:1rem">'
            + "".join(signal_card(rec) for _, rec in found.iterrows())
            + "</div>"
        )

    flagged = set(found["tab"]) if not found.empty else set()
    labels = [f"{icon} {name}" + (" ●" if name in flagged else "") for icon, name in DIALOG_TABS]
    wanted = st.session_state.get("open_tab")
    default = next((l for l, (_, n) in zip(labels, DIALOG_TABS) if n == wanted), None)

    tabs = st.tabs(labels, default=default)

    def section(tab, label, advice, chart_title, chart_sub, draw):

        with tab:

            left, right = st.columns([1, 1.25], gap="medium")

            with left:
                rec_card(label, advice)

            with right:
                with st.container(key=f"glass_dlg_{label}"):
                    card_head(chart_title, chart_sub)
                    if advice["verdict"] in NO_DATA_VERDICTS:
                        html_block(f'<div class="c-sub">{esc(advice["headline"])} — chart unavailable.</div>')
                    else:
                        draw()

    section(
        tabs[0], "Marketing", report["marketing"],
        "Revenue vs ad spend", "Daily ₹",
        lambda: chart_revenue_vs_spend(data)
    )

    inventory = report["inventory"]

    def draw_inventory():
        mini_kpis([
            ("Status", status),
            ("Restock qty", f"{inventory['restock_units']:,}", (
                [f"Velocity {row['velocity']:.1f}/day × ({pi.TARGET_COVER_DAYS} days cover + {pi.SUPPLIER_LEAD_DAYS} days lead time)",
                 f"− stock {row['stock']:,.0f}, rounded up to the next 10"], "product_insights.py")),
            ("Out-of-stock days", f"{inventory['oos_days']}", (
                [f"Days with stock = 0 (or a stock-out flag) in the last {int(row['days'])} days"],
                f"{rel(DATASET)} · stock_on_hand", None, "right")),
        ])
        chart_inventory(data)

    section(
        tabs[1], "Inventory", inventory,
        "Stock & days remaining", "Orange area = stock on hand · line = days left · red ticks = out of stock",
        draw_inventory
    )

    creative = report["creative"]

    section(
        tabs[2], "Ad creative", creative,
        "Video vs poster performance", "Video = video & influencer video · Poster = image & carousel",
        lambda: chart_creative(creative["table"], data)
    )

    section(
        tabs[3], "Abandoned purchases", report["checkout"],
        "Purchase funnel", "Click → purchase",
        lambda: chart_funnel(row)
    )

    combo = report["combo"]

    def draw_combo():

        for i, bundle in enumerate(b for b in bundles if row["product"] in (b["anchor"], b["laggard"])):
            html_block('<div class="fam-sec" style="margin-top:0">Smart bundle (portfolio-level)</div>')
            bundle_card(bundle, f"dlg_{i}", show_buttons=False)
            spacer(0.6)

        if not combo["combos"]:
            html_block('<div class="c-sub">No suitable partner found.</div>')
            return

        cards = "".join(
            f"""
<div class="card combo">
  <div class="l-icon">🎁</div>
  <div class="l-main">
    <div class="l-name" style="white-space:normal">{esc(c["name"])}</div>
    <div class="l-sub">Partner is {esc(c["partner_popularity"].lower())} ({c["partner_velocity"]:.1f} units/day) · {esc(c["discount"])} · {esc(c["note"])}</div>
  </div>
</div>
"""
            for c in combo["combos"]
        )

        html_block(cards)

        compare = pd.DataFrame(
            [{"SKU": row["label"], "sku": sku, "Units/day": row["velocity"], "Role": "This product"}]
            + [
                {"SKU": c["name"].split(" + ")[1], "sku": c["partner_sku"], "Units/day": c["partner_velocity"], "Role": "Partner"}
                for c in combo["combos"]
            ]
        )

        chart = (
            alt.Chart(compare)
            .mark_bar(cornerRadiusEnd=8, height={"band": 0.55})
            .encode(
                x=alt.X("Units/day:Q", title="Units sold per day"),
                y=alt.Y("SKU:N", title=None, sort="-x"),
                color=alt.Color("Role:N", title=None, scale=alt.Scale(domain=["This product", "Partner"], range=[CREAM, ORANGE])),
                tooltip=["SKU:N", alt.Tooltip("Units/day:Q", format=".1f"), "Role:N"]
            )
        )

        show_chart(chart, 150)

    section(
        tabs[4], "Combos", combo,
        "Popular + slow bundle ideas", "",
        draw_combo
    )

    section(
        tabs[5], "Affiliate", report["affiliate"],
        "Affiliate interactions vs revenue", "Each dot is a day with affiliate activity · dashed line = trend",
        lambda: chart_affiliate_daily(data)
    )

    section(
        tabs[6], "Sales trend", report["sales"],
        "Product sales trend", "Daily units · dashed = 7-day average",
        lambda: chart_sales_trend(data)
    )

    html_block('<div class="c-sub" style="margin-top:1rem">Rule-based on data/dataset.csv · ML engine signal shown in the header.</div>')


# ============================================================
# PAGE: PRODUCTS
# ============================================================

def _signal_skus(tier=None):

    if recommendations.empty:
        return set()

    recs = recommendations if tier is None else recommendations[recommendations["tier"] == tier]

    return set(recs["sku"])


FOCUS_FILTERS = {
    "All": lambda s: s.index == s.index,
    "High priority": lambda s: s["sku"].isin(_signal_skus("HIGH")),
    "No issues": lambda s: ~s["sku"].isin(_signal_skus()),
    "Increase budget": lambda s: s["marketing_verdict"] == "Increase",
    "Decrease budget": lambda s: s["marketing_verdict"] == "Decrease",
    "Restock first": lambda s: s["restock_first"],
    "Frequent out-of-stock": lambda s: s["frequent_oos"],
    "Clearance": lambda s: s["clearance"],
    "Slow movers": lambda s: s["popularity"] == "Slow",
}


def page_products():

    if product_summary.empty:
        st.warning("data/dataset.csv was not found — product intelligence needs the raw dataset.")
        return

    s = product_summary

    # ---------------- LEVEL 1 · AI PRIORITIES ----------------

    priorities = pe.select_priorities(recommendations)

    at_stake = recommendations["impact"].fillna(0).sum() if not recommendations.empty else 0
    affected = recommendations["sku"].nunique() if not recommendations.empty else 0

    html_block(
        '<div class="level"><span class="n">LEVEL 1</span><span class="t">AI Priorities</span>'
        '<span class="q">What needs attention right now?</span></div>'
        f'<div class="c-sub" style="margin:-.4rem 0 .9rem">{len(recommendations)} signals across {affected} of {len(s)} products'
        f' · ~₹{at_stake:,.0f} estimated impact · ranked by ₹ impact × confidence × urgency</div>'
    )

    if not priorities:
        st.success("No significant issues detected across the portfolio.")

    for start in range(0, len(priorities), 3):

        cols = st.columns(3, gap="medium")

        for offset, (col, rec) in enumerate(zip(cols, priorities[start:start + 3])):

            more = (
                f'<div class="pri-more">+{rec["more_count"]} more product{"s" if rec["more_count"] > 1 else ""} '
                f'with this issue · ~₹{rec["more_impact"]:,.0f}</div>'
                if rec["more_count"] else ""
            )
            evidence = "".join(f'<span class="ev">{esc(e)}</span>' for e in rec["evidence"][:3])

            with col:

                with st.container(key=f"glasstile_pr_{rec['tier']}_{start + offset}"):

                    html_block(
                        f"""
<div class="c-row"><span class="pri-cat">{esc(rec["category"])}</span>{rec_tier(rec)}</div>
<div class="pri-title">{esc(rec["title"])}</div>
<div class="pri-product">{esc(rec["label"])}</div>
<div class="c-sub" style="margin:0">{esc(rec["headline"])}</div>
<div class="ev-row">{evidence}</div>
<div class="pri-action"><b>Action</b>{esc(rec["action"])}</div>
<div class="pri-foot">
  <span class="impact">{rec_impact(rec)}</span>
  <span class="conf">{rec_confidence(rec, f"{rec['confidence']}%")}<span class="bar"><span style="width:{rec["confidence"]}%"></span></span></span>
</div>
{more}
"""
                    )

                    with st.container(horizontal=True, gap="small", key=f"pri_btns_{start + offset}"):
                        st.button(
                            "Details",
                            key=f"pri_open_{start + offset}",
                            icon=":material/arrow_forward:",
                            icon_position="right",
                            on_click=open_product,
                            args=(rec["sku"], rec["tab"]),
                        )
                        decision_button(rec, f"pri_{start + offset}")

    spacer(1.6)

    # ---------------- LEVEL 2 · PRODUCT INTELLIGENCE ----------------

    html_block(
        '<div class="level"><span class="n">LEVEL 2</span><span class="t">Product Intelligence</span>'
        '<span class="q">Which products need attention?</span></div>'
    )

    with st.container(key="glass_pfilters"):

        counts = {name: int(FOCUS_FILTERS[name](s).sum()) for name in FOCUS_FILTERS}

        focus = st.segmented_control(
            "Focus", list(FOCUS_FILTERS), default="All", required=True, key="product_focus",
            format_func=lambda name: name if name == "All" else f"{name} ({counts[name]})"
        )

        f2, f3 = st.columns(2, gap="small")

        with f2:
            category = st.selectbox("Category", ["All"] + sorted(s["category"].dropna().unique().tolist()))

        with f3:
            search = st.text_input("Search", placeholder="Product or SKU", key="product_search")

    view = s[FOCUS_FILTERS[focus](s)]

    if category != "All":
        view = view[view["category"] == category]

    if search:
        view = view[
            view["label"].str.contains(search, case=False, regex=False)
            | view["sku"].str.contains(search, case=False, regex=False)
        ]

    # Most urgent products first
    priority_score = (
        recommendations.groupby("sku")["score"].sum() if not recommendations.empty else pd.Series(dtype=float)
    )
    view = view.assign(_priority=view["sku"].map(priority_score).fillna(0)).sort_values(
        ["_priority", "revenue"], ascending=False
    )

    html_block(
        f'<div class="c-sub" style="margin:.9rem 0 .5rem">Showing <b>{len(view)}</b> of {len(s)} products · '
        'most urgent first</div>'
    )

    if view.empty:
        st.info("No products match this focus.")
        return

    records = list(view.iterrows())

    for start in range(0, len(records), 3):

        cols = st.columns(3, gap="medium")

        for col, (_, r) in zip(cols, records[start:start + 3]):

            status = pi.inventory_status(r)
            marketing = r["marketing_verdict"]
            found = sku_recs(r["sku"])

            tags = []

            if status in ("Out of stock", "Critical", "Low"):
                tags.append(f'<span class="sig-chip" style="--accent:{ROSE}">{esc(status.upper())}</span>')

            if not found.empty:
                top = found.sort_values("score", ascending=False).iloc[0]
                if not (top["category"] == "INVENTORY" and tags):
                    tags.append(f'<span class="sig-chip" style="--accent:{TIER_COLORS.get(top["tier"], STONE)}" '
                                f'title="{esc(top["title"])}">{esc(top["category"])}</span>')
                if len(found) > 1:
                    tags.append(f'<span class="muted" style="font-size:.72rem;align-self:center">+{len(found) - 1} more</span>')

            chips = "".join(tags[:3]) if tags else '<span class="healthy">✓ Healthy — no action needed</span>'

            with col:

                with st.container(key=f"glasstile_p_{r['sku']}"):

                    html_block(
                        f"""
<div class="pc-top">
  <div><div class="pc-name">{esc(r["label"])}</div><div class="pc-meta">{esc(r["category"])}</div></div>
</div>
<div class="pc-stats">
  <div class="pc-stat"><div class="v">{calc(fmt(r['roas_7d'], '.1f', suffix='×'), ["Average of the daily ROAS (revenue ÷ ad spend) over the last 7 days"], f"{rel(DATASET)} · roas")}</div><div class="l">ROAS 7d</div></div>
  <div class="pc-stat"><div class="v">{calc(fmt(r['velocity'], '.0f'), [f"Average daily sales velocity over {int(r['days'])} days = {r['velocity']:.1f} units/day"], f"{rel(DATASET)} · inventory_velocity_units_per_day")}</div><div class="l">Units/day</div></div>
  <div class="pc-stat"><div class="v">{calc(fmt(r['days_left'], '.0f'), [f"Stock on hand {r['stock']:,.0f} ÷ {r['velocity']:.1f} units/day (latest day)", f"= {r['days_left']:.1f} days of cover"], f"{rel(DATASET)} · days_of_inventory_remaining", align="right")}</div><div class="l">Days of stock</div></div>
</div>
<div class="sig-chips">{chips}</div>
"""
                    )

                    st.button(
                        "View insights",
                        key=f"open_{r['sku']}",
                        icon=":material/open_in_new:",
                        on_click=open_product,
                        args=(r["sku"],),
                        width="stretch"
                    )


# ============================================================
# PAGE: ANALYTICS
# ============================================================

def analytics_channel():
    """Revenue by ad platform when the data has platforms, else by ad format."""

    if not PLATFORMS.empty:
        card_head("Revenue by Channel", "Ad platform · revenue attributed in each platform's export")
        channel = (
            PLATFORMS.groupby("platform")
            .agg(revenue=("ad_revenue", "sum"), spend=("ad_spend_inr", "sum"), days=("date", "nunique"))
            .reset_index().rename(columns={"platform": "ad_type"})
        )
    elif raw_df["ad_type"].notna().any() and raw_df["ad_spend_inr"].notna().any():
        card_head("Revenue by Channel", "Channel = ad format (the data has no platform column)")
        channel = (
            raw_df.groupby("ad_type")
            .agg(revenue=("revenue_inr", "sum"), spend=("ad_spend_inr", "sum"), days=("date", "count"))
            .reset_index()
        )
    else:
        card_head("Revenue by Channel")
        html_block('<div class="c-sub">No ad platform or ad-format data uploaded — this chart is unavailable.</div>')
        return

    channel["roas"] = channel["revenue"] / channel["spend"]
    channel["label"] = channel["roas"].map(lambda v: f"{v:.1f}× ROAS")
    best = channel.sort_values("revenue", ascending=False).iloc[0]

    takeaway(
        f"<b>{esc(best['ad_type'])}</b> earns the most — "
        f"₹{best['revenue'] / 1e5:,.1f}L at {best['roas']:.1f}× return."
    )

    bars = (
        alt.Chart(channel)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, width={"band": 0.55})
        .encode(
            x=alt.X("ad_type:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("revenue:Q", title=None, axis=alt.Axis(format="~s")),
            color=alt.condition(
                alt.datum.ad_type == best["ad_type"], alt.value(ORANGE), alt.value("rgba(255,255,255,.28)")
            ),
            tooltip=[
                alt.Tooltip("ad_type:N", title="Channel"),
                alt.Tooltip("revenue:Q", title="Revenue ₹", format=",.0f"),
                alt.Tooltip("spend:Q", title="Spend ₹", format=",.0f"),
                alt.Tooltip("roas:Q", title="ROAS", format=".2f"),
            ]
        )
    )

    labels = bars.mark_text(dy=-8, color=TEXT, fontSize=11).encode(text="label:N")

    show_chart(bars + labels, 280)


def analytics_affiliate(s):

    card_head("⭐ Affiliate Performance", "Each bubble is a product · click one to open it")

    aff = s[s["affiliate"].notna()].copy()

    if aff.empty:
        html_block('<div class="c-sub">Affiliate data not uploaded — this chart is unavailable.</div>')
        return

    star = aff.sort_values("affiliate", ascending=False).iloc[0]

    takeaway(
        f"<b>{esc(star['label'])}</b> gets the most affiliate traffic "
        f"({int(star['affiliate']):,} link interactions → ₹{star['revenue'] / 1e5:,.1f}L revenue)."
    )

    chart = (
        alt.Chart(aff)
        .mark_circle(opacity=0.85, stroke="rgba(255,255,255,.7)", strokeWidth=1, cursor="pointer")
        .encode(
            x=alt.X("affiliate:Q", title="Affiliate link interactions",
                    scale=alt.Scale(zero=False, padding=16), axis=alt.Axis(tickCount=5, format="~s")),
            y=alt.Y("revenue:Q", title="Revenue (₹)", axis=alt.Axis(format="~s"), scale=alt.Scale(zero=True)),
            size=alt.Size("conversions:Q", legend=None, scale=alt.Scale(range=[80, 900])),
            color=alt.Color("category:N", title=None, scale=alt.Scale(range=[ORANGE, CREAM, TEAL, SKY, ROSE])),
            tooltip=[
                alt.Tooltip("label:N", title="Product"), alt.Tooltip("sku:N", title="SKU"),
                alt.Tooltip("affiliate:Q", title="Interactions", format=","),
                alt.Tooltip("revenue:Q", title="Revenue ₹", format=",.0f"),
                alt.Tooltip("conversions:Q", title="Conversions", format=",")
            ]
        )
    )

    clickable_chart(chart, "affiliate_chart", 340)


def analytics_revenue(s):

    # 1. Revenue vs Ad Spend ------------------------------------------------

    with st.container(key="glass_rev_spend"):

        head_left, head_right = st.columns([1.6, 1])

        products = ["All products"] + sorted(raw_df["product"].unique().tolist())

        with head_right:
            chosen = st.selectbox("Product", products, label_visibility="collapsed", key="rev_spend_product")

        data = raw_df if chosen == "All products" else raw_df[raw_df["product"] == chosen]

        with head_left:
            total_rev = data["revenue_inr"].sum()
            total_spend = data["ad_spend_inr"].sum()
            card_head("Revenue vs Ad Spend", "Daily totals for the last 28 days")

        takeaway(
            "Every <b>₹1</b> spent on ads brought back "
            + calc(f"<b>₹{(total_rev / total_spend if total_spend else 0):.2f}</b>",
                   [f"Total revenue ₹{total_rev:,.0f} ÷ total ad spend ₹{total_spend:,.0f}",
                    f"{chosen}, all {data['date'].nunique()} days"],
                   f"{rel(DATASET)} · revenue_inr, ad_spend_inr")
            + f" in revenue (₹{total_rev / 1e5:,.1f}L revenue on ₹{total_spend / 1e5:,.1f}L ad spend)."
        )

        chart_revenue_vs_spend(data, 300)

    spacer(0.9)

    left, right = st.columns([1, 1.3], gap="medium")

    # 2. Revenue by channel · 3. Affiliate performance -------------------

    with left:
        with st.container(key="glass_channel"):
            analytics_channel()

    with right:
        with st.container(key="glass_affiliate"):
            analytics_affiliate(s)

    spacer(0.9)


def analytics_inventory(s):

    # 4. Inventory + product intelligence -----------------------------------

    if s["stock"].isna().all():
        with st.container(key="glass_inventory"):
            card_head("Inventory + Product Intelligence")
            html_block('<div class="c-sub">Inventory data not uploaded — stock analysis is unavailable.</div>')
    else:
      with st.container(key="glass_inventory"):
        card_head("Inventory + Product Intelligence", "Days each product was out of stock · click a bar to open it")
        inv = s.copy()
        inv["Status"] = inv.apply(pi.inventory_status, axis=1)
        inv["OOS days"] = inv[["oos_days", "stockout_flag_days"]].max(axis=1)
        inv["Restock units"] = inv.apply(pi.restock_units, axis=1)

        out_now = int((inv["Status"] == "Out of stock").sum())
        worst = inv.sort_values("OOS days", ascending=False).iloc[0]

        takeaway(
            f"<b>{out_now} of {len(inv)}</b> products are out of stock right now. "
            f"<b>{esc(worst['label'])}</b> ran out most often ({int(worst['OOS days'])} of {int(s['days'].max())} days)."
        )

        statuses = [k for k in STATUS_COLORS if k in set(inv["Status"])]

        chart = (
            alt.Chart(inv)
            .mark_bar(cornerRadiusEnd=6, height={"band": 0.65}, cursor="pointer")
            .encode(
                x=alt.X("OOS days:Q", title=None),
                y=alt.Y("label:N", title=None, sort="-x"),
                color=alt.Color(
                    "Status:N", title="Current status",
                    scale=alt.Scale(domain=statuses, range=[STATUS_COLORS[k] for k in statuses])
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Product"), "Status:N",
                    alt.Tooltip("stock:Q", title="Stock on hand", format=","),
                    alt.Tooltip("velocity:Q", title="Units/day", format=".1f"),
                    alt.Tooltip("days_left:Q", title="Days left", format=".1f"),
                    alt.Tooltip("Restock units:Q", format=","),
                    alt.Tooltip("marketing_verdict:N", title="Ads"),
                ]
            )
        )

        clickable_chart(chart, "inventory_chart", 440)

        rows = []

        for _, r in inv.sort_values(["restock_first", "OOS days"], ascending=False).iterrows():

            rows.append([
                esc(r["label"]),
                badge(r["Status"], STATUS_COLORS.get(r["Status"], STONE)),
                f'<span class="mono">{r["stock"]:,.0f}</span>',
                f'<span class="mono">{r["velocity"]:.1f}</span>',
                f'<span class="mono">{r["days_left"]:.1f}</span>',
                f'<span class="mono">{int(r["OOS days"])}</span>',
                f'<span class="mono">{r["Restock units"]:,}</span>' if r["restock_first"] else '<span class="muted">—</span>',
                badge(r["marketing_verdict"], VERDICT_COLORS.get(r["marketing_verdict"], STONE)),
                badge("Clearance", AMBER) if r["clearance"] else badge(r["popularity"], VERDICT_COLORS.get(r["popularity"], STONE)),
            ])

        with st.expander("Inventory table — restock quantities and ad suggestions"):

            glass_table(
                ["Product", "Status", "Stock", "Units/day", "Days left", "OOS days", "Restock qty", "Ads", "Sales tier"],
                rows,
                420
            )

    spacer(0.9)


def analytics_sales(s):

    # 5. Product sales trend ------------------------------------------------

    with st.container(key="glass_sales_trend"):

        head_left, mid, head_right = st.columns([1.3, 1.6, 0.8])

        with head_left:
            card_head("Product Sales Trend", "Pick products to compare")

        options = s.sort_values("units", ascending=False)["label"].tolist()

        with mid:
            selected = st.multiselect(
                "Products", options, default=options[:3], label_visibility="collapsed",
                key="trend_products", placeholder="Choose products"
            )

        with head_right:
            metric = st.segmented_control(
                "Metric", ["Units", "Revenue"], default="Units", required=True,
                key="trend_metric", label_visibility="collapsed"
            )

        if not selected:
            st.info("Select at least one product to see its sales trend.")
            return

        label_by_sku = dict(zip(s["sku"], s["label"]))
        trend = raw_df[raw_df["sku"].map(label_by_sku).isin(selected)].copy()
        trend["Product"] = trend["sku"].map(label_by_sku)

        value = "sales_volume" if metric == "Units" else "revenue_inr"

        chart = (
            alt.Chart(trend)
            .mark_line(interpolate="monotone", strokeWidth=2.6, point=alt.OverlayMarkDef(size=28, filled=True))
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
                y=alt.Y(f"{value}:Q", title="Units per day" if metric == "Units" else "Revenue (₹)",
                        axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "Product:N", title=None,
                    scale=alt.Scale(range=[ORANGE, CREAM, TEAL, SKY, ROSE, AMBER, ORANGE_SOFT, STONE])
                ),
                tooltip=[
                    "Product:N", alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("sales_volume:Q", title="Units"),
                    alt.Tooltip("revenue_inr:Q", title="Revenue ₹", format=",.0f"),
                    alt.Tooltip("product_sales_trend:N", title="Trend")
                ]
            )
        )

        show_chart(chart, 320)

        picks = st.columns(min(len(selected), 4))

        for col, label_text in zip(picks, selected[:4]):

            sku = next(k for k, v in label_by_sku.items() if v == label_text)

            with col:
                st.button(
                    f"Insights: {label_text}",
                    key=f"trend_open_{sku}",
                    on_click=open_product,
                    args=(sku,),
                    width="stretch"
                )


def page_analytics():
    """Performance & Forecast: what happened, what the model expects, stock and sales — one page, four tabs."""

    if raw_df.empty:
        st.warning("No dataset found for this company.")
        return

    s = product_summary
    revenue = raw_df["revenue_inr"].sum(min_count=1)
    spend = raw_df["ad_spend_inr"].sum(min_count=1)
    mae = evaluation_value("MAE", ["random_forest", "model"]) if PRED_READY else None
    mae_base = evaluation_value("MAE", ["baseline"]) if PRED_READY else None

    kpi_strip([
        kpi("💰", "Revenue", fmt(revenue, ",.0f", "₹"), f"last {raw_df['date'].nunique()} days"),
        kpi("📣", "Ad spend", fmt(spend, ",.0f", "₹"), "all channels"),
        kpi("📈", "Return per ₹1 of ads", fmt(revenue / spend if pd.notna(spend) and spend else None, ".2f", "₹"), "blended ROAS"),
        kpi("🔮", "Forecast error", fmt(mae, ".1f", "±", " ROAS"),
            f"simple guess ±{mae_base:.1f}" if mae_base else "needs ≥ 10 days of ad data"),
    ])

    spacer(0.9)

    tabs = st.tabs(["💰 Revenue & channels", "🔮 Forecast", "📦 Stock", "📈 Sales trends"])

    with tabs[0]:
        analytics_revenue(s)

    with tabs[1]:
        if PRED_READY:
            forecast_view()
        else:
            html_block('<div class="c-sub">The forecast needs ad spend and revenue for at least 10 days — '
                       'not available for this data.</div>')

    with tabs[2]:
        analytics_inventory(s)

    with tabs[3]:
        analytics_sales(s)


# ============================================================
# PAGE: ACTIONS — approve → simulated execution → measure → learn
# ============================================================

OUTCOME_COLORS = {"worked": TEAL, "did not work": ROSE, "pending": AMBER, "n/a": STONE}


def page_actions():

    log = actions.load(ACTIVE)
    approved = [e for e in log if e["decision"] == "approved"]
    measured = [e for e in approved if e["outcome"] in ("worked", "did not work")]
    worked = [e for e in measured if e["outcome"] == "worked"]

    kpi_strip([
        kpi("🧠", "Recommended", f"{len(recommendations)}", "by the decision engine"),
        kpi("✅", "Approved", f"{len(approved)}", "by you · execution simulated"),
        kpi("📏", "Measured", f"{len(measured)}", "from data after the decision"),
        kpi("🎯", "Worked", f"{len(worked)} of {len(measured)}" if measured else "—", "target KPI moved the right way",
            how=(["A decision 'worked' when its target KPI moved ≥ 2% in the intended direction, "
                  "averaged over up to 7 days after the decision vs 7 days before"], "actions.py", None, "right")),
    ])

    spacer(0.8)

    takeaway(
        "Approving a recommendation logs the instruction and <b>simulates</b> execution — no ad platform is changed. "
        "When newer data for that product arrives (e.g. next week's upload), the outcome is measured and the engine's "
        "confidence for that action type moves up or down. Outcomes are observed changes, not proof of causation."
    )

    if not log:
        st.info("No decisions yet — approve one from AI Priorities (Products) or the Recommendation Center.")
        st.button("Go to AI Priorities", key="act_go", on_click=go_to, args=("Products",), type="primary")
        return

    with st.container(key="glass_action_log"):

        card_head("Decision log", "Recommendation → expected → decision → observed outcome")

        rows = []

        for e in sorted(log, key=lambda x: x["created"], reverse=True):

            before = fmt(e["baseline"], ",.2f")
            after = fmt(e.get("observed"), ",.2f")
            change = e.get("change")
            observed = (
                f'{before} → {after} <span class="muted">({change:+.0%})</span>' if change is not None
                else f'{before} → <span class="muted">waiting for data after {esc(e["data_end"])}</span>'
                if e["decision"] == "approved" else "—"
            )

            rows.append([
                f'{esc(e["title"])}<div class="l-sub">{esc(e["label"])}</div>',
                esc(e["expected"] if str(e["expected"]).startswith(("Est.", "₹")) is False else e["expected"]),
                badge("Approved · simulated" if e["decision"] == "approved" else "Dismissed",
                      TEAL if e["decision"] == "approved" else STONE),
                f'<span class="muted">{esc(e["kpi_label"])}</span><div>{observed}</div>',
                badge(e["outcome"], OUTCOME_COLORS.get(e["outcome"], STONE)),
            ])

        glass_table(["Decision", "Expected (estimate)", "Status", "Target KPI · before → after", "Outcome"], rows, 460)

    stats = actions.learning(ACTIVE)

    if stats:
        spacer(0.9)
        with st.container(key="glass_learning"):
            card_head("What the AI learned", "Measured outcomes feed back into the confidence of future recommendations")
            html_block("".join(
                f'<div class="lrow"><div class="l-main"><div class="l-name">{esc(pretty(kind))}</div>'
                f'<div class="l-sub">{w} of {n} worked → confidence for this action type '
                f'{"rises" if w / n > 0.5 else "falls" if w / n < 0.5 else "holds"}</div></div>'
                f'{badge(f"{w / n:.0%} success", TEAL if w / n >= 0.5 else ROSE)}</div>'
                for kind, (w, n) in stats.items()
            ))


# ============================================================
# PAGE: UPLOAD — drop CSVs → same decision engine
# ============================================================

FIELD_OPTIONS = ["(ignore)"] + sorted(ingest.FIELDS)


def _read_upload(file):

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    file.seek(0)
    return pd.read_csv(file, encoding_errors="replace")


def analysis_summary(company, report=None, measured=0):
    """'Analysis complete' card — computed by the same engine the dashboard uses."""

    df = pi.load_dataset(company["dataset"])
    summary = pi.build_summary(df, None)
    platforms = co.load_platforms(company)
    recs = pe.detect_recommendations(df, summary, platforms, actions.learning(company))
    found = pe.detect_anomalies(df)
    top = pe.select_priorities(recs, limit=5) if not recs.empty else []
    campaigns = 0

    for entry in co.manifest(company):
        if any(m["field"] == "campaign" for m in entry["mapping"]):
            raw = pd.read_csv(company["workspace"] / "uploads" / entry["file"])
            col = next(m["column"] for m in entry["mapping"] if m["field"] == "campaign")
            campaigns += raw[col].nunique()

    with st.container(key="glass_analysis_done"):

        html_block(
            f'<div class="level"><span class="n">ANALYSIS COMPLETE</span><span class="t">{esc(company["name"])}</span>'
            f'<span class="q">{df["date"].min():%d %b} → {df["date"].max():%d %b %Y}</span></div>'
        )

        kpi_strip([
            kpi("📦", "Products analysed", f"{df['sku'].nunique()}", f"{len(df):,} product-days"),
            kpi("📣", "Campaigns analysed", f"{campaigns}" if campaigns else "—", "from ad exports"),
            kpi("⚡", "Anomalies", f"{len(found)}", "unusual recent moves"),
            kpi("🎯", "Actionable opportunities", f"{len(recs)}", f"{int((recs['tier'] == 'HIGH').sum()) if len(recs) else 0} high priority"),
        ] + ([kpi("📏", "Outcomes measured", f"{measured}", "approved actions checked")] if measured else []))

        left, right = st.columns([1.5, 1], gap="medium")

        with left:
            card_head("Top priorities", "Same decision engine as the built-in data")
            if top:
                html_block("".join(
                    f"""
<div class="lrow" style="align-items:flex-start">
  <div class="l-icon" style="font-size:.8rem;font-weight:700">{i}</div>
  <div class="l-main">
    <div class="l-name" style="white-space:normal">{esc(rec["title"]).upper()} · {esc(rec["label"])}</div>
    <div class="l-sub">{esc(rec["headline"])}</div>
    <div class="l-sub">{esc(impact_text(rec))} · {rec["confidence"]}% confidence</div>
  </div>
  {tier_badge(rec["tier"])}
</div>
"""
                    for i, rec in enumerate(top, start=1)
                ))
            else:
                html_block('<div class="c-sub">No actionable issues found in these files.</div>')

        with right:
            card_head("What can be analysed")
            caps = (report or {}).get("capabilities") or {k: tuple(v) for k, v in (co.capabilities_of(company) or {}).items()}
            html_block(capability_list(caps))

        if report and report.get("notes"):
            html_block('<div class="c-sub" style="margin-top:.5rem">' + " · ".join(esc(n) for n in report["notes"]) + "</div>")

        st.button(f"Open {company['name']} — full analysis", key="upload_open", icon=":material/arrow_forward:",
                  icon_position="right", type="primary", on_click=open_uploaded, args=(company["slug"],))


def open_uploaded(slug):

    switch_company(slug)
    st.session_state.nav = "Products"


def page_upload():

    nonce = st.session_state.get("upload_nonce", 0)
    done = st.session_state.get("upload_done")

    if done and co.get_company(done["slug"]):
        analysis_summary(co.get_company(done["slug"]), done.get("report"), done.get("measured", 0))
        spacer(0.8)
        if st.button("Analyse more files", key="upload_again", icon=":material/upload_file:"):
            st.session_state.upload_done = None
            st.session_state.upload_nonce = nonce + 1
            st.rerun()
        return

    with st.container(key="glass_dropzone"):

        html_block(
            '<div style="text-align:center;padding:.4rem 0 .2rem">'
            '<div class="pri-cat">DROP YOUR DATA HERE</div>'
            '<div class="tb-title" style="font-size:1.35rem;margin:.35rem 0">Drag & drop CSV files — or browse</div>'
            '<div class="c-sub">Meta Ads · Google Ads · Amazon Ads · TikTok Ads · Sales / Shopify · Inventory · '
            'Customers / funnel · Pricing · Creative performance · Affiliate</div></div>'
        )

        files = st.file_uploader(
            "CSV files", type=["csv"], accept_multiple_files=True,
            key=f"uploader_{nonce}", label_visibility="collapsed",
        )

        html_block('<div class="c-sub" style="text-align:center">Try it with the simulated sample exports in '
                   '<span class="mono">sample_uploads/weeks_1-3/</span> (then add <span class="mono">week_4_followup/</span> '
                   'to the same company to see outcomes measured).</div>')

    if not files:
        return

    spacer(0.9)

    parsed, problems = [], []

    for n, file in enumerate(files):

        try:
            df = _read_upload(file)
        except Exception as error:
            problems.append(f"{file.name}: could not be read ({error})")
            continue

        if df.empty:
            problems.append(f"{file.name}: the file has no rows")
            continue

        source, platform, conf = ingest.detect_source(df, file.name)
        mapping = ingest.map_columns(df)

        with st.container(key=f"glass_file_{nonce}_{n}"):

            mapped = [m for m in mapping if m["field"]]
            mean_conf = sum(m["confidence"] for m in mapped) / len(mapped) if mapped else 0

            card_head(
                f"📄 {esc(file.name)}",
                f"Detected source: <b>{esc(source)}</b> ({conf:.0%}) · {len(df):,} rows · "
                f"{len(mapped)} of {len(mapping)} columns mapped · mapping confidence {mean_conf:.0%}",
            )

            table = pd.DataFrame({
                "Column in file": [m["column"] for m in mapping],
                "Maps to": [m["field"] or "(ignore)" for m in mapping],
                "Confidence": [round(m["confidence"] * 100) for m in mapping],
            })

            edited = st.data_editor(
                table, key=f"map_{nonce}_{n}", hide_index=True, width="stretch",
                disabled=["Column in file", "Confidence"],
                column_config={
                    "Maps to": st.column_config.SelectboxColumn(options=FIELD_OPTIONS, required=True),
                    "Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
                },
            )

            low = table[(table["Confidence"] < 80) & (table["Maps to"] != "(ignore)")]
            if len(low):
                st.warning("Please check: " + ", ".join(f"'{c}' → {f}" for c, f in zip(low["Column in file"], low["Maps to"])))

            final_mapping = [
                {"column": c, "field": None if f == "(ignore)" else f, "confidence": 1.0}
                for c, f in zip(edited["Column in file"], edited["Maps to"])
            ]

            _, findings = ingest.normalise(df, final_mapping, platform)
            icons = {"ok": "✓", "warn": "⚠", "error": "✗"}
            html_block('<div class="checks">' + "".join(
                badge(f"{icons[level]} {msg}", {"ok": TEAL, "warn": AMBER, "error": ROSE}[level]) for level, msg in findings
            ) + "</div>")

            with st.expander("Preview first rows"):
                st.dataframe(df.head(5), hide_index=True, width="stretch")

        parsed.append({"filename": file.name, "df": df, "mapping": final_mapping, "source": source, "platform": platform})
        spacer(0.6)

    for p in problems:
        st.error(p)

    if not parsed:
        return

    uploaded = [c for c in co.list_companies() if co.manifest(c)]

    with st.container(key="glass_upload_target"):

        card_head("Where should this data go?")
        options = ["New company"] + ([f"Add to {c['name']}" for c in uploaded] if uploaded else [])
        target = st.radio("Destination", options, horizontal=True, label_visibility="collapsed", key=f"dest_{nonce}")

        if target == "New company":
            name = st.text_input("Company name", value="My D2C brand", key=f"name_{nonce}")
        else:
            name = None

        go = st.button("Analyse data", key=f"analyse_{nonce}", icon=":material/bolt:", type="primary")

    if not go:
        return

    with st.status("Analysing…", expanded=True) as status:

        try:
            status.update(label="Reading data")
            st.write(f"✓ Uploaded {len(parsed)} file(s): " + ", ".join(f"{f['filename']} ({f['source']})" for f in parsed))

            status.update(label="Reconciling columns")
            if target == "New company":
                company, report = co.create_from_uploads(name, parsed)
            else:
                company = next(c for c in uploaded if f"Add to {c['name']}" == target)
                report = co.add_uploads(company, parsed)
            st.write(f"✓ Unified dataset: {len(report['dataset']):,} rows · {report['dataset']['sku'].nunique()} products · "
                     + " · ".join(report["notes"]))

            status.update(label="Calculating signals")
            for stage, line in co.run_pipeline(company):
                if line.startswith(("✓", "Test MAE")):
                    st.write(f"✓ {stage}: {line.lstrip('✓ ')}")
            st.write("✓ Signals: ROAS, contribution, CPM, CVR, velocity, stock cover, funnel")

            status.update(label="Finding anomalies · diagnosing · generating recommendations")
            measured = actions.measure(company, pd.read_csv(company["dataset"]))
            if measured:
                st.write(f"✓ Measured {measured} approved action(s) against the new data")

            st.cache_data.clear()
            status.update(label="Analysis ready", state="complete", expanded=False)

        except Exception as error:
            status.update(label=f"Analysis failed — {error}", state="error")
            return

    st.session_state.upload_done = {
        "slug": company["slug"],
        "measured": measured,
        "report": {"capabilities": report["capabilities"], "notes": report["notes"]},
    }
    st.rerun()


# ============================================================
# AI RECOMMENDATION CENTER · SMART BUNDLING · PRODUCT OVERVIEW
# ============================================================

CATEGORY_COLORS = {
    "INVENTORY": ROSE, "MARKETING": SKY, "BUDGET": TEAL, "PROFITABILITY": AMBER,
    "PRICING": AMBER, "CREATIVE": ORANGE_SOFT, "FUNNEL": "#e9a8f5", "CUSTOMER": "#e9a8f5",
    "TREND": TEAL, "AFFILIATE": SKY,
}


def rec_detail_html(rec, show_product=True):

    accent = CATEGORY_COLORS.get(rec["category"], STONE)
    evidence = "".join(f'<span class="ev">{esc(e)}</span>' for e in rec["evidence"])
    who = f'<b>{esc(rec["label"])}</b> · ' if show_product else ""

    return f"""
<div class="rc-head">
  <div class="left">
    <span class="cat-pill" style="--accent:{accent}">{esc(rec["category"])}</span>
    <span class="rc-title">{esc(rec["title"])}</span>
    {rec_tier(rec, align="left")}
  </div>
  <span class="rc-conf">Confidence<b>{rec_confidence(rec, f"{rec['confidence']}%")}</b></span>
</div>
<div class="rc-sub">{who}<span>{esc(rec["headline"])}</span></div>
<div class="ev-row">{evidence}</div>
<div class="rc-rows">
  <span class="k">Likely cause</span><span class="v">{esc(rec["cause"])}</span>
  <span class="k">Action</span><span class="v strong">{esc(rec["action"])}</span>
  <span class="k">Impact</span><span class="v">{rec_impact(rec)}</span>
</div>
"""


def rec_list(recs, prefix, in_dialog=False, show_family=True):
    """Full recommendation cards with drill-down buttons."""

    for index, rec in recs.iterrows():

        with st.container(key=f"glassrc_{rec['tier']}_{prefix}_{index}"):

            html_block(rec_detail_html(rec))

            with st.container(horizontal=True, gap="small", key=f"rcb_{prefix}_{index}"):

                if st.button(
                    f"Open {rec['tab']} insights",
                    key=f"{prefix}_open_{index}",
                    icon=":material/open_in_new:",
                ):
                    open_product(rec["sku"], rec["tab"])
                    st.rerun()

                if show_family and st.button(
                    f"All suggestions for {rec['product']}",
                    key=f"{prefix}_fam_{index}",
                    icon=":material/list:",
                ):
                    open_family(rec["product"])
                    st.rerun()

                if not in_dialog:
                    decision_button(rec, f"{prefix}_{index}")


def page_recommendations():

    if recommendations.empty:
        st.success("No significant issues detected across the portfolio.")
        return

    r = recommendations

    cat_counts = r["category"].value_counts()
    product_counts = r["product"].value_counts()

    with st.container(key="glass_rcfilters"):

        category = st.pills(
            "Category", ["All"] + cat_counts.index.tolist(), default="All", key="rc_category",
            format_func=lambda c: c if c == "All" else f"{c} · {cat_counts[c]}",
        ) or "All"

        product = st.pills(
            "Product", ["All"] + sorted(product_counts.index.tolist()), default="All", key="rc_product",
            format_func=lambda p: p if p == "All" else f"{p} · {product_counts[p]}",
        ) or "All"

        c1, c2 = st.columns([2, 1], gap="small")

        with c1:
            search = st.text_input("Search", placeholder="Product, SKU or keyword", key="rc_search")

        with c2:
            sort = st.selectbox("Sort by", ["Priority", "Impact (₹)", "Confidence"], key="rc_sort")

    view = r

    if category != "All":
        view = view[view["category"] == category]

    if product != "All":
        view = view[view["product"] == product]

    if search:
        text = (view["label"] + " " + view["sku"] + " " + view["title"] + " " + view["headline"] + " " + view["action"])
        view = view[text.str.contains(search, case=False, regex=False)]

    sort_column = {"Priority": "score", "Impact (₹)": "impact", "Confidence": "confidence"}[sort]
    view = view.sort_values(sort_column, ascending=False)

    html_block(
        f'<div class="c-sub" style="margin:.9rem 0 .6rem">{len(view)} of {len(r)} suggestions'
        + (f' · <b>{esc(product)}</b>' if product != "All" else ' · select a product to see everything we recommend for it')
        + "</div>"
    )

    if product != "All":
        st.button(
            f"Open {product} overview",
            key="rc_family_open",
            icon=":material/inventory_2:",
            on_click=open_family,
            args=(product,),
            type="primary",
        )
        spacer(0.4)

    if view.empty:
        st.info("No suggestions match these filters.")
        return

    rec_list(view, "rc", show_family=product == "All")


def bundle_card(bundle, key, show_buttons=True):

    cautions = "".join(f'<span class="caution">Caution: {esc(c)}</span>' for c in bundle["cautions"])
    how = bundle.get("calc", {"units": [], "aov": [], "margin": [], "discount": [], "confidence": []})

    html_block(
        f"""
<div class="rc-head">
  <div class="left">
    <span class="cat-pill" style="--accent:{ORANGE}">BUNDLE</span>
    <span style="font-weight:600;font-size:1.02rem;color:{ORANGE_SOFT}">{esc(bundle["name"])}</span>
  </div>
  <span class="rc-conf">Confidence<b>{calc(f"{bundle['confidence']}%", [esc(x) for x in how["confidence"]], result=f"= {bundle['confidence']}% (capped 50–95)", title="How confidence is scored", align="right")}</b></span>
</div>
<div class="rc-rows" style="margin-top:.7rem">
  <span class="k">Reason</span><span class="v">{esc(bundle["reason"])}</span>
  <span class="k">Offer</span><span class="v offer">{calc(esc(bundle["offer"]), [esc(x) for x in how["discount"] + how["units"][:1]], "bundle_engine.py", title="How the offer is set")}</span>
  <span class="k">Estimated</span><span class="v">{calc(esc(bundle["expected"]), [esc(x) for x in how["units"][1:] + how["aov"]], f"{rel(DATASET)} · last {be.WINDOW_DAYS} days")}</span>
  <span class="k">Extra margin</span><span class="v">{calc(f"~₹{bundle['extra_margin']:,.0f} per {be.WINDOW_DAYS} days after the discount", [esc(x) for x in how["margin"]], f"{rel(DATASET)} · margin_pct")}</span>
</div>
{cautions}
"""
    )

    if not show_buttons:
        return

    with st.container(horizontal=True, gap="small", key=f"bdb_{key}"):

        for role in ("anchor", "laggard"):

            if st.button(
                f"Open {bundle[role]}",
                key=f"bd_{key}_{role}",
                icon=":material/open_in_new:",
            ):
                open_family(bundle[role])
                st.rerun()


def page_bundling():

    if not bundles:
        st.info("No strong popular + slow product pairs found in the current data.")
        return

    kpi_strip([
        kpi("🎁", "Bundles suggested", f"{len(bundles)}", "one partner per product", how=(
            ["Strong products (ROAS and orders above median) paired with lagging ones",
             "Each product is used in at most one bundle"], "bundle_engine.py")),
        kpi("📦", "Extra laggard units", f"~{sum(b['extra_units'] for b in bundles):,.0f}", f"per {be.WINDOW_DAYS} days", how=(
            [" + ".join(f"{b['extra_units']:,.0f}" for b in bundles) + " (one per bundle)"],
            f"{rel(DATASET)} · last {be.WINDOW_DAYS} days")),
        kpi("💰", "Extra margin", f"~₹{sum(b['extra_margin'] for b in bundles):,.0f}", "after discounts", how=(
            [" + ".join(f"₹{b['extra_margin']:,.0f}" for b in bundles) + " (one per bundle)"],
            f"{rel(DATASET)} · last {be.WINDOW_DAYS} days")),
        kpi("⚠️", "Need restock first", f"{sum(1 for b in bundles if b['cautions'])}", "see cautions", how=(
            ["Bundles where either product has < 3 days of stock"], f"{rel(DATASET)} · stock_on_hand", None, "right")),
    ])

    html_block(
        '<div class="c-sub" style="margin:.9rem 0 .7rem">Measured: orders, ROAS, margin, price and stock (last '
        f'{be.WINDOW_DAYS} days). Attach rate is a target — the dataset has no basket data.</div>'
    )

    for i, bundle in enumerate(bundles):

        with st.container(key=f"glassbd_{i}"):
            bundle_card(bundle, i)

        spacer(0.7)


def family_dialog(product):

    variants = product_summary[product_summary["product"] == product]

    if variants.empty:
        st.warning("Product not found.")
        return

    category = variants["category"].iloc[0]
    statuses = variants.apply(pi.inventory_status, axis=1)
    before, now = variants["units_prev_7d"].sum(), variants["units_7d"].sum()
    trend = (now - before) / before if before else 0.0

    chips = []

    if statuses.isin(["Out of stock", "Critical"]).any():
        chips.append(badge("Restock", ROSE))

    chips.append(badge(f"Trend {trend:+.0%}", TEAL if trend >= 0 else ROSE))

    found = recommendations[recommendations["product"] == product] if not recommendations.empty else pd.DataFrame()

    if not found.empty and (found["kind"] == "scale_winner").any():
        chips.append(badge("Scale", TEAL))

    html_block(
        f"""
<div class="tb-title" style="font-size:1.5rem">{esc(product)}</div>
<div class="tb-sub" style="display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin:.3rem 0 1rem">
  {esc(category)} · {len(variants)} variant{"s" if len(variants) > 1 else ""} · {" ".join(chips)}
</div>
"""
    )

    data = raw_df[raw_df["product"] == product]
    last7 = data[data["date"] > data["date"].max() - pd.Timedelta(days=7)]

    mini_kpis([
        ("7-day ROAS", f"{last7['revenue_inr'].sum() / last7['ad_spend_inr'].sum():.2f}×" if last7["ad_spend_inr"].sum() else "—", (
            [f"Revenue ₹{last7['revenue_inr'].sum():,.0f} ÷ ad spend ₹{last7['ad_spend_inr'].sum():,.0f}",
             f"all {len(variants)} variants, last 7 days"], f"{rel(DATASET)}")),
        ("Revenue (28d)", f"₹{variants['revenue'].sum():,.0f}", (
            [" + ".join(f"₹{v:,.0f}" for v in variants["revenue"]) + " (one per variant)"], f"{rel(DATASET)} · revenue_inr")),
        ("Profit (28d)", f"₹{variants['profit'].sum():,.0f}", (
            [" + ".join(f"₹{v:,.0f}" for v in variants["profit"]) + " (one per variant)"], f"{rel(DATASET)} · profit_inr")),
        ("Units / day", f"{variants['velocity'].sum():.1f}", (
            [" + ".join(f"{v:.1f}" for v in variants["velocity"]) + " units/day (one per variant)"], f"{rel(DATASET)}")),
        ("Stock on hand", f"{variants['stock'].sum():,.0f}", (
            [" + ".join(f"{v:,.0f}" for v in variants["stock"]) + " units (latest day, per variant)"], f"{rel(DATASET)}", None, "right")),
    ])

    html_block('<div class="fam-sec">AI suggestions for this product</div>')

    if found.empty:
        html_block('<div class="card" style="padding:.8rem 1rem"><span class="healthy">✓ No significant issues detected</span></div>')
    else:
        rec_list(found.sort_values("score", ascending=False), "fam", in_dialog=True, show_family=False)

    related = [b for b in bundles if product in (b["anchor"], b["laggard"])]

    if related:

        html_block('<div class="fam-sec">Bundle opportunity</div>')

        for i, bundle in enumerate(related):
            with st.container(key=f"glassbd_fam_{i}"):
                bundle_card(bundle, f"fam_{i}", show_buttons=False)

    html_block('<div class="fam-sec">Variants · full insights</div>')

    with st.container(horizontal=True, gap="small", key="famv_buttons"):

        for _, v in variants.iterrows():

            if st.button(
                f"{v['label']} · {pi.inventory_status(v)}",
                key=f"famv_{v['sku']}",
                icon=":material/open_in_new:",
            ):
                open_product(v["sku"])
                st.rerun()


# ============================================================
# ROUTER
# ============================================================

PAGE_RENDERERS = {
    "Overview": page_overview,
    "Products": page_products,
    "Recommendations": page_recommendations,
    "Bundling": page_bundling,
    "Analytics": page_analytics,
    "Decide": page_decide,
    "Explain": page_explain,
    "Learn": page_learn,
    "Pipeline": page_pipeline,
    "Datasets": page_datasets,
    "Actions": page_actions,
    "Upload": page_upload,
}

NEEDS = {"Decide": ML_READY, "Explain": ML_READY, "Learn": ML_READY}

if st.session_state.get("toast"):
    st.toast(st.session_state.pop("toast"), icon="✅")

if not NEEDS.get(page, True):
    render_needs(page)
else:
    PAGE_RENDERERS[page]()


# Product insight dialog (opened from product cards or chart clicks)

if st.session_state.get("open_family") and not product_summary.empty:

    st.dialog(
        "Product overview",
        width="large",
        on_dismiss=close_product
    )(family_dialog)(st.session_state.open_family)

elif st.session_state.get("open_sku") and not product_summary.empty:

    st.dialog(
        "Product insights",
        width="large",
        on_dismiss=close_product
    )(product_dialog)(st.session_state.open_sku)


# ------------------------------------------------------------
# Previous / next navigation
# ------------------------------------------------------------

index = PAGES.index(page)

spacer(1.2)

prev_col, _, next_col = st.columns([1, 2, 1])

with prev_col:

    if index > 0:

        st.button(
            PAGES[index - 1],
            icon=":material/arrow_back:",
            on_click=go_to,
            args=(PAGES[index - 1],),
            width="stretch"
        )

with next_col:

    if index < len(PAGES) - 1:

        st.button(
            f"Next: {PAGES[index + 1]}",
            icon=":material/arrow_forward:",
            icon_position="right",
            on_click=go_to,
            args=(PAGES[index + 1],),
            width="stretch",
            type="primary"
        )


# Scroll back to the top whenever the page changes

if st.session_state.get("_last_page") != page:

    st.session_state["_last_page"] = page

    st.html(
        """
<script>
const main = document.querySelector('section.stMain') || document.querySelector('[data-testid="stAppViewContainer"]');
if (main) { main.scrollTo({top: 0, behavior: 'smooth'}); }
window.scrollTo({top: 0, behavior: 'smooth'});
</script>
""",
        unsafe_allow_javascript=True
    )


html_block(
    '<div class="footer-note">⚡ DataQuest 3.0 Hackathon Prototype · simulated scenario data · '
    'Predict → Diagnose → Decide → Learn</div>'
)
