"""app.py — AI Dividend Predictor · Streamlit Dashboard"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_NS_SUFFIX


def _validate_ticker_ns(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(DEFAULT_NS_SUFFIX):
        t = t + DEFAULT_NS_SUFFIX
    return t


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DividendAI — Smart SIP Advisor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS — full premium fintech theme
# ---------------------------------------------------------------------------

def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* Section labels */
    h2 {
        font-size: 1.15rem !important; font-weight: 700 !important;
        color: #1E293B !important; letter-spacing: 0.01em;
        margin-bottom: 1rem !important; margin-top: 0.25rem !important;
    }
    h3 { color: #1E293B !important; font-weight: 600 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F1F5F9 !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #FFFFFF; border: 1px solid #F1F5F9; border-radius: 14px;
        padding: 1.1rem 1.3rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 16px rgba(99,102,241,0.1); transform: translateY(-1px);
    }
    [data-testid="metric-container"] label {
        color: #94A3B8 !important; font-size: 0.68rem !important;
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.55rem !important; font-weight: 700 !important; color: #0F172A !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"]:focus {
        background: #6366F1 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.6rem !important;
        margin: 4px 0 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
        transition: all 0.18s !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #4F46E5 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
        box-shadow: 0 1px 4px rgba(99,102,241,0.3) !important;
    }

    /* Secondary buttons */
    .stButton > button:not([kind="primary"]),
    .stButton > button:not([kind="primary"]):focus {
        background: #F8FAFC !important;
        color: #475569 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        margin: 4px 0 !important;
        transition: all 0.15s !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #EEF2FF !important;
        color: #6366F1 !important;
        border-color: #A5B4FC !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #EEF2FF !important;
        color: #6366F1 !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        margin: 4px 0 !important;
        transition: all 0.15s !important;
    }
    .stDownloadButton > button:hover {
        background: #E0E7FF !important;
        border-color: #818CF8 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.2) !important;
    }


    /* Auto-scroll anchor for active step */
    #step-running { scroll-margin-top: 80px; }

    /* Step card loader animations */
    @keyframes shimmer { 0% { background-position: 200% center; } 100% { background-position: -200% center; } }
    @keyframes fadeSlideIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.0); }
        50% { box-shadow: 0 0 0 6px rgba(99,102,241,0.08); }
    }
    @keyframes dotBounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
        40% { transform: translateY(-3px); opacity: 1; }
    }
    .step-card-running {
        animation: pulseGlow 2s ease-in-out infinite !important;
    }
    .step-dot { display:inline-block; width:5px; height:5px; border-radius:50%;
        margin:0 1px; vertical-align:middle; }
    .step-dot:nth-child(1) { animation: dotBounce 1.2s ease-in-out 0.0s infinite; }
    .step-dot:nth-child(2) { animation: dotBounce 1.2s ease-in-out 0.2s infinite; }
    .step-dot:nth-child(3) { animation: dotBounce 1.2s ease-in-out 0.4s infinite; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #F1F5F9 !important; border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;
    }

    /* File uploader */
    [data-testid="stFileUploader"] > label {
        color: #94A3B8 !important; font-size: 0.82rem !important; font-weight: 400 !important;
    }
    [data-testid="stFileUploader"] section {
        background: #F8FAFC !important; border: 2px dashed #C7D2FE !important;
        border-radius: 12px !important; padding: 1.4rem 1.6rem !important;
    }
    [data-testid="stFileUploader"] section > div {
        color: #94A3B8 !important; font-size: 0.82rem !important;
    }
    [data-testid="stFileUploader"] section span {
        color: #94A3B8 !important;
    }

    /* Misc */
    [data-testid="stAlert"] { border-radius: 10px !important; }
    hr { border-color: #F1F5F9 !important; margin: 1.5rem 0 !important; }
    [data-testid="stExpander"] {
        background: #FFFFFF !important; border: 1px solid #F1F5F9 !important;
        border-radius: 12px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stSlider"] > div > div > div > div { background: #6366F1 !important; }

    /* Tooltip help icon — no border/bg */
    [data-testid="stTooltipIcon"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #94A3B8 !important;
        opacity: 0.7 !important;
    }
    /* Tooltip popup content box */
    [data-testid="stTooltipHoverTarget"] + div,
    div[class*="tooltip"] div[class*="body"],
    div[data-baseweb="tooltip"] {
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
        border: 1px solid #E2E8F0 !important;
        font-size: 0.75rem !important;
        line-height: 1.55 !important;
        padding: 8px 12px !important;
        max-width: 220px !important;
    }

    /* Sidebar slider labels */
    [data-testid="stSidebar"] [data-testid="stSlider"] label p {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        letter-spacing: 0.01em !important;
    }

    /* Animations */
    @keyframes fadeSlideIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    .step-card { animation: fadeSlideIn 0.32s ease forwards; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Lazy pipeline imports
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _import_pipeline():
    from config import MODEL_WEIGHTS, HIST_YEARS
    from data_pipeline.fetcher import batch_fetch, validate_ticker_ns
    from data_pipeline.features import build_feature_matrix, impute_features, build_dividend_series
    from models.random_forest import DividendRandomForest
    from models.lstm_model import DividendLSTM
    from models.finbert_sentiment import FinBERTSentimentAnalyzer
    from models.ensemble import EnsembleCombiner, ModelOutputs
    from analysis.portfolio import PortfolioAdvisor
    return (
        MODEL_WEIGHTS, HIST_YEARS,
        batch_fetch, validate_ticker_ns,
        build_feature_matrix, impute_features, build_dividend_series,
        DividendRandomForest, DividendLSTM, FinBERTSentimentAnalyzer,
        EnsembleCombiner, ModelOutputs,
        PortfolioAdvisor,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Dict[str, float]:
    locked = st.session_state.get("analyzing", False)

    st.sidebar.markdown(
        "<div style='margin-bottom:8px;'>"
        "<h1 style='color:#6366F1;font-size:1.05rem;font-weight:800;margin:0 0 1px 0;'>DividendAI<span style='color:#F59E0B;'>.</span></h1>"
        "<p style='color:#94A3B8;font-size:0.75rem;font-style:italic;margin:0;'>"
        "Because your dividends deserve a PhD.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "<p style='color:#0F172A;font-size:0.95rem;font-weight:800;letter-spacing:-0.01em;margin-bottom:4px;'>"
        "Ensemble Engine</p>"
        "<p style='color:#94A3B8;font-size:0.7rem;margin-top:-4px;margin-bottom:6px;'>"
        "Adjust how much each AI model influences the final SIP call.</p>",
        unsafe_allow_html=True,
    )

    rf_w = st.sidebar.slider("Random Forest", 0.0, 1.0, 0.70, 0.05, disabled=locked,
        help="Trained on 19 financial features — ROE, FCF yield, dividend CAGR, payout ratio, D/E ratio. "
             "Classifies each stock as Increase / Hold / Reduce based on fundamentals.")
    lstm_w = st.sidebar.slider("Pattern Memory", 0.0, 1.0, 0.15, 0.05, disabled=locked,
        help="MLP trained on 10Y quarterly dividend sequences. Detects momentum — "
             "consistent raising streaks vs irregular payout patterns.")
    finbert_w = st.sidebar.slider("News Sentiment", 0.0, 1.0, 0.10, 0.05, disabled=locked,
        help="Finance-tuned BERT model reads recent headlines. Captures earnings surprises, "
             "dividend announcements, and analyst guidance in real time.")

    base_w = max(0.0, round(1.0 - rf_w - lstm_w - finbert_w, 4))
    st.sidebar.markdown(
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;"
        f"padding:5px 10px;margin-top:-4px;margin-bottom:8px;'>"
        f"<span style='color:#94A3B8;font-size:0.7rem;'>Base Rate Prior</span>"
        f"<span style='color:#94A3B8;font-size:0.68rem;'> — historical NSE dividend raise frequency (~55%), "
        f"used as a tiebreaker when models disagree</span><br>"
        f"<span style='color:#3B82F6;font-weight:700;font-size:0.8rem;'>auto: {base_w:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    news_lookback_days = st.sidebar.number_input(
        "News lookback (days)", min_value=3, max_value=90, value=5, step=1,
        help="Headlines older than this many days are ignored by FinBERT.", disabled=locked,
    )
    st.sidebar.divider()

    if st.sidebar.button("Reset Defaults", use_container_width=True, disabled=locked):
        st.rerun()

    return {"random_forest": rf_w, "lstm": lstm_w, "finbert": finbert_w, "base": base_w}, int(news_lookback_days)


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown("""
    <div style="background:#FFFFFF;
        border:1px solid #E8EDFB;border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;
        box-shadow:0 2px 12px rgba(99,102,241,0.07);">
        <div style="margin-bottom:0.5rem;">
            <h1 style="margin:0 0 0.25rem 0;font-size:2rem;font-weight:800;
                background:linear-gradient(100deg,#6366F1,#8B5CF6,#06B6D4);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                DividendAI<span style="-webkit-text-fill-color:#F59E0B;">.</span>
            </h1>
            <p style="margin:0 0 0.6rem 0;color:#94A3B8;font-size:0.85rem;line-height:1.7;max-width:580px;">
                Upload your NSE dividend portfolio and get AI-driven recommendations
                on whether to increase, hold, or reduce each monthly SIP —
                backed by 10 years of live market data.
            </p>
            <p style="margin:0;color:#94A3B8;font-size:0.68rem;letter-spacing:0.08em;">
                SMART SIP ADVISOR &nbsp;·&nbsp; 4 AI MODELS &nbsp;·&nbsp; NSE INDIA
            </p>
        </div>
        <div style="display:flex;gap:2.5rem;margin-top:1.4rem;flex-wrap:wrap;">
            <div>
                <div style="color:#6366F1;font-size:1.3rem;font-weight:800;">4</div>
                <div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;">AI Models</div>
            </div>
            <div>
                <div style="color:#059669;font-size:1.3rem;font-weight:800;">19</div>
                <div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;">Financial Factors</div>
            </div>
            <div>
                <div style="color:#0891B2;font-size:1.3rem;font-weight:800;">10Y</div>
                <div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;">Historical Data</div>
            </div>
            <div>
                <div style="color:#7C3AED;font-size:1.3rem;font-weight:800;">Live</div>
                <div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.06em;">NSE Prices</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------

def render_upload_section() -> Optional[pd.DataFrame]:
    st.markdown(
        "<h2>01 · Upload Portfolio</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Drop your SIP CSV — columns: **ticker**, **monthly_sip**",
            type=["csv"],
            help="Use NSE symbols like HINDZINC.NS (or just HINDZINC — .NS auto-added)",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sip.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                sample_csv = f.read()
            st.download_button(
                "Download Sample CSV",
                data=sample_csv,
                file_name="sample_sip.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if uploaded is None:
        st.markdown("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:1rem 1.5rem;color:#64748B;font-size:0.85rem;">
            Download the sample CSV above to get started with
            <strong style="color:#475569;">HINDZINC, ITC, VEDL, TCS, INFY</strong> and 7 more NSE dividend stocks.
        </div>
        """, unsafe_allow_html=True)
        return None

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "monthly_sip" not in df.columns:
        st.error("CSV must have columns: **ticker** and **monthly_sip**")
        return None

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["monthly_sip"] = pd.to_numeric(df["monthly_sip"], errors="coerce")
    df = df.dropna(subset=["monthly_sip"])
    df = df[df["monthly_sip"] > 0]
    if df.empty:
        st.error("No valid holdings found in CSV.")
        return None

    df["ticker"] = df["ticker"].apply(_validate_ticker_ns)

    # Portfolio summary cards
    total_sip = df["monthly_sip"].sum()
    n = len(df)
    st.markdown(f"""
    <div style="display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap;">
        <div style="background:#FFFFFF;border:1px solid #E8EDFB;
            border-radius:12px;padding:0.8rem 1.2rem;flex:1;min-width:140px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#94A3B8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;">Holdings</div>
            <div style="color:#6366F1;font-size:1.6rem;font-weight:700;">{n}</div>
        </div>
        <div style="background:#FFFFFF;border:1px solid #E8EDFB;
            border-radius:12px;padding:0.8rem 1.2rem;flex:1;min-width:140px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#94A3B8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;">Monthly SIP</div>
            <div style="color:#059669;font-size:1.6rem;font-weight:700;">&#8377;{total_sip:,.0f}</div>
        </div>
        <div style="background:#FFFFFF;border:1px solid #E8EDFB;
            border-radius:12px;padding:0.8rem 1.2rem;flex:1;min-width:140px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#94A3B8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;">Yearly SIP</div>
            <div style="color:#0891B2;font-size:1.6rem;font-weight:700;">&#8377;{total_sip*12:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive ticker cards
    max_sip = df["monthly_sip"].max()
    total_sip = df["monthly_sip"].sum()
    COLORS = ["#6366F1","#06B6D4","#10B981","#F59E0B","#F472B6","#A78BFA","#EF4444","#84CC16"]
    cards_html = '<div style="display:flex;flex-direction:column;gap:2px;margin-top:0.4rem;">'
    for idx, row in df.iterrows():
        t = str(row["ticker"]).replace(".NS", "")
        sip = float(row["monthly_sip"])
        pct = sip / total_sip * 100
        bar_w = int(sip / max_sip * 100)
        color = COLORS[idx % len(COLORS)]
        cards_html += (
            f'<div style="background:#FFFFFF;border:1px solid #F1F5F9;border-radius:8px;'
            f'padding:5px 10px;display:flex;align-items:center;gap:10px;">'
            f'<div style="background:{color}55;width:6px;height:28px;border-radius:3px;flex-shrink:0;"></div>'
            f'<span style="font-weight:600;color:#1E293B;font-size:0.82rem;width:90px;flex-shrink:0;">{t}</span>'
            f'<div style="flex:1;background:#F1F5F9;border-radius:99px;height:4px;">'
            f'<div style="background:{color}88;width:{bar_w}%;height:4px;border-radius:99px;"></div>'
            f'</div>'
            f'<span style="color:#94A3B8;font-size:0.72rem;width:36px;text-align:right;flex-shrink:0;">{pct:.0f}%</span>'
            f'<span style="color:{color}CC;font-weight:700;font-size:0.82rem;width:70px;text-align:right;flex-shrink:0;">&#8377;{sip:,.0f}</span>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
    return df


# ---------------------------------------------------------------------------
# Animated step renderer
# ---------------------------------------------------------------------------

STEP_META = [
    ("01", "Fetching NSE Market Data",     "Pulling 10Y price, dividends & financials from Yahoo Finance", "#D97706"),
    ("02", "Engineering 19 Features",      "ROE, D/E, FCF yield, dividend CAGR, sector encoding + more",  "#0891B2"),
    ("03", "Cleaning Feature Matrix",      "Batch median imputation for missing financial values",         "#64748B"),
    ("04", "Building RF Training Data",    "Rolling annual snapshots → 3Y forward dividend CAGR labels",  "#059669"),
    ("05", "Training Random Forest",       "200 trees · balanced class weights · 19 features",            "#059669"),
    ("06", "RF Predicting Probabilities",  "P(Increase) / P(Hold) / P(Reduce) per ticker",                "#059669"),
    ("07", "Building Dividend Series",     "Resampling to quarterly frequency for pattern model",         "#7C3AED"),
    ("08", "Training Pattern Model",       "MLP learning 10Y quarterly dividend momentum sequences",      "#7C3AED"),
    ("09", "Pattern Model Predicting",     "Sequence-based P(next quarter up) per ticker",                "#7C3AED"),
    ("10", "Scanning News Sentiment",      "FinBERT reading latest headlines for dividend signals",       "#DB2777"),
    ("11", "Ensemble Combining",           "Weighted blend: RF 70% · Pattern 15% · News 10% · Prior 5%", "#D97706"),
    ("12", "Generating Recommendations",   "Translating scores to SIP actions with reasoning",            "#059669"),
]


def _render_step_card(num: str, title: str, detail: str, color: str,
                      status: str, detail_lines: List[str]):
    import html as _html

    if status == "done":
        card_class = "step-card"
        card_style = (
            f"background:#FFFFFF;border:1px solid {color}22;"
            f"border-left:4px solid {color};border-radius:12px;"
            f"padding:0.65rem 1rem;margin-bottom:6px;"
        )
        num_style = (
            f"background:{color};color:#FFFFFF;font-size:0.85rem;font-weight:700;"
            f"min-width:28px;height:28px;border-radius:99px;display:flex;"
            f"align-items:center;justify-content:center;flex-shrink:0;"
        )
        num_content = "&#10003;"  # checkmark
        title_style = f"color:#1E293B;font-size:0.84rem;font-weight:600;"
        badge_html = (
            f'<span style="background:{color}18;color:{color};font-size:0.6rem;'
            f'font-weight:700;padding:2px 8px;border-radius:99px;letter-spacing:0.04em;">done</span>'
        )
        detail_color = "#94A3B8"

    elif status == "running":
        card_class = "step-card step-card-running"
        card_style = (
            f"background:linear-gradient(to right,{color}06,#FFFFFF);"
            f"border:1px solid {color}55;border-left:4px solid {color};"
            f"border-radius:12px;padding:0.65rem 1rem;margin-bottom:6px;"
            f"scroll-margin-top:80px;"
        )
        num_style = (
            f"background:{color}18;color:{color};font-size:0.68rem;font-weight:800;"
            f"min-width:28px;height:28px;border-radius:99px;display:flex;"
            f"align-items:center;justify-content:center;flex-shrink:0;"
        )
        num_content = num
        title_style = f"color:#0F172A;font-size:0.84rem;font-weight:700;"
        badge_html = (
            f'<span style="color:{color};font-size:0.6rem;font-weight:700;'
            f'display:inline-flex;align-items:center;gap:3px;">'
            f'<span class="step-dot" style="background:{color};"></span>'
            f'<span class="step-dot" style="background:{color};"></span>'
            f'<span class="step-dot" style="background:{color};"></span>'
            f'</span>'
        )
        detail_color = "#64748B"

    else:  # pending
        card_class = "step-card"
        card_style = (
            "background:#FAFAFA;border:1px solid #F1F5F9;"
            "border-radius:12px;padding:0.65rem 1rem;margin-bottom:6px;"
        )
        num_style = (
            "background:#F1F5F9;color:#CBD5E1;font-size:0.68rem;font-weight:700;"
            "min-width:28px;height:28px;border-radius:99px;display:flex;"
            "align-items:center;justify-content:center;flex-shrink:0;"
        )
        num_content = num
        title_style = "color:#CBD5E1;font-size:0.84rem;font-weight:500;"
        badge_html = ""
        detail_color = "#E2E8F0"

    sub_html = ""
    for item in detail_lines:
        # item is either (msg, is_raw_html) tuple or a plain string (legacy)
        if isinstance(item, tuple):
            msg_text, is_raw = item
        else:
            msg_text, is_raw = item, False
        content = str(msg_text) if is_raw else _html.escape(str(msg_text))
        sub_html += (
            f'<div style="color:#64748B;font-size:0.72rem;margin-top:4px;'
            f'padding:2px 0 2px 8px;border-left:2px solid #E2E8F0;">'
            f'{content}</div>'
        )

    anchor = ' id="step-running"' if status == "running" else ""
    card = (
        f'<div class="{card_class}"{anchor} style="{card_style}">'
        f'<div style="display:flex;align-items:center;gap:0.75rem;">'
        f'<div style="{num_style}">{num_content}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1px;">'
        f'<span style="{title_style}">{_html.escape(title)}</span>'
        f'{badge_html}'
        f'</div>'
        f'<div style="color:{detail_color};font-size:0.72rem;">{_html.escape(detail)}</div>'
        f'{sub_html}'
        f'</div></div></div>'
    )
    st.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 2 — Pipeline
# ---------------------------------------------------------------------------

def run_analysis_pipeline(
    sip_df: pd.DataFrame,
    weight_overrides: Dict[str, float],
    news_lookback_days: int = 90,
) -> Optional[Tuple]:

    (
        MODEL_WEIGHTS, HIST_YEARS,
        batch_fetch, validate_ticker_ns,
        build_feature_matrix, impute_features, build_dividend_series,
        DividendRandomForest, DividendLSTM, FinBERTSentimentAnalyzer,
        EnsembleCombiner, ModelOutputs,
        PortfolioAdvisor,
    ) = _import_pipeline()

    tickers = sip_df["ticker"].tolist()

    # Custom progress indicator (no bar — just pct + label)
    progress_bar = st.empty()

    def _set_progress(pct: float, label: str):
        pct_int = int(pct * 100)
        done = pct_int >= 100
        pct_color = "#059669" if done else "#6366F1"
        pct_label = "Complete" if done else f"Step {max(1, round(pct * 13))} of 13"
        progress_bar.markdown(
            f'<div style="display:flex;align-items:baseline;gap:0.6rem;'
            f'padding:0.3rem 0 0.5rem 0;">'
            f'<span style="color:{pct_color};font-size:1.5rem;font-weight:800;'
            f'line-height:1;font-family:Inter,sans-serif;">{pct_int}%</span>'
            f'<span style="color:#94A3B8;font-size:0.72rem;font-weight:400;">'
            f'{pct_label} &nbsp;·&nbsp; {label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _set_progress(0, "Initialising AI engines...")

    # Full-width step slots — one per pipeline step
    step_slots = [st.empty() for _ in range(12)]

    # data_lines: (step_n, msg_or_html, is_raw_html)
    data_lines: List[tuple] = []
    _current_running: List[int] = [0]  # mutable container so closures can update it

    def refresh_steps(running_idx: int):
        _current_running[0] = running_idx
        for i, (num, title, detail, color) in enumerate(STEP_META):
            if i < running_idx:
                status = "done"
            elif i == running_idx:
                status = "running"
            else:
                status = "pending"
            lines = [(m, raw) for sn, m, raw in data_lines if sn == i + 1]
            with step_slots[i]:
                _render_step_card(num, title, detail, color, status, lines)

    def log_data(step_n: int, msg: str, raw_html: bool = False):
        """Append a line to a step card and re-render only that card."""
        data_lines.append((step_n, msg, raw_html))
        i = step_n - 1
        num, title, detail, color = STEP_META[i]
        # card is either running (if it's current) or done (if we're past it)
        status = "running" if i >= _current_running[0] else "done"
        lines = [(m, raw) for sn, m, raw in data_lines if sn == step_n]
        with step_slots[i]:
            _render_step_card(num, title, detail, color, status, lines)

    try:
        # Step 1: Fetch
        refresh_steps(0)
        _set_progress(1 / 12, "Fetching NSE market data...")
        all_ticker_data = batch_fetch(tickers)
        for t, td in all_ticker_data.items():
            has_fins = bool(not td.financials.empty) if hasattr(td, "financials") else False
            log_data(1, f"{t}: {len(td.history)} rows | {len(td.dividends)} dividends | fins={'Y' if has_fins else 'N'}")
        failed = [t for t, td in all_ticker_data.items() if td.history.empty]
        if failed:
            log_data(1, f"No data: {', '.join(failed)}")

        # Step 2: Features
        refresh_steps(1)
        _set_progress(2 / 12, "Engineering financial features...")
        feature_df = build_feature_matrix(all_ticker_data)
        log_data(2, f"Matrix: {feature_df.shape[0]} stocks x {feature_df.shape[1]} features")

        # Step 3: Impute
        refresh_steps(2)
        _set_progress(3 / 12, "Cleaning feature matrix...")
        clean_feature_df = impute_features(feature_df)
        nan_count = int(feature_df.isna().sum().sum())
        log_data(3, f"Imputed {nan_count} missing values via batch median")

        # Step 4: RF training data
        refresh_steps(3)
        _set_progress(4 / 12, "Building training dataset...")
        rf_model = DividendRandomForest()
        X_train, y_train = rf_model.build_training_data(all_ticker_data, clean_feature_df)
        if len(X_train):
            log_data(4, f"{len(X_train)} rows | Reduce:{int((y_train == 0).sum())} Hold:{int((y_train == 1).sum())} Increase:{int((y_train == 2).sum())}")
        else:
            log_data(4, "Insufficient history - uniform fallback active")

        # Step 5: Train RF
        refresh_steps(4)
        _set_progress(5 / 12, "Training Random Forest...")
        rf_model.train(X_train, y_train)
        log_data(5, f"Trained={rf_model.is_trained} | 200 estimators | balanced weights")

        # Step 6: RF predict
        refresh_steps(5)
        _set_progress(6 / 12, "RF predicting probabilities...")
        rf_probas = rf_model.predict_proba(clean_feature_df.values)
        for i, t in enumerate(tickers):
            log_data(6, f"{t}  down:{rf_probas[i][0]:.2f}  flat:{rf_probas[i][1]:.2f}  up:{rf_probas[i][2]:.2f}")

        # Step 7: Dividend series
        refresh_steps(6)
        _set_progress(7 / 12, "Building dividend time-series...")
        div_series_dict = {}
        for ticker, td in all_ticker_data.items():
            series = build_dividend_series(td)
            if len(series) > 0:
                div_series_dict[ticker] = series
                log_data(7, f"{ticker}: {len(series)} quarters")
            else:
                log_data(7, f"{ticker}: no dividend history")

        # Step 8: Train sequence model
        refresh_steps(7)
        _set_progress(8 / 12, "Training pattern model...")
        lstm_model = DividendLSTM()
        X_seq, y_seq = lstm_model.prepare_sequences(div_series_dict)
        log_data(8, f"{len(X_seq) if X_seq is not None else 0} sequences")
        lstm_model.train(X_seq, y_seq)
        log_data(8, f"Trained={lstm_model.is_trained}")

        # Step 9: Sequence predict
        refresh_steps(8)
        _set_progress(9 / 12, "Pattern model predicting...")
        lstm_probas = {}
        for ticker in tickers:
            series = div_series_dict.get(ticker)
            lstm_probas[ticker] = lstm_model.predict_proba(series) if series is not None else 0.5
            log_data(9, f"{ticker}: {lstm_probas[ticker]:.3f}")

        # Step 10: FinBERT — fetch headlines per ticker and stream them to the live feed
        refresh_steps(9)
        _set_progress(10 / 12, "Scanning news sentiment...")
        finbert = FinBERTSentimentAnalyzer(lookback_days=news_lookback_days)
        finbert_scores = {}
        news_articles_map = {}   # ticker → list of {title, url, score, label}
        for ticker in tickers:
            import html as _html
            articles = finbert.fetch_news_with_urls(ticker)  # list of {title, url}
            headlines = [a["title"] for a in articles]
            ticker_safe = _html.escape(ticker)
            if articles:
                log_data(10, f"{ticker_safe}  {len(articles)} headline{'s' if len(articles) != 1 else ''} found", raw_html=True)
                for idx_a, art in enumerate(articles, 1):
                    title_safe = _html.escape(art["title"][:72] + ("..." if len(art["title"]) > 72 else ""))
                    url = art.get("url", "")
                    if url:
                        link_html = (
                            f'<span style="color:#94A3B8;">[{idx_a}]</span> '
                            f'<a href="{_html.escape(url)}" target="_blank" '
                            f'style="color:#6366F1;text-decoration:none;">{title_safe}</a>'
                        )
                    else:
                        link_html = f'<span style="color:#94A3B8;">[{idx_a}]</span> {title_safe}'
                    log_data(10, link_html, raw_html=True)
            else:
                log_data(10, f"{ticker_safe}  no headlines in last {news_lookback_days}d", raw_html=True)
            score = finbert.get_sentiment_score(ticker, headlines)
            finbert_scores[ticker] = score
            s_color = "#059669" if score > 0.6 else ("#DC2626" if score < 0.4 else "#94A3B8")
            s_label = "bullish" if score > 0.6 else ("bearish" if score < 0.4 else "neutral")
            log_data(10,
                f'<strong style="color:{s_color};">{ticker_safe}: {score:.3f} ({s_label})</strong>',
                raw_html=True)
            news_articles_map[ticker] = [
                {**a, "sentiment_score": score, "sentiment_label": s_label}
                for a in articles
            ]
        st.session_state["news_articles_map"] = news_articles_map

        # Step 11: Ensemble
        refresh_steps(10)
        _set_progress(11 / 12, "Combining all models...")
        combiner = EnsembleCombiner(weights=weight_overrides)
        model_outputs = []
        for i, ticker in enumerate(tickers):
            mo = ModelOutputs(
                ticker=ticker,
                rf_proba=rf_probas[i] if i < len(rf_probas) else np.array([1 / 3, 1 / 3, 1 / 3]),
                lstm_proba=lstm_probas.get(ticker, 0.5),
                finbert_score=finbert_scores.get(ticker, 0.5),
                rf_available=rf_model.is_trained,
                lstm_available=lstm_model.is_trained,
                finbert_available=finbert.is_loaded,
            )
            model_outputs.append(mo)
        ensemble_results = combiner.combine_batch(model_outputs)
        for r in ensemble_results:
            arrow = "^" if r.recommendation == "Increase" else ("v" if r.recommendation == "Reduce" else "=")
            adj_str = f"{r.sip_adjustment_pct:+.0%}" if r.sip_adjustment_pct != 0 else "±0%"
            log_data(11, f"{r.ticker}: score={r.ensemble_score:.3f} {arrow} {r.recommendation} (SIP {adj_str})")

        # Step 12: Recommendations
        refresh_steps(11)
        _set_progress(12 / 12, "Generating SIP recommendations...")
        advisor = PortfolioAdvisor(ensemble_results)
        recommendations = advisor.generate_recommendations(sip_df)
        for r in recommendations:
            log_data(12, f"{r.ticker}: Rs{r.current_monthly_sip:.0f} -> Rs{r.recommended_monthly_sip:.0f}")

        # All done
        refresh_steps(12)
        _set_progress(1.0, "Analysis complete — scroll down for results!")

        # Persist step data so cards stay visible after st.rerun()
        st.session_state["pipeline_step_data"] = data_lines[:]

        return ensemble_results, all_ticker_data, recommendations, rf_model

    except Exception as e:
        import traceback
        log_data(1, f"ERROR: {e}")
        log_data(1, traceback.format_exc()[:500])
        progress_bar.empty()
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        return None


# ---------------------------------------------------------------------------
# Section 3 — Recommendations
# ---------------------------------------------------------------------------

def render_recommendations(recommendations):
    import html as html_lib

    from analysis.portfolio import PortfolioAdvisor
    advisor = PortfolioAdvisor([])
    totals = advisor.get_portfolio_totals(recommendations)

    # Header row: title | sort | download
    h_title, h_sort, h_dl = st.columns([4, 2, 1])
    with h_title:
        st.markdown("<h2 style='margin-bottom:0;'>03 · SIP Recommendations</h2>", unsafe_allow_html=True)
    with h_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["AI Score (high to low)", "AI Score (low to high)",
             "Ticker A-Z", "Current SIP (high to low)", "Change (high to low)"],
            label_visibility="collapsed",
        )
    with h_dl:
        # Build CSV first so button is always present
        csv_rows = []
        for r in recommendations:
            csv_rows.append({
                "Ticker": r.ticker,
                "Recommendation": r.recommendation,
                "AI Score": round(r.ensemble_score, 3),
                "Confidence": f"{int(r.confidence * 100)}%",
                "Current Monthly SIP (Rs)": int(r.current_monthly_sip),
                "Recommended Monthly SIP (Rs)": int(r.recommended_monthly_sip),
                "Change (Rs)": int(r.recommended_monthly_sip - r.current_monthly_sip),
                "Reasoning": r.reasoning,
            })
        csv_df = pd.DataFrame(csv_rows)
        st.download_button(
            "Export CSV",
            data=csv_df.to_csv(index=False),
            file_name="sip_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Sort
    sort_key_map = {
        "AI Score (high to low)":    lambda r: -r.ensemble_score,
        "AI Score (low to high)":    lambda r: r.ensemble_score,
        "Ticker A-Z":                lambda r: r.ticker,
        "Current SIP (high to low)": lambda r: -r.current_monthly_sip,
        "Change (high to low)":      lambda r: -(r.recommended_monthly_sip - r.current_monthly_sip),
    }
    sorted_recs = sorted(recommendations, key=sort_key_map[sort_by])

    # Metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Monthly SIP", f"Rs {totals['current_total_sip']:,.0f}")
    _chg = totals['change_amount']
    c2.metric("Recommended Monthly SIP", f"Rs {totals['recommended_total_sip']:,.0f}",
              delta=f"{'+' if _chg >= 0 else '-'}Rs {abs(_chg):,.0f}")
    c3.metric("Portfolio Change", f"{totals['change_pct']:+.1%}")
    c4.metric("Holdings Analyzed", str(len(recommendations)))

    st.markdown("<br>", unsafe_allow_html=True)

    # Horizontal stacked cards
    ACTION_CONFIG = {
        "Increase": {
            "color": "#059669", "bg": "#F0FDF4", "border": "#BBF7D0",
            "label": "Increase SIP", "badge_bg": "#DCFCE7",
        },
        "Hold": {
            "color": "#B45309", "bg": "#FFFBEB", "border": "#FDE68A",
            "label": "Hold SIP", "badge_bg": "#FEF9C3",
        },
        "Reduce": {
            "color": "#DC2626", "bg": "#FFF1F2", "border": "#FECACA",
            "label": "Reduce SIP", "badge_bg": "#FFE4E6",
        },
    }

    for rec in sorted_recs:
        cfg = ACTION_CONFIG.get(rec.recommendation, ACTION_CONFIG["Hold"])
        bar_pct = int(rec.ensemble_score * 100)
        conf_pct = int(rec.confidence * 100)
        change_amt = int(rec.recommended_monthly_sip - rec.current_monthly_sip)
        change_str = f"+Rs {change_amt:,}" if change_amt >= 0 else f"-Rs {abs(change_amt):,}"
        change_color = "#059669" if change_amt > 0 else ("#DC2626" if change_amt < 0 else "#64748B")
        # Escape reasoning to prevent HTML breakage
        reasoning_safe = html_lib.escape(rec.reasoning)

        ticker_short = html_lib.escape(rec.ticker.replace('.NS', ''))
        card_html = (
            f'<div style="background:{cfg["bg"]};border:1px solid {cfg["border"]};'
            f'border-radius:14px;padding:1rem 1.4rem;margin-bottom:10px;">'

            f'<div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">'

            f'<div style="min-width:130px;">'
            f'<div style="color:#64748B;font-size:0.67rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:4px;">{ticker_short}</div>'
            f'<span style="background:{cfg["badge_bg"]};color:{cfg["color"]};'
            f'font-size:0.75rem;font-weight:700;padding:3px 10px;border-radius:99px;">'
            f'{html_lib.escape(cfg["label"].upper())}</span>'
            f'</div>'

            f'<div style="display:flex;align-items:center;gap:1rem;flex:1;min-width:220px;">'
            f'<div><div style="color:#94A3B8;font-size:0.62rem;text-transform:uppercase;">Current</div>'
            f'<div style="color:#475569;font-size:1.1rem;font-weight:600;">&#8377;{rec.current_monthly_sip:,.0f}</div></div>'
            f'<div style="color:#CBD5E1;font-size:1.3rem;">&#8594;</div>'
            f'<div><div style="color:#94A3B8;font-size:0.62rem;text-transform:uppercase;">Recommended</div>'
            f'<div style="color:{cfg["color"]};font-size:1.1rem;font-weight:700;">&#8377;{rec.recommended_monthly_sip:,.0f}</div></div>'
            f'<div style="margin-left:0.5rem;">'
            f'<div style="color:#94A3B8;font-size:0.62rem;text-transform:uppercase;">Change</div>'
            f'<div style="color:{change_color};font-size:0.95rem;font-weight:700;">{html_lib.escape(change_str)}</div></div>'
            f'</div>'

            f'<div style="text-align:right;min-width:100px;">'
            f'<div style="color:#94A3B8;font-size:0.62rem;text-transform:uppercase;margin-bottom:2px;">AI Score</div>'
            f'<div style="color:{cfg["color"]};font-size:1.5rem;font-weight:800;line-height:1;">{rec.ensemble_score:.2f}</div>'
            f'<div style="color:#94A3B8;font-size:0.65rem;margin-top:2px;">Conf: '
            f'<strong style="color:{cfg["color"]};">{conf_pct}%</strong></div>'
            f'</div>'

            f'</div>'

            f'<div style="background:rgba(0,0,0,0.07);border-radius:99px;height:3px;margin:10px 0 8px 0;">'
            f'<div style="background:{cfg["color"]};height:3px;border-radius:99px;width:{bar_pct}%;"></div></div>'

            f'<div style="background:rgba(255,255,255,0.65);border-radius:8px;padding:6px 10px;'
            f'font-size:0.76rem;color:#475569;line-height:1.55;">{reasoning_safe}</div>'

            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.caption("Increase / Hold / Reduce · Not financial advice.")


# ---------------------------------------------------------------------------
# Section 4 — Backtest chart
# ---------------------------------------------------------------------------

def render_dividend_history(all_ticker_data):
    st.markdown("<h2>04 · Dividend History</h2>", unsafe_allow_html=True)

    tickers = list(all_ticker_data.keys())
    labels = [t.replace(".NS", "") for t in tickers]
    selected_label = st.selectbox("Select stock", labels, key="div_hist_select", label_visibility="collapsed")
    selected_ticker = tickers[labels.index(selected_label)]
    td = all_ticker_data[selected_ticker]

    if not td.has_dividends or td.dividends.empty:
        st.info(f"No dividend history available for {selected_label}.")
        return

    divs = td.dividends.copy()
    divs.index = pd.to_datetime(divs.index)
    if divs.index.tz is not None:
        divs.index = divs.index.tz_localize(None)
    divs = divs.sort_index()

    # Last 5 years
    cutoff = divs.index[-1] - pd.DateOffset(years=10)
    divs = divs[divs.index >= cutoff]

    # Group by year
    annual = divs.groupby(divs.index.year).sum().reset_index()
    annual.columns = ["Year", "Div_Per_Share"]
    annual["YoY_Growth"] = annual["Div_Per_Share"].pct_change() * 100
    annual["Payments"] = divs.groupby(divs.index.year).count().values

    # Bar chart with color coding
    bar_colors = []
    for i, row in annual.iterrows():
        if i == 0 or pd.isna(row["YoY_Growth"]):
            bar_colors.append("#C7D2FE")   # light indigo — first/unknown
        elif row["YoY_Growth"] > 0:
            bar_colors.append("#6EE7B7")   # light green — growth
        elif row["YoY_Growth"] < 0:
            bar_colors.append("#FCA5A5")   # light red — cut
        else:
            bar_colors.append("#E2E8F0")   # light gray — flat

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annual["Year"].astype(str),
        y=annual["Div_Per_Share"],
        marker_color=bar_colors,
        text=[f"Rs{v:.2f}" for v in annual["Div_Per_Share"]],
        textposition="outside",
        textfont=dict(color="#475569", size=11),
        hovertemplate="<b>%{x}</b><br>Rs %{y:.2f} per share<extra></extra>",
        name="Annual Dividend",
    ))
    fig.update_layout(
        paper_bgcolor="#F5F7FA", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color="#475569"),
        title=dict(text=f"{selected_label} — Annual Dividend Per Share (Last 5 Years)", font=dict(size=13, color="#1E293B")),
        xaxis=dict(gridcolor="#E2E8F0", showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="#E2E8F0", showgrid=True, zeroline=False, tickprefix="Rs "),
        showlegend=False, height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    table_rows = ""
    for _, row in annual.iterrows():
        yoy = row["YoY_Growth"]
        if pd.isna(yoy):
            yoy_str = "—"
            yoy_color = "#94A3B8"
        elif yoy > 0:
            yoy_str = f"+{yoy:.1f}%"
            yoy_color = "#059669"
        elif yoy < 0:
            yoy_str = f"{yoy:.1f}%"
            yoy_color = "#DC2626"
        else:
            yoy_str = "0.0%"
            yoy_color = "#94A3B8"
        table_rows += (
            f'<tr style="border-bottom:1px solid #F1F5F9;">'
            f'<td style="padding:6px 12px;font-weight:600;color:#1E293B;">{int(row["Year"])}</td>'
            f'<td style="padding:6px 12px;color:#475569;">&#8377;{row["Div_Per_Share"]:.2f}</td>'
            f'<td style="padding:6px 12px;color:{yoy_color};font-weight:600;">{yoy_str}</td>'
            f'<td style="padding:6px 12px;color:#64748B;">{int(row["Payments"])}x</td>'
            f'</tr>'
        )
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<thead><tr style="background:#F8FAFC;">'
        '<th style="padding:6px 12px;text-align:left;color:#64748B;font-weight:600;">Year</th>'
        '<th style="padding:6px 12px;text-align:left;color:#64748B;font-weight:600;">Div / Share</th>'
        '<th style="padding:6px 12px;text-align:left;color:#64748B;font-weight:600;">YoY Growth</th>'
        '<th style="padding:6px 12px;text-align:left;color:#64748B;font-weight:600;">Payments</th>'
        f'</tr></thead><tbody>{table_rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_backtest_chart(backtest_results):
    st.markdown("<h2>04 · SIP Performance &amp; 1-Year Projection</h2>", unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;'
        'padding:0.7rem 1rem;font-size:0.78rem;color:#1E40AF;margin-bottom:1rem;">'
        '<strong>How to read:</strong> Left of the dashed line = actual historical SIP performance (same for both). '
        'Right of the dashed line = 1-year projection using this stock\'s historical price trend, '
        'with AI-recommended SIP applied immediately. Diamonds = dividend reinvestments.</div>',
        unsafe_allow_html=True,
    )

    if not backtest_results:
        st.info("No backtest data available.")
        return

    tickers = list(backtest_results.keys())
    labels = [t.replace(".NS", "") for t in tickers]
    selected_label = st.selectbox(
        "Select stock", labels, key="backtest_ticker_select",
        label_visibility="collapsed",
    )
    selected_ticker = tickers[labels.index(selected_label)]
    r = backtest_results[selected_ticker]

    COLOR_BASELINE = "#94A3B8"
    COLOR_AI = "#6366F1"
    COLOR_DIV = "#F59E0B"

    fig = go.Figure()

    # Baseline — dotted gray line
    fig.add_trace(go.Scatter(
        x=r.dates, y=r.baseline_values,
        name="Baseline SIP",
        line=dict(color=COLOR_BASELINE, dash="dot", width=2),
        hovertemplate="<b>Baseline</b><br>%{x}<br>Rs %{y:,.0f}<extra></extra>",
    ))

    # AI-guided — solid indigo line
    fig.add_trace(go.Scatter(
        x=r.dates, y=r.ai_adjusted_values,
        name="AI-Guided SIP",
        line=dict(color=COLOR_AI, width=2.5),
        hovertemplate="<b>AI-Guided</b><br>%{x}<br>Rs %{y:,.0f}<extra></extra>",
    ))

    # Dividend dots
    if r.dividend_events:
        div_dates = [e[0] for e in r.dividend_events]
        div_vals  = [e[1] for e in r.dividend_events]
        div_amts  = [e[2] for e in r.dividend_events]
        fig.add_trace(go.Scatter(
            x=div_dates, y=div_vals,
            name="Dividend Reinvested",
            mode="markers",
            marker=dict(color=COLOR_DIV, size=9, symbol="diamond",
                        line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="<b>Dividend</b><br>%{x}<br>Rs %{customdata:,.0f} reinvested<extra></extra>",
            customdata=div_amts,
        ))

    today_date = r.dates[r.today_index] if r.today_index < len(r.dates) else None

    shapes = []
    annotations = []
    if today_date:
        shapes.append(dict(
            type="line", x0=today_date, x1=today_date,
            y0=0, y1=1, yref="paper",
            line=dict(color="#64748B", width=1.5, dash="dash"),
        ))
        annotations.append(dict(
            x=today_date, y=1, yref="paper",
            text="Today", showarrow=False,
            font=dict(size=11, color="#64748B"),
            xanchor="left", yanchor="top",
            xshift=6,
        ))

    rec = r.cagr_ai - r.cagr_baseline
    alpha_color = "#059669" if rec >= 0 else "#DC2626"
    alpha_text = f"Projected alpha: {rec*100:+.1f}pp"

    fig.update_layout(
        paper_bgcolor="#F5F7FA",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color="#475569"),
        title=dict(
            text=f"{selected_label}  ·  hist CAGR {r.historical_cagr*100:.1f}%  ·  div yield {r.avg_div_yield*100:.1f}%  ·  <span style='color:{alpha_color}'>{alpha_text}</span>",
            font=dict(size=12, color="#1E293B"),
        ),
        xaxis=dict(gridcolor="#E2E8F0", showgrid=True, zeroline=False,
                   title_font=dict(color="#64748B")),
        yaxis=dict(gridcolor="#E2E8F0", showgrid=True, zeroline=False,
                   tickprefix="Rs ", title_font=dict(color="#64748B")),
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        legend=dict(
            bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1,
            font=dict(size=11), orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        height=460,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    from analysis.backtest import SIPBacktester
    summary_df = SIPBacktester().compute_portfolio_summary(backtest_results)

    def style_alpha(val):
        if isinstance(val, str) and val.startswith("+"):
            return "color:#059669;font-weight:700"
        if isinstance(val, str) and val.startswith("-"):
            return "color:#DC2626;font-weight:700"
        return ""

    styled = summary_df.style.map(style_alpha, subset=["Alpha"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 5 — Diagnostics
# ---------------------------------------------------------------------------

def render_diagnostics(rf_model, ensemble_results, all_ticker_data):
    with st.expander("Model Diagnostics & Data Quality", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("RF Feature Importances")
            fi = rf_model.get_feature_importance()
            if fi.empty:
                st.info("RF not trained.")
            else:
                fi_df = fi.reset_index()
                fi_df.columns = ["Feature", "Importance"]
                fi_df["Importance"] = fi_df["Importance"].map(lambda x: f"{x:.4f}")
                st.dataframe(fi_df, hide_index=True, use_container_width=True)

        with c2:
            st.subheader("Model Weights Used")
            if ensemble_results:
                r = ensemble_results[0]
                w = r.model_weights_used
                wdf = pd.DataFrame({
                    "Model": list(w.keys()),
                    "Weight": [f"{v:.0%}" for v in w.values()],
                })
                st.dataframe(wdf, hide_index=True)
                st.subheader("Score Breakdown (top stock)")
                sdf = pd.DataFrame({
                    "Model": ["Random Forest", "Pattern Memory", "FinBERT", "Ensemble"],
                    "Score": [
                        f"{r.rf_score:.3f}",
                        f"{r.lstm_score:.3f}",
                        f"{r.finbert_score:.3f}",
                        f"{r.ensemble_score:.3f}",
                    ],
                })
                st.dataframe(sdf, hide_index=True)

        # News contributions per ticker
        st.subheader("News Sentiment — Articles Scanned")
        news_map = st.session_state.get("news_articles_map", {})
        if not news_map:
            st.info("No news data available.")
        else:
            tickers_with_news = list(news_map.keys())
            labels = [t.replace(".NS", "") for t in tickers_with_news]
            selected = st.selectbox("Ticker", labels, key="diag_news_select", label_visibility="collapsed")
            sel_ticker = tickers_with_news[labels.index(selected)]
            articles = news_map.get(sel_ticker, [])
            if not articles:
                st.info(f"No headlines found for {selected}.")
            else:
                score = finbert_scores_from_articles = articles[0].get("sentiment_score", 0.5)
                label = articles[0].get("sentiment_label", "neutral")
                s_color = "#059669" if score > 0.6 else ("#DC2626" if score < 0.4 else "#94A3B8")
                st.markdown(
                    f'<p style="font-size:0.8rem;color:#64748B;">Overall sentiment: '
                    f'<strong style="color:{s_color};">{label} ({score:.3f})</strong> '
                    f'from {len(articles)} article{"s" if len(articles) != 1 else ""}</p>',
                    unsafe_allow_html=True,
                )
                rows_html = ""
                for i, art in enumerate(articles, 1):
                    title = art.get("title", "")
                    url = art.get("url", "")
                    title_cell = (
                        f'<a href="{url}" target="_blank" '
                        f'style="color:#6366F1;text-decoration:none;font-size:0.8rem;">{title}</a>'
                        if url else f'<span style="font-size:0.8rem;color:#475569;">{title}</span>'
                    )
                    rows_html += (
                        f'<tr style="border-bottom:1px solid #F1F5F9;">'
                        f'<td style="padding:5px 8px;color:#94A3B8;font-size:0.75rem;width:24px;">{i}</td>'
                        f'<td style="padding:5px 8px;">{title_cell}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<tbody>{rows_html}</tbody></table>',
                    unsafe_allow_html=True,
                )

        st.subheader("Data Fetch Quality")
        rows = []
        for t, td in all_ticker_data.items():
            has_price = not td.history.empty if hasattr(td, "history") else False
            has_fins = not td.financials.empty if hasattr(td, "financials") else False
            has_divs = len(td.dividends) > 0 if hasattr(td, "dividends") else False
            rows.append({
                "Ticker": t,
                "Prices":     "Y" if has_price else "N",
                "Financials": "Y" if has_fins else "N",
                "Dividends":  "Y" if has_divs else "N",
                "Errors": ", ".join(td.fetch_errors) if td.fetch_errors else "None",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _inject_css()
    render_hero()

    weight_overrides, news_lookback_days = render_sidebar()
    sip_df = render_upload_section()
    if sip_df is None:
        return

    st.divider()
    st.markdown("<h2>02 · Run AI Analysis</h2>", unsafe_allow_html=True)

    analyzing = st.session_state.get("analyzing", False)

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        if analyzing:
            st.button("Analyzing...", disabled=True, type="primary", use_container_width=True)
        else:
            run_clicked = st.button("Analyze Portfolio", type="primary", use_container_width=True)
            if run_clicked:
                st.session_state["analyzing"] = True
                st.rerun()
    with info_col:
        if analyzing:
            st.markdown(
                '<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:10px;'
                'padding:0.75rem 1rem;color:#4338CA;font-size:0.82rem;">'
                'Running all 4 AI models — please wait, this takes ~30 seconds...</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
                'padding:0.75rem 1rem;color:#64748B;font-size:0.82rem;">'
                'Takes <strong style="color:#475569;">~30 seconds</strong>'
                ' &nbsp;·&nbsp; All 4 AI models train on live NSE data'
                ' &nbsp;·&nbsp; Results cached in session</div>',
                unsafe_allow_html=True,
            )

    if analyzing:
        results = run_analysis_pipeline(sip_df, weight_overrides, news_lookback_days)
        if results is None:
            st.session_state["analyzing"] = False
            return
        st.session_state["pipeline_results"] = results
        st.session_state["last_sip_df"] = sip_df.copy()
        st.session_state["analyzing"] = False
        st.rerun()

    # Re-render completed step cards after rerun (persisted from last pipeline run)
    step_data = st.session_state.get("pipeline_step_data")
    if step_data and not analyzing:
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:0.6rem;'
            'padding:0.3rem 0 0.5rem 0;">'
            '<span style="color:#059669;font-size:1.5rem;font-weight:800;'
            'line-height:1;font-family:Inter,sans-serif;">100%</span>'
            '<span style="color:#94A3B8;font-size:0.72rem;font-weight:400;">'
            'Complete &nbsp;·&nbsp; Analysis complete — scroll down for results!</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        for i, (num, title, detail, color) in enumerate(STEP_META):
            lines = [(m, raw) for sn, m, raw in step_data if sn == i + 1]
            _render_step_card(num, title, detail, color, "done", lines)

    if "pipeline_results" not in st.session_state:
        return

    ensemble_results, all_ticker_data, recommendations, rf_model = (
        st.session_state["pipeline_results"]
    )

    st.divider()
    render_recommendations(recommendations)

    st.divider()
    render_dividend_history(all_ticker_data)

    st.divider()
    render_diagnostics(rf_model, ensemble_results, all_ticker_data)

    st.divider()
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.72rem;text-align:center;'>"
        "DividendAI &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; MLP &nbsp;·&nbsp; "
        "HuggingFace FinBERT &nbsp;·&nbsp; yfinance &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly"
        "<br>Not financial advice. Consult a SEBI-registered advisor before changing SIP allocations.</p>",
        unsafe_allow_html=True,
    )


main()
