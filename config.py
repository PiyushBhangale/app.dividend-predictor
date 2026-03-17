# config.py — Single source of truth for all tuneable parameters

# ---------------------------------------------------------------------------
# Model ensemble weights (must sum to 1.0)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = {
    "random_forest": 0.70,
    "lstm": 0.15,
    "finbert": 0.10,
    "base": 0.05,  # long-run dividend base rate prior
}

# ---------------------------------------------------------------------------
# Decision thresholds on ensemble score [0, 1]
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "increase": 0.58,   # score >= 0.58  → Increase SIP
    "hold":     0.42,   # 0.42 <= score < 0.58 → Hold
    # score < 0.42 → Reduce
}

# ---------------------------------------------------------------------------
# SIP adjustment multipliers applied at annual rebalance
# ---------------------------------------------------------------------------
SIP_ADJUSTMENTS = {
    "Increase": 0.15,   # +15% of current monthly SIP
    "Hold":     0.00,
    "Reduce":  -0.20,   # -20% of current monthly SIP
}

# ---------------------------------------------------------------------------
# 19 feature names in canonical order (used by RF and imputation)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "roe",
    "debt_equity",
    "dividend_yield",
    "div_cagr_3y",
    "div_cagr_5y",
    "payout_ratio",
    "eps_growth_3y",
    "revenue_growth_3y",
    "ocf_margin",
    "fcf_yield",
    "log_market_cap",
    "pe_ratio",
    "pb_ratio",
    "current_ratio",
    "interest_coverage",
    "net_profit_margin",
    "asset_turnover",
    "sector_encoded",
    "consecutive_div_years",
]

# ---------------------------------------------------------------------------
# Sector encoding map for NSE stocks
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "Basic Materials":        0,
    "Energy":                 1,
    "Financial Services":     2,
    "Healthcare":             3,
    "Industrials":            4,
    "Technology":             5,
    "Consumer Defensive":     6,
    "Consumer Cyclical":      7,
    "Utilities":              8,
    "Real Estate":            9,
    "Communication Services": 10,
    "Unknown":                11,
}

# ---------------------------------------------------------------------------
# LSTM configuration
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN = 8        # number of quarters in each input sequence
LSTM_EPOCHS  = 30       # training epochs
LSTM_BATCH   = 16       # batch size
MIN_SEQUENCES = 10      # minimum total sequences needed to train LSTM

# ---------------------------------------------------------------------------
# FinBERT configuration
# ---------------------------------------------------------------------------
FINBERT_MODEL          = "ProsusAI/finbert"
NEWS_MAX_ARTICLES      = 10  # max headlines per ticker
NEWS_LOOKBACK_DAYS     = 5  # only consider news published within this many days

# ---------------------------------------------------------------------------
# Data fetching configuration
# ---------------------------------------------------------------------------
HIST_YEARS           = 10   # years of historical data to download
MIN_DIVIDEND_RECORDS = 4    # min annual dividends required for LSTM series
DEFAULT_NS_SUFFIX    = ".NS"

# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------
BACKTEST_REBALANCE_FREQ = "A"   # annual SIP rebalance
BACKTEST_YEARS          = 3    # years of history to simulate (keep HIST_YEARS >= this)

# ---------------------------------------------------------------------------
# RF training configuration
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS  = 200
RF_RANDOM_STATE  = 42
RF_LABEL_CAGR_THRESHOLD = 0.05   # 3Y forward CAGR > 5% → label Increase

# ---------------------------------------------------------------------------
# Display rounding for recommended SIP amounts (INR)
# ---------------------------------------------------------------------------
SIP_ROUND_TO = 100   # round to nearest ₹100
