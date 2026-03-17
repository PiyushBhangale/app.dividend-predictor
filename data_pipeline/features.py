"""data_pipeline/features.py — Feature engineering for the dividend predictor.

Transforms raw TickerData into the 19-feature numeric vector consumed by the
Random Forest model. Also produces quarterly dividend time-series for LSTM.

Key design decisions:
- Every feature function returns NaN when data is insufficient — never raises.
- Imputation is batch-wise (across all tickers) to avoid per-ticker leakage.
- Dividend series is quarterly-resampled and gap-filled with 0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import FEATURE_NAMES, SECTOR_MAP, HIST_YEARS
from data_pipeline.fetcher import TickerData
from utils.logger import get_logger, log_missing_data

logger = get_logger("features")

NAN = float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_vector(td: TickerData) -> Dict[str, float]:
    """Compute all 19 features for a single ticker.

    Returns a dict with every key from FEATURE_NAMES.
    Any feature that cannot be computed is NaN — impute_features() handles it.
    """
    info = td.info
    ticker = td.ticker

    features: Dict[str, float] = {}

    # 1. ROE
    features["roe"] = _safe_info(info, "returnOnEquity", ticker, "roe")

    # 2. Debt / Equity
    features["debt_equity"] = _safe_info(info, "debtToEquity", ticker, "debt_equity")
    if not math.isnan(features["debt_equity"]):
        features["debt_equity"] = features["debt_equity"] / 100.0  # yf returns as %

    # 3. Current dividend yield
    features["dividend_yield"] = _safe_info(info, "dividendYield", ticker, "dividend_yield")

    # 4 & 5. Dividend CAGR (3Y and 5Y)
    div_series = _annual_dividend_totals(td.dividends)
    features["div_cagr_3y"] = compute_cagr(div_series, years=3)
    features["div_cagr_5y"] = compute_cagr(div_series, years=5)

    # 6. Payout ratio
    features["payout_ratio"] = _safe_info(info, "payoutRatio", ticker, "payout_ratio")

    # 7. EPS growth 3Y
    features["eps_growth_3y"] = _income_stmt_growth(td.financials, "Net Income", years=3)

    # 8. Revenue growth 3Y
    features["revenue_growth_3y"] = _income_stmt_growth(td.financials, "Total Revenue", years=3)

    # 9. OCF margin
    features["ocf_margin"] = _ocf_margin(td)

    # 10. FCF yield
    features["fcf_yield"] = _fcf_yield(td)

    # 11. Log market cap
    mc = _safe_info(info, "marketCap", ticker, "log_market_cap")
    features["log_market_cap"] = math.log(mc) if not math.isnan(mc) and mc > 0 else NAN

    # 12. P/E ratio
    features["pe_ratio"] = _safe_info(info, "trailingPE", ticker, "pe_ratio")

    # 13. P/B ratio
    features["pb_ratio"] = _safe_info(info, "priceToBook", ticker, "pb_ratio")

    # 14. Current ratio
    features["current_ratio"] = _safe_info(info, "currentRatio", ticker, "current_ratio")

    # 15. Interest coverage
    features["interest_coverage"] = _interest_coverage(td)

    # 16. Net profit margin
    features["net_profit_margin"] = _safe_info(info, "profitMargins", ticker, "net_profit_margin")

    # 17. Asset turnover
    features["asset_turnover"] = _asset_turnover(td)

    # 18. Sector encoded
    features["sector_encoded"] = float(encode_sector(info))

    # 19. Consecutive dividend years
    features["consecutive_div_years"] = float(_consecutive_div_years(td.dividends))

    return features


def build_feature_matrix(all_ticker_data: Dict[str, TickerData]) -> pd.DataFrame:
    """Build a (n_tickers × 19) feature DataFrame for the full batch."""
    rows = {}
    for ticker, td in all_ticker_data.items():
        rows[ticker] = build_feature_vector(td)
    df = pd.DataFrame(rows).T  # shape: (n_tickers, 19)
    df = df[FEATURE_NAMES]     # enforce canonical column order
    return df


def impute_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Batch-wise median imputation for NaN values.

    Applied column-by-column. Falls back to 0.0 if the entire column is NaN
    (e.g., every ticker is missing P/E).
    """
    df = feature_df.copy()
    for col in df.columns:
        if df[col].isna().all():
            df[col] = 0.0
            logger.warning("Column %s entirely NaN — filled with 0.0", col)
        elif df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def build_dividend_series(td: TickerData, freq: str = "Q") -> pd.Series:
    """Resample dividend history to the given frequency (default: quarterly).

    Gaps are filled with 0 (no dividend paid that period).
    Returns an empty Series if there is insufficient history.
    """
    if not td.has_dividends or len(td.dividends) < 2:
        return pd.Series(dtype=float)

    try:
        divs = td.dividends.copy()
        divs.index = pd.to_datetime(divs.index)
        # Sum dividends within each period (some stocks pay quarterly)
        resampled = divs.resample(freq).sum().fillna(0.0)
        return resampled
    except Exception as e:
        logger.warning("build_dividend_series failed for %s: %s", td.ticker, e)
        return pd.Series(dtype=float)


def compute_cagr(series: pd.Series, years: int) -> float:
    """Compute CAGR over the last `years` of annual dividend totals.

    Returns NaN if:
    - fewer than (years + 1) data points are available
    - start value is 0 (can't compute growth from zero)
    """
    if series is None or len(series) < years + 1:
        return NAN
    try:
        end_val = series.iloc[-1]
        start_val = series.iloc[-(years + 1)]
        if start_val <= 0:
            return NAN
        return (end_val / start_val) ** (1.0 / years) - 1.0
    except Exception:
        return NAN


def encode_sector(info: Dict) -> int:
    """Map yfinance 'sector' string to integer via SECTOR_MAP."""
    sector = info.get("sector", "Unknown")
    return SECTOR_MAP.get(sector, SECTOR_MAP["Unknown"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_info(info: Dict, key: str, ticker: str, feature_name: str) -> float:
    """Get a numeric value from the info dict, returning NaN if missing/None."""
    val = info.get(key)
    if val is None:
        log_missing_data(ticker, feature_name, NAN)
        return NAN
    try:
        return float(val)
    except (TypeError, ValueError):
        log_missing_data(ticker, feature_name, NAN)
        return NAN


def _annual_dividend_totals(dividends: pd.Series) -> pd.Series:
    """Aggregate dividends into annual totals."""
    if dividends is None or len(dividends) == 0:
        return pd.Series(dtype=float)
    try:
        divs = dividends.copy()
        divs.index = pd.to_datetime(divs.index)
        return divs.resample("A").sum()
    except Exception:
        return pd.Series(dtype=float)


def _income_stmt_growth(financials: pd.DataFrame, row_key: str, years: int) -> float:
    """Compute CAGR for an income statement line item over `years` years.

    yfinance financials columns are in descending date order (most recent first).
    """
    if financials.empty:
        return NAN

    # Find row by partial match (handles minor name variations)
    matching = [r for r in financials.index if row_key.lower() in r.lower()]
    if not matching:
        return NAN

    row = financials.loc[matching[0]]
    # Drop NaN and convert to numeric
    row = pd.to_numeric(row, errors="coerce").dropna()

    if len(row) < years + 1:
        return NAN

    # Columns are newest→oldest; reverse for chronological order
    row = row.iloc[::-1]
    return compute_cagr(row, years=years)


def _ocf_margin(td: TickerData) -> float:
    """Operating Cash Flow / Total Revenue."""
    if td.cashflow.empty or td.financials.empty:
        return NAN
    try:
        ocf_row = _find_row(td.cashflow, "Operating Cash Flow")
        rev_row = _find_row(td.financials, "Total Revenue")
        if ocf_row is None or rev_row is None:
            return NAN
        ocf = pd.to_numeric(ocf_row, errors="coerce").iloc[0]
        rev = pd.to_numeric(rev_row, errors="coerce").iloc[0]
        if rev == 0 or math.isnan(ocf) or math.isnan(rev):
            return NAN
        return float(ocf / rev)
    except Exception:
        return NAN


def _fcf_yield(td: TickerData) -> float:
    """Free Cash Flow / Market Cap."""
    mc = td.info.get("marketCap")
    if mc is None or mc == 0:
        return NAN
    if td.cashflow.empty:
        return NAN
    try:
        fcf_row = _find_row(td.cashflow, "Free Cash Flow")
        if fcf_row is None:
            # Fallback: OCF - Capex
            ocf_row = _find_row(td.cashflow, "Operating Cash Flow")
            cap_row = _find_row(td.cashflow, "Capital Expenditure")
            if ocf_row is None or cap_row is None:
                return NAN
            ocf = pd.to_numeric(ocf_row, errors="coerce").iloc[0]
            cap = pd.to_numeric(cap_row, errors="coerce").iloc[0]
            fcf = ocf - abs(cap)
        else:
            fcf = pd.to_numeric(fcf_row, errors="coerce").iloc[0]
        if math.isnan(fcf):
            return NAN
        return float(fcf / mc)
    except Exception:
        return NAN


def _interest_coverage(td: TickerData) -> float:
    """EBIT / Interest Expense."""
    if td.financials.empty:
        return NAN
    try:
        ebit_row = _find_row(td.financials, "EBIT")
        int_row = _find_row(td.financials, "Interest Expense")
        if ebit_row is None or int_row is None:
            return NAN
        ebit = pd.to_numeric(ebit_row, errors="coerce").iloc[0]
        interest = abs(pd.to_numeric(int_row, errors="coerce").iloc[0])
        if interest == 0 or math.isnan(ebit) or math.isnan(interest):
            return NAN
        return float(ebit / interest)
    except Exception:
        return NAN


def _asset_turnover(td: TickerData) -> float:
    """Revenue / Total Assets."""
    if td.financials.empty or td.balance_sheet.empty:
        return NAN
    try:
        rev_row = _find_row(td.financials, "Total Revenue")
        asset_row = _find_row(td.balance_sheet, "Total Assets")
        if rev_row is None or asset_row is None:
            return NAN
        rev = pd.to_numeric(rev_row, errors="coerce").iloc[0]
        assets = pd.to_numeric(asset_row, errors="coerce").iloc[0]
        if assets == 0 or math.isnan(rev) or math.isnan(assets):
            return NAN
        return float(rev / assets)
    except Exception:
        return NAN


def _consecutive_div_years(dividends: pd.Series) -> int:
    """Count consecutive calendar years (counting backward) with ≥1 dividend."""
    if dividends is None or len(dividends) == 0:
        return 0
    try:
        divs = dividends.copy()
        divs.index = pd.to_datetime(divs.index)
        annual = divs.resample("A").sum()
        count = 0
        for val in annual.iloc[::-1]:
            if val > 0:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _find_row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    """Find a row in a financial DataFrame by partial case-insensitive key match."""
    if df.empty:
        return None
    matches = [r for r in df.index if key.lower() in str(r).lower()]
    if not matches:
        return None
    return df.loc[matches[0]]
