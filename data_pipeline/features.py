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

    Each yf.info-dependent feature has a statement-based fallback so the
    pipeline works correctly when yf.info is blocked (e.g. on Streamlit Cloud).
    """
    info = td.info
    ticker = td.ticker

    features: Dict[str, float] = {}

    # Precompute shared values used in multiple features
    current_price = _current_price(td)
    shares = _shares(td)

    # 1. ROE — fallback: Net Income / Stockholders Equity
    roe = _safe_info(info, "returnOnEquity", ticker, "roe")
    if math.isnan(roe):
        ni = _stmt_value(td.financials, "Net Income")
        eq = _stmt_value(td.balance_sheet, "Stockholders Equity")
        if not math.isnan(ni) and not math.isnan(eq) and eq != 0:
            roe = ni / eq
    features["roe"] = roe

    # 2. Debt / Equity — fallback: Total Debt / Stockholders Equity
    de = _safe_info(info, "debtToEquity", ticker, "debt_equity")
    if not math.isnan(de):
        de = de / 100.0  # yf returns as percentage
    else:
        debt = _stmt_value(td.balance_sheet, "Total Debt")
        eq   = _stmt_value(td.balance_sheet, "Stockholders Equity")
        if not math.isnan(debt) and not math.isnan(eq) and eq != 0:
            de = debt / eq
    features["debt_equity"] = de

    # 3. Dividend yield — fallback: annual DPS / current price
    dy = _safe_info(info, "dividendYield", ticker, "dividend_yield")
    if math.isnan(dy) and current_price > 0 and len(td.dividends) > 0:
        try:
            divs = td.dividends.copy()
            divs.index = pd.to_datetime(divs.index)
            annual_dps = float(divs.resample("A").sum().iloc[-1])
            dy = annual_dps / current_price
        except Exception:
            pass
    features["dividend_yield"] = dy

    # 4 & 5. Dividend CAGR (from dividends series — always reliable)
    div_series = _annual_dividend_totals(td.dividends)
    features["div_cagr_3y"] = compute_cagr(div_series, years=3)
    features["div_cagr_5y"] = compute_cagr(div_series, years=5)

    # 6. Payout ratio — fallback: Dividends Paid (cashflow) / Net Income
    pr = _safe_info(info, "payoutRatio", ticker, "payout_ratio")
    if math.isnan(pr):
        div_paid = abs(_stmt_value(td.cashflow, "Common Stock Dividend"))
        ni       = _stmt_value(td.financials, "Net Income")
        if math.isnan(div_paid):
            # alternate cashflow row name
            div_paid = abs(_stmt_value(td.cashflow, "Cash Dividends Paid"))
        if not math.isnan(div_paid) and not math.isnan(ni) and ni > 0:
            pr = div_paid / ni
    features["payout_ratio"] = pr

    # 7 & 8. Growth from income statement (always from financials)
    features["eps_growth_3y"]     = _income_stmt_growth(td.financials, "Net Income",     years=3)
    features["revenue_growth_3y"] = _income_stmt_growth(td.financials, "Total Revenue",  years=3)

    # 9. OCF margin (from cashflow + financials)
    features["ocf_margin"] = _ocf_margin(td)

    # 10. FCF yield — uses fast_info/price as market cap fallback
    features["fcf_yield"] = _fcf_yield(td, current_price, shares)

    # 11. Log market cap — fast_info fallback, then price × shares
    mc = _safe_info(info, "marketCap", ticker, "log_market_cap")
    if math.isnan(mc) or mc <= 0:
        mc = td.fast_info.get("market_cap") or 0
    if (mc is None or mc <= 0) and current_price > 0 and shares > 0:
        mc = current_price * shares
    features["log_market_cap"] = math.log(float(mc)) if mc and mc > 0 else NAN

    # 12. P/E ratio — fallback: price / (Net Income / shares)
    pe = _safe_info(info, "trailingPE", ticker, "pe_ratio")
    if math.isnan(pe) and current_price > 0 and shares > 0:
        ni = _stmt_value(td.financials, "Net Income")
        if not math.isnan(ni) and ni > 0:
            eps = ni / shares
            if eps > 0:
                pe = current_price / eps
    features["pe_ratio"] = pe

    # 13. P/B ratio — fallback: price / (Stockholders Equity / shares)
    pb = _safe_info(info, "priceToBook", ticker, "pb_ratio")
    if math.isnan(pb) and current_price > 0 and shares > 0:
        eq = _stmt_value(td.balance_sheet, "Stockholders Equity")
        if not math.isnan(eq) and eq > 0:
            bvps = eq / shares
            if bvps > 0:
                pb = current_price / bvps
    features["pb_ratio"] = pb

    # 14. Current ratio — fallback: Current Assets / Current Liabilities
    cr = _safe_info(info, "currentRatio", ticker, "current_ratio")
    if math.isnan(cr):
        ca = _stmt_value(td.balance_sheet, "Current Assets")
        cl = _stmt_value(td.balance_sheet, "Current Liabilities")
        if not math.isnan(ca) and not math.isnan(cl) and cl != 0:
            cr = ca / cl
    features["current_ratio"] = cr

    # 15. Interest coverage (from financials)
    features["interest_coverage"] = _interest_coverage(td)

    # 16. Net profit margin — fallback: Net Income / Total Revenue
    npm = _safe_info(info, "profitMargins", ticker, "net_profit_margin")
    if math.isnan(npm):
        ni  = _stmt_value(td.financials, "Net Income")
        rev = _stmt_value(td.financials, "Total Revenue")
        if not math.isnan(ni) and not math.isnan(rev) and rev != 0:
            npm = ni / rev
    features["net_profit_margin"] = npm

    # 17. Asset turnover (from financials + balance sheet)
    features["asset_turnover"] = _asset_turnover(td)

    # 18. Sector encoded (info only — "Unknown" if info blocked)
    features["sector_encoded"] = float(encode_sector(info))

    # 19. Consecutive dividend years (from dividends series — always reliable)
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


def _fcf_yield(td: TickerData, current_price: float = 0.0, shares: float = 0.0) -> float:
    """Free Cash Flow / Market Cap.

    Market cap priority: yf.info → fast_info → price × shares.
    """
    mc = td.info.get("marketCap") or td.fast_info.get("market_cap") or 0
    if (not mc or mc <= 0) and current_price > 0 and shares > 0:
        mc = current_price * shares
    if not mc or mc <= 0 or td.cashflow.empty:
        return NAN
    try:
        fcf_row = _find_row(td.cashflow, "Free Cash Flow")
        if fcf_row is None:
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


def _stmt_value(df: pd.DataFrame, key: str) -> float:
    """Get the most recent scalar value from a financial statement row. Returns NaN if not found."""
    row = _find_row(df, key)
    if row is None:
        return NAN
    try:
        val = pd.to_numeric(row, errors="coerce").iloc[0]
        return float(val) if not pd.isna(val) else NAN
    except Exception:
        return NAN


def _current_price(td: TickerData) -> float:
    """Latest closing price from history. Falls back to fast_info.last_price."""
    try:
        if not td.history.empty:
            col = "Close" if "Close" in td.history.columns else td.history.columns[0]
            p = float(td.history[col].dropna().iloc[-1])
            if p > 0:
                return p
    except Exception:
        pass
    return float(td.fast_info.get("last_price") or 0)


def _shares(td: TickerData) -> float:
    """Shares outstanding: fast_info → balance sheet implied shares."""
    shares = td.fast_info.get("shares") or td.info.get("sharesOutstanding")
    if shares and float(shares) > 0:
        return float(shares)
    # Balance sheet fallback: equity / book_value_per_share is circular, so skip
    return 0.0
