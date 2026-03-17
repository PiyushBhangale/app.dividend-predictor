"""data_pipeline/fetcher.py — Safe yfinance data layer.

All yfinance calls are isolated here. Every method wraps network/API calls
in try/except and returns a typed empty value on failure — never raises to callers.
Missing fields are recorded in TickerData.fetch_errors for downstream logging.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from config import HIST_YEARS, DEFAULT_NS_SUFFIX
from utils.logger import get_logger, log_fetch_error

logger = get_logger("fetcher")


# ---------------------------------------------------------------------------
# TickerData — the central data container passed between all pipeline stages
# ---------------------------------------------------------------------------

@dataclass
class TickerData:
    """All raw yfinance data for a single NSE ticker.

    Every field has a typed empty default so callers can always check
    len/empty without guarding against None.
    """
    ticker: str
    info: Dict = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividends: pd.Series = field(default_factory=pd.Series)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetch_errors: List[str] = field(default_factory=list)

    @property
    def has_financials(self) -> bool:
        return not self.financials.empty

    @property
    def has_dividends(self) -> bool:
        return len(self.dividends) > 0

    @property
    def has_price_history(self) -> bool:
        return not self.history.empty


# ---------------------------------------------------------------------------
# StockFetcher
# ---------------------------------------------------------------------------

class StockFetcher:
    """Fetches and assembles TickerData for a list of NSE tickers."""

    def __init__(self, tickers: List[str], period_years: int = HIST_YEARS):
        self.tickers = [validate_ticker_ns(t) for t in tickers]
        self.period_years = period_years

    def fetch_all(self) -> Dict[str, TickerData]:
        """Fetch all data for every ticker. Returns dict keyed by ticker symbol."""
        results: Dict[str, TickerData] = {}
        for ticker in self.tickers:
            logger.info("Fetching data for %s", ticker)
            td = self._fetch_one(ticker)
            results[ticker] = td
        return results

    def _fetch_one(self, ticker: str) -> TickerData:
        td = TickerData(ticker=ticker)

        td.info = self._safe_fetch_info(ticker, td)
        td.history, td.dividends = self._safe_fetch_history(ticker, td)
        td.financials, td.balance_sheet, td.cashflow = self._safe_fetch_financials(
            ticker, td
        )

        if td.fetch_errors:
            logger.warning(
                "Ticker %s had %d fetch errors: %s",
                ticker, len(td.fetch_errors), td.fetch_errors
            )
        return td

    # ------------------------------------------------------------------
    # Safe fetch helpers — each returns a typed empty value on failure
    # ------------------------------------------------------------------

    def _safe_fetch_info(self, ticker: str, td: TickerData) -> Dict:
        try:
            info = yf.Ticker(ticker).info
            if not info or "regularMarketPrice" not in info and "currentPrice" not in info and "marketCap" not in info:
                # yfinance returns a minimal dict for unknown tickers
                td.fetch_errors.append("info_incomplete")
                logger.warning("Ticker %s returned incomplete info dict", ticker)
            return info if info else {}
        except Exception as e:
            log_fetch_error(ticker, "fetch_info", e)
            td.fetch_errors.append("info")
            return {}

    def _safe_fetch_history(
        self, ticker: str, td: TickerData
    ) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            t = yf.Ticker(ticker)
            period = f"{self.period_years}y"
            hist = t.history(period=period, auto_adjust=True)
            if hist.empty:
                td.fetch_errors.append("history_empty")
                return pd.DataFrame(), pd.Series(dtype=float)

            # Extract dividend series from history
            if "Dividends" in hist.columns:
                divs = hist["Dividends"]
                divs = divs[divs > 0]  # drop zero-dividend rows
            else:
                divs = pd.Series(dtype=float)
                td.fetch_errors.append("dividends_column_missing")

            return hist, divs
        except Exception as e:
            log_fetch_error(ticker, "fetch_history", e)
            td.fetch_errors.append("history")
            return pd.DataFrame(), pd.Series(dtype=float)

    def _safe_fetch_financials(
        self, ticker: str, td: TickerData
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        empty = pd.DataFrame()
        try:
            t = yf.Ticker(ticker)
            financials = _safe_df(t, "financials", ticker, td)
            balance_sheet = _safe_df(t, "balance_sheet", ticker, td)
            cashflow = _safe_df(t, "cashflow", ticker, td)
            return financials, balance_sheet, cashflow
        except Exception as e:
            log_fetch_error(ticker, "fetch_financials", e)
            td.fetch_errors.extend(["financials", "balance_sheet", "cashflow"])
            return empty, empty, empty


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_df(t: yf.Ticker, attr: str, ticker: str, td: TickerData) -> pd.DataFrame:
    """Get a DataFrame attribute from a yf.Ticker, returning empty DF on any error."""
    try:
        df = getattr(t, attr)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            td.fetch_errors.append(f"{attr}_empty")
            return pd.DataFrame()
        return df
    except Exception as e:
        log_fetch_error(ticker, attr, e)
        td.fetch_errors.append(attr)
        return pd.DataFrame()


def validate_ticker_ns(ticker: str) -> str:
    """Ensure ticker ends with .NS (NSE India suffix). Appends if missing."""
    ticker = ticker.strip().upper()
    if not ticker.endswith(DEFAULT_NS_SUFFIX):
        ticker = ticker + DEFAULT_NS_SUFFIX
    return ticker


def batch_fetch(
    tickers: List[str], period_years: int = HIST_YEARS
) -> Dict[str, TickerData]:
    """Module-level convenience function. Returns TickerData dict for all tickers."""
    fetcher = StockFetcher(tickers=tickers, period_years=period_years)
    return fetcher.fetch_all()
