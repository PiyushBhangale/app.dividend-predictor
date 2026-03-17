"""analysis/backtest.py — SIP projection engine.

Shows two sections per ticker:
  A. Historical (past BACKTEST_YEARS): same flat SIP for both strategies —
     illustrates what the stock has done and what a flat SIP would be worth today.
  B. Forward projection (next PROJECTION_YEARS): flat SIP vs AI-adjusted SIP,
     using the stock's historical price CAGR to simulate future price growth.
     Dividend yield is estimated from historical data and reinvested monthly.

The AI recommendation is applied ONLY to the forward section, which is the
correct use: the signal is about future SIP sizing, not rewriting the past.
A vertical "Today" marker separates the two sections in the chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BACKTEST_REBALANCE_FREQ, BACKTEST_YEARS, HIST_YEARS, SIP_ADJUSTMENTS
from data_pipeline.fetcher import TickerData
from models.ensemble import EnsembleResult
from utils.logger import get_logger

logger = get_logger("backtest")

PROJECTION_YEARS = 1   # forward projection = ~1 dividend cycle


@dataclass
class BacktestResult:
    """Per-ticker backtest + projection output."""
    ticker: str

    # Combined timeline (historical + projected), monthly date strings
    dates: List[str]

    # Portfolio values — both strategies identical during historical section;
    # diverge during the projected section
    baseline_values: List[float]
    ai_adjusted_values: List[float]

    # Index into dates/values where "today" falls (projection starts here)
    today_index: int

    # Totals
    total_invested_baseline: float
    total_invested_ai: float
    final_value_baseline: float
    final_value_ai: float

    # CAGR over the projected period only
    cagr_baseline: float
    cagr_ai: float

    # Dividend dots during historical section (on the historical SIP line)
    dividend_events: List[Tuple[str, float, float]] = field(default_factory=list)
    # (date_str, portfolio_value, dividend_income)

    # Metadata
    historical_cagr: float = 0.0      # annualised price return over hist window
    avg_div_yield: float = 0.0        # average annual dividend yield


class SIPBacktester:
    """Monthly SIP simulator: historical context + forward projection."""

    def __init__(self, rebalance_freq: str = BACKTEST_REBALANCE_FREQ):
        self.rebalance_freq = rebalance_freq

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        sip_holdings: pd.DataFrame,
        ensemble_results: List[EnsembleResult],
        all_ticker_data: Dict[str, TickerData],
    ) -> Dict[str, BacktestResult]:
        ensemble_map: Dict[str, EnsembleResult] = {r.ticker: r for r in ensemble_results}
        results: Dict[str, BacktestResult] = {}

        for _, row in sip_holdings.iterrows():
            ticker = str(row["ticker"])
            monthly_sip = float(row["monthly_sip"])

            if ticker not in all_ticker_data:
                logger.warning("Backtest: no data for %s — skipping", ticker)
                continue

            td = all_ticker_data[ticker]
            if not td.has_price_history:
                logger.warning("Backtest: empty price history for %s — skipping", ticker)
                continue

            ensemble_result = ensemble_map.get(ticker)
            adj_pct = ensemble_result.sip_adjustment_pct if ensemble_result else 0.0

            result = self._run_ticker(ticker, monthly_sip, adj_pct, td)
            if result:
                results[ticker] = result

        return results

    # ------------------------------------------------------------------
    # Per-ticker: historical SIP + forward projection
    # ------------------------------------------------------------------

    def _run_ticker(
        self,
        ticker: str,
        monthly_sip: float,
        sip_adjustment_pct: float,
        td: TickerData,
    ) -> Optional[BacktestResult]:
        try:
            price_series = self._prepare_prices(td)
            if price_series.empty or len(price_series) < 12:
                logger.warning("Backtest: %s has <12 monthly prices — skipping", ticker)
                return None

            monthly_hist = price_series.resample("MS").first().ffill()

            # ---- Historical dividend series (YYYY-MM → per-share amount) ----
            div_series = self._prepare_dividends(td, price_series.index)

            # ---- Estimate stock parameters from history ----
            hist_cagr = self._estimate_price_cagr(monthly_hist)
            avg_div_yield = self._estimate_div_yield(monthly_hist, div_series)
            monthly_price_growth = (1 + hist_cagr) ** (1 / 12)

            last_price = float(monthly_hist.iloc[-1])
            today = monthly_hist.index[-1]

            # ---- SECTION A: Historical simulation (flat SIP, same for both) ----
            hist_dates, hist_vals, inv_hist, _, div_events = self._simulate_sip(
                monthly_hist, div_series, monthly_sip,
                adjustment_schedule={}, track_dividends=True,
            )

            if not hist_vals:
                return None

            portfolio_at_today = hist_vals[-1]
            shares_at_today = portfolio_at_today / last_price

            today_index = len(hist_dates) - 1  # index of "today" in combined array

            # ---- SECTION B: Forward projection ----
            proj_months = PROJECTION_YEARS * 12

            # Build projected price series using historical CAGR
            proj_dates = pd.date_range(
                start=today + pd.DateOffset(months=1),
                periods=proj_months,
                freq="MS",
            )
            proj_prices = pd.Series(
                [last_price * (monthly_price_growth ** (i + 1)) for i in range(proj_months)],
                index=proj_dates,
            )

            # Projected dividend per share (monthly average yield)
            monthly_div_per_share = (avg_div_yield / 12)

            # AI adjustment starts from month 1 of projection (the recommendation is for NOW)
            ai_schedule = {}
            if sip_adjustment_pct != 0.0:
                adjusted = round(monthly_sip * (1 + sip_adjustment_pct), -2)
                adjusted = max(adjusted, 100.0)
                ai_schedule[proj_dates[0]] = adjusted

            baseline_proj_vals, inv_b_proj, baseline_shares = self._simulate_projection(
                proj_prices, monthly_sip, {}, shares_at_today,
                monthly_div_per_share,
            )
            ai_proj_vals, inv_ai_proj, _ = self._simulate_projection(
                proj_prices, monthly_sip, ai_schedule, shares_at_today,
                monthly_div_per_share,
            )

            # ---- Combine historical + projected ----
            all_dates = [str(d.date()) for d in hist_dates] + [str(d.date()) for d in proj_dates]
            # Historical section: both lines follow same path
            baseline_all = hist_vals + baseline_proj_vals
            ai_all = hist_vals + ai_proj_vals

            # CAGR over projected section only
            years_proj = PROJECTION_YEARS
            cagr_b = self._compute_cagr(inv_b_proj, baseline_proj_vals[-1], years_proj) if baseline_proj_vals else 0.0
            cagr_ai = self._compute_cagr(inv_ai_proj, ai_proj_vals[-1], years_proj) if ai_proj_vals else 0.0

            logger.info(
                "Projection | %s | hist_cagr=%.1f%% | adj=%.0f%% | "
                "proj_baseline=%.0f | proj_ai=%.0f",
                ticker, hist_cagr * 100, sip_adjustment_pct * 100,
                baseline_proj_vals[-1] if baseline_proj_vals else 0,
                ai_proj_vals[-1] if ai_proj_vals else 0,
            )

            return BacktestResult(
                ticker=ticker,
                dates=all_dates,
                baseline_values=baseline_all,
                ai_adjusted_values=ai_all,
                today_index=today_index,
                total_invested_baseline=inv_hist + inv_b_proj,
                total_invested_ai=inv_hist + inv_ai_proj,
                final_value_baseline=baseline_proj_vals[-1] if baseline_proj_vals else 0.0,
                final_value_ai=ai_proj_vals[-1] if ai_proj_vals else 0.0,
                cagr_baseline=cagr_b,
                cagr_ai=cagr_ai,
                dividend_events=div_events,
                historical_cagr=hist_cagr,
                avg_div_yield=avg_div_yield,
            )
        except Exception as e:
            logger.warning("Backtest failed for %s: %s", ticker, e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Historical SIP simulation (uses real prices + real dividends)
    # ------------------------------------------------------------------

    def _simulate_sip(
        self,
        monthly_prices: pd.Series,
        div_series: pd.Series,
        initial_monthly: float,
        adjustment_schedule: Dict,
        track_dividends: bool = False,
    ) -> Tuple[List, List[float], float, float, List]:
        dates, values = [], []
        total_invested = 0.0
        total_divs = 0.0
        dividend_events = []
        shares_held = 0.0
        current_monthly = initial_monthly

        for month_date, price in monthly_prices.items():
            if price <= 0 or np.isnan(price):
                continue
            if month_date in adjustment_schedule:
                current_monthly = adjustment_schedule[month_date]
            shares_held += current_monthly / price
            total_invested += current_monthly

            month_str = month_date.strftime("%Y-%m")
            month_divs = div_series.get(month_str, 0.0)
            if month_divs > 0:
                div_income = shares_held * month_divs
                shares_held += div_income / price
                total_divs += div_income
                if track_dividends:
                    dividend_events.append((str(month_date.date()), shares_held * price, div_income))

            values.append(shares_held * price)
            dates.append(month_date)

        return dates, values, total_invested, total_divs, dividend_events

    # ------------------------------------------------------------------
    # Forward projection simulation (uses synthetic price growth)
    # ------------------------------------------------------------------

    def _simulate_projection(
        self,
        proj_prices: pd.Series,
        initial_monthly: float,
        adjustment_schedule: Dict,
        starting_shares: float,
        monthly_div_per_share: float,
    ) -> Tuple[List[float], float, float]:
        """Simulate SIP forward using projected prices. Returns (values, total_invested, final_shares)."""
        values = []
        total_invested = 0.0
        shares_held = starting_shares
        current_monthly = initial_monthly

        for month_date, price in proj_prices.items():
            if price <= 0:
                continue
            if month_date in adjustment_schedule:
                current_monthly = adjustment_schedule[month_date]
            shares_held += current_monthly / price
            total_invested += current_monthly

            # Estimated dividend reinvestment
            if monthly_div_per_share > 0:
                div_income = shares_held * price * monthly_div_per_share
                shares_held += div_income / price

            values.append(shares_held * price)

        return values, total_invested, shares_held

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_prices(self, td: TickerData) -> pd.Series:
        hist = td.history
        if hist.empty:
            return pd.Series(dtype=float)
        close_col = "Close" if "Close" in hist.columns else hist.columns[0]
        prices = hist[close_col].dropna()
        prices.index = pd.to_datetime(prices.index)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        prices = prices.sort_index()
        cutoff = prices.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
        return prices[prices.index >= cutoff]

    def _prepare_dividends(self, td: TickerData, price_index: pd.DatetimeIndex) -> pd.Series:
        if not td.has_dividends:
            return pd.Series(dtype=float)
        try:
            divs = td.dividends.copy()
            divs.index = pd.to_datetime(divs.index)
            divs = divs.sort_index()
            return divs.groupby(divs.index.strftime("%Y-%m")).sum()
        except Exception:
            return pd.Series(dtype=float)

    def _estimate_price_cagr(self, monthly_prices: pd.Series) -> float:
        """Annualised price return over the historical window."""
        if len(monthly_prices) < 2:
            return 0.06  # 6% fallback
        try:
            start = float(monthly_prices.iloc[0])
            end = float(monthly_prices.iloc[-1])
            years = len(monthly_prices) / 12.0
            if start <= 0 or years <= 0:
                return 0.06
            return float((end / start) ** (1.0 / years) - 1.0)
        except Exception:
            return 0.06

    def _estimate_div_yield(self, monthly_prices: pd.Series, div_series: pd.Series) -> float:
        """Estimate average annual dividend yield from historical data."""
        if div_series.empty or monthly_prices.empty:
            return 0.0
        try:
            total_divs = float(div_series.sum())
            avg_price = float(monthly_prices.mean())
            years = len(monthly_prices) / 12.0
            if avg_price <= 0 or years <= 0:
                return 0.0
            return float((total_divs / years) / avg_price)
        except Exception:
            return 0.0

    def _build_adjustment_schedule(
        self,
        date_index: pd.DatetimeIndex,
        initial_monthly: float,
        sip_adjustment_pct: float,
    ) -> Dict:
        schedule = {}
        if sip_adjustment_pct == 0.0:
            return schedule
        monthly_years = sorted(set(d.year for d in date_index))
        current_amount = initial_monthly
        for year in monthly_years:
            jan_date = pd.Timestamp(year=year, month=1, day=1)
            if jan_date in pd.DatetimeIndex(date_index):
                current_amount = round(current_amount * (1 + sip_adjustment_pct), -2)
                current_amount = max(current_amount, 100.0)
                schedule[jan_date] = current_amount
        return schedule

    @staticmethod
    def _compute_cagr(invested: float, final_value: float, years: float) -> float:
        if invested <= 0 or years <= 0:
            return 0.0
        try:
            return float((final_value / invested) ** (1.0 / years) - 1.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Portfolio-level summary
    # ------------------------------------------------------------------

    def compute_portfolio_summary(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker":             ticker,
                "Hist CAGR":          f"{r.historical_cagr * 100:.1f}%",
                "Avg Div Yield":      f"{r.avg_div_yield * 100:.1f}%",
                "Proj Final (Flat)":  f"Rs{r.final_value_baseline:,.0f}",
                "Proj Final (AI)":    f"Rs{r.final_value_ai:,.0f}",
                "Proj CAGR (Flat)":   f"{r.cagr_baseline * 100:.1f}%",
                "Proj CAGR (AI)":     f"{r.cagr_ai * 100:.1f}%",
                "Alpha":              f"{(r.cagr_ai - r.cagr_baseline) * 100:+.1f}pp",
            })
        return pd.DataFrame(rows)
