"""analysis/portfolio.py — Portfolio recommendation layer.

Translates raw EnsembleResult scores into actionable SIP recommendations
expressed in INR, rounded to the nearest ₹100.

Also generates a human-readable reasoning string explaining the AI's
confidence and key signal for each stock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config import SIP_ROUND_TO
from models.ensemble import EnsembleResult
from utils.logger import get_logger

logger = get_logger("portfolio")


@dataclass
class PortfolioRecommendation:
    """Final actionable recommendation for a single holding."""
    ticker: str
    current_monthly_sip: float
    recommended_monthly_sip: float
    change_amount: float            # recommended - current (signed)
    change_pct: float               # sip_adjustment_pct from ensemble
    recommendation: str             # "Increase" | "Hold" | "Reduce"
    ensemble_score: float           # 0-1 probability score
    confidence: float               # model agreement (0-1, higher = more confident)
    reasoning: str                  # human-readable one-liner


class PortfolioAdvisor:
    """Generates portfolio-level SIP recommendations from ensemble results."""

    def __init__(self, ensemble_results: List[EnsembleResult]):
        self._ensemble_map: Dict[str, EnsembleResult] = {
            r.ticker: r for r in ensemble_results
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate_recommendations(
        self, sip_holdings: pd.DataFrame
    ) -> List[PortfolioRecommendation]:
        """Merge sip_holdings with ensemble results to produce recommendations.

        Args:
            sip_holdings: DataFrame with columns [ticker, monthly_sip]

        Returns:
            List of PortfolioRecommendation, one per holding.
            Holdings with no ensemble result get a "Hold" recommendation.
        """
        recommendations = []

        for _, row in sip_holdings.iterrows():
            ticker = str(row["ticker"])
            current_sip = float(row["monthly_sip"])

            result = self._ensemble_map.get(ticker)

            if result is None:
                logger.warning("No ensemble result for %s — defaulting to Hold", ticker)
                rec = PortfolioRecommendation(
                    ticker=ticker,
                    current_monthly_sip=current_sip,
                    recommended_monthly_sip=current_sip,
                    change_amount=0.0,
                    change_pct=0.0,
                    recommendation="Hold",
                    ensemble_score=0.5,
                    confidence=0.0,
                    reasoning="Insufficient data — holding current SIP allocation.",
                )
            else:
                adj_pct = result.sip_adjustment_pct
                raw_recommended = current_sip * (1.0 + adj_pct)
                recommended_sip = self._round_to_nearest(raw_recommended, SIP_ROUND_TO)
                recommended_sip = max(recommended_sip, SIP_ROUND_TO)  # floor ₹100

                rec = PortfolioRecommendation(
                    ticker=ticker,
                    current_monthly_sip=current_sip,
                    recommended_monthly_sip=recommended_sip,
                    change_amount=recommended_sip - current_sip,
                    change_pct=adj_pct,
                    recommendation=result.recommendation,
                    ensemble_score=result.ensemble_score,
                    confidence=result.confidence,
                    reasoning=self._build_reasoning(result),
                )

            recommendations.append(rec)
            logger.info(
                "Recommendation | %s | %s | ₹%.0f → ₹%.0f | score=%.3f",
                ticker, rec.recommendation,
                current_sip, rec.recommended_monthly_sip, rec.ensemble_score,
            )

        return recommendations

    def to_dataframe(self, recs: List[PortfolioRecommendation]) -> pd.DataFrame:
        """Return a display-ready DataFrame from a list of recommendations."""
        rows = []
        for r in recs:
            rows.append({
                "Ticker":           r.ticker,
                "Current SIP":      f"₹{r.current_monthly_sip:,.0f}",
                "Recommended SIP":  f"₹{r.recommended_monthly_sip:,.0f}",
                "Change":           f"{'+'if r.change_amount >= 0 else ''}₹{r.change_amount:,.0f}",
                "Action":           r.recommendation,
                "AI Score":         f"{r.ensemble_score:.2f}",
                "Confidence":       f"{r.confidence:.0%}",
                "Reasoning":        r.reasoning,
            })
        return pd.DataFrame(rows)

    def get_portfolio_totals(
        self, recs: List[PortfolioRecommendation]
    ) -> Dict:
        """Return portfolio-level totals for display in Streamlit metrics."""
        current_total = sum(r.current_monthly_sip for r in recs)
        recommended_total = sum(r.recommended_monthly_sip for r in recs)
        change = recommended_total - current_total
        change_pct = (change / current_total) if current_total > 0 else 0.0
        return {
            "current_total_sip":      current_total,
            "recommended_total_sip":  recommended_total,
            "change_amount":          change,
            "change_pct":             change_pct,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_to_nearest(value: float, nearest: int) -> float:
        """Round value to the nearest `nearest` (e.g., 100 → ₹100 increments)."""
        return round(value / nearest) * nearest

    @staticmethod
    def _build_reasoning(result: EnsembleResult) -> str:
        """Generate a concise one-line explanation for the recommendation."""
        score = result.ensemble_score
        conf = result.confidence
        rec = result.recommendation

        if rec == "Increase":
            if conf >= 0.75:
                return f"High-conviction dividend growth signal (score {score:.2f}, models agree)."
            elif conf >= 0.5:
                return f"Moderate growth signal (score {score:.2f}); most models bullish."
            else:
                return f"Mild growth signal (score {score:.2f}); models partially disagree."

        elif rec == "Hold":
            if conf >= 0.7:
                return f"Stable dividend expected (score {score:.2f}); no clear catalyst."
            else:
                return f"Mixed signals (score {score:.2f}); hold and review next quarter."

        else:  # Reduce
            if conf >= 0.65:
                return f"Dividend reduction risk detected (score {score:.2f}); reduce exposure."
            else:
                return f"Weak dividend outlook (score {score:.2f}); consider reducing SIP."
