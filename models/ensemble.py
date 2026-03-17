"""models/ensemble.py — Weighted ensemble combiner for all 4 AI models.

Combines Random Forest (70%), LSTM (15%), FinBERT (10%), and a base rate prior (5%)
into a single ensemble score per ticker.

Key feature: dynamic weight redistribution when any model is unavailable
(e.g., LSTM has insufficient data, FinBERT import failed). The remaining
available models' weights are rescaled to sum to 1.0.

Output: EnsembleResult per ticker with score, recommendation, confidence,
and per-model score breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from config import MODEL_WEIGHTS, THRESHOLDS, SIP_ADJUSTMENTS
from utils.logger import get_logger

logger = get_logger("ensemble")

# Long-run base rate: NSE blue chips raise dividends ~55% of the time historically
BASE_RATE = 0.55


@dataclass
class ModelOutputs:
    """Raw outputs from all 3 trained models for a single ticker."""
    ticker: str
    rf_proba: np.ndarray        # shape (3,): [P(Reduce), P(Hold), P(Increase)]
    lstm_proba: float           # P(increase) from LSTM  [0, 1]
    finbert_score: float        # news sentiment score   [0, 1]
    rf_available: bool = True
    lstm_available: bool = True
    finbert_available: bool = True


@dataclass
class EnsembleResult:
    """Final prediction and recommendation for a single ticker."""
    ticker: str
    ensemble_score: float           # weighted score in [0, 1]
    recommendation: str             # "Increase" | "Hold" | "Reduce"
    sip_adjustment_pct: float       # e.g., +0.15, 0.0, -0.20
    confidence: float               # 1 - stddev of model scores (higher = more confident)
    rf_score: float                 # RF P(Increase) used in ensemble
    lstm_score: float               # LSTM P(increase)
    finbert_score: float            # FinBERT sentiment score
    model_weights_used: Dict[str, float] = field(default_factory=dict)


class EnsembleCombiner:
    """Combines model outputs into final dividend predictions."""

    def __init__(self, weights: Dict[str, float] = None):
        self._base_weights = weights or MODEL_WEIGHTS.copy()

    # ------------------------------------------------------------------
    # Single ticker combination
    # ------------------------------------------------------------------

    def combine(self, outputs: ModelOutputs) -> EnsembleResult:
        """Produce a final EnsembleResult for one ticker."""

        # Normalise all model outputs to [0, 1] scale (P(Increase))
        rf_score = float(outputs.rf_proba[2]) if outputs.rf_available else None
        lstm_score = outputs.lstm_proba if outputs.lstm_available else None
        finbert_score = outputs.finbert_score if outputs.finbert_available else None
        base_score = BASE_RATE

        availability = {
            "random_forest": outputs.rf_available,
            "lstm": outputs.lstm_available,
            "finbert": outputs.finbert_available,
            "base": True,  # base rate is always available
        }
        scores = {
            "random_forest": rf_score,
            "lstm": lstm_score,
            "finbert": finbert_score,
            "base": base_score,
        }

        # Redistribute weights for unavailable models
        effective_weights = self._redistribute_weights(availability)

        # Weighted sum
        ensemble_score = 0.0
        for model_name, weight in effective_weights.items():
            score = scores[model_name]
            if score is not None:
                ensemble_score += weight * score

        ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))

        # Map score to recommendation
        recommendation = self._score_to_recommendation(ensemble_score)
        sip_adjustment = SIP_ADJUSTMENTS.get(recommendation, 0.0)

        # Confidence: inverse of spread between available model scores
        available_scores = [s for s in [rf_score, lstm_score, finbert_score] if s is not None]
        if len(available_scores) >= 2:
            spread = float(np.std(available_scores))
            confidence = float(np.clip(1.0 - spread, 0.0, 1.0))
        else:
            confidence = 0.5  # only one or zero models — moderate confidence

        logger.info(
            "Ensemble | %s | score=%.3f | rec=%s | conf=%.3f | rf=%s | lstm=%s | finbert=%s",
            outputs.ticker, ensemble_score, recommendation, confidence,
            f"{rf_score:.3f}" if rf_score is not None else "n/a",
            f"{lstm_score:.3f}" if lstm_score is not None else "n/a",
            f"{finbert_score:.3f}" if finbert_score is not None else "n/a",
        )

        return EnsembleResult(
            ticker=outputs.ticker,
            ensemble_score=ensemble_score,
            recommendation=recommendation,
            sip_adjustment_pct=sip_adjustment,
            confidence=confidence,
            rf_score=rf_score if rf_score is not None else 0.333,
            lstm_score=lstm_score if lstm_score is not None else 0.5,
            finbert_score=finbert_score if finbert_score is not None else 0.5,
            model_weights_used=effective_weights,
        )

    def combine_batch(self, outputs_list: List[ModelOutputs]) -> List[EnsembleResult]:
        """Combine all tickers. Returns results sorted by ensemble_score descending."""
        results = [self.combine(o) for o in outputs_list]
        results.sort(key=lambda r: r.ensemble_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Weight redistribution
    # ------------------------------------------------------------------

    def _redistribute_weights(
        self, availability: Dict[str, bool]
    ) -> Dict[str, float]:
        """Redistribute weights from unavailable models to available ones.

        Example: if LSTM (15%) unavailable, its weight is split proportionally
        among RF (70%) and FinBERT (10%) → RF gets 70/(70+10)*15% extra.
        Base rate always contributes.
        """
        available = {k: v for k, v in self._base_weights.items() if availability.get(k, False)}
        unavailable_total = sum(v for k, v in self._base_weights.items() if not availability.get(k, False))

        if not available:
            # All models unavailable — use uniform
            return {k: 0.25 for k in self._base_weights}

        available_total = sum(available.values())

        if unavailable_total == 0:
            # All available — use base weights unchanged
            return dict(self._base_weights)

        # Redistribute proportionally
        redistributed = {}
        for model_name, base_w in self._base_weights.items():
            if availability.get(model_name, False):
                # This model gets its own weight + proportional share of unavailable weight
                extra = (base_w / available_total) * unavailable_total
                redistributed[model_name] = base_w + extra
            else:
                redistributed[model_name] = 0.0

        # Normalise to sum exactly to 1.0
        total = sum(redistributed.values())
        redistributed = {k: v / total for k, v in redistributed.items()}
        return redistributed

    # ------------------------------------------------------------------
    # Score to recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_recommendation(score: float) -> str:
        if score >= THRESHOLDS["increase"]:
            return "Increase"
        elif score >= THRESHOLDS["hold"]:
            return "Hold"
        else:
            return "Reduce"
