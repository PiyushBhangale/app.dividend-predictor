"""models/lstm_model.py — Sequence pattern model for dividend cycle prediction (15% weight).

Replaces TensorFlow LSTM with sklearn MLPClassifier trained on flattened + enriched
quarterly dividend sequences. Achieves the same goal (learn NSE dividend momentum
patterns across tickers) with zero heavy dependencies and no CPU instruction issues.

Input features per sequence (seq_len=8 quarters):
  - Raw normalised dividend values  (8 features)
  - Slope (linear regression over window)
  - Momentum (last quarter vs first)
  - Volatility (std dev of window)
  = seq_len + 3 features total

Label: 1 if next quarter dividend > current quarter, else 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import LSTM_SEQ_LEN, MIN_SEQUENCES
from utils.logger import get_logger, log_model_skip

logger = get_logger("lstm")

NEUTRAL_PROBA = 0.5


class DividendLSTM:
    """MLP-based sequence model that predicts dividend increase probability."""

    def __init__(self, seq_len: int = LSTM_SEQ_LEN):
        self.seq_len = seq_len
        self._model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        self._scaler = StandardScaler()
        self._trained = False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_sequences(
        self,
        dividend_series_dict: Dict[str, pd.Series],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build (X, y) from all tickers' quarterly dividend series.

        X shape: (n_sequences, seq_len + 3)
        y shape: (n_sequences,) — 1 if next quarter > current, else 0

        Returns (None, None) if total usable sequences < MIN_SEQUENCES.
        """
        X_all, y_all = [], []

        for ticker, series in dividend_series_dict.items():
            if series is None or len(series) < self.seq_len + 1:
                log_model_skip(
                    ticker, "lstm",
                    f"only {len(series) if series is not None else 0} quarterly records"
                )
                continue

            values = series.values.astype(float)
            max_val = values.max()
            if max_val == 0:
                continue
            values = values / max_val

            for i in range(len(values) - self.seq_len):
                window = values[i: i + self.seq_len]
                next_val = values[i + self.seq_len]
                label = 1.0 if next_val > window[-1] else 0.0
                X_all.append(self._extract_features(window))
                y_all.append(label)

        if len(X_all) < MIN_SEQUENCES:
            logger.warning(
                "Sequence model skipped — only %d sequences (need %d)", len(X_all), MIN_SEQUENCES
            )
            return None, None

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        logger.info("Sequence model: X=%s, pos_rate=%.2f", X.shape, y.mean())
        return X, y

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """Convert a normalised window into a flat feature vector."""
        n = len(window)
        slope = float(np.polyfit(range(n), window, 1)[0])
        momentum = float(window[-1] - window[0])
        volatility = float(window.std())
        return np.concatenate([window, [slope, momentum, volatility]])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
    ) -> None:
        """Fit the MLP. No-op if X/y are None or labels have only one class."""
        if X is None or y is None or len(X) < MIN_SEQUENCES:
            logger.warning("Sequence model training skipped — no sequences available")
            return
        if len(np.unique(y)) < 2:
            logger.warning("Sequence model skipped — only one class in training labels")
            return

        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._trained = True
        logger.info(
            "Sequence model trained: %d samples, %d iterations",
            len(X), self._model.n_iter_
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, series: pd.Series) -> float:
        """Predict P(next quarter dividend > current) for one ticker.

        Returns NEUTRAL_PROBA (0.5) if model untrained or series too short.
        """
        if not self._trained:
            return NEUTRAL_PROBA
        if series is None or len(series) < self.seq_len:
            return NEUTRAL_PROBA

        try:
            values = series.values.astype(float)
            max_val = values.max()
            if max_val == 0:
                return NEUTRAL_PROBA
            values = values / max_val

            window = values[-self.seq_len:]
            feat = self._extract_features(window).reshape(1, -1)
            feat_scaled = self._scaler.transform(feat)
            classes = list(self._model.classes_)
            proba = self._model.predict_proba(feat_scaled)[0]
            p_increase = float(proba[classes.index(1)]) if 1 in classes else NEUTRAL_PROBA
            return float(np.clip(p_increase, 0.0, 1.0))
        except Exception as e:
            logger.warning("Sequence model predict_proba failed: %s", e)
            return NEUTRAL_PROBA

    @property
    def is_trained(self) -> bool:
        return self._trained
