"""models/random_forest.py — Random Forest dividend classifier (70% ensemble weight).

Training strategy:
- Uses rolling annual snapshots from historical data (each year = one training row per ticker).
- Label: 3Y forward annual dividend CAGR > RF_LABEL_CAGR_THRESHOLD → class 2 (Increase)
         3Y forward CAGR < 0 → class 0 (Reduce)
         else → class 1 (Hold)
- Predicts on the most recent feature snapshot for each ticker.

Fallback: if fewer than 2 training rows can be assembled, predict_proba returns
          uniform [0.33, 0.33, 0.33] and a warning is logged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_NAMES,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    RF_LABEL_CAGR_THRESHOLD,
    HIST_YEARS,
)
from data_pipeline.features import (
    build_feature_vector,
    build_feature_matrix,
    impute_features,
    compute_cagr,
    _annual_dividend_totals,
)
from data_pipeline.fetcher import TickerData
from utils.logger import get_logger, log_model_skip

logger = get_logger("random_forest")

# Label encoding
LABEL_REDUCE   = 0
LABEL_HOLD     = 1
LABEL_INCREASE = 2
UNIFORM_PROBA  = np.array([[1/3, 1/3, 1/3]], dtype=float)


class DividendRandomForest:
    """Random Forest classifier for dividend increase/hold/reduce prediction."""

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        random_state: int = RF_RANDOM_STATE,
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",  # handles imbalanced labels
            min_samples_leaf=2,
        )
        self._scaler = StandardScaler()
        self._trained = False
        self._feature_names = FEATURE_NAMES

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        all_ticker_data: Dict[str, TickerData],
        current_feature_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) training arrays from historical annual snapshots.

        For each ticker we create one row per historical year using a rolling
        window of features proxied from the current info dict (static features
        don't change year-to-year in this simple approach) plus the dividend
        series up to that year. Labels come from the 3Y forward dividend CAGR.

        Returns (empty arrays, empty arrays) if fewer than 2 rows are assembled.
        """
        X_rows, y_rows = [], []

        for ticker, td in all_ticker_data.items():
            ann_divs = _annual_dividend_totals(td.dividends)
            if len(ann_divs) < 5:
                log_model_skip(ticker, "random_forest", f"only {len(ann_divs)} annual dividend records")
                continue

            # Slide a window: for each year index t where we have t+3 years ahead
            years = list(ann_divs.index)
            for i in range(len(years) - 3):
                past_divs = ann_divs.iloc[:i + 1]
                future_divs = ann_divs.iloc[i: i + 4]  # 3-year forward window

                # Label from 3Y forward CAGR
                cagr = compute_cagr(future_divs, years=3)
                if np.isnan(cagr):
                    continue
                if cagr > RF_LABEL_CAGR_THRESHOLD:
                    label = LABEL_INCREASE
                elif cagr < 0:
                    label = LABEL_REDUCE
                else:
                    label = LABEL_HOLD

                # Features: use current financial ratios (static proxy)
                # plus time-varying dividend features recomputed on past_divs
                if ticker in current_feature_df.index:
                    row = current_feature_df.loc[ticker].copy()
                else:
                    row = pd.Series({f: np.nan for f in FEATURE_NAMES})

                # Override the dividend CAGR features with historical values
                row["div_cagr_3y"] = compute_cagr(past_divs, years=3)
                row["div_cagr_5y"] = compute_cagr(past_divs, years=5)
                row["consecutive_div_years"] = float(sum(1 for v in past_divs.iloc[::-1] if v > 0))

                X_rows.append(row.values.astype(float))
                y_rows.append(label)

        if len(X_rows) < 2:
            logger.warning(
                "Only %d training rows assembled — RF will use uniform fallback", len(X_rows)
            )
            return np.empty((0, len(FEATURE_NAMES))), np.empty(0)

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Impute NaNs in training data using column medians
        for col_i in range(X.shape[1]):
            col = X[:, col_i]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                X[:, col_i] = 0.0
            elif nan_mask.any():
                X[nan_mask, col_i] = np.nanmedian(col)

        logger.info("RF training data: %d rows, %d features, label dist: %s",
                    X.shape[0], X.shape[1],
                    {LABEL_REDUCE: (y==0).sum(), LABEL_HOLD: (y==1).sum(), LABEL_INCREASE: (y==2).sum()})
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Random Forest. Skips if X is empty."""
        if X.shape[0] < 2:
            logger.warning("RF training skipped — insufficient data")
            return
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._trained = True
        logger.info("RF trained on %d samples. OOB not available (use CV for validation).", X.shape[0])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for shape (n_samples, 3).

        Column order: [P(Reduce), P(Hold), P(Increase)]
        Returns uniform probabilities if model is not trained.
        """
        if not self._trained or X.shape[0] == 0:
            n = max(X.shape[0], 1)
            return np.tile(UNIFORM_PROBA, (n, 1))

        # Impute NaNs before scaling
        X_clean = X.copy()
        for col_i in range(X_clean.shape[1]):
            col = X_clean[:, col_i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                X_clean[nan_mask, col_i] = 0.0

        X_scaled = self._scaler.transform(X_clean)
        probas = self._model.predict_proba(X_scaled)

        # sklearn returns columns in sorted class order; ensure (Reduce, Hold, Increase)
        classes = list(self._model.classes_)
        full_probas = np.zeros((probas.shape[0], 3))
        for i, cls in enumerate(classes):
            full_probas[:, cls] = probas[:, i]
        return full_probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels [0=Reduce, 1=Hold, 2=Increase]."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending. Empty series if untrained."""
        if not self._trained:
            return pd.Series(dtype=float)
        importances = self._model.feature_importances_
        return pd.Series(importances, index=FEATURE_NAMES).sort_values(ascending=False)

    @property
    def is_trained(self) -> bool:
        return self._trained
