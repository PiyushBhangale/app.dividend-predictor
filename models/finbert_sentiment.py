"""models/finbert_sentiment.py — FinBERT news sentiment analyzer (10% ensemble weight).

Uses ProsusAI/finbert to classify financial news headlines as positive/negative/neutral.
Maps classifications to a dividend bullishness score in [0, 1]:
  1.0 = very bullish (all positive)
  0.5 = neutral (no news or mixed)
  0.0 = very bearish (all negative)

Design decisions:
- Lazy-loaded: 400MB+ model downloads on first "Analyze" click, not app startup.
- Always returns a score for every ticker (0.5 neutral fallback on any failure).
- NSE news via yf.Ticker.news is sparse; this is expected and not logged as error.
- transformers import is guarded — graceful degradation if not installed.
"""

from __future__ import annotations

import os
# Block TensorFlow before transformers loads — TF crashes with AVX SIGABRT on some machines
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Offline mode: only activate when explicitly set (e.g. local dev after first download).
# On Streamlit Cloud or fresh environments the model downloads from HuggingFace on first use.
# To enable locally after downloading: set TRANSFORMERS_OFFLINE=1 in your shell.
if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

from typing import Dict, List, Optional

import yfinance as yf

from config import FINBERT_MODEL, NEWS_MAX_ARTICLES, NEWS_LOOKBACK_DAYS
from utils.logger import get_logger, log_model_skip

logger = get_logger("finbert")

NEUTRAL_SCORE = 0.5


class FinBERTSentimentAnalyzer:
    """FinBERT-based news sentiment scorer for NSE dividend stocks."""

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        max_articles: int = NEWS_MAX_ARTICLES,
        lookback_days: int = NEWS_LOOKBACK_DAYS,
    ):
        self.model_name = model_name
        self.max_articles = max_articles
        self.lookback_days = lookback_days
        self._pipeline = None
        self._model_loaded = False
        self._transformers_available = self._check_transformers()

    # ------------------------------------------------------------------
    # Availability guards
    # ------------------------------------------------------------------

    @staticmethod
    def _check_transformers() -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "transformers not installed — FinBERT will use neutral fallback (0.5)"
            )
            return False

    def load_model(self) -> bool:
        """Lazy-load the FinBERT pipeline. Returns True on success."""
        if self._model_loaded:
            return True
        if not self._transformers_available:
            return False
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT model '%s' (first run may download ~400MB)...", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            self._model_loaded = True
            logger.info("FinBERT model loaded successfully")
            return True
        except Exception as e:
            logger.warning("FinBERT load_model failed: %s", e)
            self._pipeline = None
            return False

    # ------------------------------------------------------------------
    # News fetching
    # ------------------------------------------------------------------

    def fetch_news_with_urls(self, ticker: str) -> List[dict]:
        """Fetch recent news headlines + URLs for a ticker.

        Returns list of {"title": str, "url": str} dicts, filtered by lookback window.
        Falls back to fetch_news_headlines structure if URLs unavailable.
        """
        try:
            import datetime
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=self.lookback_days)
            cutoff_ts = int(cutoff.timestamp())

            t = yf.Ticker(ticker)
            news = t.news
            if not news:
                return []

            articles = []
            for item in news:
                content = item.get("content") or {}

                # Date filter
                pub_ts = item.get("providerPublishTime") or item.get("providerPublishedAt")
                if pub_ts is None:
                    pub_date_str = content.get("pubDate") or content.get("displayTime") or ""
                    if pub_date_str:
                        try:
                            pub_ts = int(datetime.datetime.fromisoformat(
                                pub_date_str.replace("Z", "+00:00")
                            ).timestamp())
                        except Exception:
                            pub_ts = None
                if pub_ts is not None and pub_ts < cutoff_ts:
                    continue

                title = (
                    content.get("title")
                    or item.get("title")
                    or item.get("headline")
                    or ""
                )
                if not title:
                    continue

                # URL: prefer canonicalUrl → clickThroughUrl → link
                url = (
                    (content.get("canonicalUrl") or {}).get("url", "")
                    or (content.get("clickThroughUrl") or {}).get("url", "")
                    or item.get("link", "")
                    or ""
                )

                articles.append({"title": str(title).strip(), "url": url})
                if len(articles) >= self.max_articles:
                    break

            return articles
        except Exception as e:
            logger.debug("fetch_news_with_urls failed for %s: %s", ticker, e)
            return []

    def fetch_news_headlines(self, ticker: str) -> List[str]:
        """Fetch recent news headlines for a ticker via yfinance.

        Handles both old yfinance structure (item["title"]) and new structure
        (item["content"]["title"]) introduced in yfinance >= 0.2.50.

        Returns a list of up to max_articles headline strings.
        Returns [] if no news or on any error.
        """
        try:
            import datetime
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=self.lookback_days)
            cutoff_ts = int(cutoff.timestamp())

            t = yf.Ticker(ticker)
            news = t.news
            if not news:
                logger.info("No news returned by yfinance for %s", ticker)
                return []

            headlines = []
            skipped_old = 0
            for item in news:
                # date filter — yfinance stores pubDate as unix timestamp in content or providerPublishTime
                content = item.get("content") or {}
                pub_ts = item.get("providerPublishTime") or item.get("providerPublishedAt")
                if pub_ts is None:
                    # new structure: content.pubDate is ISO string e.g. "2025-03-01T10:00:00Z"
                    pub_date_str = content.get("pubDate") or content.get("displayTime") or ""
                    if pub_date_str:
                        try:
                            pub_ts = int(datetime.datetime.fromisoformat(
                                pub_date_str.replace("Z", "+00:00")
                            ).timestamp())
                        except Exception:
                            pub_ts = None

                if pub_ts is not None and pub_ts < cutoff_ts:
                    skipped_old += 1
                    continue

                title = (
                    content.get("title")
                    or item.get("title")
                    or item.get("headline")
                    or ""
                )
                if title:
                    headlines.append(str(title).strip())
                if len(headlines) >= self.max_articles:
                    break

            logger.info(
                "Fetched %d headlines for %s (skipped %d older than %d days)",
                len(headlines), ticker, skipped_old, self.lookback_days,
            )
            return headlines
        except Exception as e:
            logger.debug("fetch_news_headlines failed for %s: %s", ticker, e)
            return []

    # ------------------------------------------------------------------
    # Sentiment scoring
    # ------------------------------------------------------------------

    def get_sentiment_score(self, ticker: str, headlines: List[str]) -> float:
        """Compute dividend bullishness score from a list of headlines.

        Scoring:
          - Run each headline through FinBERT → label + confidence score
          - positive label  → contribution = score
          - negative label  → contribution = -score
          - neutral label   → contribution = 0
          - Aggregate: mean contribution, scale to [0, 1]

        Returns NEUTRAL_SCORE if headlines is empty or model not loaded.
        """
        if not headlines:
            return NEUTRAL_SCORE

        if not self._model_loaded:
            loaded = self.load_model()
            if not loaded:
                log_model_skip(ticker, "finbert", "model not available")
                return NEUTRAL_SCORE

        try:
            contributions = []
            for headline in headlines:
                result = self._pipeline(headline[:512])[0]
                label = result["label"].lower()
                score = float(result["score"])
                if label == "positive":
                    contributions.append(score)
                elif label == "negative":
                    contributions.append(-score)
                else:
                    contributions.append(0.0)

            if not contributions:
                return NEUTRAL_SCORE

            mean_contribution = sum(contributions) / len(contributions)
            # Scale from [-1, 1] to [0, 1]
            scaled = (mean_contribution + 1.0) / 2.0
            return float(max(0.0, min(1.0, scaled)))

        except Exception as e:
            logger.warning("FinBERT inference failed for %s: %s", ticker, e)
            return NEUTRAL_SCORE

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def analyze_batch(self, tickers: List[str]) -> Dict[str, float]:
        """Return sentiment scores for all tickers.

        Always returns a score for every ticker (neutral fallback ensures this).
        """
        scores: Dict[str, float] = {}
        for ticker in tickers:
            headlines = self.fetch_news_headlines(ticker)
            scores[ticker] = self.get_sentiment_score(ticker, headlines)
            logger.info(
                "FinBERT score | %s | headlines=%d | score=%.3f",
                ticker, len(headlines), scores[ticker]
            )
        return scores

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded
