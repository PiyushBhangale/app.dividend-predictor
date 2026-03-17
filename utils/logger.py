"""utils/logger.py — Structured logging helpers for the dividend predictor pipeline.

Every data-fetch miss or model fallback must be logged, not swallowed.
All loggers write to stdout so they appear in Streamlit's terminal output.
"""

import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger writing to stdout.

    Format: [LEVEL] [module] message
    Uses a NullHandler guard so importing this module never adds duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_missing_data(ticker: str, field: str, fallback: Any) -> None:
    """Log a WARNING when a data field is missing for a ticker.

    Args:
        ticker:   NSE ticker symbol e.g. 'HINDZINC.NS'
        field:    The missing field name e.g. 'pe_ratio'
        fallback: The value being used in place of the missing data
    """
    logger = get_logger("data_pipeline")
    logger.warning(
        "Missing data | ticker=%s | field=%s | fallback=%s",
        ticker, field, fallback
    )


def log_model_skip(ticker: str, model: str, reason: str) -> None:
    """Log a WARNING when a model is skipped for a ticker.

    Args:
        ticker: NSE ticker symbol
        model:  Model name e.g. 'lstm', 'finbert'
        reason: Human-readable reason e.g. 'insufficient dividend history (3 records)'
    """
    logger = get_logger("models")
    logger.warning(
        "Model skipped | ticker=%s | model=%s | reason=%s",
        ticker, model, reason
    )


def log_fetch_error(ticker: str, operation: str, error: Exception) -> None:
    """Log a WARNING when a yfinance fetch fails.

    Args:
        ticker:    NSE ticker symbol
        operation: Description of the failed operation e.g. 'fetch_info'
        error:     The caught exception
    """
    logger = get_logger("fetcher")
    logger.warning(
        "Fetch error | ticker=%s | op=%s | error=%s",
        ticker, operation, str(error)
    )
