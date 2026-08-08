"""
Structured logging configuration shared by the whole backend.

A single place to switch formats (plain text in development, JSON in
production) without touching call sites.
"""

import logging
import sys

from app.core.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int | None = None) -> None:
    """Configure root logging once at application startup."""
    resolved = level if level is not None else (
        logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO
    )
    logging.basicConfig(
        level=resolved,
        format=_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
