"""Loguru-based logging setup. Idempotent."""

from __future__ import annotations

import sys

from loguru import logger

from core.settings import get_settings

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


def setup_logging() -> None:
    """Configure loguru with the level from settings.

    Naturally idempotent: ``logger.remove()`` clears existing handlers
    before re-installing ours, so any number of calls converges to the
    same end state.
    """
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=_LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=False,  # don't dump variable values — risk of leaking secrets
    )
