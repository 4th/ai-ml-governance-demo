# src/utils/logging_utils.py
from __future__ import annotations

from loguru import logger

from src.config import LOG_DIR


# Configure a global logger once
logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)


def get_logger(name: str):
    """
    Returns a logger bound with a module/context name.
    """
    return logger.bind(module=name)
