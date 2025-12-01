# load model
# src/models/load_model.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.config import MODEL_PATH
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache()
def load_sklearn_model(path: Path | None = None) -> Any:
    """
    Load and cache a scikit-learn model from disk.
    """
    model_path = path or MODEL_PATH
    if not model_path.exists():
        msg = f"Model file not found at {model_path}. Have you run train_example_model?"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading model from {}", model_path)
    model = joblib.load(model_path)
    return model
