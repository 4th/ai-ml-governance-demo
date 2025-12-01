# src/config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (safe for local dev; in prod use real secrets manager)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "src" / "models" / "model.pkl"))
ONNX_MODEL_PATH = Path(
    os.getenv("ONNX_MODEL_PATH", BASE_DIR / "src" / "models" / "model.onnx")
)

# Logging
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# LLM config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API config
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
