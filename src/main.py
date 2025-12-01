# src/main.py
from __future__ import annotations

import uvicorn

from src.api.fastapi_app import app
from src.config import API_HOST, API_PORT, DEBUG


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=DEBUG)
