# src/api/fastapi_app.py
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generative.llm_rag_app import answer_with_rag
from src.models.load_model import load_sklearn_model
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI/ML Governance Demo API",
    description="Demo API exposing a classical ML model and a simple RAG endpoint.",
    version="0.1.0",
)

# Load model at startup
model = None


class PredictRequest(BaseModel):
    feature1: float = Field(..., description="First numeric feature")
    feature2: float = Field(..., description="Second numeric feature")
    feature3: float = Field(..., description="Third numeric feature")


class PredictResponse(BaseModel):
    prediction: int
    probability: float


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=3)


class RAGResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = None


@app.on_event("startup")
def on_startup():
    global model
    try:
        model = load_sklearn_model()
        logger.info("Model loaded successfully on startup.")
    except FileNotFoundError as e:
        logger.warning("Model not found at startup: {}", e)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    X = np.array([[req.feature1, req.feature2, req.feature3]])
    if not hasattr(model, "predict_proba"):
        logger.error("Model does not implement predict_proba.")
        raise HTTPException(status_code=500, detail="Model misconfigured.")

    proba = model.predict_proba(X)[0]
    pred = int(proba.argmax())
    logger.info(
        "Prediction requested: features={}, prediction={}, probability={}",
        [req.feature1, req.feature2, req.feature3],
        pred,
        float(proba.max()),
    )
    return PredictResponse(prediction=pred, probability=float(proba.max()))


@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    logger.info("RAG question received: {}", req.question)
    answer, sources = answer_with_rag(req.question)
    return RAGResponse(answer=answer, sources=sources)
