from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import (
    DATA_PATH,
    RISK_MODEL_PATH,
    RETRIEVER_PATH,
    TAGGER_PATH,
    query_documents,
    train_all,
)


app = FastAPI(
    title="IDCS - Investment Document Classification System",
    description=(
        "RAG-style investment research retrieval, ML document tagging, "
        "and chronic disease risk prediction API."
    ),
    version="2.0.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class TagRequest(BaseModel):
    text: str


class RiskRequest(BaseModel):
    age: float
    a1c: float
    systolic_bp: float
    bmi: float
    daily_steps: float


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "tagger_available": TAGGER_PATH.exists(),
        "retriever_available": RETRIEVER_PATH.exists(),
        "risk_model_available": RISK_MODEL_PATH.exists(),
        "dataset_available": Path(DATA_PATH).exists(),
    }


@app.post("/train")
def train() -> dict:
    return train_all()


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        return query_documents(request.question, request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Run /train before querying.") from exc


@app.post("/tag-document")
def tag_document(request: TagRequest) -> dict:
    if not TAGGER_PATH.exists():
        raise HTTPException(status_code=409, detail="Run /train before tagging documents.")
    model = joblib.load(TAGGER_PATH)
    label = model.predict([request.text])[0]
    confidence: Optional[float] = None
    if hasattr(model[-1], "predict_proba"):
        confidence = float(model.predict_proba([request.text]).max())
    return {"label": label, "confidence": confidence}


@app.post("/predict-risk")
def predict_risk(request: RiskRequest) -> dict:
    if not RISK_MODEL_PATH.exists():
        raise HTTPException(status_code=409, detail="Run /train before predicting risk.")
    artifact = joblib.load(RISK_MODEL_PATH)
    features = artifact["features"]
    model = artifact["model"]
    row = pd.DataFrame([request.dict()])[features]
    risk_flag = int(model.predict(row)[0])
    probability = None
    if hasattr(model[-1], "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])
    return {"high_risk": bool(risk_flag), "risk_probability": probability}
