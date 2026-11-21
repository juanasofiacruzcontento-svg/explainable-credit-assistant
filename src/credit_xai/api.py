"""API REST con FastAPI.

Avvio:
    uvicorn credit_xai.api:app --reload
Documentazione interattiva: http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from credit_xai import config, rag, service
from credit_xai.features import CATEGORICAL_FEATURES

_NUMERIC_FIELDS = {
    "duration_months": (int, Field(..., ge=1, le=120, description="Durata in mesi")),
    "credit_amount": (int, Field(..., ge=100, le=100_000, description="Importo in DM")),
    "installment_rate": (int, Field(..., ge=1, le=4, description="Rata in % del reddito, 1-4")),
    "residence_since": (int, Field(..., ge=1, le=4, description="Anni di residenza, 1-4")),
    "age": (int, Field(..., ge=18, le=100)),
    "existing_credits": (int, Field(..., ge=1, le=10)),
    "people_liable": (int, Field(..., ge=1, le=10)),
}
_CATEGORICAL_FIELDS = {
    f: (Literal[tuple(config.CODEBOOK[f].values())], Field(...)) for f in CATEGORICAL_FEATURES
}

# Schema generato dalla configurazione: i valori ammessi restano allineati al codebook.
CreditApplication = create_model("CreditApplication", **_NUMERIC_FIELDS, **_CATEGORICAL_FIELDS)


class Question(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)


app = FastAPI(
    title="Explainable Credit Assistant",
    description="Credit scoring con spiegazioni SHAP, controfattuali e narrazione LLM verificata.",
    version="1.0.0",
)


def _features(application: BaseModel) -> dict:
    return application.model_dump()


@app.get("/health")
def health() -> dict:
    try:
        model = service.get_model()
        return {"status": "ok", "model_version": model.trained_at, "threshold": model.threshold}
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/options")
def options() -> dict:
    return service.options()


@app.get("/example")
def example(rejected: bool | None = None) -> dict:
    return service.example(rejected=rejected)


@app.post("/predict")
def predict(application: CreditApplication) -> dict:
    return service.predict(_features(application))


@app.post("/explain")
def explain(application: CreditApplication, counterfactuals: bool = True) -> dict:
    return service.explain(_features(application), counterfactuals=counterfactuals)


@app.post("/ask")
def ask(q: Question) -> dict:
    return rag.answer(q.question)
