"""Logica applicativa condivisa da API e interfaccia Streamlit."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.explain.counterfactuals import CounterfactualGenerator
from credit_xai.explain.shap_explainer import ShapExplainer
from credit_xai.features import MODEL_FEATURES, NUMERIC_FEATURES
from credit_xai.llm.narrator import narrate
from credit_xai.model import get_model
from credit_xai.monitoring import log_decision


@lru_cache(maxsize=1)
def get_explainer() -> ShapExplainer:
    return ShapExplainer(get_model())


@lru_cache(maxsize=1)
def get_cf_generator() -> CounterfactualGenerator:
    return CounterfactualGenerator(get_model(), load_processed("train"))


def to_frame(features: dict) -> pd.DataFrame:
    row = pd.DataFrame([{f: features[f] for f in MODEL_FEATURES}])
    for col in NUMERIC_FEATURES:
        row[col] = row[col].astype(int)
    return row


def predict(features: dict, log: bool = True) -> dict:
    model = get_model()
    row = to_frame(features)
    p = float(model.prob_default(row).iloc[0])
    approved = p < model.threshold
    decision_id = log_decision(features, p, approved, model.trained_at) if log else None
    return {
        "decision_id": decision_id,
        "approved": approved,
        "prob_default": round(p, 4),
        "threshold": model.threshold,
        "model_version": model.trained_at,
    }


def explain(features: dict, counterfactuals: bool = True, log: bool = True) -> dict:
    decision = predict(features, log=log)
    row = to_frame(features)
    explanation = get_explainer().explain_one(row)
    cfs = (
        get_cf_generator().generate(row) if counterfactuals and not explanation["approved"] else []
    )
    return {
        **decision,
        "contributions": explanation["contributions"],
        "base_value": explanation["base_value"],
        "counterfactuals": cfs,
        "narration": narrate(explanation, cfs),
    }


def example(seed: int | None = None, rejected: bool | None = None) -> dict:
    """Una richiesta reale dal test set (utile per demo e test)."""
    test = load_processed("test")
    if rejected is not None:
        approved = get_model().approve(test)
        test = test[~approved] if rejected else test[approved]
    row = test.sample(1, random_state=seed).iloc[0]
    return {f: (int(row[f]) if f in NUMERIC_FEATURES else str(row[f])) for f in MODEL_FEATURES}


def options() -> dict:
    """Valori ammessi per le feature categoriche (per form e validazione)."""
    return {f: list(config.CODEBOOK[f].values()) for f in MODEL_FEATURES if f in config.CODEBOOK}
