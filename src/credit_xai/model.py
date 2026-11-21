"""Interfaccia unica al modello addestrato: probabilità, decisione, metadati."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_xai import config
from credit_xai.features import MODEL_FEATURES


@dataclass
class CreditModel:
    pipeline: Pipeline
    threshold: float
    metrics: dict
    trained_at: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> CreditModel:
        path = path or config.MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} non trovato: esegui `python -m credit_xai.train`")
        art = joblib.load(path)
        return cls(art["pipeline"], art["threshold"], art.get("metrics", {}), art["trained_at"])

    def prob_default(self, X: pd.DataFrame) -> pd.Series:
        """Probabilità stimata che il richiedente non ripaghi."""
        return pd.Series(self.pipeline.predict_proba(X[MODEL_FEATURES])[:, 1], index=X.index)

    def approve(self, X: pd.DataFrame) -> pd.Series:
        """True = credito concesso (probabilità di default sotto la soglia)."""
        return self.prob_default(X) < self.threshold

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Interfaccia sklearn-like: 1 = rifiuto (default previsto), 0 = concessione."""
        return (~self.approve(X)).astype(int)


@lru_cache(maxsize=1)
def get_model() -> CreditModel:
    """Carica il modello una sola volta (usato da API e app)."""
    return CreditModel.load()
