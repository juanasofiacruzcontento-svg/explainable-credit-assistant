"""Spiegazioni SHAP riportate alle feature originali.

Il modello vede colonne one-hot (es. 'savings_< 100 DM'); per le persone ha senso
parlare di 'risparmi'. Qui sommiamo i contributi SHAP delle colonne one-hot
della stessa feature: per TreeSHAP la somma è additiva, quindi resta esatta.

Convenzione: SHAP > 0 = aumenta il rischio di default (spinge verso il rifiuto).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import shap

from credit_xai import config
from credit_xai.features import MODEL_FEATURES, NUMERIC_FEATURES, original_feature
from credit_xai.model import CreditModel


@dataclass
class Contribution:
    feature: str
    label: str
    value: str
    shap: float
    direction: str  # "aumenta il rischio" | "riduce il rischio"


class ShapExplainer:
    def __init__(self, model: CreditModel):
        self.model = model
        self.prep = model.pipeline.named_steps["prep"]
        self.explainer = shap.TreeExplainer(model.pipeline.named_steps["clf"])
        self.encoded_names = list(self.prep.get_feature_names_out())
        self.groups = [original_feature(n) for n in self.encoded_names]

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Valori SHAP (log-odds di default) aggregati per feature originale."""
        Xt = self.prep.transform(X[MODEL_FEATURES])
        values = np.asarray(self.explainer.shap_values(Xt))
        encoded = pd.DataFrame(values, columns=self.encoded_names, index=X.index)
        return encoded.T.groupby(self.groups).sum().T[MODEL_FEATURES]

    @property
    def base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])

    def explain_one(self, row: pd.DataFrame) -> dict:
        """Spiegazione completa di una singola richiesta (DataFrame di una riga)."""
        if len(row) != 1:
            raise ValueError("explain_one richiede esattamente una riga")
        sv = self.shap_values(row).iloc[0].sort_values(key=np.abs, ascending=False)
        contributions = [
            Contribution(
                feature=f,
                label=config.FEATURE_LABELS.get(f, f),
                value=_fmt(row.iloc[0][f], f),
                shap=round(float(v), 4),
                direction="aumenta il rischio" if v > 0 else "riduce il rischio",
            )
            for f, v in sv.items()
        ]
        p = float(self.model.prob_default(row).iloc[0])
        return {
            "prob_default": round(p, 4),
            "threshold": self.model.threshold,
            "approved": p < self.model.threshold,
            "base_value": round(self.base_value, 4),
            "contributions": [asdict(c) for c in contributions],
        }

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Importanza globale: media del valore assoluto SHAP per feature."""
        return self.shap_values(X).abs().mean().sort_values(ascending=False)


def top_features(explanation: dict, k: int = 5) -> list[str]:
    return [c["feature"] for c in explanation["contributions"][:k]]


def stability(
    explainer: ShapExplainer,
    row: pd.DataFrame,
    n: int = 30,
    noise: float = 0.05,
    k: int = 5,
    seed: int = 0,
) -> float:
    """Quanto è stabile la spiegazione sotto piccole perturbazioni?

    Perturba le feature numeriche di ±noise (relativo) e misura la sovrapposizione
    media (Jaccard) tra le top-k feature originali e quelle perturbate. 1 = stabile.
    """
    rng = np.random.default_rng(seed)
    base = set(top_features(explainer.explain_one(row), k))
    perturbed = pd.concat([row] * n, ignore_index=True)
    for col in NUMERIC_FEATURES:
        factor = 1 + rng.uniform(-noise, noise, size=n)
        perturbed[col] = np.maximum(1, np.round(perturbed[col] * factor)).astype(int)
    sv = explainer.shap_values(perturbed)
    scores = []
    for _, r in sv.iterrows():
        top = set(r.abs().sort_values(ascending=False).index[:k])
        scores.append(len(base & top) / len(base | top))
    return float(np.mean(scores))


def _fmt(value, feature: str) -> str:
    if feature == "duration_months":
        return f"{int(value)} mesi"
    if feature == "credit_amount":
        return f"{int(value):,} DM".replace(",", ".")
    if feature == "age":
        return f"{int(value)} anni"
    if feature == "installment_rate":
        return f"{int(value)} (scala 1-4)"
    return str(value)
