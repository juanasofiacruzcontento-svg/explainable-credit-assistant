"""Controfattuali con DiCE: 'cosa dovrebbe cambiare per ottenere il credito?'.

Vincoli:
- si possono cambiare solo le feature in config.ACTIONABLE_FEATURES;
- un controfattuale è valido solo se il NOSTRO modello, con la NOSTRA soglia
  (non 0.5), concederebbe il credito;
- si preferiscono i controfattuali con meno modifiche.
"""

from __future__ import annotations

import warnings

import dice_ml
import pandas as pd

from credit_xai import config
from credit_xai.explain.shap_explainer import _fmt
from credit_xai.features import MODEL_FEATURES, NUMERIC_FEATURES
from credit_xai.model import CreditModel


class CounterfactualGenerator:
    def __init__(self, model: CreditModel, train: pd.DataFrame):
        self.model = model
        data = train[MODEL_FEATURES + [config.TARGET]]
        self.dice = dice_ml.Dice(
            dice_ml.Data(
                dataframe=data,
                continuous_features=NUMERIC_FEATURES,
                outcome_name=config.TARGET,
            ),
            dice_ml.Model(model=model.pipeline, backend="sklearn"),
            method="random",
        )

    def generate(self, row: pd.DataFrame, n: int = 3, seed: int = 0) -> list[dict]:
        """Restituisce fino a n controfattuali validi, dal più semplice."""
        query = row[MODEL_FEATURES]
        if bool(self.model.approve(query).iloc[0]):
            return []  # già approvato: niente da cambiare
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.dice.generate_counterfactuals(
                query,
                total_CFs=n * 5,
                desired_class=0,
                stopping_threshold=min(0.99, 1 - self.model.threshold + 0.02),
                features_to_vary=config.ACTIONABLE_FEATURES,
                random_seed=seed,
                verbose=False,
            )
        cfs = result.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            return []
        cfs = cfs[MODEL_FEATURES].reset_index(drop=True)
        probs = self.model.prob_default(cfs)
        original = query.iloc[0]

        out = []
        for i, cf in cfs.iterrows():
            if probs[i] >= self.model.threshold:
                continue  # non abbastanza per la nostra soglia
            if not _plausible(original, cf):
                continue
            changes = [
                {
                    "feature": f,
                    "label": config.FEATURE_LABELS.get(f, f),
                    "from": _fmt(original[f], f),
                    "to": _fmt(cf[f], f),
                }
                for f in MODEL_FEATURES
                if str(original[f]) != str(cf[f])
            ]
            if changes:
                out.append({"prob_default": round(float(probs[i]), 4), "changes": changes})

        # dedup e ordinamento: meno modifiche prima, poi probabilità più bassa
        unique = {tuple((c["feature"], c["to"]) for c in o["changes"]): o for o in out}
        ranked = sorted(unique.values(), key=lambda o: (len(o["changes"]), o["prob_default"]))
        return ranked[:n]


def _plausible(original: pd.Series, cf: pd.Series) -> bool:
    """Scarta modifiche che non avrebbe senso consigliare (vedi config)."""
    for feat, banned in config.IMPLAUSIBLE_TARGETS.items():
        if str(cf[feat]) != str(original[feat]) and cf[feat] in banned:
            return False
    return all(cf[feat] <= original[feat] for feat in config.ONLY_DECREASE)
