"""Selezione delle feature e preprocessing per il modello."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_xai import config

CATEGORICAL_FEATURES = [c for c in config.CODEBOOK if c not in config.EXCLUDED_FROM_MODEL]
NUMERIC_FEATURES = list(config.NUMERIC_FEATURES)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa le feature del modello dal target (gli attributi sensibili restano fuori)."""
    return df[MODEL_FEATURES].copy(), df[config.TARGET].astype(int)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """One-hot per le categoriche; standardizzazione delle numeriche solo se serve.

    La regressione logistica è sensibile alla scala delle variabili, gli alberi
    (XGBoost) no: per loro le numeriche passano invariate.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def original_feature(encoded_name: str) -> str:
    """Riporta una colonna one-hot (es. 'savings_< 100 DM') alla feature originale."""
    for feat in CATEGORICAL_FEATURES:
        if encoded_name.startswith(f"{feat}_"):
            return feat
    return encoded_name
