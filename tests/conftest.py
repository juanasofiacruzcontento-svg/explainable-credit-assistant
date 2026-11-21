"""Fixture condivise.

I test usano un file sintetico nello stesso formato del German Credit di UCI e
cartelle temporanee: non servono rete, chiavi API né i dati veri. Girano in CI.
"""

import numpy as np
import pandas as pd
import pytest

from credit_xai import config


def make_synthetic_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for col in config.RAW_COLUMNS:
        if col in config.CODEBOOK:
            data[col] = rng.choice(list(config.CODEBOOK[col]), size=n)
        elif col == "personal_status_sex":
            data[col] = rng.choice(list(config.PERSONAL_STATUS_SEX), size=n)
        elif col == "class":
            data[col] = 1  # calcolato sotto, con un segnale realistico
        elif col == "age":
            data[col] = rng.integers(19, 75, size=n)
        elif col == "credit_amount":
            data[col] = rng.integers(250, 18000, size=n)
        elif col == "duration_months":
            data[col] = rng.integers(4, 72, size=n)
        else:
            data[col] = rng.integers(1, 5, size=n)
    df = pd.DataFrame(data, columns=config.RAW_COLUMNS)

    # Rischio sintetico: conto in rosso, prestiti lunghi e pochi risparmi
    # aumentano la probabilità di default (come nei dati reali).
    logit = (
        -1.4
        + 1.2 * (df["checking_status"] == "A11")
        - 1.0 * (df["checking_status"] == "A14")
        + 0.03 * (df["duration_months"] - 24)
        + 0.6 * (df["savings"] == "A61")
        - 0.8 * (df["credit_history"] == "A34")
        - 0.015 * (df["age"] - 35)
    )
    p_bad = 1 / (1 + np.exp(-logit))
    df["class"] = np.where(rng.random(n) < p_bad, 2, 1)
    return df


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "german.data"
    make_synthetic_raw().to_csv(path, sep=" ", header=False, index=False)
    return path


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Progetto completo in una cartella temporanea: dati, modello e report."""
    from credit_xai import data, model, monitoring, train

    root = tmp_path_factory.mktemp("project")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RAW_FILE", root / "raw" / "german.data")
    mp.setattr(config, "PROCESSED_DIR", root / "processed")
    mp.setattr(config, "MODELS_DIR", root / "models")
    mp.setattr(config, "MODEL_FILE", root / "models" / "model.joblib")
    mp.setattr(config, "REPORTS_DIR", root / "reports")
    mp.setattr(config, "METRICS_FILE", root / "reports" / "metrics.json")
    mp.setattr(monitoring, "DECISIONS_LOG", root / "logs" / "decisions.jsonl")
    mp.setenv("LLM_PROVIDER", "template")

    config.RAW_FILE.parent.mkdir(parents=True)
    make_synthetic_raw(600).to_csv(config.RAW_FILE, sep=" ", header=False, index=False)
    data.build_dataset()
    train.train_all(n_trials=2)
    model.get_model.cache_clear()
    yield root
    model.get_model.cache_clear()
    mp.undo()


@pytest.fixture(scope="session")
def credit_model(project):
    from credit_xai.model import CreditModel

    return CreditModel.load()
