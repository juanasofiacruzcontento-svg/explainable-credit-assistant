"""Monitoraggio: log delle decisioni e data drift (PSI).

- Ogni decisione dell'API viene scritta in logs/decisions.jsonl (tracciabilità,
  cfr. AI Act art. 12).
- Il drift si misura con il Population Stability Index tra i dati di training
  (riferimento) e quelli recenti. Regola pratica: PSI < 0.1 stabile,
  0.1-0.2 da osservare, > 0.2 drift significativo.

Uso:
    python -m credit_xai.monitoring              # confronta train vs richieste loggate
    python -m credit_xai.monitoring --simulate   # demo: simula un drift sul test set

Per report grafici più ricchi si può usare Evidently sugli stessi due DataFrame.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

DECISIONS_LOG = config.PROJECT_ROOT / "logs" / "decisions.jsonl"
EPS = 1e-4


def log_decision(features: dict, prob_default: float, approved: bool, model_version: str) -> str:
    """Aggiunge una decisione al log e restituisce il suo id."""
    DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    decision_id = str(uuid.uuid4())
    record = {
        "id": decision_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": model_version,
        "features": features,
        "prob_default": round(float(prob_default), 4),
        "approved": bool(approved),
    }
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return decision_id


def load_logged() -> pd.DataFrame:
    if not DECISIONS_LOG.exists():
        return pd.DataFrame(columns=MODEL_FEATURES)
    rows = [json.loads(line) for line in DECISIONS_LOG.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame([r["features"] for r in rows])


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    return _psi(ref, cur)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref, cur)


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MODEL_FEATURES:
        fn = psi_numeric if col in NUMERIC_FEATURES else psi_categorical
        value = fn(reference[col], current[col])
        status = "stabile" if value < 0.1 else "da osservare" if value < 0.2 else "DRIFT"
        rows.append({"feature": col, "psi": round(value, 4), "stato": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Demo: importi più alti del 40% e più richieste senza risparmi."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["credit_amount"] = (out["credit_amount"] * 1.4).round().astype(int)
    mask = rng.random(len(out)) < 0.3
    out.loc[mask, "savings"] = config.CODEBOOK["savings"]["A61"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    reference = load_processed("train")
    if args.simulate:
        current = simulate_drift(load_processed("test"))
        source = "test set con drift simulato"
    else:
        current = load_logged()
        source = f"{len(current)} richieste in {DECISIONS_LOG.name}"
        if len(current) < 30:
            print(f"Solo {len(current)} richieste loggate: servono almeno 30. Prova --simulate.")
            return
    report = drift_report(reference, current[MODEL_FEATURES])
    print(f"Drift: train vs {source}\n")
    print(report.to_string(index=False))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.REPORTS_DIR / "drift.csv", index=False)
    unused = [c for c in CATEGORICAL_FEATURES if c not in current.columns]
    if unused:
        print("Colonne mancanti:", unused)


if __name__ == "__main__":
    main()
