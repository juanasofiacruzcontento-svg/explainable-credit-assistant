"""Training: baseline logistica, XGBoost con tuning Optuna, soglia a costo minimo.

Uso:
    python -m credit_xai.train              # 40 trial di Optuna
    python -m credit_xai.train --trials 100
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.features import MODEL_FEATURES, build_preprocessor, get_X_y

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)


# --------------------------------------------------------------------- modelli
def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def build_xgb(**params) -> Pipeline:
    defaults = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    defaults.update(params)
    return Pipeline(
        [("prep", build_preprocessor(scale_numeric=False)), ("clf", XGBClassifier(**defaults))]
    )


def tune_xgb(X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    """Cerca gli iperparametri che massimizzano l'AUC in cross-validation."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        }
        return cross_val_score(build_xgb(**params), X, y, cv=CV, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------- metriche
def expected_cost(y_true, y_pred) -> float:
    """Costo medio per richiesta secondo la matrice dei costi UCI."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = config.COST_FALSE_NEGATIVE * fn + config.COST_FALSE_POSITIVE * fp
    return float(cost / len(y_true))


def choose_threshold(y_true, proba) -> float:
    """Soglia sulla probabilità di default che minimizza il costo atteso.

    Sopra la soglia la richiesta è rifiutata. Va scelta su predizioni
    out-of-fold del train, mai sul test.
    """
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y_true, proba, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "expected_cost": round(expected_cost(y_true, y_pred), 4),
        "approval_rate": round(float((y_pred == 0).mean()), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------- MLflow
def _log_mlflow(name: str, params: dict, metrics: dict) -> None:
    """Logga su MLflow se installato (pip install mlflow); altrimenti non fa nulla."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"file:{config.PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("explainable-credit-assistant")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})


# --------------------------------------------------------------------- pipeline
def fit_and_evaluate(name: str, model: Pipeline, X_tr, y_tr, X_te, y_te) -> tuple[Pipeline, dict]:
    oof = cross_val_predict(model, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_tr, oof)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "threshold": round(threshold, 3),
        "cv_auc": round(float(roc_auc_score(y_tr, oof)), 4),
    }
    metrics.update(evaluate(y_te, proba, threshold))
    print(
        f"{name:<10} AUC test {metrics['auc']:.3f} | F1 {metrics['f1']:.3f} | "
        f"costo medio {metrics['expected_cost']:.3f} | soglia {threshold:.2f}"
    )
    return model, metrics


def train_all(n_trials: int = 40) -> dict:
    X_tr, y_tr = get_X_y(load_processed("train"))
    X_te, y_te = get_X_y(load_processed("test"))

    results = {}
    _, m = fit_and_evaluate("logistica", build_logreg(), X_tr, y_tr, X_te, y_te)
    results["logistic_regression"] = m
    _log_mlflow("logistic_regression", {"class_weight": "balanced"}, m)

    print(f"Tuning XGBoost con Optuna ({n_trials} trial)...")
    best = tune_xgb(X_tr, y_tr, n_trials=n_trials)
    xgb, m = fit_and_evaluate("xgboost", build_xgb(**best), X_tr, y_tr, X_te, y_te)
    m["params"] = best
    results["xgboost"] = m
    _log_mlflow("xgboost", best, m)

    # Il modello servito è XGBoost: serve un modello ad albero per TreeSHAP.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": xgb,
            "threshold": m["threshold"],
            "features": MODEL_FEATURES,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "metrics": m,
        },
        config.MODEL_FILE,
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Modello salvato in {config.MODEL_FILE}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    train_all(parser.parse_args().trials)


if __name__ == "__main__":
    main()
