"""Fairness audit con Fairlearn e confronto con una mitigazione.

Convenzioni: y = 1 cattivo pagatore; predizione 1 = rifiuto.
Metriche per gruppo, nel linguaggio del credito:
- approval_rate: quota di richieste approvate;
- false_rejection_rate: buoni pagatori rifiutati (danno al cliente);
- false_approval_rate: cattivi pagatori approvati (danno alla banca);
- auc: capacità di ordinare il rischio dentro il gruppo.

Uso:
    python -m credit_xai.fairness      # scrive reports/fairness.json e docs/fairness_report.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.features import get_X_y
from credit_xai.model import CreditModel


def _approval_rate(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) == 0))


def _false_rejection_rate(y_true, y_pred) -> float:
    good = np.asarray(y_true) == 0
    return float(np.mean(np.asarray(y_pred)[good] == 1)) if good.any() else np.nan


def _false_approval_rate(y_true, y_pred) -> float:
    bad = np.asarray(y_true) == 1
    return float(np.mean(np.asarray(y_pred)[bad] == 0)) if bad.any() else np.nan


METRICS = {
    "approval_rate": _approval_rate,
    "false_rejection_rate": _false_rejection_rate,
    "false_approval_rate": _false_approval_rate,
}


def audit_attribute(y_true, y_pred, proba, sensitive: pd.Series) -> dict:
    """Metriche per gruppo + differenze riassuntive per un attributo sensibile."""
    frame = MetricFrame(metrics=METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    by_group = frame.by_group.copy()
    if set(by_group.index) <= set(config.AGE_GROUPS):  # ordine crescente delle fasce d'età
        by_group = by_group.reindex([g for g in config.AGE_GROUPS if g in by_group.index])
    by_group["n"] = sensitive.value_counts().reindex(by_group.index)
    by_group["auc"] = [
        roc_auc_score(y_true[sensitive == g], proba[sensitive == g])
        if y_true[sensitive == g].nunique() == 2
        else np.nan
        for g in by_group.index
    ]
    rates = by_group["approval_rate"]
    # demographic_parity_difference usa la selection rate (= rifiuti): la differenza
    # tra gruppi è identica a quella dei tassi di approvazione.
    return {
        "by_group": by_group.round(4).reset_index().to_dict(orient="records"),
        "demographic_parity_difference": round(
            float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        "equalized_odds_difference": round(
            float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)), 4
        ),
        # Regola dei 4/5: rapporto tra il tasso di approvazione minimo e massimo.
        "disparate_impact_ratio": round(float(rates.min() / rates.max()), 4)
        if rates.max() > 0
        else None,
    }


def audit(model: CreditModel, df: pd.DataFrame) -> dict:
    X, y = get_X_y(df)
    proba = model.prob_default(X)
    pred = model.predict(X)
    return {
        attr: audit_attribute(y, pred, proba, df[attr].astype(str))
        for attr in config.SENSITIVE_ATTRIBUTES
    }


class _Float64Proba(ClassifierMixin, BaseEstimator):
    """Adatta un modello già addestrato per Fairlearn.

    XGBoost restituisce probabilità float32; Fairlearn vi scrive valori float64,
    cosa che pandas 3 rifiuta. Il wrapper restituisce float64.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.asarray(self.pipeline.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def mitigate(
    model: CreditModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    attribute: str = "age_group",
    constraint: str = "equalized_odds",
) -> dict:
    """Post-processing con ThresholdOptimizer: soglie diverse per gruppo.

    Attenzione: richiede l'attributo sensibile al momento della decisione,
    cosa che può essere a sua volta problematica sul piano legale. Lo usiamo
    per misurare il compromesso accuratezza/equità, non come scelta di produzione.
    """
    X_tr, y_tr = get_X_y(train)
    X_te, y_te = get_X_y(test)
    opt = ThresholdOptimizer(
        estimator=_Float64Proba(model.pipeline).fit(None),
        constraints=constraint,
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    opt.fit(X_tr, y_tr, sensitive_features=train[attribute].astype(str).to_numpy())
    pred = pd.Series(
        opt.predict(
            X_te,
            sensitive_features=test[attribute].astype(str).to_numpy(),
            random_state=config.RANDOM_STATE,
        ),
        index=X_te.index,
    ).astype(int)
    proba = model.prob_default(X_te)
    before = audit_attribute(y_te, model.predict(X_te), proba, test[attribute].astype(str))
    after = audit_attribute(y_te, pred, proba, test[attribute].astype(str))

    def acc(p):
        return round(float((p == y_te).mean()), 4)

    return {
        "attribute": attribute,
        "constraint": constraint,
        "accuracy_before": acc(model.predict(X_te)),
        "accuracy_after": acc(pred),
        "equalized_odds_difference_before": before["equalized_odds_difference"],
        "equalized_odds_difference_after": after["equalized_odds_difference"],
        "demographic_parity_difference_before": before["demographic_parity_difference"],
        "demographic_parity_difference_after": after["demographic_parity_difference"],
    }


def write_report(results: dict, mitigation: dict, n_test: int) -> str:
    lines = [
        "# Fairness report",
        "",
        "Generato automaticamente da `python -m credit_xai.fairness` sul test set "
        f"({n_test} richieste). Gli attributi sensibili **non** sono feature del modello.",
        "",
        "Lettura: `approval_rate` = quota approvata; `false_rejection_rate` = buoni "
        "pagatori rifiutati; `false_approval_rate` = cattivi pagatori approvati. "
        "Un `disparate_impact_ratio` sotto 0,8 viola la regola dei 4/5.",
        "",
    ]
    for attr, res in results.items():
        table = pd.DataFrame(res["by_group"])
        lines += [
            f"## {attr}",
            "",
            table.to_markdown(index=False),
            "",
            f"- Differenza di demographic parity: **{res['demographic_parity_difference']}**",
            f"- Differenza di equalized odds: **{res['equalized_odds_difference']}**",
            f"- Disparate impact ratio: **{res['disparate_impact_ratio']}**",
            "",
        ]
    m = mitigation
    lines += [
        f"## Mitigazione: ThresholdOptimizer ({m['constraint']}) su `{m['attribute']}`",
        "",
        "Prima = modello con soglia unica a costo minimo; dopo = soglie per gruppo "
        "scelte per l'accuratezza bilanciata sotto il vincolo di equità. Le due "
        "accuratezze non vanno lette come un confronto puro: guardare soprattutto le "
        "differenze tra gruppi.",
        "",
        "| Metrica | Prima | Dopo |",
        "| --- | --- | --- |",
        f"| Accuratezza | {m['accuracy_before']} | {m['accuracy_after']} |",
        f"| Equalized odds diff. | {m['equalized_odds_difference_before']} "
        f"| {m['equalized_odds_difference_after']} |",
        f"| Demographic parity diff. | {m['demographic_parity_difference_before']} "
        f"| {m['demographic_parity_difference_after']} |",
        "",
        "## Limiti",
        "",
        "- Gruppi piccoli: con poche decine di richieste per gruppo le differenze "
        "possono essere rumore. Riportare sempre la colonna `n`.",
        "- La mitigazione usa l'attributo sensibile al momento della decisione: "
        "valutarne la liceità prima di un uso reale.",
        "- Escludere un attributo non basta: altre variabili possono farne da proxy.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    model = CreditModel.load()
    train, test = load_processed("train"), load_processed("test")
    results = audit(model, test)
    mitigation = mitigate(model, train, test)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "fairness.json").write_text(
        json.dumps({"audit": results, "mitigation": mitigation}, indent=2, ensure_ascii=False)
    )
    docs = config.PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "fairness_report.md").write_text(
        write_report(results, mitigation, len(test)), encoding="utf-8"
    )
    for attr, res in results.items():
        print(
            f"{attr:<15} DP diff {res['demographic_parity_difference']:.3f} | "
            f"EO diff {res['equalized_odds_difference']:.3f} | "
            f"DI ratio {res['disparate_impact_ratio']}"
        )
    print("Report scritto in docs/fairness_report.md")


if __name__ == "__main__":
    main()
