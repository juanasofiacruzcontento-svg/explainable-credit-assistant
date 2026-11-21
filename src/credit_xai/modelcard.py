"""Genera docs/model_card.md dai risultati salvati in reports/.

La struttura segue "Model Cards for Model Reporting" (Mitchell et al., 2019)
ed è organizzata per coprire gli elementi di documentazione richiesti ai sistemi
ad alto rischio dall'AI Act (artt. 11, 13, 14, 15).
"""

from __future__ import annotations

import json

from credit_xai import config
from credit_xai.features import MODEL_FEATURES


def _load(name: str) -> dict:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def build() -> str:
    metrics = _load("metrics.json")
    fairness = _load("fairness.json")
    faith = _load("faithfulness.json")
    xgb = metrics.get("xgboost", {})
    lr = metrics.get("logistic_regression", {})

    def row(name, key):
        return f"| {name} | {lr.get(key, '-')} | {xgb.get(key, '-')} |"

    fair_lines = []
    for attr, res in fairness.get("audit", {}).items():
        fair_lines.append(
            f"| {attr} | {res['demographic_parity_difference']} | "
            f"{res['equalized_odds_difference']} | {res['disparate_impact_ratio']} |"
        )

    return f"""# Model card: Explainable Credit Assistant

> Progetto dimostrativo su dati pubblici. Non usare per decisioni reali di credito.

## 1. Scopo e uso previsto

- **Compito**: stimare la probabilità che un richiedente non ripaghi un prestito
  (classificazione binaria) e spiegare la decisione.
- **Utenti previsti**: analisti del credito, come supporto e non sostituto della
  valutazione umana (AI Act, art. 14).
- **Usi non previsti**: decisioni completamente automatiche senza revisione umana;
  popolazioni o periodi diversi dai dati di addestramento.
- **Classificazione normativa**: valutazione del merito creditizio di persone fisiche,
  sistema ad alto rischio (AI Act, Allegato III, punto 5(b)).

## 2. Dati

- **Fonte**: Statlog German Credit (UCI), 1.000 richieste, Germania, anni '70-'90,
  importi in marchi tedeschi (DM). Tasso di cattivi pagatori circa 30%.
- **Split**: 80% train / 20% test, stratificato sul target.
- **Feature usate ({len(MODEL_FEATURES)})**: {", ".join(MODEL_FEATURES)}.
- **Escluse dal modello**: {", ".join(config.EXCLUDED_FROM_MODEL)}
  (attributi protetti e `personal_status`, che codifica anche il sesso).
- **Età**: mantenuta come feature perché considerata un fattore di rischio
  legittimo nel credito; il suo effetto è monitorato nel fairness audit
  (`age_group`).
- **Limiti**: dataset piccolo, datato e non rappresentativo della popolazione
  attuale; etichette di qualità non documentata.

## 3. Modello

- **Modello servito**: XGBoost dentro una pipeline scikit-learn (one-hot encoding),
  iperparametri scelti con Optuna (AUC in cross-validation a 5 fold).
- **Baseline**: regressione logistica con classi bilanciate.
- **Soglia di decisione**: {xgb.get("threshold", "-")} sulla probabilità di default,
  scelta su predizioni out-of-fold per minimizzare il costo atteso, con la matrice dei
  costi UCI: concedere credito a un cattivo pagatore costa
  {config.COST_FALSE_NEGATIVE:g}, rifiutarlo a un buon pagatore costa
  {config.COST_FALSE_POSITIVE:g}.

## 4. Prestazioni (test set)

| Metrica | Logistica | XGBoost |
| --- | --- | --- |
{row("AUC", "auc")}
{row("AUC in CV (train)", "cv_auc")}
{row("F1 (classe cattivo pagatore)", "f1")}
{row("Precision", "precision")}
{row("Recall", "recall")}
{row("Brier score", "brier")}
{row("Costo medio per richiesta", "expected_cost")}
{row("Tasso di approvazione", "approval_rate")}

## 5. Equità

Valutata sul test set per gruppi di sesso, fascia d'età e lavoratore straniero.
Dettagli in [fairness_report.md](fairness_report.md).

| Attributo | Diff. demographic parity | Diff. equalized odds | Disparate impact ratio |
| --- | --- | --- | --- |
{chr(10).join(fair_lines) or "| - | - | - | - |"}

Un disparate impact ratio sotto 0,8 indica una possibile discriminazione indiretta
da approfondire prima di qualsiasi uso.

## 6. Spiegabilità

- **Locale**: TreeSHAP, contributi aggregati per feature originale (esatti grazie
  all'additività).
- **Controfattuali**: DiCE, limitati alle feature su cui il cliente può agire
  ({", ".join(config.ACTIONABLE_FEATURES)}), validati con la soglia reale del modello
  e filtrati per plausibilità.
- **Narrazione**: testo generato da template o LLM e verificato automaticamente
  contro i valori SHAP (fattori inventati, direzione, copertura, attributi protetti,
  coerenza dell'esito).
- **Fedeltà misurata**: {json.dumps(faith, ensure_ascii=False) if faith else "esegui `python -m credit_xai.evaluation.faithfulness`"}

## 7. Supervisione umana e trasparenza

- Ogni spiegazione ricorda al cliente il diritto a un riesame umano
  (GDPR art. 22, direttiva (UE) 2023/2225 art. 18, AI Act art. 86).
- Ogni decisione è registrata in `logs/decisions.jsonl` con id, timestamp, versione
  del modello, input e output (AI Act, art. 12).
- Il drift dei dati è monitorato con il PSI (`python -m credit_xai.monitoring`).

## 8. Rischi e limiti noti

- Le spiegazioni SHAP descrivono il modello, non la causalità nel mondo reale.
- I controfattuali indicano cosa cambierebbe la previsione del modello, non garantiscono
  l'approvazione e possono proporre modifiche ampie (es. durate molto più brevi).
- Variabili come `job`, `housing` o `property` possono fare da proxy di attributi protetti.
- Un LLM può produrre spiegazioni plausibili ma infedeli: per questo ogni narrazione è
  verificata e, in caso di errore del provider, si usa il template.
"""


def main() -> None:
    out = config.PROJECT_ROOT / "docs" / "model_card.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"Model card scritta in {out}")


if __name__ == "__main__":
    main()
