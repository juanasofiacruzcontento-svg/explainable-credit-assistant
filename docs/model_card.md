# Model card: Explainable Credit Assistant

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
- **Feature usate (18)**: duration_months, credit_amount, installment_rate, residence_since, age, existing_credits, people_liable, checking_status, credit_history, purpose, savings, employment_since, other_debtors, property, other_installment_plans, housing, job, telephone.
- **Escluse dal modello**: sex, personal_status, foreign_worker, age_group
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
- **Soglia di decisione**: 0.11 sulla probabilità di default,
  scelta su predizioni out-of-fold per minimizzare il costo atteso, con la matrice dei
  costi UCI: concedere credito a un cattivo pagatore costa
  5, rifiutarlo a un buon pagatore costa
  1.

## 4. Prestazioni (test set)

| Metrica | Logistica | XGBoost |
| --- | --- | --- |
| AUC | 0.8099 | 0.8036 |
| AUC in CV (train) | 0.7593 | 0.7951 |
| F1 (classe cattivo pagatore) | 0.5492 | 0.5654 |
| Precision | 0.3985 | 0.4122 |
| Recall | 0.8833 | 0.9 |
| Brier score | 0.1831 | 0.1586 |
| Costo medio per richiesta | 0.575 | 0.535 |
| Tasso di approvazione | 0.335 | 0.345 |

## 5. Equità

Valutata sul test set per gruppi di sesso, fascia d'età e lavoratore straniero.
Dettagli in [fairness_report.md](fairness_report.md).

| Attributo | Diff. demographic parity | Diff. equalized odds | Disparate impact ratio |
| --- | --- | --- | --- |
| sex | 0.0954 | 0.119 | 0.745 |
| age_group | 0.2381 | 0.3032 | 0.4621 |
| foreign_worker | 0.1598 | 0.1556 | 0.6804 |

Un disparate impact ratio sotto 0,8 indica una possibile discriminazione indiretta
da approfondire prima di qualsiasi uso.

## 6. Spiegabilità

- **Locale**: TreeSHAP, contributi aggregati per feature originale (esatti grazie
  all'additività).
- **Controfattuali**: DiCE, limitati alle feature su cui il cliente può agire
  (duration_months, credit_amount, checking_status, savings, installment_rate, other_debtors, existing_credits, other_installment_plans, purpose), validati con la soglia reale del modello
  e filtrati per plausibilità.
- **Narrazione**: testo generato da template o LLM e verificato automaticamente
  contro i valori SHAP (fattori inventati, direzione, copertura, attributi protetti,
  coerenza dell'esito).
- **Fedeltà misurata**: {"n": 100, "faithful_rate": 1.0, "unsupported_rate": 0.0, "direction_error_rate": 0.0, "mean_coverage_top3": 1.0, "sensitive_leak_rate": 0.0, "outcome_consistency_rate": 1.0, "provider": "template"}

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
