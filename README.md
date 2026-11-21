# Explainable Credit Assistant

Un sistema di credit scoring che **decide**, **spiega** ogni decisione in linguaggio
naturale e **verifica automaticamente** che la spiegazione sia fedele al modello.
Include fairness audit, controfattuali, RAG sulla normativa, API, interfaccia,
monitoraggio e una model card in stile EU AI Act (il credit scoring è un sistema
ad alto rischio, Allegato III punto 5(b)).

> Progetto di portfolio su dati pubblici. Non usare per decisioni reali.

<!-- Dopo il deploy: aggiungi qui una GIF della demo e il link live -->

## Cosa dimostra

| Area | Implementazione |
| --- | --- |
| ML | Baseline logistica vs XGBoost, tuning con Optuna, soglia scelta per **costo minimo** (matrice dei costi UCI 5:1) |
| Explainability | TreeSHAP aggregato per feature originale, test di stabilità, controfattuali DiCE vincolati e filtrati per plausibilità |
| Fairness | Audit con Fairlearn (sesso, età, lavoratore straniero), regola dei 4/5, mitigazione con ThresholdOptimizer |
| LLM | Narratore multi-provider (Anthropic, OpenAI, Ollama) con output JSON strutturato e fallback offline |
| Evaluation | **Controllo automatico di fedeltà**: fattori inventati, direzione sbagliata, copertura, attributi protetti, esito |
| RAG | Retrieval TF-IDF su note normative (AI Act, GDPR art. 22, direttiva credito ai consumatori) con citazioni |
| Produzione | FastAPI con validazione, Streamlit, Docker, CI, log delle decisioni, drift con PSI |
| Governance | Model card e fairness report generati dai risultati |

## Architettura

```
richiesta ──► FastAPI / Streamlit
                 │
                 ├─► modello XGBoost ──► probabilità + decisione (soglia a costo minimo)
                 ├─► SHAP ──────────────► contributi dei fattori
                 ├─► DiCE ──────────────► cosa cambiare per ottenere il credito
                 ├─► narratore (LLM o template) ──► testo per il cliente
                 │        └─► controllo di fedeltà vs SHAP
                 └─► log decisioni ──► monitoraggio drift (PSI)

domanda normativa ──► RAG (TF-IDF su knowledge/) ──► risposta con fonti
```

## Avvio rapido (Windows PowerShell)

Un comando alla volta, dalla cartella del progetto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m credit_xai.pipeline
```

Su macOS/Linux attiva l'ambiente con `source .venv/bin/activate`.

La pipeline scarica i dati da UCI, addestra i modelli e scrive report e model card
(qualche minuto). Poi:

```powershell
streamlit run app/streamlit_app.py          # interfaccia: http://localhost:8501
uvicorn credit_xai.api:app --reload         # API: http://localhost:8000/docs
pytest                                      # 39 test, non servono rete né chiavi
jupyter lab                                 # notebook di analisi
```

### Jupyter dice "No module named ..."?

Jupyter sta usando un Python diverso da `.venv`. Registra l'ambiente come kernel:

```powershell
python -m ipykernel install --user --name credit-xai --display-name "Python (credit-xai)"
```

poi in Jupyter scegli **Kernel → Change Kernel → Python (credit-xai)**.

## Comandi singoli

| Comando | Cosa fa |
| --- | --- |
| `python -m credit_xai.data` | scarica e pulisce i dati, crea train/test |
| `python -m credit_xai.train --trials 100` | addestra logistica e XGBoost |
| `python -m credit_xai.fairness` | audit di equità → `docs/fairness_report.md` |
| `python -m credit_xai.evaluation.faithfulness --n 100` | valuta la fedeltà delle spiegazioni |
| `python -m credit_xai.modelcard` | rigenera `docs/model_card.md` |
| `python -m credit_xai.monitoring --simulate` | demo di rilevamento del drift |
| `python -m credit_xai.rag "domanda"` | interroga le note normative |

## Usare un LLM

Senza configurazione il narratore usa un template a regole (offline, fedele per
costruzione). Per usare un LLM copia `.env.example` in `.env` e scegli il provider:

```
LLM_PROVIDER=ollama        # gratuito e locale: https://ollama.com, poi `ollama pull llama3.1`
# oppure anthropic / openai con la relativa API key
```

Poi rilancia `python -m credit_xai.evaluation.faithfulness --n 100` e confronta il tasso
di fedeltà con quello del template: è il risultato più interessante da raccontare.

## Notebook

| Notebook | Contenuto |
| --- | --- |
| `01_eda` | qualità dei dati, sbilanciamento, gruppi sensibili, proxy |
| `02_model` | ROC, calibrazione, costo al variare della soglia, confusione |
| `03_explainability` | importanza globale, spiegazione locale, controfattuali, stabilità |
| `04_fairness_and_faithfulness` | audit, peso dell'età, mitigazione, casi non fedeli |

## Scelte di progetto

- **Attributi sensibili esclusi dal modello**: `sex`, `foreign_worker` e
  `personal_status` (nel dataset originale codifica anche il sesso). Restano nei dati
  solo per l'audit.
- **Età mantenuta** come fattore di rischio, ma monitorata: è la scelta più discutibile
  ed è documentata nella model card.
- **Soglia non a 0,5**: scelta su predizioni out-of-fold per minimizzare il costo atteso.
- **Controfattuali solo su variabili attuabili**: niente "cambi la sua età" o
  "chiuda il conto corrente".
- **Ogni spiegazione LLM è verificata**: un testo convincente ma infedele è peggio di
  nessuna spiegazione.

## Deploy

- **Docker**: `docker compose up --build` avvia API (porta 8000) e interfaccia (8501);
  al primo avvio esegue la pipeline.
- **Hugging Face Spaces** (gratuito): crea uno Space di tipo Docker o Streamlit, carica
  il repository insieme a `models/` e `reports/` generati in locale, e imposta
  `app/streamlit_app.py` come file principale.

## Struttura

```
src/credit_xai/
  config.py, data.py, features.py      dati e feature
  train.py, model.py                   modelli e soglia
  explain/                             SHAP e controfattuali
  fairness.py                          audit e mitigazione
  llm/                                 client multi-provider, prompt, narratore
  evaluation/faithfulness.py           controllo di fedeltà
  rag.py                               domande sulla normativa
  service.py, api.py                   logica applicativa e API
  monitoring.py                        log delle decisioni e drift
  modelcard.py, pipeline.py            documentazione e orchestrazione
app/streamlit_app.py                   interfaccia
knowledge/                             note normative indicizzate dal RAG
notebooks/                             analisi
tests/                                 test (dati sintetici, girano in CI)
docs/                                  model card e fairness report (generati)
```

## Dati

[Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
UCI Machine Learning Repository: 1.000 richieste, 20 attributi, licenza CC BY 4.0.

## Licenza

MIT
