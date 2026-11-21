#!/bin/sh
# Se il modello non esiste ancora (primo avvio), esegue la pipeline completa.
set -e
if [ ! -f models/model.joblib ]; then
  echo "Modello non trovato: eseguo la pipeline (dati, training, report)..."
  python -m credit_xai.pipeline --trials 30 --n-eval 50
fi
exec "$@"
