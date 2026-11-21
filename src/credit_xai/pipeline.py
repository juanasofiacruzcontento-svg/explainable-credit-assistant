"""Esegue tutto il progetto in ordine: dati -> modello -> fairness -> fedeltà -> model card.

Uso:
    python -m credit_xai.pipeline
    python -m credit_xai.pipeline --trials 100 --n-eval 200
"""

from __future__ import annotations

import argparse
import warnings

from credit_xai import data, fairness, modelcard, train
from credit_xai.evaluation import faithfulness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40, help="trial di Optuna")
    parser.add_argument(
        "--n-eval", type=int, default=100, help="casi per la valutazione di fedeltà"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    print("== 1/5 Dati")
    data.build_dataset()
    print("\n== 2/5 Training")
    train.train_all(args.trials)
    print("\n== 3/5 Fairness audit")
    fairness.main()
    print("\n== 4/5 Valutazione di fedeltà delle spiegazioni")
    summary = faithfulness.run(args.n_eval)
    print(summary)
    print("\n== 5/5 Model card")
    modelcard.main()
    print("\nFatto. Avvia l'interfaccia con: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
