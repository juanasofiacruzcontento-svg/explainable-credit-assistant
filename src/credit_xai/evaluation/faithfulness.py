"""La spiegazione in linguaggio naturale è fedele al modello?

Per ogni spiegazione generata controlliamo, rispetto ai valori SHAP:
1. unsupported: fattori citati che non sono tra le top-k feature SHAP
   (o che non esistono proprio: allucinazioni);
2. direction_errors: fattori citati con la direzione sbagliata
   (es. "i risparmi l'hanno penalizzata" quando SHAP dice che riducono il rischio);
3. coverage: quota delle top-3 feature SHAP effettivamente citate;
4. sensitive_leak: il testo nomina attributi protetti che il modello non usa;
5. outcome_consistent: il testo dichiara l'esito giusto.

Una spiegazione è "fedele" se supera tutti i controlli. Il report riporta i tassi
su N casi, così si può confrontare il template con diversi LLM e prompt.

Uso:
    python -m credit_xai.evaluation.faithfulness --n 100
"""

from __future__ import annotations

import argparse
import json
import re

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.features import MODEL_FEATURES

TOP_K = 5
SENSITIVE_TERMS = [
    r"\bsesso\b",
    r"\bgenere\b",
    r"\bdonna\b",
    r"\buomo\b",
    r"\bfemmin",
    r"\bmaschi",
    r"\bstranier",
    r"\bnazionalit",
    r"\bstato civile\b",
    r"\bsposat",
    r"\bdivorziat",
    r"\betnia\b",
    r"\borigine\b",
]

REJECTION_PATTERNS = [
    r"\bnon\b[^.]{0,25}\b(approvat|concess|accolt|approvar|conceder|accoglier)",
    r"\brifiutat",
    r"\brespint",
    r"\bnegat",
    r"\besito negativo",
]
APPROVAL_PATTERNS = [r"\bapprovat", r"\bconcess", r"\baccolt", r"\besito positivo"]


def check(explanation: dict, narration: dict, k: int = TOP_K) -> dict:
    contributions = {c["feature"]: c for c in explanation["contributions"]}
    top_k = [c["feature"] for c in explanation["contributions"][:k]]
    top_3 = top_k[:3]
    cited = narration.get("cited_features", []) or []
    text = narration.get("text", "") or ""

    unsupported, direction_errors = [], []
    for item in cited:
        feat = item.get("feature")
        if feat not in MODEL_FEATURES or feat not in top_k:
            unsupported.append(feat)
            continue
        if item.get("direction") != contributions[feat]["direction"]:
            direction_errors.append(feat)

    cited_set = {i.get("feature") for i in cited}
    coverage = len(cited_set & set(top_3)) / len(top_3) if top_3 else 1.0

    lowered = text.lower()
    sensitive = [t for t in SENSITIVE_TERMS if re.search(t, lowered)]

    says_rejected = any(re.search(p, lowered) for p in REJECTION_PATTERNS)
    says_approved = any(re.search(p, lowered) for p in APPROVAL_PATTERNS) and not says_rejected
    outcome_consistent = says_approved if explanation["approved"] else says_rejected

    faithful = (
        not unsupported
        and not direction_errors
        and coverage >= 2 / 3
        and not sensitive
        and outcome_consistent
    )
    return {
        "faithful": faithful,
        "unsupported": unsupported,
        "direction_errors": direction_errors,
        "coverage_top3": round(coverage, 3),
        "sensitive_leak": sensitive,
        "outcome_consistent": outcome_consistent,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "faithful_rate": round(sum(r["faithful"] for r in results) / n, 3),
        "unsupported_rate": round(sum(bool(r["unsupported"]) for r in results) / n, 3),
        "direction_error_rate": round(sum(bool(r["direction_errors"]) for r in results) / n, 3),
        "mean_coverage_top3": round(sum(r["coverage_top3"] for r in results) / n, 3),
        "sensitive_leak_rate": round(sum(bool(r["sensitive_leak"]) for r in results) / n, 3),
        "outcome_consistency_rate": round(sum(r["outcome_consistent"] for r in results) / n, 3),
    }


def run(n: int = 100, with_counterfactuals: bool = True) -> dict:
    """Genera n spiegazioni sul test set con il narratore configurato e le valuta."""
    from credit_xai.explain.counterfactuals import CounterfactualGenerator
    from credit_xai.explain.shap_explainer import ShapExplainer
    from credit_xai.llm import client
    from credit_xai.llm.narrator import narrate
    from credit_xai.model import CreditModel

    model = CreditModel.load()
    explainer = ShapExplainer(model)
    train, test = load_processed("train"), load_processed("test")
    cfg = CounterfactualGenerator(model, train) if with_counterfactuals else None
    sample = test.sample(min(n, len(test)), random_state=config.RANDOM_STATE)

    results, cases = [], []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        row_df = row.to_frame().T.astype(sample.dtypes.to_dict())
        explanation = explainer.explain_one(row_df)
        cfs = cfg.generate(row_df) if cfg and not explanation["approved"] else []
        narration = narrate(explanation, cfs)
        result = check(explanation, narration)
        results.append(result)
        cases.append({"explanation": explanation, "narration": narration, "check": result})
        print(f"\r{i}/{len(sample)}", end="", flush=True)
    print()

    summary = summarize(results) | {
        "provider": ":".join(filter(None, [client.provider(), client.model_name()]))
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "faithfulness.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    with open(config.REPORTS_DIR / "faithfulness_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--no-cf", action="store_true", help="salta i controfattuali (più veloce)")
    args = parser.parse_args()
    summary = run(args.n, with_counterfactuals=not args.no_cf)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
