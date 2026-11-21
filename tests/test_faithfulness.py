"""Il controllo di fedeltà deve accettare spiegazioni corrette e bocciare quelle sbagliate."""

import pytest

from credit_xai.evaluation.faithfulness import check, summarize
from credit_xai.llm.narrator import build_prompt, narrate_template

EXPLANATION = {
    "approved": False,
    "prob_default": 0.62,
    "threshold": 0.35,
    "contributions": [
        {
            "feature": "checking_status",
            "label": "saldo del conto corrente",
            "value": "< 0 DM",
            "shap": 0.9,
            "direction": "aumenta il rischio",
        },
        {
            "feature": "duration_months",
            "label": "durata del prestito",
            "value": "48 mesi",
            "shap": 0.5,
            "direction": "aumenta il rischio",
        },
        {
            "feature": "savings",
            "label": "risparmi",
            "value": ">= 1000 DM",
            "shap": -0.3,
            "direction": "riduce il rischio",
        },
        {
            "feature": "age",
            "label": "età",
            "value": "40 anni",
            "shap": 0.1,
            "direction": "aumenta il rischio",
        },
        {
            "feature": "purpose",
            "label": "scopo del prestito",
            "value": "auto nuova",
            "shap": 0.05,
            "direction": "aumenta il rischio",
        },
        {
            "feature": "job",
            "label": "tipo di lavoro",
            "value": "qualificato",
            "shap": 0.01,
            "direction": "aumenta il rischio",
        },
    ],
}
CFS = [
    {
        "prob_default": 0.2,
        "changes": [
            {
                "feature": "duration_months",
                "label": "durata del prestito",
                "from": "48 mesi",
                "to": "12 mesi",
            }
        ],
    }
]


def good_narration():
    return {
        "text": "La sua richiesta non è stata approvata a causa del saldo del conto e della durata.",
        "cited_features": [
            {"feature": "checking_status", "direction": "aumenta il rischio"},
            {"feature": "duration_months", "direction": "aumenta il rischio"},
            {"feature": "savings", "direction": "riduce il rischio"},
        ],
    }


def test_template_is_faithful():
    assert check(EXPLANATION, narrate_template(EXPLANATION, CFS))["faithful"]


def test_good_narration_passes():
    assert check(EXPLANATION, good_narration())["faithful"]


def test_detects_hallucinated_feature():
    n = good_narration()
    n["cited_features"].append({"feature": "income", "direction": "aumenta il rischio"})
    r = check(EXPLANATION, n)
    assert not r["faithful"] and "income" in r["unsupported"]


def test_detects_feature_outside_top_k():
    n = good_narration()
    n["cited_features"].append({"feature": "job", "direction": "aumenta il rischio"})
    assert "job" in check(EXPLANATION, n)["unsupported"]


def test_detects_wrong_direction():
    n = good_narration()
    n["cited_features"][2]["direction"] = "aumenta il rischio"
    r = check(EXPLANATION, n)
    assert not r["faithful"] and r["direction_errors"] == ["savings"]


def test_detects_low_coverage():
    n = good_narration()
    n["cited_features"] = n["cited_features"][:1]
    assert not check(EXPLANATION, n)["faithful"]


@pytest.mark.parametrize(
    "text",
    [
        "Non è stata approvata anche per via del suo sesso.",
        "Essendo lavoratore straniero, la richiesta non è stata approvata.",
    ],
)
def test_detects_sensitive_leak(text):
    n = good_narration() | {"text": text}
    assert check(EXPLANATION, n)["sensitive_leak"]


def test_detects_wrong_outcome():
    n = good_narration() | {"text": "Ottime notizie: la sua richiesta è stata approvata."}
    assert not check(EXPLANATION, n)["outcome_consistent"]


def test_rejection_paraphrase_is_recognised():
    n = good_narration() | {"text": "Purtroppo non possiamo concedere il finanziamento."}
    assert check(EXPLANATION, n)["outcome_consistent"]


def test_prompt_contains_facts():
    prompt = build_prompt(EXPLANATION, CFS)
    assert 'feature="checking_status"' in prompt and "NON APPROVATA" in prompt
    assert "12 mesi" in prompt
    assert 'feature="job"' not in prompt  # solo top-5


def test_summarize():
    s = summarize([check(EXPLANATION, good_narration())] * 3)
    assert s["n"] == 3 and s["faithful_rate"] == 1.0


def test_llm_narration_is_parsed_and_checked(monkeypatch):
    """Simula un LLM che inverte la direzione di un fattore: il controllo lo deve trovare."""
    from credit_xai.llm import client, narrator

    fake = (
        'Ecco la risposta:\n```json\n{"text": "La richiesta non è stata approvata: '
        'i suoi risparmi hanno pesato contro.", "cited_features": ['
        '{"feature": "checking_status", "direction": "aumenta il rischio"}, '
        '{"feature": "duration_months", "direction": "aumenta il rischio"}, '
        '{"feature": "savings", "direction": "aumenta il rischio"}]}\n```'
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(client, "chat", lambda *a, **k: fake)
    out = narrator.narrate(EXPLANATION, CFS)
    assert out["provider"].startswith("anthropic")
    result = check(EXPLANATION, out)
    assert not result["faithful"] and result["direction_errors"] == ["savings"]


def test_llm_failure_falls_back_to_template(monkeypatch):
    from credit_xai.llm import client, narrator

    def boom(*a, **k):
        raise ConnectionError("rete assente")

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(client, "chat", boom)
    out = narrator.narrate(EXPLANATION, CFS)
    assert "fallback" in out["provider"] and check(EXPLANATION, out)["faithful"]
