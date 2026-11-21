"""Dalla spiegazione SHAP + controfattuali a un testo per il cliente.

Due modalità:
- template: testo costruito con regole, deterministico e fedele per costruzione
  (baseline e fallback offline);
- LLM: testo più naturale, ma va verificato (vedi evaluation/faithfulness.py).

L'output ha sempre la stessa forma:
    {"text": str, "cited_features": [{"feature": str, "direction": str}], "provider": str}
"""

from __future__ import annotations

from credit_xai.llm import client
from credit_xai.llm.prompts import NARRATOR_SYSTEM, NARRATOR_USER

TOP_K = 5


def _factors(explanation: dict, k: int = TOP_K) -> list[dict]:
    return explanation["contributions"][:k]


def build_prompt(explanation: dict, counterfactuals: list[dict], k: int = TOP_K) -> str:
    factors = "\n".join(
        f'- feature="{c["feature"]}" | {c["label"]}: {c["value"]} -> {c["direction"]}'
        for c in _factors(explanation, k)
    )
    if counterfactuals:
        cfs = "\n".join(
            "- " + "; ".join(f"{ch['label']}: da {ch['from']} a {ch['to']}" for ch in cf["changes"])
            for cf in counterfactuals
        )
    else:
        cfs = "- nessuna (richiesta già approvata o nessuna alternativa trovata)"
    return NARRATOR_USER.format(
        outcome="APPROVATA" if explanation["approved"] else "NON APPROVATA",
        prob=explanation["prob_default"],
        threshold=explanation["threshold"],
        factors=factors,
        counterfactuals=cfs,
    )


def narrate_template(explanation: dict, counterfactuals: list[dict], k: int = 3) -> dict:
    """Spiegazione a regole: cita le top-k feature con la direzione corretta."""
    factors = _factors(explanation, k)
    against = [c for c in factors if c["direction"] == "aumenta il rischio"]
    favour = [c for c in factors if c["direction"] == "riduce il rischio"]

    def fmt(cs):
        return ", ".join(f"{c['label']} ({c['value']})" for c in cs)

    if explanation["approved"]:
        parts = ["La sua richiesta di credito è stata approvata."]
        if favour:
            parts.append(f"Hanno pesato a suo favore soprattutto: {fmt(favour)}.")
        if against:
            parts.append(f"Hanno invece aumentato il rischio stimato: {fmt(against)}.")
    else:
        parts = ["La sua richiesta di credito non è stata approvata."]
        if against:
            parts.append(f"I fattori che hanno pesato di più sono: {fmt(against)}.")
        if favour:
            parts.append(f"Hanno giocato a suo favore: {fmt(favour)}.")
        if counterfactuals:
            changes = "; ".join(
                f"{ch['label']} da {ch['from']} a {ch['to']}"
                for ch in counterfactuals[0]["changes"]
            )
            parts.append(
                f"Secondo il modello, l'esito potrebbe cambiare con queste modifiche: {changes}."
            )
    parts.append("Può sempre chiedere che la decisione sia riesaminata da una persona.")
    return {
        "text": " ".join(parts),
        "cited_features": [{"feature": c["feature"], "direction": c["direction"]} for c in factors],
        "provider": "template",
    }


def narrate(explanation: dict, counterfactuals: list[dict] | None = None) -> dict:
    """Usa l'LLM configurato; se non configurato o in errore, ricade sul template."""
    counterfactuals = counterfactuals or []
    if client.provider() == "template":
        return narrate_template(explanation, counterfactuals)
    try:
        raw = client.chat(NARRATOR_SYSTEM, build_prompt(explanation, counterfactuals))
        out = client.extract_json(raw)
        return {
            "text": str(out.get("text", "")),
            "cited_features": list(out.get("cited_features", [])),
            "provider": f"{client.provider()}:{client.model_name()}",
        }
    except Exception as exc:  # rete, chiave mancante, JSON non valido...
        fallback = narrate_template(explanation, counterfactuals)
        fallback["provider"] = f"template (fallback: {type(exc).__name__})"
        return fallback
