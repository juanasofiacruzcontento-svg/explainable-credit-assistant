"""Client LLM minimale e intercambiabile, via HTTP (nessun SDK obbligatorio).

Configurazione con variabili d'ambiente (o file .env, vedi .env.example):
    LLM_PROVIDER = template | anthropic | openai | ollama   (default: template)
    LLM_MODEL    = nome del modello (default diverso per provider)
    ANTHROPIC_API_KEY / OPENAI_API_KEY
    OLLAMA_HOST  = http://localhost:11434

Con "template" nessun LLM viene chiamato: il progetto funziona offline e i test
non dipendono da chiavi API.
"""

from __future__ import annotations

import json
import os
import re

import httpx

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}
TIMEOUT = 60.0


def _load_dotenv() -> None:
    """Legge un eventuale file .env nella root del progetto (senza dipendenze)."""
    from credit_xai.config import PROJECT_ROOT

    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "template").lower()


def model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider(), "")


def chat(system: str, user: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
    """Invia un messaggio al provider configurato e restituisce il testo."""
    p = provider()
    if p == "anthropic":
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model_name(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if p == "openai":
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": model_name(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if p == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = httpx.post(
            f"{host}/api/chat",
            json={
                "model": model_name(),
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=TIMEOUT * 3,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise RuntimeError(f"Provider '{p}' non supportato per chat(); usa il narratore template")


def extract_json(text: str) -> dict:
    """Estrae il primo oggetto JSON da una risposta (anche se circondato da testo)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Nessun JSON nella risposta del modello")
    return json.loads(text[start : end + 1])
