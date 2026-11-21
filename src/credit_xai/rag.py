"""RAG leggero sulla normativa: retrieval TF-IDF + risposta con citazioni.

Indicizza tutti i file .md in knowledge/, spezzati per sezione (## ...).
TF-IDF non richiede modelli da scaricare: per un corpus piccolo e tecnico funziona
bene. Per corpora grandi si può sostituire con embeddings (sentence-transformers)
e un vector store (ChromaDB) mantenendo la stessa interfaccia `retrieve()`.

Uso:
    python -m credit_xai.rag "Ho diritto a una spiegazione se mi rifiutano il prestito?"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from credit_xai import config
from credit_xai.llm import client
from credit_xai.llm.prompts import RAG_SYSTEM, RAG_USER


@dataclass
class Chunk:
    source: str
    title: str
    text: str


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    directory = directory or config.KNOWLEDGE_DIR
    chunks = []
    for path in sorted(directory.glob("*.md")):
        for section in re.split(r"\n(?=## )", path.read_text(encoding="utf-8")):
            if not section.startswith("## "):
                continue
            title, _, body = section.partition("\n")
            chunks.append(Chunk(path.name, title.removeprefix("## ").strip(), body.strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Nessun documento in knowledge/")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode"
        )
        self.matrix = self.vectorizer.fit_transform(f"{c.title}\n{c.text}" for c in chunks)

    def retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_chunks())


def answer(question: str, k: int = 3) -> dict:
    """Risposta con fonti. Senza LLM restituisce i passaggi più pertinenti."""
    hits = get_retriever().retrieve(question, k)
    sources = [{"title": c.title, "score": round(s, 3), "text": c.text} for c, s in hits]
    if not hits:
        return {"answer": "Non ho trovato passaggi pertinenti.", "sources": [], "provider": "-"}

    if client.provider() != "template":
        context = "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in hits)
        try:
            text = client.chat(RAG_SYSTEM, RAG_USER.format(context=context, question=question))
            return {
                "answer": text.strip(),
                "sources": sources,
                "provider": f"{client.provider()}:{client.model_name()}",
            }
        except Exception as exc:
            provider = f"estrattivo (fallback: {type(exc).__name__})"
    else:
        provider = "estrattivo"

    best = hits[0][0]
    return {
        "answer": f"{best.title}\n{best.text}\n\n(Sintesi non ufficiale, non è consulenza legale.)",
        "sources": sources,
        "provider": provider,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Ho diritto a una spiegazione se mi rifiutano il credito?"
    res = answer(q)
    print(res["answer"])
    print("\nFonti:", ", ".join(s["title"] for s in res["sources"]))
