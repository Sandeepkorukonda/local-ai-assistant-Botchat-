"""Embeddings and retrieval (simple in-memory RAG)."""
import numpy as np
import ollama

import config


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 32):  # batch to keep requests small
        vectors += ollama.embed(model=config.EMBED_MODEL, input=texts[i:i + 32])["embeddings"]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def retrieve(question: str, chunks, vectors: np.ndarray, k: int = config.TOP_K):
    q = embed([question])[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]


def build_rag_prompt(question: str, hits) -> str:
    context = "\n\n".join(f"[Page {p}] {t}" for p, t in hits)
    return ("Answer ONLY using the context below. If the answer isn't there, "
            f"say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}")


def sources_text(hits) -> str:
    pages = ", ".join(str(p) for p in sorted({p for p, _ in hits}))
    return f"\n\n*Sources: pages {pages}*"