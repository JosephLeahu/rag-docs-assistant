"""Retrievers: BM25 (implemented from scratch), dense embeddings, and hybrid fusion.

All retrievers share one interface:  search(query, k) -> list[(chunk_index, score)]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "so", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your", "about", "after", "before", "also", "more", "most", "no", "not", "only", "other", "over", "same", "some", "very"}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        # Very light stemming: plural -> singular ("lahars" -> "lahar", not "glass" -> "glas").
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]: ...


class BM25:
    """Okapi BM25.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        n = len(self.docs)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        # BM25+ style idf that never goes negative.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, i: int) -> float:
        d, dl = self.docs[i], self.lengths[i]
        s = 0.0
        for t in tokenize(query):
            tf = d.get(t, 0)
            if tf:
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


EmbedFn = Callable[[list[str]], np.ndarray]


class DenseRetriever:
    """Cosine-similarity search over embeddings. `embed` is any function texts -> (n, dim) array."""

    def __init__(self, texts: Sequence[str], embed: EmbedFn):
        self.embed = embed
        self.matrix = self._normalize(embed(list(texts)))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = self._normalize(self.embed([query]))[0]
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine ranked lists without needing comparable scores: score = sum 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], pool: int = 20):
        self.retrievers, self.pool = retrievers, pool

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        return reciprocal_rank_fusion([r.search(query, self.pool) for r in self.retrievers])[:k]


def sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbedFn:
    """Local, free embeddings (pip install sentence-transformers). Downloads the model on first use."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, convert_to_numpy=True)
