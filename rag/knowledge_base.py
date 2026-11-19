"""Builds chunks from a docs folder and wires up the chosen retriever."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_document, load_documents
from .retrieval import BM25, DenseRetriever, EmbedFn, HybridRetriever, Retriever

DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass
class SearchHit:
    rank: int
    score: float
    chunk: Chunk


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], retriever: Retriever):
        self.chunks = chunks
        self.retriever = retriever

    @classmethod
    def from_folder(
        cls,
        folder: Path = DEFAULT_DOCS,
        mode: str | None = None,
        embed: EmbedFn | None = None,
        max_chars: int = 900,
    ) -> KnowledgeBase:
        mode = mode or os.getenv("RETRIEVER", "bm25")
        chunks: list[Chunk] = []
        for doc, text in load_documents(Path(folder)):
            chunks.extend(chunk_document(doc, text, max_chars=max_chars))
        if not chunks:
            raise ValueError(f"no documents found in {folder}")

        # Index the heading with the text: headings carry a lot of signal.
        texts = [f"{c.title}. {c.heading}. {c.text}" for c in chunks]
        if mode == "bm25":
            retriever: Retriever = BM25(texts)
        elif mode in {"dense", "hybrid"}:
            if embed is None:
                from .retrieval import sentence_transformer_embedder
                embed = sentence_transformer_embedder()
            dense = DenseRetriever(texts, embed)
            retriever = dense if mode == "dense" else HybridRetriever([BM25(texts), dense])
        else:
            raise ValueError(f"unknown retriever mode {mode!r}")
        return cls(chunks, retriever)

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        return [
            SearchHit(rank=r, score=round(s, 4), chunk=self.chunks[i])
            for r, (i, s) in enumerate(self.retriever.search(query, k), start=1)
        ]
