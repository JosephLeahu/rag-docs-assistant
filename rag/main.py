"""FastAPI app + CLI for the document Q&A assistant."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import Answerer
from .knowledge_base import KnowledgeBase

STATIC = Path(__file__).resolve().parent.parent / "static"


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


def create_app(kb: KnowledgeBase | None = None, client=None) -> FastAPI:
    kb = kb or KnowledgeBase.from_folder()
    answerer = Answerer(kb, client=client)
    app = FastAPI(title="Volcano Docs Q&A", version="1.0.0")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(kb.chunks), "llm": answerer.client is not None}

    @app.get("/search")
    def search(q: str, k: int = 4):
        if not q.strip():
            raise HTTPException(422, "q must not be empty")
        return [{"rank": h.rank, "score": h.score, **h.chunk.to_dict()} for h in kb.search(q, min(k, 20))]

    @app.post("/ask")
    def ask(body: Question):
        return asdict(answerer.ask(body.question))

    return app
