from types import SimpleNamespace as NS

import numpy as np
import pytest

from evals.run_eval import load_questions, retrieval_metrics
from rag.answer import NO_ANSWER, Answerer, build_prompt, parse_citations
from rag.chunking import chunk_document
from rag.knowledge_base import KnowledgeBase
from rag.retrieval import BM25, DenseRetriever, HybridRetriever, reciprocal_rank_fusion, tokenize


# ---- chunking ----------------------------------------------------------------

def test_chunks_respect_headings_and_size():
    text = "# Title\n\n## A\n\n" + "\n\n".join(["alpha " * 40] * 5) + "\n\n## B\n\nshort b"
    chunks = chunk_document("doc", text, max_chars=500)
    assert {c.heading for c in chunks} == {"A", "B"}
    assert all(c.title == "Title" for c in chunks)
    assert len([c for c in chunks if c.heading == "A"]) > 1
    # overlap: last paragraph of a chunk starts the next one within a section
    a = [c for c in chunks if c.heading == "A"]
    assert a[0].text.split("\n\n")[-1] == a[1].text.split("\n\n")[0]


def test_chunk_ids_unique():
    kb = KnowledgeBase.from_folder()
    ids = [c.id for c in kb.chunks]
    assert len(ids) == len(set(ids)) and "README" not in {c.doc for c in kb.chunks}


# ---- retrieval -----------------------------------------------------------------

def test_tokenize_stems_plurals_and_drops_stopwords():
    assert tokenize("The lahars and the glass") == ["lahar", "glass"]


def test_bm25_ranks_relevant_doc_first():
    bm = BM25(["volcanic ash damages jet engines", "lahars are mudflows", "the cat sat"])
    assert bm.search("ash in jet engines")[0][0] == 0
    assert bm.search("zebra") == []


def test_bm25_idf_rewards_rare_terms():
    bm = BM25(["common rare", "common", "common", "common"])
    assert bm.idf["rare"] > bm.idf["common"]


def fake_embed(texts):
    vocab = ["ash", "lahar", "mudflow", "engine", "seismic"]
    return np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float)


def test_dense_and_hybrid():
    texts = ["ash in engine", "lahar mudflow", "seismic station"]
    dense = DenseRetriever(texts, fake_embed)
    assert dense.search("mudflow", 1)[0][0] == 1
    hybrid = HybridRetriever([BM25(texts), dense])
    assert hybrid.search("seismic", 1)[0][0] == 2


def test_rrf_prefers_items_ranked_high_in_both():
    fused = reciprocal_rank_fusion([[(1, 9), (2, 8)], [(1, 0.9), (3, 0.5)]])
    assert fused[0][0] == 1


def test_real_corpus_retrieval_quality():
    kb = KnowledgeBase.from_folder()
    m = retrieval_metrics(kb, load_questions())
    assert m["recall@3"] >= 0.9  # regression guard


# ---- answering ---------------------------------------------------------------------

class FakeClient:
    def __init__(self, text):
        self.text, self.messages, self.last = text, self, None

    def create(self, **kw):
        self.last = kw
        return NS(content=[NS(type="text", text=self.text)])


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_folder()


def test_parse_citations():
    assert parse_citations("A [1]. B [3][1].") == [1, 3]


def test_prompt_contains_numbered_sources(kb):
    p = build_prompt("q?", kb.search("lahar", 2))
    assert '<source id="1"' in p and '<source id="2"' in p and p.endswith("Question: q?")


def test_answer_with_valid_citation(kb):
    a = Answerer(kb, client=FakeClient("57 people were killed [1].")).ask("How many people died at St. Helens in 1980?")
    assert a.grounded and a.citations[0]["doc"] == "mount-st-helens-1980"


def test_hallucinated_citation_is_flagged(kb):
    a = Answerer(kb, client=FakeClient("Something [9].")).ask("What is a lahar?")
    assert a.grounded is False and a.citations == []


def test_refusal_counts_as_grounded(kb):
    a = Answerer(kb, client=FakeClient(NO_ANSWER)).ask("What is a lahar?")
    assert a.grounded is True


def test_retrieval_only_mode(kb, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = Answerer(kb).ask("Where is the Cascades Volcano Observatory?")
    assert a.answer.startswith("(retrieval-only mode)")
    assert a.citations[0]["doc"] == "cascades-volcano-observatory"
