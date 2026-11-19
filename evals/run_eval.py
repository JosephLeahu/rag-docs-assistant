"""Evaluate retrieval (always) and answers (when ANTHROPIC_API_KEY is set).

    python -m evals.run_eval                    # BM25 retrieval metrics
    python -m evals.run_eval --mode hybrid      # needs sentence-transformers
    python -m evals.run_eval --answers          # also grade Claude's answers

Retrieval metrics use the questions that have an expected_doc:
  recall@k  share of questions where a chunk from the right doc is in the top k
  MRR       mean of 1/rank of the first correct chunk (0 if not found)
Answer metrics:
  grounded_rate     answers whose citations all point at real sources
  cited_correct     answers citing a chunk from the expected doc
  refusal_accuracy  out-of-scope questions correctly declined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.answer import NO_ANSWER, Answerer
from rag.knowledge_base import KnowledgeBase

HERE = Path(__file__).resolve().parent


def load_questions(path: Path = HERE / "questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_metrics(kb: KnowledgeBase, questions: list[dict], ks=(1, 3, 5)) -> dict:
    answerable = [q for q in questions if q["expected_doc"]]
    hits_at = {k: 0 for k in ks}
    rr_total, misses = 0.0, []
    for q in answerable:
        docs = [h.chunk.doc for h in kb.search(q["question"], max(ks))]
        rank = next((i for i, d in enumerate(docs, start=1) if d == q["expected_doc"]), None)
        rr_total += 1 / rank if rank else 0.0
        for k in ks:
            hits_at[k] += int(rank is not None and rank <= k)
        if rank != 1:
            misses.append({"id": q["id"], "question": q["question"], "expected": q["expected_doc"],
                           "got": docs[:3], "rank": rank})
    n = len(answerable)
    return {"n": n, **{f"recall@{k}": round(hits_at[k] / n, 3) for k in ks},
            "mrr": round(rr_total / n, 3), "not_rank_1": misses}


def answer_metrics(answerer: Answerer, questions: list[dict]) -> dict:
    grounded = cited_ok = refusals_ok = n_ans = n_oos = 0
    rows = []
    for q in questions:
        a = answerer.ask(q["question"])
        if q["expected_doc"]:
            n_ans += 1
            grounded += int(a.grounded and bool(a.citations))
            cited_ok += int(any(c["doc"] == q["expected_doc"] for c in a.citations))
        else:
            n_oos += 1
            refusals_ok += int(a.answer.strip() == NO_ANSWER)
        rows.append({"id": q["id"], "question": q["question"], "answer": a.answer,
                     "cited": [c["doc"] for c in a.citations]})
    return {
        "grounded_rate": round(grounded / n_ans, 3) if n_ans else None,
        "cited_correct": round(cited_ok / n_ans, 3) if n_ans else None,
        "refusal_accuracy": round(refusals_ok / n_oos, 3) if n_oos else None,
        "answers": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()

    kb = KnowledgeBase.from_folder(mode=args.mode)
    questions = load_questions()
    report = {"mode": args.mode, "chunks": len(kb.chunks), "retrieval": retrieval_metrics(kb, questions)}
    if args.answers:
        answerer = Answerer(kb)
        if answerer.client is None:
            raise SystemExit("Set ANTHROPIC_API_KEY to grade answers.")
        report["answers"] = answer_metrics(answerer, questions)

    out = HERE / "results" / f"{args.mode}{'_answers' if args.answers else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    r = report["retrieval"]
    print(f"mode={args.mode} chunks={report['chunks']} questions={r['n']}")
    print(f"  recall@1={r['recall@1']}  recall@3={r['recall@3']}  recall@5={r['recall@5']}  MRR={r['mrr']}")
    if args.answers:
        a = report["answers"]
        print(f"  grounded={a['grounded_rate']}  cited_correct={a['cited_correct']}  refusal_acc={a['refusal_accuracy']}")


if __name__ == "__main__":
    main()
