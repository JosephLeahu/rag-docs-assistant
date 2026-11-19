"""Ask the docs from the terminal:  python -m rag.cli "What is a lahar?" """
from __future__ import annotations

import argparse

from .answer import Answerer
from .knowledge_base import KnowledgeBase


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the docs from the terminal")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--mode", default=None, choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args()
    result = Answerer(KnowledgeBase.from_folder(mode=args.mode)).ask(" ".join(args.question))
    print(result.answer, "\n")
    for c in result.citations:
        print(f"  [{c['n']}] {c['title']} > {c['section']}")


if __name__ == "__main__":
    main()
