"""Load documents and split them into overlapping, heading-aware chunks."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str          # "<doc>#<n>"
    doc: str         # file stem
    title: str       # document title (first H1) or file stem
    heading: str     # nearest section heading
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Return (doc_name, text) for every .md / .txt / .pdf file in `folder`."""
    docs = []
    for path in sorted(folder.iterdir()):
        if path.name.lower() == "readme.md" or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append((path.stem, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            docs.append((path.stem, _read_pdf(path)))
    return docs


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # optional dependency

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs. Non-markdown text becomes one section."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if any(s.strip() for s in buf):
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if any(s.strip() for s in buf):
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def chunk_document(doc: str, text: str, max_chars: int = 900, overlap_paragraphs: int = 1) -> list[Chunk]:
    """Pack paragraphs into chunks of up to `max_chars`, never crossing a heading.

    Consecutive chunks inside a section share `overlap_paragraphs` paragraphs so a
    fact split across a boundary is still retrievable.
    """
    title_match = re.search(r"^#\s+(.*)", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else doc
    chunks: list[Chunk] = []

    for heading, body in _sections(text):
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current: list[str] = []
        for para in paras:
            if current and len("\n\n".join(current + [para])) > max_chars:
                chunks.append(_make(doc, title, heading, current, len(chunks)))
                current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current.append(para)
        if current:
            chunks.append(_make(doc, title, heading, current, len(chunks)))
    return chunks


def _make(doc: str, title: str, heading: str, paras: list[str], n: int) -> Chunk:
    return Chunk(id=f"{doc}#{n}", doc=doc, title=title, heading=heading or title, text="\n\n".join(paras))
