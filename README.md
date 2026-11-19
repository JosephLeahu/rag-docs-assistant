# RAG Document Assistant

A Python/FastAPI retrieval-augmented question-answering prototype for volcano-related reference documents. Combines BM25 and embedding-based retrieval with Claude-generated answers and source references.

## Features
- Document ingestion and chunking
- Keyword and semantic retrieval
- Source-referenced answers
- Retrieval evaluation and automated tests

## Local setup
```bash
pip install -r requirements.txt
uvicorn rag.main:app --reload
```

**Note:** Retrieved citation IDs are checked, but this alone does not prove each cited passage supports the answer. Full source upload is pending.
