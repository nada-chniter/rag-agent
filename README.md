# Enterprise RAG Agent

A retrieval-augmented generation agent that answers questions from your own
document set, using Claude with tool-calling to decide *when* and *what* to search —
not a fixed one-shot retrieve-then-generate pipeline.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# 1. Indexing the documents (put .pdf, .txt, .md, .csv files in a folder)
python main.py ingest ./sample_data

# 2. Chatting with the agent
python main.py chat
```

The first run downloads the local embedding model (~90MB, one-time) and
caches it. After that, embedding runs fully offline.

## How it works

| Stage | File | What happens |
|---|---|---|
| Ingest | `ingestion.py` | Load files, split into overlapping text chunks |
| Embed & store | `vectorstore.py` | Chunks → vectors (local model) → ChromaDB |
| Agent loop | `rag_agent.py` | Claude gets a `search_knowledge_base` tool and decides when to call it, possibly multiple times, before answering |
| CLI | `main.py` | Wires it together |


Letting Claude issue follow-up searches materially improves
the accuracy over naive RAG.
The agent pattern matters for enterprise Q&A specifically because real
questions are often multi-hop.


## Security notes for real enterprise data

- Embeddings run **locally** in this setup (sentence-transformers) — no
  document text is sent anywhere just to generate embeddings.
- Only the **retrieved chunks** (not the whole database) are sent to the
  Claude API as part of the prompt when answering a question.
- Keep `.env` out of version control (add it to `.gitignore`).
- If data is highly sensitive, consider Anthropic's data handling policies
  for the API tier you're using, and apply document-level access control
  before ingestion (i.e., don't index documents a given user shouldn't see —
  this simple version doesn't implement per-user permissions).

## Free, legitimate datasets to practice on

Real internal company data obviously isn't public, but these are solid free
stand-ins that resemble enterprise data (financial filings, internal comms,
support tickets, policy docs) for building and testing your pipeline:

