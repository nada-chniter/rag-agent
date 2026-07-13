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
# 1. Index your documents (put .pdf, .txt, .md, .csv files in a folder)
python main.py ingest ./sample_data

# 2. Chat with the agent
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

The agent pattern matters for enterprise Q&A specifically because real
questions are often multi-hop ("What's our refund policy for enterprise
customers who signed before 2023?") — a single fixed retrieval often misses
half the answer. Letting Claude issue follow-up searches materially improves
accuracy over naive RAG.

## Connecting to a real enterprise database (not just files)

This starter kit ingests files. To point it at a live database:

- **SQL databases**: don't embed raw rows. Instead, either (a) write a
  `sql_to_documents.py` that periodically exports rows as text documents
  (e.g. one document per customer/ticket/record) and re-runs `ingest`, or
  (b) give the agent a **second tool**, `query_sql_database`, that runs
  read-only SQL and returns results directly — better for precise numeric
  lookups than semantic search.
- **Confluence / SharePoint / Notion**: use their APIs to pull pages as
  markdown/text, then run through the same `ingest` pipeline.
- **Incremental updates**: track a `last_indexed` timestamp per source and
  only re-embed changed documents — re-embedding everything on a schedule
  gets expensive fast.

## Security notes for real enterprise data

- Embeddings run **locally** in this setup (sentence-transformers) — no
  document text is sent anywhere just to generate embeddings.
- Only the **retrieved chunks** (not your whole database) are sent to the
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

- **SEC EDGAR** (sec.gov/edgar) — every public company's 10-K/10-Q filings,
  free, no signup. Great for financial/enterprise-report style RAG.
- **Enron Email Dataset** (cs.cmu.edu/~enron) — ~500k real corporate emails,
  the classic dataset for testing RAG/search on internal business
  communication.
- **Hugging Face Datasets Hub** (huggingface.co/datasets) — search
  "customer support", "helpdesk", "legal contracts", "financial reports" for
  ready-made enterprise-style corpora.
- **Kaggle Datasets** (kaggle.com/datasets) — search "company", "support
  tickets", "invoices" — many are cleaned and ready to use.
- **Data.gov / data.europa.eu** — government open data, useful for testing
  ingestion at scale with structured + unstructured mixed content.
- **UCI Machine Learning Repository** — smaller, well-documented datasets,
  good for quick pipeline tests before scaling up.

All of the above are free, publicly licensed for research/practice, and
don't require handling real PII — good for building confidence in your
pipeline before pointing it at anything sensitive.
