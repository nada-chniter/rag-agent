"""
config.py
---------
Every setting the agent depends on lives here. Centralizing config means
you never hunt through multiple files to change a chunk size or model name.

We load secrets (API keys) from a .env file rather than hardcoding them —
never commit real keys to source control.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the project root if present

# --- LLM (generation) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GENERATION_MODEL = "claude-sonnet-4-6"   # good balance of quality/cost for RAG
MAX_TOKENS = 1024

# --- Embeddings (local, free, no API calls, no data leaves your machine) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, strong for retrieval

# --- Chunking ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap so context isn't cut mid-idea

# --- Vector store ---
VECTOR_DB_PATH = "./chroma_db"     # local persistent storage
COLLECTION_NAME = "enterprise_kb"

# --- Retrieval ---
TOP_K = 5               # how many chunks to retrieve per search
MAX_AGENT_SEARCH_STEPS = 4   # cap on how many times the agent can search per query

if not ANTHROPIC_API_KEY:
    print("⚠️  ANTHROPIC_API_KEY not set. Set it in a .env file before running.")
