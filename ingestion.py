"""
ingestion.py
------------
Step 1 of the pipeline: turn raw files into clean text chunks ready to embed.

Why chunk at all? LLMs and embedding models work best on small, semantically
coherent pieces of text. A 50-page PDF as one "chunk" would (a) blow past
embedding model limits and (b) return low-precision search results, because
the vector would represent an average of many unrelated topics.

We use a simple sliding-window chunker with overlap: overlap prevents an
important sentence from being split awkwardly across two chunks and losing
context on both sides.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    source: str      # filename it came from
    chunk_id: int     # position within that file


def load_text_from_file(path: Path) -> str:
    """Extract raw text depending on file type."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix in (".txt", ".md", ".csv"):
        return path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sliding-window chunker.
    Moves forward by (chunk_size - overlap) each step, so consecutive
    chunks share `overlap` characters of context.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_and_chunk_directory(directory: str) -> list[Chunk]:
    """
    Walk a directory, load every supported file, chunk it, and return
    a flat list of Chunk objects ready for embedding.
    """
    all_chunks = []
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    supported = (".pdf", ".txt", ".md", ".csv")
    files = [f for f in directory_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"⚠️  No supported files found in {directory} (looked for {supported})")

    for file_path in files:
        try:
            raw_text = load_text_from_file(file_path)
            pieces = chunk_text(raw_text)
            for i, piece in enumerate(pieces):
                all_chunks.append(Chunk(text=piece, source=file_path.name, chunk_id=i))
            print(f"  ✓ {file_path.name}: {len(pieces)} chunks")
        except Exception as e:
            print(f"  ✗ Skipped {file_path.name}: {e}")

    return all_chunks
