"""
vectorstore.py
--------------
Step 2 & 3: embed chunks into vectors, store/search them.

We use:
- sentence-transformers for embeddings: runs LOCALLY on your machine.
  No text ever leaves your network to generate embeddings — important
  when the source is sensitive enterprise data.
- ChromaDB as the vector store: lightweight, persists to a local folder,
  no external service to manage.

A vector store lets us do semantic search: instead of matching exact
keywords, it finds chunks whose *meaning* is closest to the query's meaning.
"""

import chromadb
from chromadb.utils import embedding_functions

from config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH, COLLECTION_NAME, TOP_K
from ingestion import Chunk


class VectorStore:
    def __init__(self):
        # PersistentClient writes to disk so you don't re-embed everything
        # every time you start the program.
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

        # This embedding function downloads the model once (cached locally)
        # and runs it on CPU/GPU without any network calls after that.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: list[Chunk]):
        """Embed and store chunks. Chroma handles the embedding call internally."""
        if not chunks:
            return

        self.collection.add(
            ids=[f"{c.source}-{c.chunk_id}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_id": c.chunk_id} for c in chunks],
        )
        print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'.")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Semantic search: returns the top_k chunks most similar in meaning
        to the query, along with their source metadata.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "source": meta["source"], "distance": dist})
        return hits

    def count(self) -> int:
        return self.collection.count()
