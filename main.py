"""
main.py
-------
Command-line entry point.

Usage:
    python main.py ingest ./sample_data      # index documents (run once, or whenever data changes)
    python main.py chat                       # interactive Q&A loop
"""

import sys

from ingestion import load_and_chunk_directory
from vectorstore import VectorStore
from rag_agent import RAGAgent


def cmd_ingest(directory: str):
    print(f"Loading and chunking files from: {directory}")
    chunks = load_and_chunk_directory(directory)
    print(f"Total chunks: {len(chunks)}")

    store = VectorStore()
    store.add_chunks(chunks)
    print(f"Vector store now contains {store.count()} chunks total.")


def cmd_chat():
    agent = RAGAgent()
    print(f"Knowledge base has {agent.store.count()} chunks indexed.")
    print("Type your question (or 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = agent.ask(question)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python main.py ingest <directory>\n  python main.py chat")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a directory: python main.py ingest ./sample_data")
            sys.exit(1)
        cmd_ingest(sys.argv[2])

    elif command == "chat":
        cmd_chat()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
