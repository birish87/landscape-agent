"""
data/seed_vector_db.py
======================
Seeds your ChromaDB vector database with the knowledge documents.

Run this ONCE before using the semantic RAG version:
    python src/data/seed_vector_db.py

WHAT THIS DOES:
1. Reads all documents from knowledge_docs.json
2. Generates vector embeddings for each document using the Anthropic API
   (or optionally OpenAI's text-embedding-3-small — your choice)
3. Stores them in ChromaDB (local file-based vector database)

After seeding, replace retrieve_knowledge_simple() calls in
tools/tools.py with retrieve_knowledge_semantic() defined below.

VECTOR DATABASE CONCEPTS:
- Embedding: A list of ~1000-1500 numbers that represent the meaning
  of a piece of text. Similar texts have similar numbers (close in space).
- Vector search: Given a query embedding, find the stored embeddings
  with the smallest distance (most similar meaning).
- ChromaDB: A free, open-source vector database that runs locally.
  Perfect for development. Persists to disk at ./chroma_db/

UPGRADING TO PINECONE (production):
    import pinecone
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("landscape-knowledge")
    index.upsert(vectors=[(doc["id"], embedding, {"content": doc["content"]}) for doc, embedding in zip(docs, embeddings)])

    # Query:
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    retrieved = results["matches"][0]["metadata"]["content"]
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DOCS_PATH = Path(__file__).parent / "knowledge_docs.json"
CHROMA_DB_PATH = str(Path(__file__).parent.parent.parent / "chroma_db")


def seed_chromadb():
    """
    Load documents from knowledge_docs.json into ChromaDB.
    ChromaDB generates its own embeddings using sentence-transformers.
    """
    try:
        import chromadb
    except ImportError:
        print("ChromaDB not installed. Run: pip install chromadb")
        return

    print(f"Loading documents from {DOCS_PATH}...")
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    # Initialize ChromaDB (persists to disk)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Create or get the collection
    # ChromaDB uses sentence-transformers by default for embeddings
    # This is free and runs locally — no API key needed for basic use
    collection = client.get_or_create_collection(
        name="landscape_knowledge",
        metadata={"description": "Landscaping service docs and procedures"}
    )

    # Add all documents
    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=[doc["content"] for doc in docs],
        metadatas=[{
            "topic": doc["topic"],
            "title": doc["title"],
            "source": doc["source"]
        } for doc in docs]
    )

    print(f"✓ Seeded {len(docs)} documents into ChromaDB at {CHROMA_DB_PATH}")
    print("\nTest query:")
    results = collection.query(
        query_texts=["when should I aerate in Ohio"],
        n_results=1
    )
    print(f"  Query: 'when should I aerate in Ohio'")
    print(f"  Best match: {results['metadatas'][0][0]['title']}")
    print(f"  Distance: {results['distances'][0][0]:.4f}")
    print("\n✓ ChromaDB is ready! Update tools.py to use retrieve_knowledge_semantic().")


def retrieve_knowledge_semantic(query: str, top_k: int = 1) -> str:
    """
    Production RAG: semantic search using ChromaDB embeddings.

    This replaces retrieve_knowledge_simple() in tools/tools.py.
    Handles synonyms, related concepts, and paraphrases that
    keyword search would miss.

    Examples of queries keyword search FAILS on but semantic HANDLES:
    - "When do I cut the lawn?" → matches "lawn_aeration" / "lawn_treatment"
    - "My tree is sick" → matches "tree_health"
    - "What do I put on the grass in winter?" → matches "lawn_treatment" (winterizer)

    Args:
        query:  Natural language question
        top_k:  Number of documents to retrieve

    Returns:
        Concatenated content of the top-k most relevant documents
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection("landscape_knowledge")

        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs_content = []
        for i, (doc_content, metadata) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            source = metadata.get("source", "Knowledge base")
            docs_content.append(f"[Source: {source}]\n{doc_content}")

        combined = "\n\n---\n\n".join(docs_content)
        print(f"  [RAG] retrieve_knowledge_semantic('{query[:30]}...') → {results['metadatas'][0][0]['title']}")
        return combined

    except Exception as e:
        print(f"  [RAG] ChromaDB error: {e}. Falling back to keyword search.")
        from tools.tools import retrieve_knowledge_simple
        return retrieve_knowledge_simple(query)


if __name__ == "__main__":
    seed_chromadb()
