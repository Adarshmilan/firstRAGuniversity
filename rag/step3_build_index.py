"""
STEP 3 - EMBED + STORE IN VECTOR DB  (run this ONCE, and again when data changes)
-------------------------------------
Goal: turn each chunk into a vector (list of numbers that captures meaning)
and save chunks + vectors in ChromaDB on disk.

Why vectors? "When is CSE HOD available?" and "CSE HOD Free Hours" share few
exact words, but their vectors are close -> semantic search finds it.

Run:   python -m rag.step3_build_index
(First run downloads the embedding model, ~90 MB.)
"""
import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from rag.step1_load_documents import load_documents
from rag.step2_split_chunks import make_chunks


def build_index():
    chunks = make_chunks(load_documents())
    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} ...")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([c["text"] for c in chunks], show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Start fresh each time so old/deleted chunks don't stay behind
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity = standard for text
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )
    print(f"Saved {collection.count()} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
