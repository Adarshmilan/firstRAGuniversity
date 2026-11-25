"""
STEP 4 - RETRIEVE  (the "R" in RAG)
-----------------
Goal: given a question, find the TOP_K most similar chunks.

How: embed the question with the SAME model used in step 3,
then ask ChromaDB for the nearest vectors.

Test retrieval ALONE before adding the LLM. If the right chunk isn't
in the results, the LLM can't answer correctly no matter how good it is.

Try it:   python -m rag.step4_retrieve "When is the CSE HOD free?"
"""
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question, k=TOP_K):
    """Returns a list of dicts: [{"text", "source", "section", "score"}, ...]"""
    query_vector = _get_model().encode([question]).tolist()
    result = _get_collection().query(query_embeddings=query_vector, n_results=k)

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "section": meta["section"],
            "score": round(1 - distance, 3),  # cosine distance -> similarity (higher = better)
        })
    return hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "When is the CSE HOD free?"
    print(f"Question: {q}\n")
    for h in retrieve(q):
        print(f"[{h['score']}] {h['source']} -> {h['section']}")
