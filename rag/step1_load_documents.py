"""
STEP 1 - LOAD DOCUMENTS
-----------------------
Goal: read every .md file in data/ and keep its text + where it came from.

Why metadata ("source")? So the chatbot can show "Answer from: 02_fee_structure.md"
and so you can debug which file a wrong answer came from.

Try it:   python -m rag.step1_load_documents
"""
from rag.config import DATA_DIR


def load_documents():
    """Returns a list of dicts: [{"text": "...", "source": "file.md"}, ...]"""
    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({"text": text, "source": path.name})
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for d in docs:
        print(f"  - {d['source']}: {len(d['text'])} characters")
