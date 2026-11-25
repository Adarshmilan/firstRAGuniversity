"""
STEP 2 - SPLIT INTO CHUNKS
--------------------------
Goal: cut each document into small, self-contained pieces.

Why? The LLM only gets the TOP_K most relevant pieces, not the whole data.
A chunk = one topic (e.g. "CSE Department" or "Hostel Fee").

Strategy:
  1. Split on markdown headings (## and ###). Each section = one chunk.
  2. Put the heading path at the top of every chunk, e.g.
       "Marwadi University Departments > Computer Science and Engineering (CSE) Department"
     so the chunk still makes sense when read alone.
  3. If a section is still too long, cut it into overlapping windows.

Try it:   python -m rag.step2_split_chunks
"""
from rag.config import CHUNK_OVERLAP_CHARS, MAX_CHUNK_CHARS
from rag.step1_load_documents import load_documents


def split_by_headings(text):
    """Yield (heading_path, body) for each section of a markdown file."""
    headings = {}          # level -> heading text, e.g. {1: "Fees", 2: "Hostel Fee"}
    current_lines = []
    current_path = ""

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            return (current_path, body)
        return None

    for line in text.splitlines():
        if line.startswith("#"):
            # A new heading starts -> finish the previous section
            section = flush()
            if section:
                yield section
            current_lines = []

            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            headings[level] = title
            # forget deeper headings from the previous section
            for deeper in [lvl for lvl in headings if lvl > level]:
                del headings[deeper]
            current_path = " > ".join(headings[lvl] for lvl in sorted(headings))
        else:
            current_lines.append(line)

    section = flush()
    if section:
        yield section


def split_long_text(text, max_chars, overlap):
    """Cut text into windows of max_chars with some overlap, breaking at newlines."""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to end at a newline so we don't cut mid-line
        newline = text.rfind("\n", start, end)
        if newline > start and end < len(text):
            end = newline
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def make_chunks(documents):
    """Returns a list of dicts: [{"id", "text", "source", "section"}, ...]"""
    chunks = []
    for doc in documents:
        for i, (path, body) in enumerate(split_by_headings(doc["text"])):
            for j, piece in enumerate(split_long_text(body, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)):
                chunks.append({
                    "id": f"{doc['source']}::{i}::{j}",
                    "text": f"{path}\n\n{piece}",   # heading path + content
                    "source": doc["source"],
                    "section": path,
                })
    return chunks


if __name__ == "__main__":
    chunks = make_chunks(load_documents())
    print(f"Made {len(chunks)} chunks\n")
    # Look at a few chunks with your own eyes - this is the best debugging you can do.
    for c in chunks[:3]:
        print("=" * 60)
        print(f"[{c['id']}]")
        print(c["text"])
