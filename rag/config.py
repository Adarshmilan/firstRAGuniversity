"""
config.py - all settings in one place so the other files stay clean.
Change values here instead of hunting through code.
"""
from pathlib import Path

# Folder layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # your .md files live here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # vector DB is saved here (auto-created)
COLLECTION_NAME = "college_docs"

# Chunking
MAX_CHUNK_CHARS = 1500    # a section bigger than this gets split further
CHUNK_OVERLAP_CHARS = 150 # overlap so a sentence cut in half isn't lost

# Embeddings - small, free, runs on CPU. Downloads (~90 MB) on first use.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
TOP_K = 4  # how many chunks to send to the LLM

# LLM (Google Gemini). Needs GEMINI_API_KEY in the environment (it's in .env).
# "gemini-flash-latest" always points to Google's newest Flash model - fast and cheap.
# Other options your key can use: "gemini-2.5-flash", "gemini-pro-latest".
LLM_MODEL = "gemini-3.8-flash"
# If the main model says "too many requests" (429) or is overloaded (503),
# we retry a few times, then try these models in order. Each model has its own
# free-tier quota, so switching usually gets an answer.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
MAX_RETRIES = 3          # tries per model before moving to the next one
RETRY_BASE_SECONDS = 2   # wait 2s, 4s, 8s ... between tries
MAX_TOKENS = 1024

FALLBACK_CONTACT = "the Admin Office at 0731-2400100"
