# Marwadi University Chatbot (RAG)

```
data/*.md ──► step1 load ──► step2 chunk ──► step3 embed + store (chroma_db/)
                                                        │
user question ──► step4 retrieve top-k chunks ◄─────────┘
                        │
                        ▼
               step5 Gemini answers using ONLY those chunks ──► app.py (chat UI)
```

## Setup (do this once)

```bash
cd ~/projects/Mchat
uv add -r requirements.txt      # installs libraries into Mchat/.venv + records them in pyproject.toml
```

The Gemini API key lives in `.env` (`GEMINI_API_KEY=...`). `.env` is in `.gitignore` - never commit it.
Pass it to any command with `uv run --env-file .env ...`.

## Run it, step by step

All commands are run from the `Mchat/` folder (not inside `rag/`). Prefix each with `uv run --env-file .env`.

| Step | Command | What you should see |
|------|---------|---------------------|
| 1 | `python -m rag.step1_load_documents` | 8 files loaded |
| 2 | `python -m rag.step2_split_chunks` | chunk count + 3 sample chunks |
| 3 | `python -m rag.step3_build_index` | `chroma_db/` folder created |
| 4 | `python -m rag.step4_retrieve "When is the CSE HOD free?"` | `04_departments_and_hods.md` on top |
| 5 | `python -m rag.step5_generate "What is the B.Tech fee?"` | an answer from Gemini |
| 6 | `streamlit run rag/app.py` | chat page at http://localhost:8501 |
| 7 | `python -m rag.step7_evaluate` (add `--answers` to test Gemini too) | PASS/FAIL list |

Re-run step 3 every time you edit a file in `data/`.
