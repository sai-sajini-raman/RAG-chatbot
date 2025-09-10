# RAG Chatbot — Excel → Chroma → LlamaIndex → GPT-4o (via token)

**Overview**
This project is a Retrieval-Augmented Generation (RAG) chatbot that:
- Ingests a single Excel file as the knowledge base.
- Chunks rows into embeddings-friendly text (keeps sheet/row/column metadata).
- Uses a local HuggingFace sentence-transformer for embeddings.
- Stores embeddings in a local Chroma vector DB.
- Builds a LlamaIndex index for orchestration.
- Uses an LLM (GPT-4o) via an API token (configured in `config.py` / environment).
- Streamlit frontend with chat UI and monitoring dashboard.

**Important**
- This project expects a single Excel file located at `data/MAL-Food-SC.xlsx`.
- Set your API token in environment variable `GITHUB_TOKEN` or `OPENAI_API_KEY`.
- If you plan to use a GitHub Marketplace token as the LLM bearer token, place it in `GITHUB_TOKEN`.

---
## Quick setup (tested on Python 3.10+)

1. Clone or unzip this project.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your Excel knowledge file at `data/MAL-Food-SC.xlsx`.
   - The ingestion script expects one Excel file with one or more sheets.
5. Configure API token:
   ```bash
   set GITHUB_TOKEN=your_token_here        # Windows (cmd)
   ```
   Alternatively set `OPENAI_API_KEY`.
6. Run ingestion (creates embeddings + populates Chroma + builds LlamaIndex):
   ```bash
   python ingest.py
   ```
7. Run Streamlit UI:
   ```bash
   streamlit run main.py
   ```

---
## Files produced
- `main.py` — Streamlit app (chat UI + metrics)
- `ingest.py` — parse Excel, chunk text, create embeddings, populate Chroma and LlamaIndex
- `rag_pipeline.py` — retrieval + LLM call + prompt logic
- `monitoring.py` — logging of queries, latency, and quality (CSV & SQLite)
- `config.py` — configuration file (paths, model names, tokens)
- `requirements.txt` — pip dependencies
- `logs/` — created at runtime for CSV/SQLite logs
- `chroma_db/` — created by Chroma for persisted embeddings
- `data/MAL-Food-SC.xlsx` — (user-supplied) knowledge source

---
## Notes, caveats, and extension ideas
- The code attempts to support using either `OPENAI_API_KEY` or `GITHUB_TOKEN` as the bearer token for requests to OpenAI-style chat endpoints. If you truly have a different LLM endpoint, adjust `rag_pipeline.py` accordingly.
- Hallucination control: when retrieval returns no relevant chunks (or below threshold), the assistant will reply:
  ```
  I don’t know, not in knowledge base.
  ```
- Trust layer: every response includes sources (sheet, row index, column headers).
- The LlamaIndex is used as an orchestration layer; Chroma is used for persisted similarity search.

If you run into issues, paste any error here and I'll help debug.
