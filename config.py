import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# Path to the single Excel file (user should supply)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
EXCEL_PATH = DATA_DIR / "MAL-Food-SC.xlsx"

# Chroma DB persistent directory
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

# LlamaIndex storage (can be the same dir)
INDEX_DIR = BASE_DIR / "index_data"
INDEX_DIR.mkdir(exist_ok=True)

# Sentence transformers model for embeddings
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Embedding dimensions (all-MiniLM-L6-v2 -> 384)
EMBEDDING_DIM = 384

# LLM settings (user should set env var GITHUB_TOKEN or OPENAI_API_KEY)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # per user's request (Marketplace token)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Which bearer token to prefer (github or openai). If both present, GITHUB_TOKEN is used.
BEARER_TOKEN = GITHUB_TOKEN or OPENAI_API_KEY

# LLM endpoint (OpenAI-compatible). Adjust if you have a different endpoint.
OPENAI_CHAT_COMPLETION_URL = "https://models.github.ai/inference"

# Retrieval settings
TOP_K = 5

# Chunking
CHUNK_WORD_SIZE = 450   # ~500 tokens conservatively (word-based approx)
CHUNK_WORD_OVERLAP = 50

# Monitoring
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CSV_LOG_PATH = LOG_DIR / "queries.csv"
SQLITE_DB = LOG_DIR / "monitoring.db"

# Streamlit
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
