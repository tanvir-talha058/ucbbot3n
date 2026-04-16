"""
config/settings.py
------------------
Central configuration module for UCB Bank RAG Chatbot.
All values are loaded from the .env file using python-dotenv.
No hardcoded values — everything is environment-driven.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Qdrant vector database settings
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "ucb_bank")

# ---------------------------------------------------------------------------
# Ollama LLM settings
# ---------------------------------------------------------------------------
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:latest")

# ---------------------------------------------------------------------------
# Embedding model (HuggingFace)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "8"))

_confidence_threshold_raw = os.getenv("CONFIDENCE_THRESHOLD", "0.3").strip()
try:
    _confidence_threshold = float(_confidence_threshold_raw)
except ValueError:
    _confidence_threshold = 0.3

# Keep the threshold in a valid probability range even if the environment
# contains a stale or experimental value.
CONFIDENCE_THRESHOLD: float = (
    _confidence_threshold if 0.0 <= _confidence_threshold <= 1.0 else 0.3
)

# ---------------------------------------------------------------------------
# Scraper settings
# ---------------------------------------------------------------------------
UCB_BASE_URL: str = os.getenv("UCB_BASE_URL", "https://www.ucb.com.bd")

# ---------------------------------------------------------------------------
# OCR / Tesseract settings
# ---------------------------------------------------------------------------
TESSERACT_PATH: str = os.getenv(
    "TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

# ---------------------------------------------------------------------------
# Developer / local test mode
# LOCAL_TEST_MODE=true → use smaller batches and shorter timeouts for fast testing
# ---------------------------------------------------------------------------
LOCAL_TEST_MODE: bool = os.getenv("LOCAL_TEST_MODE", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Derived / computed paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "ucb_raw.json"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "ucb_chunks.json"
VOCAB_PATH: Path = DATA_DIR / "processed" / "bm25_vocab.json"

# Embedding vector dimension for multilingual-e5-large
EMBEDDING_DIM: int = 1024

# BM25 sparse vector size (vocabulary size for sparse index)
SPARSE_VECTOR_SIZE: int = 30000

# Scraper: polite crawl delay in seconds
CRAWL_DELAY: float = 1.5

# Scraper: max pages to crawl.
# None = no limit (crawl every page on the site).
# Set MAX_PAGES env var to an integer to cap it (useful for quick tests).
_max_pages_env = os.getenv("MAX_PAGES", "")
MAX_PAGES: int | None = int(_max_pages_env) if _max_pages_env.strip().isdigit() else None

# Conversation memory window (number of past exchanges to keep)
MEMORY_WINDOW: int = 5

# Batch size for embedding computation
EMBED_BATCH_SIZE: int = 16 if LOCAL_TEST_MODE else 64
