"""
embeddings/embedder.py
----------------------
Multilingual embedding pipeline using intfloat/multilingual-e5-large.

Loads the HuggingFace model, auto-detects GPU (CUDA) or falls back to CPU,
batches the chunk texts, computes dense embeddings, caches results to disk,
and returns embeddings ready for Qdrant indexing.

Usage:
  python embeddings/embedder.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional in lightweight test envs
    SentenceTransformer = None

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    LOCAL_TEST_MODE,
    PROCESSED_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.embedder")

# Cache file next to processed data
CACHE_PATH: Path = PROCESSED_DATA_PATH.parent / "embeddings_cache.pkl"


def get_device() -> str:
    """
    Detect and return the best available compute device.

    Prefers CUDA GPU for fast batch embedding. Falls back to CPU when
    no CUDA-capable device is found (e.g. on CI or laptop without GPU).

    Returns:
        Device string: "cuda" or "cpu".
    """
    if torch is not None and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️  GPU detected: {gpu_name} ({vram:.1f} GB VRAM) → using CUDA")
    else:
        device = "cpu"
        logger.warning("⚠️  No CUDA GPU found — using CPU (will be slow)")
    return device


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model from HuggingFace or local cache.

    The multilingual-e5-large model produces 1024-dimensional vectors and
    supports 100+ languages including Bengali.

    Args:
        model_name: HuggingFace model identifier (e.g. "intfloat/multilingual-e5-large").
        device: Compute device ("cuda" or "cpu").

    Returns:
        Loaded SentenceTransformer model ready for inference.
    """
    logger.info(f"📦 Loading embedding model: {model_name} on {device}")
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not available in this environment. "
            "Install the project dependencies to run embedding generation."
        )
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"✅ Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as exc:
        logger.error(f"❌ Failed to load model {model_name}: {exc}", exc_info=True)
        raise


def prepare_texts_for_e5(chunks: list[dict]) -> list[str]:
    """
    Prefix each chunk text with "passage: " as required by multilingual-e5 models.

    The E5 family uses asymmetric embedding: queries are prefixed with "query: "
    and passages/documents with "passage: ". This is essential for correct
    retrieval performance.

    Args:
        chunks: List of chunk dicts, each with a "text" key.

    Returns:
        List of prefixed strings ready for embedding.
    """
    # multilingual-e5 requires "passage: " prefix for document texts
    return [f"passage: {chunk['text']}" for chunk in chunks]


def compute_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Compute dense embeddings for a list of texts in batches.

    Processes texts in mini-batches to manage GPU VRAM. Embeddings are
    L2-normalised so cosine similarity == dot product (faster retrieval).

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of prefixed text strings to embed.
        batch_size: Number of texts per GPU batch.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    logger.info(f"🔢 Embedding {len(texts)} texts (batch_size={batch_size})")

    try:
        # encode() returns a numpy array of shape (N, embedding_dim)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalise for cosine similarity
            convert_to_numpy=True,
        )
        logger.info(f"✅ Embeddings computed: shape {embeddings.shape}")
        # Convert numpy array rows to Python lists for JSON-serialisability
        return embeddings.tolist()
    except Exception as exc:
        logger.error(f"❌ Embedding failed: {exc}", exc_info=True)
        raise


def load_cache(cache_path: Path) -> Optional[dict]:
    """
    Load cached embeddings from disk if available.

    Args:
        cache_path: Path to the pickle cache file.

    Returns:
        Dictionary with keys "chunks" and "embeddings", or None if not found.
    """
    if cache_path.exists():
        logger.info(f"💾 Loading embeddings from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"⚠️  Cache load failed ({exc}), recomputing.")
    return None


def save_cache(chunks: list[dict], embeddings: list[list[float]], cache_path: Path) -> None:
    """
    Save computed embeddings and chunk metadata to a pickle cache.

    Args:
        chunks: List of chunk dicts.
        embeddings: Corresponding embedding vectors.
        cache_path: Destination file path.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    logger.info(f"💾 Embeddings cached to {cache_path}")


def embed_chunks(
    chunks: list[dict],
    force_recompute: bool = False,
) -> tuple[list[dict], list[list[float]]]:
    """
    Main embedding function: load model, embed all chunks, cache results.

    Checks disk cache first (unless force_recompute=True). If the number
    of cached chunks matches, returns cached results instantly.

    Args:
        chunks: List of processed chunk dicts from ucb_chunks.json.
        force_recompute: If True, skip cache and always recompute.

    Returns:
        Tuple of (chunks, embeddings) where embeddings[i] corresponds to chunks[i].
    """
    # --- Check cache ---
    if not force_recompute:
        cached = load_cache(CACHE_PATH)
        if cached:
            # Validate by chunk count AND source file mtime so a re-scrape with
            # the same number of pages doesn't silently serve stale embeddings.
            cache_mtime = CACHE_PATH.stat().st_mtime if CACHE_PATH.exists() else 0
            source_mtime = PROCESSED_DATA_PATH.stat().st_mtime if PROCESSED_DATA_PATH.exists() else 0
            count_ok = len(cached.get("chunks", [])) == len(chunks)
            mtime_ok = cache_mtime >= source_mtime
            if count_ok and mtime_ok:
                logger.info("✅ Using cached embeddings (count and mtime match).")
                return cached["chunks"], cached["embeddings"]
            else:
                reason = "count mismatch" if not count_ok else "source file newer than cache"
                logger.info(f"🔄 Cache invalid ({reason}) — recomputing embeddings.")

    # --- Compute fresh embeddings ---
    device = get_device()
    model = load_embedding_model(EMBEDDING_MODEL, device)
    texts = prepare_texts_for_e5(chunks)
    embeddings = compute_embeddings(model, texts)

    # --- Cache to disk ---
    save_cache(chunks, embeddings, CACHE_PATH)

    return chunks, embeddings


def load_chunks(path: Path) -> list[dict]:
    """
    Load processed chunk data from the JSON file produced by chunker.py.

    Args:
        path: Path to ucb_chunks.json.

    Returns:
        List of chunk dicts.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed chunks not found at {path}. "
            "Run the chunker first: python preprocessing/chunker.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"📂 Loaded {len(chunks)} chunks from {path}")
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: load chunks, compute embeddings, save cache.

    In LOCAL_TEST_MODE, only the first 100 chunks are embedded to
    speed up development iteration cycles.
    """
    logger.info("=" * 60)
    logger.info("UCB Bank Embedder — Starting")
    logger.info(f"Model  : {EMBEDDING_MODEL}")
    logger.info(f"Cache  : {CACHE_PATH}")
    logger.info(f"Local test mode: {LOCAL_TEST_MODE}")
    logger.info("=" * 60)

    try:
        chunks = load_chunks(PROCESSED_DATA_PATH)
        logger.info(f"📦 Embedding all {len(chunks)} chunks (no limit)")

        # Force recompute because the corpus has changed
        chunks, embeddings = embed_chunks(chunks, force_recompute=True)

        logger.info(
            f"🎉 Done! {len(embeddings)} embeddings ready. "
            f"Dim: {len(embeddings[0]) if embeddings else 'N/A'}"
        )

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Embedder failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
