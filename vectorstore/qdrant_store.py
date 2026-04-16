"""
vectorstore/qdrant_store.py
----------------------------
Qdrant vector store manager for UCB Bank RAG Chatbot.

Handles:
- Connecting to the Qdrant Docker instance
- Creating a collection with both dense (E5) and sparse (BM25) vectors
- Indexing all processed chunks with their embeddings
- Providing hybrid search (dense + BM25) via Reciprocal Rank Fusion

Usage:
  python vectorstore/qdrant_store.py
"""

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    EMBEDDING_DIM,
    PROCESSED_DATA_PATH,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SPARSE_VECTOR_SIZE,
    TOP_K_RETRIEVAL,
    VOCAB_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.qdrant_store")

# Batch size for Qdrant upsert calls
UPSERT_BATCH_SIZE = 100


def get_qdrant_client():
    """
    Create and return a QdrantClient connected to the Docker instance.

    Returns:
        QdrantClient instance.

    Raises:
        ConnectionError: If Qdrant is unreachable.
    """
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        # Quick health check
        client.get_collections()
        logger.info(f"✅ Connected to Qdrant at {QDRANT_URL}")
        return client
    except Exception as exc:
        raise ConnectionError(
            f"Cannot connect to Qdrant at {QDRANT_URL}. "
            "Is 'docker-compose up -d' running? Error: {exc}"
        ) from exc


def create_collection(client, collection_name: str, recreate: bool = False) -> None:
    """
    Create (or recreate) a Qdrant collection with dense + sparse vector config.

    Dense vectors use COSINE distance (E5 embeddings are L2-normalised so
    cosine == dot product). Sparse vectors use DOT distance for BM25 scores.

    Args:
        client: Active QdrantClient.
        collection_name: Name of the collection to create.
        recreate: If True, delete existing collection before creating.
    """
    from qdrant_client.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.info(f"🗑️  Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"✅ Collection '{collection_name}' already exists. Skipping creation.")
            return

    logger.info(f"📦 Creating collection: {collection_name}")
    logger.info(f"   Dense dim: {EMBEDDING_DIM}, Sparse size: {SPARSE_VECTOR_SIZE}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Dense E5 embedding vectors
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # Sparse BM25 vectors for keyword matching
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(f"✅ Collection '{collection_name}' created successfully.")


def build_bm25_sparse_vector(text: str, vocab: dict[str, int]) -> dict[str, Any]:
    """
    Build a sparse BM25-style vector for a text chunk.

    Computes TF (term frequency) scores for each token found in the
    vocabulary. Returns indices and values for Qdrant's sparse vector format.

    Args:
        text: The text to vectorise.
        vocab: Mapping of token → index in the sparse vocabulary.

    Returns:
        Dict with "indices" and "values" lists for SparseVector.
    """
    from collections import Counter
    import math

    tokens = text.lower().split()
    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    indices = []
    values = []

    for token, count in token_counts.items():
        if token in vocab:
            # BM25-like TF score: log(1 + tf)
            tf = math.log(1 + count / max(total_tokens, 1))
            indices.append(vocab[token])
            values.append(float(tf))

    return {"indices": indices, "values": values}


def build_vocabulary(chunks: list[dict], max_size: int = SPARSE_VECTOR_SIZE) -> dict[str, int]:
    """
    Build a token-to-index vocabulary from all chunk texts.

    Selects the `max_size` most frequent tokens across the entire corpus.
    This vocabulary is used to create sparse BM25 vectors.

    Args:
        chunks: List of chunk dicts with "text" fields.
        max_size: Maximum vocabulary size (caps sparse vector dimension).

    Returns:
        Dictionary mapping token string → integer index.
    """
    from collections import Counter

    logger.info("🔤 Building BM25 vocabulary from corpus...")
    all_tokens: list[str] = []

    for chunk in chunks:
        # Tokenise by whitespace and lowercase for language-agnostic matching
        tokens = chunk["text"].lower().split()
        all_tokens.extend(tokens)

    # Keep only the most frequent tokens to bound vocabulary size
    counter = Counter(all_tokens)
    most_common = counter.most_common(max_size)
    vocab = {token: idx for idx, (token, _) in enumerate(most_common)}

    logger.info(f"✅ Vocabulary built: {len(vocab)} unique tokens")
    return vocab


def index_chunks(
    client,
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str,
) -> None:
    """
    Index all chunks into Qdrant with dense + sparse vectors and metadata.

    Processes chunks in batches of UPSERT_BATCH_SIZE to avoid memory issues
    and improve throughput.

    Args:
        client: Active QdrantClient.
        chunks: List of chunk dicts (with text, url, title, language, etc.).
        embeddings: Dense embedding vectors, one per chunk.
        collection_name: Target Qdrant collection.
    """
    from qdrant_client.models import PointStruct, SparseVector

    logger.info(f"📤 Indexing {len(chunks)} chunks into '{collection_name}'...")

    # Build sparse vocabulary from all chunks and persist it for retrieval-time use
    vocab = build_vocabulary(chunks)
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    logger.info(f"💾 BM25 vocabulary saved to {VOCAB_PATH}")

    total_indexed = 0

    # Process in batches
    for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch_chunks = chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
        batch_embeddings = embeddings[batch_start: batch_start + UPSERT_BATCH_SIZE]

        points = []
        for chunk, dense_vec in zip(batch_chunks, batch_embeddings):
            # Build sparse vector for this chunk
            sparse_data = build_bm25_sparse_vector(chunk["text"], vocab)

            # Skip if sparse vector is empty (chunk has no vocab tokens)
            if not sparse_data["indices"]:
                sparse_data = {"indices": [0], "values": [0.0]}

            point = PointStruct(
                id=str(uuid.uuid4()),          # unique UUID for each chunk
                vector={
                    "dense": dense_vec,         # E5 embedding
                    "sparse": SparseVector(     # BM25 sparse vector
                        indices=sparse_data["indices"],
                        values=sparse_data["values"],
                    ),
                },
                payload={
                    # Store full metadata for retrieval and display
                    "text": chunk["text"],
                    "url": chunk.get("url", ""),
                    "title": chunk.get("title", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "language": chunk.get("language", "english"),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            )
            points.append(point)

        # Upsert batch to Qdrant
        client.upsert(collection_name=collection_name, points=points)
        total_indexed += len(points)
        logger.info(
            f"   ✅ Indexed batch {batch_start // UPSERT_BATCH_SIZE + 1}: "
            f"{total_indexed}/{len(chunks)} chunks"
        )

    logger.info(f"🎉 All {total_indexed} chunks indexed successfully.")


def get_collection_stats(client, collection_name: str) -> dict:
    """
    Retrieve basic statistics about the Qdrant collection.

    Args:
        client: Active QdrantClient.
        collection_name: Name of the collection to query.

    Returns:
        Dict with total_points and collection_name.
    """
    try:
        info = client.get_collection(collection_name)
        return {
            "collection_name": collection_name,
            "total_points": info.points_count,
            "status": str(info.status),
        }
    except Exception as exc:
        logger.error(f"❌ Failed to get stats: {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: load embeddings cache, create Qdrant collection,
    and index all chunks.
    """
    import pickle

    # Path to the embeddings cache created by embedder.py
    cache_path = PROCESSED_DATA_PATH.parent / "embeddings_cache.pkl"

    logger.info("=" * 60)
    logger.info("UCB Bank Qdrant Store — Starting")
    logger.info(f"Qdrant URL : {QDRANT_URL}")
    logger.info(f"Collection : {QDRANT_COLLECTION}")
    logger.info("=" * 60)

    # Load cached embeddings
    if not cache_path.exists():
        logger.error(
            f"❌ Embeddings cache not found at {cache_path}. "
            "Run embedder first: python embeddings/embedder.py"
        )
        sys.exit(1)

    logger.info(f"📂 Loading embeddings cache from {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    chunks: list[dict] = cache["chunks"]
    embeddings: list[list[float]] = cache["embeddings"]
    logger.info(f"✅ Loaded {len(chunks)} chunks with {len(embeddings[0])}-dim embeddings")

    try:
        # Connect to Qdrant
        client = get_qdrant_client()

        # Always recreate to ensure a clean index with the latest data
        create_collection(client, QDRANT_COLLECTION, recreate=True)

        # Index all chunks
        index_chunks(client, chunks, embeddings, QDRANT_COLLECTION)

        # Final stats
        stats = get_collection_stats(client, QDRANT_COLLECTION)
        logger.info(f"📊 Collection stats: {stats}")

    except ConnectionError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Qdrant store failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
