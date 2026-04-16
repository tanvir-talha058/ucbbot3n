"""
retrieval/retriever.py
----------------------
Hybrid retrieval with dense + BM25 search and CrossEncoder re-ranking.

Pipeline:
  1. Embed the user query with multilingual-e5-large (prefixed "query: ").
  2. Run dense vector search against Qdrant.
  3. Run sparse BM25 search against Qdrant.
  4. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF).
  5. Re-rank the top-N fused results with a CrossEncoder model.
  6. Return the top RERANK_TOP_K results with confidence scores.

Usage:
  from retrieval.retriever import UCBRetriever
  retriever = UCBRetriever()
  results = retriever.retrieve("What are UCB loan products?")
"""

import logging
import math
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CONFIDENCE_THRESHOLD,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_K,
    PROCESSED_DATA_PATH,
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
logger = logging.getLogger("ucb.retriever")

# CrossEncoder model for re-ranking (multilingual, free, local)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# RRF constant — higher k reduces the impact of rank differences
RRF_K = 60

EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_PATH.parent / "embeddings_cache.pkl"


def _resolve_compute_device(role: str) -> str:
    """
    Resolve compute device for retriever components.

    Priority:
      1) Role-specific env var (EMBEDDING_DEVICE / RERANKER_DEVICE)
      2) Global env var (RETRIEVER_DEVICE)
      3) Auto-detect (cuda if available else cpu)
    """
    role_key = "EMBEDDING_DEVICE" if role == "embedding" else "RERANKER_DEVICE"
    requested = os.getenv(role_key, os.getenv("RETRIEVER_DEVICE", "auto")).strip().lower()

    if requested in {"cpu", "cuda"}:
        return requested

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _normalize_search_results(results: Any) -> list[dict]:
    """Normalize Qdrant search/query_points results into a common list format."""
    if results is None:
        return []

    points = getattr(results, "points", results)
    normalized = []
    for r in points:
        normalized.append(
            {
                "id": str(getattr(r, "id", "")),
                "score": float(getattr(r, "score", 0.0) or 0.0),
                "payload": getattr(r, "payload", {}) or {},
            }
        )
    return normalized

_LOCAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "show", "tell", "the", "to", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "u", "about", "that", "this", "there",
    "any", "can", "could", "would", "should", "i'm", "im", "whats", "what's",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight local retrieval."""
    raw_tokens = re.findall(r"[a-z0-9\u0980-\u09FF']+", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        if not token or token in _LOCAL_STOPWORDS:
            continue

        # Light stemming so singular/plural and verb forms match better in
        # offline lexical retrieval (e.g., application/applications/apply).
        norm = token
        for suffix in ("ing", "ation", "ations", "ed", "es", "s"):
            if len(norm) > 5 and norm.endswith(suffix):
                norm = norm[: -len(suffix)]
                break
        tokens.append(norm)
    return tokens


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Combine dense and sparse result lists using Reciprocal Rank Fusion.

    RRF score for a document d: sum over ranked lists L of 1 / (k + rank(d, L))

    Args:
        dense_results: Ranked list from dense vector search (each has "id", "payload").
        sparse_results: Ranked list from BM25 sparse search.
        k: RRF smoothing constant (default 60 per the original RRF paper).

    Returns:
        Merged and re-ranked list of result dicts sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}  # id → payload dict

    # Accumulate RRF scores from dense list
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = result

    # Accumulate RRF scores from sparse list
    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = result

    # Sort by descending RRF score
    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    fused = []
    for doc_id in sorted_ids:
        result = dict(doc_map[doc_id])
        result["rrf_score"] = scores[doc_id]
        fused.append(result)

    return fused


class UCBRetriever:
    """
    Hybrid retriever for UCB Bank RAG chatbot.

    Combines dense semantic search (multilingual-e5-large) with sparse BM25
    keyword search, then re-ranks using a CrossEncoder for precision.

    Attributes:
        _embedding_model: Loaded SentenceTransformer for query encoding.
        _reranker: Loaded CrossEncoder for re-ranking.
        _qdrant_client: Connected QdrantClient.
    """

    def __init__(self) -> None:
        """
        Initialise the retriever by loading all models and connecting to Qdrant.

        Models are loaded lazily on first call to avoid slowing startup.
        """
        self._embedding_model = None   # loaded on first use
        self._reranker = None           # loaded on first use
        self._qdrant_client = None      # connected on first use
        self._vocab: Optional[dict] = None  # loaded on first sparse search
        self._local_corpus: Optional[dict[str, Any]] = None
        logger.info("UCBRetriever initialised (models will load on first query)")

    def _get_embedding_model(self):
        """
        Lazily load the SentenceTransformer embedding model.

        Device is auto-selected by default (CUDA preferred), and can be
        overridden by EMBEDDING_DEVICE or RETRIEVER_DEVICE env vars.

        Returns:
            Loaded SentenceTransformer model on CPU.
        """
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                logger.warning(
                    "⚠️  sentence-transformers unavailable. "
                    "Dense retrieval will be disabled: %s",
                    exc,
                )
                self._embedding_model = None
                return None

            device = _resolve_compute_device("embedding")
            logger.info(f"📦 Loading embedding model {EMBEDDING_MODEL} on {device}")
            try:
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            except Exception as exc:
                logger.warning(
                    "⚠️  Could not load embedding model %s. Dense retrieval disabled: %s",
                    EMBEDDING_MODEL,
                    exc,
                )
                self._embedding_model = None
        return self._embedding_model

    def _get_reranker(self):
        """
        Lazily load the CrossEncoder re-ranking model.

        Device is auto-selected by default (CUDA preferred), and can be
        overridden by RERANKER_DEVICE or RETRIEVER_DEVICE env vars.

        Returns:
            Loaded CrossEncoder model on CPU.
        """
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                logger.warning(
                    "⚠️  sentence-transformers unavailable. "
                    "Re-ranking will be disabled: %s",
                    exc,
                )
                self._reranker = None
                return None

            device = _resolve_compute_device("reranker")
            logger.info(f"📦 Loading re-ranker: {RERANKER_MODEL} on {device}")
            try:
                self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)
            except Exception as exc:
                logger.warning(
                    "⚠️  Could not load reranker %s. Re-ranking disabled: %s",
                    RERANKER_MODEL,
                    exc,
                )
                self._reranker = None
                return None
        return self._reranker

    def _get_qdrant_client(self):
        """
        Lazily connect to Qdrant and return the client.

        Returns:
            Connected QdrantClient.
        """
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient

            self._qdrant_client = QdrantClient(url=QDRANT_URL, timeout=30)
            logger.info(f"✅ Connected to Qdrant at {QDRANT_URL}")
        return self._qdrant_client

    def _embed_query(self, query: str) -> list[float]:
        """
        Encode a user query into a dense embedding vector.

        Prefixes with "query: " as required by the multilingual-e5 model family
        for asymmetric (query vs passage) retrieval.

        Args:
            query: Raw user query string.

        Returns:
            Normalised embedding vector as a list of floats.
        """
        model = self._get_embedding_model()
        if model is None:
            return None
        # E5 requires "query: " prefix for query-side embedding
        prefixed = f"query: {query}"
        vector = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def _dense_search(self, query_vector: Optional[list[float]], top_k: int) -> list[dict]:
        """
        Perform dense vector search in Qdrant using the query embedding.

        Args:
            query_vector: L2-normalised query embedding.
            top_k: Number of results to return.

        Returns:
            List of result dicts with id, score, and payload.
        """
        if not query_vector:
            return []

        client = self._get_qdrant_client()

        try:
            from qdrant_client.models import NamedVector
            named = NamedVector(name="dense", vector=query_vector)
            if hasattr(client, "search"):
                results = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=named,
                    limit=top_k,
                    with_payload=True,
                )
            else:
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=named,
                    limit=top_k,
                    with_payload=True,
                )
            return _normalize_search_results(results)
        except Exception as exc:
            logger.error(f"❌ Dense search failed: {exc}")
            return []

    def _get_vocab(self) -> dict:
        """
        Lazily load the BM25 vocabulary saved by the indexer.

        Returns an empty dict (disabling sparse search) if the vocab file
        does not exist yet — this happens before the first indexing run.

        Returns:
            token → integer-index mapping.
        """
        if self._vocab is None:
            if VOCAB_PATH.exists():
                import json as _json
                with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                    self._vocab = _json.load(f)
                logger.info(f"📖 BM25 vocab loaded: {len(self._vocab)} tokens")
            else:
                logger.warning(
                    f"⚠️  BM25 vocab not found at {VOCAB_PATH}. "
                    "Sparse search disabled until you re-run qdrant_store.py."
                )
                self._vocab = {}
        return self._vocab

    def _get_local_corpus(self) -> Optional[dict[str, Any]]:
        """
        Load the local cached corpus used when Qdrant or embedding services are
        unavailable.

        The repository already contains `data/processed/embeddings_cache.pkl`
        and `data/processed/ucb_chunks.json`, so we can still do meaningful
        retrieval without external services.
        """
        if self._local_corpus is not None:
            return self._local_corpus

        chunks: list[dict] = []
        embeddings: list[list[float]] = []

        if EMBEDDINGS_CACHE_PATH.exists():
            try:
                with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                chunks = cached.get("chunks", [])
                embeddings = cached.get("embeddings", [])
                logger.info(
                    "📦 Local cache loaded: %d chunks, %d embeddings",
                    len(chunks),
                    len(embeddings),
                )
            except Exception as exc:
                logger.warning(
                    "⚠️  Failed to load embeddings cache %s: %s",
                    EMBEDDINGS_CACHE_PATH,
                    exc,
                )

        if not chunks and PROCESSED_DATA_PATH.exists():
            try:
                import json as _json
                with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
                    chunks = _json.load(f)
                logger.info("📚 Local chunk corpus loaded from %s", PROCESSED_DATA_PATH)
            except Exception as exc:
                logger.warning(
                    "⚠️  Failed to load processed chunks %s: %s",
                    PROCESSED_DATA_PATH,
                    exc,
                )

        if not chunks:
            self._local_corpus = None
            return None

        token_counts: list[Counter[str]] = []
        doc_freq: Counter[str] = Counter()
        doc_lengths: list[int] = []

        for chunk in chunks:
            tokens = _tokenize(chunk.get("text", ""))
            counts = Counter(tokens)
            token_counts.append(counts)
            doc_freq.update(counts.keys())
            doc_lengths.append(sum(counts.values()))

        total_docs = len(chunks)
        avgdl = sum(doc_lengths) / total_docs if total_docs else 1.0
        default_idf = math.log((total_docs + 0.5) / 0.5 + 1.0)
        idf = {
            token: math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            for token, df in doc_freq.items()
        }

        self._local_corpus = {
            "chunks": chunks,
            "embeddings": embeddings,
            "token_counts": token_counts,
            "doc_freq": doc_freq,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl or 1.0,
            "idf": idf,
            "default_idf": default_idf,
        }
        return self._local_corpus

    def _local_search(self, query: str, top_k: int) -> list[dict]:
        """
        Search the cached corpus locally with a BM25-like lexical scorer.

        This is the offline fallback used when the full vector stack is not
        available. It is not as strong as the Qdrant path, but it keeps the bot
        useful and grounded in the corpus.
        """
        corpus = self._get_local_corpus()
        if not corpus:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        credit_card_apply_intent = (
            ("credit" in query_token_set and "card" in query_token_set)
            and any(t.startswith("appl") for t in query_token_set)
        )

        k1 = 1.5
        b = 0.75
        avgdl = corpus["avgdl"] or 1.0
        scores: list[tuple[float, int]] = []

        for idx, counts in enumerate(corpus["token_counts"]):
            dl = max(corpus["doc_lengths"][idx], 1)
            score = 0.0
            chunk = corpus["chunks"][idx]
            title = chunk.get("title", "").lower()
            url = chunk.get("url", "").lower()
            for token in query_tokens:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = corpus["idf"].get(token, corpus["default_idf"])
                score += idf * (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * dl / avgdl)
                )

            if not score and credit_card_apply_intent:
                if (
                    "documentation-requirements" in url
                    or "documents required" in title
                    or "choose-your-card" in url
                ):
                    score = 0.20

            if not score:
                continue

            if any(token in title for token in query_tokens):
                score *= 1.15
            if any(token in url for token in query_tokens):
                score *= 1.10

            # Query-intent boost for application requirements pages.
            if credit_card_apply_intent:
                if "/cards/" in url:
                    score *= 1.20
                if "prepaid" in url or "payroll" in url:
                    score *= 0.55
                if (
                    "documentation-requirements" in url
                    or "documents required" in title
                    or "choose-your-card" in url
                    or "important-information" in url
                ):
                    score *= 2.10
            scores.append((score, idx))

        if not scores:
            return []

        scores.sort(key=lambda item: item[0], reverse=True)
        top = scores[:top_k]
        max_score = top[0][0] if top and top[0][0] > 0 else 1.0

        results: list[dict] = []
        for score, idx in top:
            chunk = corpus["chunks"][idx]
            normalized = score / max_score if max_score else 0.0
            payload = {
                "text": chunk.get("text", ""),
                "url": chunk.get("url", ""),
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index", idx),
                "language": chunk.get("language", "english"),
                "chunk_id": chunk.get("chunk_id", idx),
            }
            results.append(
                {
                    "id": str(payload["chunk_id"]),
                    "score": score,
                    "payload": payload,
                    "rrf_score": normalized,
                    "rerank_score": normalized,
                    "above_threshold": normalized >= CONFIDENCE_THRESHOLD,
                }
            )

        return results

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        """
        Perform BM25 sparse keyword search in Qdrant.

        Builds a sparse query vector by tokenising the query and matching
        against the indexed BM25 vocabulary that was persisted during indexing.
        Uses the same token→index mapping as the indexer so scores align.

        Args:
            query: Raw query string (tokenised internally).
            top_k: Number of results to return.

        Returns:
            List of result dicts with id, score, and payload.
        """
        from qdrant_client.models import SparseVector
        import math
        from collections import Counter

        vocab = self._get_vocab()
        if not vocab:
            return []  # vocab not built yet, skip sparse search

        client = self._get_qdrant_client()

        # Build sparse query vector using the same vocab as the indexer
        tokens = query.lower().split()
        token_counts = Counter(tokens)
        total = len(tokens)

        indices = []
        values = []
        for token, count in token_counts.items():
            if token not in vocab:
                continue  # out-of-vocabulary token — skip
            token_idx = vocab[token]
            tf = math.log(1 + count / max(total, 1))
            indices.append(token_idx)
            values.append(float(tf))

        if not indices:
            return []

        try:
            from qdrant_client.models import NamedSparseVector
            sparse_query = NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=indices, values=values),
            )
            if hasattr(client, "search"):
                results = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=sparse_query,
                    limit=top_k,
                    with_payload=True,
                )
            else:
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=sparse_query,
                    limit=top_k,
                    with_payload=True,
                )
            return _normalize_search_results(results)
        except Exception as exc:
            logger.error(f"❌ Sparse search failed: {exc}")
            return []

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Re-rank candidate passages using a CrossEncoder for fine-grained scoring.

        CrossEncoder reads query + passage jointly to compute a relevance score,
        producing more accurate ranking than bi-encoder similarity alone.

        Args:
            query: Original user query.
            candidates: List of candidate result dicts (from RRF fusion).

        Returns:
            Re-ranked list sorted by descending CrossEncoder score,
            each result augmented with "rerank_score".
        """
        if not candidates:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            logger.warning("⚠️  Re-ranker unavailable. Using RRF order.")
            return candidates

        # Build (query, passage) pairs for the CrossEncoder
        pairs = [(query, c["payload"].get("text", "")) for c in candidates]

        try:
            scores = reranker.predict(pairs)
        except Exception as exc:
            logger.error(f"❌ Re-ranker failed: {exc}. Using RRF scores.")
            # Fall back to RRF order without re-ranking
            return candidates

        # Attach CrossEncoder scores and sort descending
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        rerank_top_k: int = RERANK_TOP_K,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline: embed → search → fuse → rerank.

        Returns up to `rerank_top_k` results. Each result includes:
          - text: the chunk text
          - url: source URL
          - title: page title
          - language: detected language
          - rerank_score: CrossEncoder confidence score
          - above_threshold: True if score ≥ CONFIDENCE_THRESHOLD

        Args:
            query: User's question in any supported language.
            top_k: Number of candidates to retrieve from each search type.
            rerank_top_k: Final number of results after re-ranking.

        Returns:
            List of result dicts, sorted by relevance.
        """
        logger.info(f"🔍 Retrieving for: {query[:80]}...")

        # Step 1: Embed query
        query_vector = self._embed_query(query)

        # Step 2: Dense + sparse search in parallel
        dense_results = self._dense_search(query_vector, top_k)
        sparse_results = self._sparse_search(query, top_k)

        logger.info(
            f"   Dense: {len(dense_results)} results, "
            f"Sparse: {len(sparse_results)} results"
        )

        # Step 3: Fuse with Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        logger.info(f"   RRF fusion: {len(fused)} unique candidates")

        if not fused:
            local_results = self._local_search(query, top_k=max(top_k, rerank_top_k))
            if local_results:
                logger.info(
                    "✅ Returning %d local corpus results (offline fallback)",
                    len(local_results[:rerank_top_k]),
                )
                return local_results[:rerank_top_k]
            logger.warning("⚠️  No retrieval results from online or local corpus paths.")
            return []

        # Step 3b: Deduplicate by canonical URL (removes http/https duplicates)
        seen_urls: set[str] = set()
        deduped = []
        for result in fused:
            url = result.get("payload", {}).get("url", "")
            canonical = url.replace("http://", "https://").rstrip("/")
            if canonical not in seen_urls:
                seen_urls.add(canonical)
                deduped.append(result)
        fused = deduped

        # Step 4: Re-rank top candidates with CrossEncoder
        top_candidates = fused[:rerank_top_k * 2]  # give reranker 2× candidates
        reranked = self._rerank(query, top_candidates)

        # Step 5: Return top rerank_top_k with confidence flags
        final_results = reranked[:rerank_top_k]
        for r in final_results:
            r["above_threshold"] = r.get("rerank_score", 0.0) >= CONFIDENCE_THRESHOLD

        logger.info(
            f"✅ Returning {len(final_results)} results. "
            f"Above threshold: {sum(1 for r in final_results if r['above_threshold'])}"
        )
        return final_results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    retriever = UCBRetriever()
    test_queries = [
        "What are UCB Bank's loan products?",
        "UCB ব্যাংকের সুদের হার কত?",
        "UCB er loan nite ki ki lagbe?",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = retriever.retrieve(q, top_k=5, rerank_top_k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  [{i}] score={r.get('rerank_score', 0):.3f} | "
                f"{r['payload'].get('url', '')} | "
                f"{r['payload'].get('text', '')[:80]}..."
            )
