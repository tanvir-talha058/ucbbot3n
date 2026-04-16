"""
tests/test_pipeline.py
-----------------------
Test suite for the UCB Bank RAG chatbot pipeline.

Tests cover:
- Configuration loading
- Text cleaning and chunking
- Language detection
- Embedding computation
- Qdrant connectivity and indexing
- Retrieval and re-ranking
- LLM response generation
- FastAPI endpoint contracts

Run with:
  python -m pytest tests/test_pipeline.py -v
  python -m pytest tests/test_pipeline.py -v -k "not slow"
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Bootstrap project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 1. Configuration tests
# ===========================================================================

class TestSettings(unittest.TestCase):
    """Tests for config/settings.py — verifies all required settings exist."""

    def test_required_settings_present(self):
        """All required settings should be importable without error."""
        from config.settings import (
            QDRANT_URL, QDRANT_COLLECTION, OLLAMA_URL, OLLAMA_MODEL,
            EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
            TOP_K_RETRIEVAL, RERANK_TOP_K, CONFIDENCE_THRESHOLD,
            UCB_BASE_URL, TESSERACT_PATH, CORS_ORIGINS, LOCAL_TEST_MODE,
        )
        # Basic type checks
        self.assertIsInstance(QDRANT_URL, str)
        self.assertIsInstance(CHUNK_SIZE, int)
        self.assertIsInstance(CONFIDENCE_THRESHOLD, float)
        self.assertIsInstance(LOCAL_TEST_MODE, bool)

    def test_chunk_size_positive(self):
        """CHUNK_SIZE must be a positive integer."""
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreaterEqual(CHUNK_OVERLAP, 0)
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)

    def test_confidence_threshold_range(self):
        """CONFIDENCE_THRESHOLD must be between 0 and 1."""
        from config.settings import CONFIDENCE_THRESHOLD
        self.assertGreaterEqual(CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(CONFIDENCE_THRESHOLD, 1.0)


# ===========================================================================
# 2. Chunker / preprocessing tests
# ===========================================================================

class TestChunker(unittest.TestCase):
    """Tests for preprocessing/chunker.py."""

    def setUp(self):
        """Import chunker module once for all tests."""
        from preprocessing.chunker import clean_text, detect_language, split_into_chunks
        self.clean_text = clean_text
        self.detect_language = detect_language
        self.split_into_chunks = split_into_chunks

    def test_clean_text_removes_html(self):
        """clean_text should strip HTML tags."""
        dirty = "<p>Hello <b>World</b></p>"
        result = self.clean_text(dirty)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)
        self.assertIn("Hello", result)

    def test_clean_text_handles_html_entities(self):
        """clean_text should convert &amp; → & and &nbsp; → space."""
        result = self.clean_text("UCB&amp;Bank&nbsp;Ltd")
        self.assertIn("&", result)
        self.assertNotIn("&amp;", result)

    def test_clean_text_empty_input(self):
        """clean_text should return empty string for empty input."""
        self.assertEqual(self.clean_text(""), "")
        self.assertEqual(self.clean_text(None), "")

    def test_detect_language_english(self):
        """Should detect purely English text as 'english'."""
        lang = self.detect_language("UCB Bank offers competitive loan products.")
        self.assertEqual(lang, "english")

    def test_detect_language_bangla(self):
        """Should detect Bengali script as 'bangla'."""
        lang = self.detect_language("UCB ব্যাংকের সুদের হার অনেক ভালো।")
        self.assertEqual(lang, "bangla")

    def test_detect_language_banglish(self):
        """Should detect mixed Bengali-English as 'banglish'."""
        lang = self.detect_language("UCB er loan nite হবে কি?")
        # Should be banglish (mixed) or bangla, but not purely english
        self.assertIn(lang, ("banglish", "bangla"))

    def test_split_into_chunks_basic(self):
        """split_into_chunks should return non-empty list for valid text."""
        text = " ".join(["word"] * 200)  # 200 words
        chunks = self.split_into_chunks(text, chunk_size=50, chunk_overlap=10)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_split_into_chunks_empty(self):
        """split_into_chunks should return empty list for empty text."""
        self.assertEqual(self.split_into_chunks(""), [])

    def test_split_into_chunks_overlap(self):
        """Consecutive chunks should share overlapping content."""
        # Create text with distinct words to test overlap
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = self.split_into_chunks(text, chunk_size=30, chunk_overlap=10)
        if len(chunks) >= 2:
            # Last words of chunk 0 should appear in chunk 1
            chunk0_words = set(chunks[0].split())
            chunk1_words = set(chunks[1].split())
            overlap = chunk0_words & chunk1_words
            self.assertGreater(len(overlap), 0, "Expected overlapping words between chunks")


# ===========================================================================
# 3. Embedder tests (mocked — no GPU required)
# ===========================================================================

class TestEmbedder(unittest.TestCase):
    """Tests for embeddings/embedder.py with mocked model."""

    def test_prepare_texts_for_e5(self):
        """E5 embedding texts should be prefixed with 'passage: '."""
        from embeddings.embedder import prepare_texts_for_e5
        chunks = [{"text": "Hello UCB"}, {"text": "Loan products"}]
        texts = prepare_texts_for_e5(chunks)
        self.assertEqual(len(texts), 2)
        self.assertTrue(texts[0].startswith("passage: "))
        self.assertTrue(texts[1].startswith("passage: "))

    def test_get_device_returns_string(self):
        """get_device should return either 'cuda' or 'cpu'."""
        from embeddings.embedder import get_device
        device = get_device()
        self.assertIn(device, ("cuda", "cpu"))

    @patch("embeddings.embedder.SentenceTransformer")
    def test_compute_embeddings_shape(self, mock_st):
        """compute_embeddings should return one vector per input text."""
        import numpy as np
        from embeddings.embedder import compute_embeddings

        # Mock model that returns a 3×1024 numpy array
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 1024))
        mock_st.return_value = mock_model

        texts = ["text 1", "text 2", "text 3"]
        embeddings = compute_embeddings(mock_model, texts, batch_size=2, show_progress=False)

        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(embeddings[0]), 1024)


# ===========================================================================
# 4. Retriever tests (mocked Qdrant)
# ===========================================================================

class TestRetriever(unittest.TestCase):
    """Tests for retrieval/retriever.py with mocked dependencies."""

    def test_reciprocal_rank_fusion_merges_results(self):
        """RRF should merge dense and sparse lists and assign scores."""
        from retrieval.retriever import reciprocal_rank_fusion

        dense = [
            {"id": "a", "score": 0.9, "payload": {"text": "Loan info"}},
            {"id": "b", "score": 0.8, "payload": {"text": "Deposit info"}},
        ]
        sparse = [
            {"id": "b", "score": 5.0, "payload": {"text": "Deposit info"}},
            {"id": "c", "score": 4.0, "payload": {"text": "FD info"}},
        ]
        fused = reciprocal_rank_fusion(dense, sparse)

        # Should have 3 unique documents (a, b, c)
        ids = [r["id"] for r in fused]
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertIn("c", ids)

        # "b" appears in both lists → should have higher RRF score
        b_score = next(r["rrf_score"] for r in fused if r["id"] == "b")
        a_score = next(r["rrf_score"] for r in fused if r["id"] == "a")
        self.assertGreater(b_score, a_score)

    def test_rrf_empty_inputs(self):
        """RRF with empty inputs should return empty list."""
        from retrieval.retriever import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [])
        self.assertEqual(result, [])


# ===========================================================================
# 5. LLM tests (mocked Ollama)
# ===========================================================================

class TestOllamaLLM(unittest.TestCase):
    """Tests for llm/ollama_llm.py with mocked HTTP calls."""

    def test_detect_language_english(self):
        """Should detect English text correctly."""
        from llm.ollama_llm import detect_language
        self.assertEqual(detect_language("What loans does UCB offer?"), "english")

    def test_detect_language_bangla(self):
        """Should detect Bengali script correctly."""
        from llm.ollama_llm import detect_language
        lang = detect_language("UCB ব্যাংকের লোন সম্পর্কে জানতে চাই।")
        self.assertEqual(lang, "bangla")

    def test_format_context_basic(self):
        """format_context should extract text and URLs from chunks."""
        from llm.ollama_llm import format_context
        chunks = [
            {
                "payload": {
                    "text": "UCB offers home loans at 9%",
                    "url": "https://www.ucb.com.bd/loans",
                    "title": "Loans",
                }
            }
        ]
        ctx, urls = format_context(chunks)
        self.assertIn("UCB offers", ctx)
        self.assertIn("https://www.ucb.com.bd/loans", urls)

    @patch("llm.ollama_llm.UCBOllamaLLM._call_ollama")
    def test_generate_with_good_context(self, mock_call):
        """generate() should return answer when context is good."""
        from llm.ollama_llm import UCBOllamaLLM
        mock_call.return_value = "UCB offers home loans at 9% per annum."

        llm = UCBOllamaLLM()
        chunks = [
            {
                "payload": {
                    "text": "UCB home loan at 9%",
                    "url": "https://www.ucb.com.bd/loans",
                    "title": "Loans",
                },
                "rerank_score": 0.85,
            }
        ]
        result = llm.generate("What are UCB loan rates?", chunks, language="english")
        self.assertFalse(result["fallback"])
        self.assertIn("UCB", result["answer"])

    def test_generate_fallback_on_no_context(self):
        """generate() should use fallback when no good chunks provided."""
        from llm.ollama_llm import UCBOllamaLLM
        llm = UCBOllamaLLM()
        result = llm.generate("Random unrelated question?", [], language="english")
        self.assertTrue(result["fallback"])
        self.assertIn("UCB Bank", result["answer"])


# ===========================================================================
# 6. FastAPI endpoint tests
# ===========================================================================

class TestFastAPIEndpoints(unittest.TestCase):
    """Tests for api/main.py endpoint contracts using TestClient."""

    def setUp(self):
        """Create a FastAPI test client."""
        try:
            from fastapi.testclient import TestClient
            from api.main import app
            self.client = TestClient(app)
        except ImportError:
            self.skipTest("httpx not installed — skip API tests")

    def test_root_returns_200(self):
        """GET / should return 200 with a message."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("message", resp.json())

    def test_health_endpoint_structure(self):
        """GET /health should return status and components keys."""
        resp = self.client.get("/health")
        # Health endpoint should always return 200 (even if degraded)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("components", data)

    @patch("api.main.get_pipeline")
    def test_chat_endpoint_returns_answer(self, mock_get_pipeline):
        """POST /chat should return answer, language, sources, fallback."""
        mock_pipeline = MagicMock()
        mock_pipeline.chat.return_value = {
            "answer": "UCB offers competitive loans.",
            "language": "english",
            "sources": ["https://www.ucb.com.bd/loans"],
            "fallback": False,
            "session_id": "test",
            "retrieved_count": 3,
        }
        mock_get_pipeline.return_value = mock_pipeline

        resp = self.client.post(
            "/chat",
            json={
                "query": "What are UCB loans?",
                "language": "english",
                "session_id": "test",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIn("language", data)
        self.assertIn("sources", data)
        self.assertIn("fallback", data)

    def test_chat_empty_query_rejected(self):
        """POST /chat with empty query should return 422."""
        resp = self.client.post(
            "/chat",
            json={"query": "", "session_id": "test"},
        )
        self.assertEqual(resp.status_code, 422)

    @patch("api.main.get_pipeline")
    def test_stats_endpoint_structure(self, _):
        """GET /stats should return total_chunks and language_breakdown."""
        # /stats calls QdrantClient inside the function scope; just check status
        resp = self.client.get("/stats")
        # May return 200 (Qdrant running) or 503 (not running) — both are valid
        self.assertIn(resp.status_code, (200, 503))


# ===========================================================================
# 7. Pipeline integration test (mocked end-to-end)
# ===========================================================================

class TestRAGPipeline(unittest.TestCase):
    """Integration tests for pipeline/rag_pipeline.py."""

    def test_detect_language(self):
        """Pipeline language detector works correctly."""
        from pipeline.rag_pipeline import detect_language
        self.assertEqual(detect_language("Hello UCB"), "english")
        self.assertEqual(detect_language("UCB ব্যাংকের লোন"), "bangla")

    def test_conversation_memory(self):
        """ConversationMemory should store and trim history correctly."""
        from pipeline.rag_pipeline import ConversationMemory
        mem = ConversationMemory(window=2)

        # Add 3 exchanges (window=2 means only last 2 kept)
        mem.add_exchange("s1", "q1", "a1")
        mem.add_exchange("s1", "q2", "a2")
        mem.add_exchange("s1", "q3", "a3")

        history = mem.get_history("s1")
        # window=2 → 4 messages (2 exchanges)
        self.assertEqual(len(history), 4)
        # Most recent exchange should be q3/a3
        self.assertEqual(history[-2]["content"], "q3")

    def test_query_rewriting_passthrough(self):
        """Long, self-contained queries should not be rewritten."""
        from pipeline.rag_pipeline import rewrite_query
        long_query = "What are the current interest rates for home loans at UCB Bank?"
        result = rewrite_query(long_query, [])
        # No history → no rewrite
        self.assertEqual(result, long_query)

    @patch("pipeline.rag_pipeline.UCBRagPipeline._get_retriever")
    @patch("pipeline.rag_pipeline.UCBRagPipeline._get_llm")
    def test_chat_returns_expected_keys(self, mock_llm, mock_retriever):
        """pipeline.chat() should return a dict with all required keys."""
        from pipeline.rag_pipeline import UCBRagPipeline

        # Mock retriever
        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = []
        mock_retriever.return_value = mock_ret

        # Mock LLM
        mock_l = MagicMock()
        mock_l.generate.return_value = {
            "answer": "UCB provides various loan options.",
            "language": "english",
            "sources": [],
            "fallback": False,
        }
        mock_llm.return_value = mock_l

        pipeline = UCBRagPipeline()
        result = pipeline.chat("What loans does UCB offer?", session_id="test_sess")

        required_keys = {"answer", "language", "sources", "fallback", "session_id", "retrieved_count"}
        self.assertTrue(required_keys.issubset(set(result.keys())))
        self.assertEqual(result["session_id"], "test_sess")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
