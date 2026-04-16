"""
monitoring/phoenix_monitor.py
------------------------------
Arize Phoenix integration for local LLM observability and tracing.

Provides:
- Phoenix session setup (local, free, no cloud account needed)
- LangChain callback handler for automatic trace capture
- Span logging for retrieval and generation steps
- Helper to log individual query traces manually

Phoenix Dashboard: http://localhost:6006 (after docker-compose up -d)

Usage:
  from monitoring.phoenix_monitor import setup_phoenix, get_tracer
  setup_phoenix()  # call once at startup
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.phoenix")

# Phoenix endpoint (matches docker-compose.yml service)
PHOENIX_ENDPOINT = "http://localhost:6006"


def setup_phoenix() -> bool:
    """
    Initialise Arize Phoenix tracing for the RAG pipeline.

    Configures the OpenTelemetry tracer to send spans to the local
    Phoenix Docker instance. Should be called once at application startup.

    Returns:
        True if setup succeeded, False if Phoenix is unavailable.
    """
    try:
        import phoenix as px
        from phoenix.otel import register

        logger.info(f"🔭 Setting up Arize Phoenix at {PHOENIX_ENDPOINT}...")

        # Register the Phoenix tracer provider
        # This configures OpenTelemetry to send spans to Phoenix
        tracer_provider = register(
            project_name="ucb-bank-rag",
            endpoint=f"{PHOENIX_ENDPOINT}/v1/traces",
        )

        logger.info("✅ Arize Phoenix tracing enabled. Dashboard: http://localhost:6006")
        return True

    except ImportError:
        logger.warning(
            "⚠️  arize-phoenix not installed. "
            "Install with: pip install arize-phoenix. "
            "Monitoring will be disabled."
        )
        return False
    except Exception as exc:
        logger.warning(
            f"⚠️  Phoenix setup failed: {exc}. "
            "Is Phoenix running? Try: docker-compose up -d"
        )
        return False


def get_langchain_callback():
    """
    Return a Phoenix LangChain callback handler for automatic tracing.

    When passed to LangChain chains/runnables, this handler automatically
    captures input/output, latency, token counts, and intermediate steps.

    Returns:
        Phoenix LangChain callback handler, or None if unavailable.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        # Instrument LangChain globally (captures all LangChain calls)
        LangChainInstrumentor().instrument()
        logger.info("✅ LangChain Phoenix instrumentation enabled")
        return None  # Instrumentation is global, no explicit callback needed
    except ImportError:
        logger.warning("⚠️  openinference-instrumentation-langchain not installed.")
        return None
    except Exception as exc:
        logger.warning(f"⚠️  LangChain instrumentation failed: {exc}")
        return None


def log_rag_trace(
    query: str,
    language: str,
    retrieved_chunks: list[dict],
    answer: str,
    fallback: bool,
    latency_ms: float,
    session_id: str = "default",
) -> None:
    """
    Manually log a RAG query trace to Phoenix using OpenTelemetry spans.

    This provides visibility into each step of the RAG pipeline:
    query, retrieval, generation, and final response.

    Args:
        query: User query string.
        language: Detected language.
        retrieved_chunks: List of retrieved chunk dicts.
        answer: Generated answer text.
        fallback: Whether fallback was used.
        latency_ms: Total pipeline latency in milliseconds.
        session_id: Conversation session identifier.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        tracer = trace.get_tracer("ucb.rag_pipeline")

        # Create a root span for the full RAG query
        with tracer.start_as_current_span(
            "rag_query",
            kind=SpanKind.SERVER,
        ) as span:
            # Set span attributes for Phoenix UI
            span.set_attribute("query", query)
            span.set_attribute("language", language)
            span.set_attribute("session_id", session_id)
            span.set_attribute("fallback", fallback)
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("retrieved_count", len(retrieved_chunks))
            span.set_attribute("answer_length", len(answer))

            # Top retrieved chunk info
            if retrieved_chunks:
                top = retrieved_chunks[0]
                payload = top.get("payload", {})
                span.set_attribute("top_chunk_url", payload.get("url", ""))
                span.set_attribute(
                    "top_chunk_score",
                    top.get("rerank_score", top.get("rrf_score", 0.0))
                )

        logger.debug(f"📊 Trace logged: session={session_id}, latency={latency_ms:.0f}ms")

    except Exception as exc:
        # Monitoring should never crash the main application
        logger.debug(f"Phoenix trace logging skipped: {exc}")


def check_phoenix_health() -> bool:
    """
    Check if the Phoenix Docker service is reachable.

    Returns:
        True if Phoenix is running and responding, False otherwise.
    """
    try:
        import httpx
        resp = httpx.get(f"{PHOENIX_ENDPOINT}/healthz", timeout=3)
        is_healthy = resp.status_code == 200
        if is_healthy:
            logger.info(f"✅ Phoenix is healthy at {PHOENIX_ENDPOINT}")
        else:
            logger.warning(f"⚠️  Phoenix returned {resp.status_code}")
        return is_healthy
    except Exception as exc:
        logger.warning(f"⚠️  Phoenix not reachable at {PHOENIX_ENDPOINT}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Entry point — run directly to test Phoenix connectivity
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("UCB Bank Phoenix Monitor — Testing Connection")
    logger.info(f"Phoenix URL: {PHOENIX_ENDPOINT}")
    logger.info("=" * 60)

    # Check health first
    if not check_phoenix_health():
        logger.error(
            "❌ Phoenix is not running. Start it with:\n"
            "   docker-compose up -d\n"
            "   Then visit http://localhost:6006"
        )
        sys.exit(1)

    # Test setup
    ok = setup_phoenix()
    if ok:
        logger.info("🎉 Phoenix monitoring is fully operational!")
        logger.info(f"   Dashboard: {PHOENIX_ENDPOINT}")

        # Log a test trace
        log_rag_trace(
            query="Test query for Phoenix",
            language="english",
            retrieved_chunks=[],
            answer="Test answer",
            fallback=True,
            latency_ms=100.0,
            session_id="monitor_test",
        )
        logger.info("✅ Test trace logged. Check the Phoenix dashboard.")
    else:
        logger.error("❌ Phoenix setup failed. Check the logs above.")
        sys.exit(1)
