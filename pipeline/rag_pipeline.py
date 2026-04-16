"""
pipeline/rag_pipeline.py
------------------------
LangChain-orchestrated RAG pipeline for UCB Bank chatbot.

Full pipeline flow:
  1. Receive user query + session_id
  2. Detect language (English / Bangla / Banglish)
  3. Rewrite query using conversation history for follow-up questions
  4. Embed query and run hybrid retrieval
  5. Re-rank retrieved chunks with CrossEncoder
  6. Generate answer with Ollama / Qwen2.5
  7. Store exchange in session memory
  8. Return structured response

Usage:
  from pipeline.rag_pipeline import UCBRagPipeline
  pipeline = UCBRagPipeline()
  response = pipeline.chat("What are UCB loan products?", session_id="abc123")
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MEMORY_WINDOW,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROCESSED_DATA_PATH,
    RERANK_TOP_K,
    TOP_K_RETRIEVAL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.rag_pipeline")


def detect_language(text: str) -> str:
    """
    Detect the language of a query string.

    Checks Unicode character ranges for Bengali script first, then uses
    langdetect for Latin-script text disambiguation.

    Args:
        text: Input query string.

    Returns:
        Language string: "english", "bangla", or "banglish".
    """
    if not text:
        return "english"

    # Count Bengali Unicode characters (U+0980 to U+09FF)
    bangla_chars = sum(1 for ch in text if 0x0980 <= ord(ch) <= 0x09FF)
    ratio = bangla_chars / len(text)

    if ratio > 0.30:
        return "bangla"
    if ratio > 0.05:
        return "banglish"

    # Detect Banglish using a small set of Bengali grammatical markers that
    # never appear in English text: verb forms (-te/-bo/-un), Bengali question
    # words, and common pronouns written in Roman script.
    _BANGLISH_MARKERS = {
        "korte", "korun", "korbo", "korben",           # do/doing
        "nite", "nibo", "niben",                        # take
        "dite", "dibo", "diben",                        # give
        "khulte", "khulbo",                             # open
        "bolte", "bolun", "bolbo",                      # say/tell
        "hobe", "hoye", "hobena",                       # will be
        "lagbe", "lage", "lagena",                      # need
        "nai", "nei", "achhe",                          # not / exists
        "apni", "amra", "tumi",                         # you / we / you (informal)
        "keno", "kothay", "kokhon", "kivabe",           # why/where/when/how
        "ki",                                            # what (standalone)
    }
    if set(text.lower().split()) & _BANGLISH_MARKERS:
        return "banglish"

    try:
        from langdetect import detect
        code = detect(text)
        if code == "bn":
            return "bangla" if bangla_chars > 0 else "banglish"
        return "english"
    except Exception:
        return "english"


class ConversationMemory:
    """
    Simple in-memory conversation history manager.

    Maintains a sliding window of the last MEMORY_WINDOW exchanges per session.
    Each exchange is a dict with "role" ("user" or "assistant") and "content".

    Attributes:
        window: Maximum number of exchanges to retain per session.
        _store: Dict mapping session_id → list of message dicts.
    """

    def __init__(self, window: int = MEMORY_WINDOW) -> None:
        """
        Initialise the memory manager.

        Args:
            window: Maximum number of past exchanges to retain.
        """
        self.window = window
        # defaultdict auto-creates an empty list for new sessions
        self._store: dict[str, list[dict]] = defaultdict(list)

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """
        Record a user-assistant exchange in the session history.

        Trims history to the last `window` exchanges after adding.

        Args:
            session_id: Unique identifier for the conversation session.
            user_msg: User's message text.
            assistant_msg: Assistant's generated response text.
        """
        history = self._store[session_id]
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        # Keep only the last window * 2 messages (window exchanges)
        self._store[session_id] = history[-(self.window * 2):]

    def get_history(self, session_id: str) -> list[dict]:
        """
        Retrieve the conversation history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        return self._store.get(session_id, [])

    def clear(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session to clear.
        """
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"🗑️  Cleared memory for session: {session_id}")


# Banking-specific terms the spell checker must never "correct"
_BANKING_TERMS = {
    "ucb", "dps", "nrb", "sme", "emi", "atm", "nfcd", "rfcd", "uepfd",
    "taqwa", "ayma", "upay", "unet", "visa", "mastercard", "chequebook",
    "mudaraba", "probashi", "swadhin", "superflex", "prothom", "shomota",
    "youngsters", "pensioner", "millionaire", "maximizer", "banglish",
    "wasa", "brac", "bnp", "offshor", "offshore", "installment",
    "cheque", "cheques", "clearance", "encashment", "prepaid",
}

# Module-level singleton — SpellChecker loads word lists from disk; creating
# it once at startup avoids ~100-200ms I/O cost on every query.
_spell_checker = None


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker()
    return _spell_checker


def correct_typos(query: str) -> str:
    """
    Correct common English spelling mistakes in the query before retrieval.
    Skips correction for Bangla/Banglish text and known banking terms.

    Args:
        query: Raw user query.

    Returns:
        Query with typos corrected where safe to do so.
    """
    # Skip if query has Bangla Unicode characters
    if any(0x0980 <= ord(ch) <= 0x09FF for ch in query):
        return query

    try:
        spell = _get_spell_checker()
        words = query.split()
        corrected = []
        for word in words:
            clean = word.lower().strip("?!.,")
            # Never correct banking-specific terms or short words
            if clean in _BANKING_TERMS or len(clean) <= 3:
                corrected.append(word)
            elif spell.unknown([clean]):
                suggestion = spell.correction(clean)
                if suggestion and suggestion != clean:
                    # Preserve original casing style
                    corrected.append(suggestion)
                    logger.info(f"✏️  Typo corrected: '{word}' → '{suggestion}'")
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        return " ".join(corrected)
    except ImportError:
        return query  # spellchecker not installed, skip silently


def rewrite_query(query: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up query to be self-contained using conversation history.

    For follow-up questions like "What about the interest rate?" the retriever
    needs a complete query like "What is UCB Bank's home loan interest rate?"
    to find the right context.

    Strategy: If the query is short or starts with pronouns/references,
    prepend the last user question to provide context.

    Args:
        query: Current user query (may be a follow-up).
        history: List of previous message dicts for this session.

    Returns:
        Rewritten query string suitable for retrieval.
    """
    # Only rewrite if there's prior history
    if not history:
        return query

    # Short queries or queries starting with reference words likely need context
    follow_up_indicators = [
        "it", "its", "that", "this", "they", "them", "those", "these",
        "what about", "and", "also", "tell me more", "আর", "এটা", "সেটা",
        "er", "tar", "seta", "eta",
    ]

    query_lower = query.lower().strip()
    is_follow_up = (
        # Only treat very short queries (≤3 words) as follow-ups, not 5.
        # "What is DPS?" is 3 words and standalone — must not be rewritten.
        # "It" or "That one?" are clearly follow-ups.
        (len(query.split()) <= 3) or
        any(query_lower.startswith(word) for word in follow_up_indicators)
    )

    if is_follow_up:
        # Find the last user message for context
        last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if last_user_msgs:
            # Prepend context from last question to make query self-contained
            rewritten = f"{last_user_msgs[-1]} {query}"
            logger.info(f"🔄 Query rewritten: '{query}' → '{rewritten[:80]}...'")
            return rewritten

    return query


# Vague queries that need clarification before retrieval
_VAGUE_PATTERNS = [
    "tell me about them", "tell me about those", "what about them",
    "tell me more about that", "explain that", "what is that",
    "what are they", "describe them", "more info", "more information",
    "details please", "give me details", "what do you mean",
    "tell me about accounts", "tell me about account", "info about accounts",
    "about accounts", "about banking", "about loans", "about cards",
]

_VAGUE_CLARIFICATION = {
    "english": "Could you please clarify which specific product or service you'd like to know about? For example, are you asking about loans, cards, savings accounts, DPS, or something else?",
    "bangla": "আপনি কোন নির্দিষ্ট পণ্য বা সেবা সম্পর্কে জানতে চান? যেমন — লোন, কার্ড, সেভিংস অ্যাকাউন্ট, DPS, নাকি অন্য কিছু?",
    "banglish": "Apni ki specific kono product ba service er kotha jante chan? Jemon — loan, card, savings account, DPS, naaki onno kichu?",
}

def _is_vague(query: str) -> bool:
    """Return True if the query is too vague to retrieve meaningful results."""
    q = query.lower().strip()
    return any(pattern in q for pattern in _VAGUE_PATTERNS)


_CASUAL_PATTERNS = {
    "greeting": (
        "hi", "hello", "hey", "howdy", "good morning", "good afternoon",
        "good evening", "salam", "assalamu alaikum", "helo", "হ্যালো", "আসসালামু আলাইকুম"
    ),
    "how_are_you": ("how are you", "how r u", "how are u", "whats up", "what's up", "sup", "কেমন আছেন"),
    "thanks": ("thank you", "thanks", "thank u", "dhonnobad", "shukriya", "ty", "ধন্যবাদ"),
    "bye": ("bye", "goodbye", "see you", "see ya", "cya", "alvida", "বিদায়", "আল্লাহ হাফেজ"),
    "who_are_you": (
        "who are you", "what are you", "are you a bot", "are you ai", "are you human",
        "tell me about yourself", "tell me about urself", "tell me about your self",
        "introduce yourself", "introduce urself", "what can you do", "what do you do", "আপনি কে",
        "tumi ke", "ke tumi", "tumi ki help korte paro", "tumi ki korte paro", "tomar kaj ki"
    ),
}

_CASUAL_REPLIES = {
    "english": {
        "greeting": "Hello! I'm the UCB Bank virtual assistant. How can I help you today?",
        "how_are_you": "I'm doing great, thank you for asking! How can I assist you with UCB Bank services today?",
        "thanks": "You're welcome! Is there anything else I can help you with?",
        "bye": "Goodbye! Have a great day. Feel free to come back if you have any questions about UCB Bank.",
        "who_are_you": "I'm UCB Bank's AI-powered virtual assistant. I can answer questions about UCB Bank's products, services, loans, cards, accounts, and more!",
    },
    "bangla": {
        "greeting": "স্বাগতম! আমি UCB ব্যাংকের ভার্চুয়াল সহকারী। আজ আপনাকে কীভাবে সাহায্য করতে পারি?",
        "how_are_you": "আমি ভালো আছি, ধন্যবাদ। UCB ব্যাংকের কোন সেবা সম্পর্কে জানতে চান?",
        "thanks": "আপনাকে স্বাগতম। আরও কোনো বিষয়ে সাহায্য লাগলে জানাতে পারেন।",
        "bye": "ধন্যবাদ। ভালো থাকবেন। UCB সংক্রান্ত যেকোনো প্রশ্নে আবার আসবেন।",
        "who_are_you": "আমি UCB ব্যাংকের AI-চালিত ভার্চুয়াল সহকারী। UCB-এর পণ্য, সেবা, ঋণ, কার্ড ও অ্যাকাউন্ট সম্পর্কে তথ্য দিতে পারি।",
    },
    "banglish": {
        "greeting": "Assalamu alaikum! Ami UCB Bank er virtual assistant. Ajke apnake kivabe help korte pari?",
        "how_are_you": "Ami bhalo achi, dhonnobad! UCB Bank er kon service niye help lagbe?",
        "thanks": "Welcome! Aro kono information lagle bolben.",
        "bye": "Dhonnobad. Bhalo thakben. UCB related question thakle abar ashben.",
        "who_are_you": "Ami UCB Bank er AI-powered virtual assistant. UCB er products, services, loan, card ar account niye help korte pari.",
    },
}

def _casual_reply(query: str, language: str) -> str | None:
    """Return a canned reply for casual/greeting messages, or None for banking queries."""
    q = query.lower().strip().rstrip("!?.،")
    replies = _CASUAL_REPLIES.get(language, _CASUAL_REPLIES["english"])

    # Exact match first
    for intent, triggers in _CASUAL_PATTERNS.items():
        if q in triggers:
            return replies[intent]

    # Fuzzy: check if any trigger phrase appears as a dominant part of the query.
    # Only match if the trigger phrase accounts for >50% of the query length —
    # prevents "thanks" from hijacking mid-conversation messages like
    # "thanks for the info, what about the interest rate?"
    for intent, triggers in _CASUAL_PATTERNS.items():
        for trigger in triggers:
            if len(trigger) >= 6 and trigger in q and len(trigger) / max(len(q), 1) > 0.5:
                return replies[intent]

    return None


class UCBRagPipeline:
    """
    Main RAG pipeline orchestrating retrieval and generation for UCB Bank.

    Integrates:
    - UCBRetriever: Hybrid dense + BM25 search with CrossEncoder reranking
    - UCBOllamaLLM: Qwen2.5 LLM with multilingual prompt templates
    - ConversationMemory: Per-session conversation history

    The pipeline is designed to be imported by the FastAPI server and
    called once per chat request.
    """

    def __init__(self) -> None:
        """
        Initialise the pipeline with lazy-loaded components.

        Components are loaded on first use to avoid blocking server startup.
        """
        self._retriever = None    # UCBRetriever, loaded on first use
        self._llm = None          # UCBOllamaLLM, loaded on first use
        self._memory = ConversationMemory(window=MEMORY_WINDOW)
        logger.info("UCBRagPipeline initialised (components load on first query)")

    def _get_retriever(self):
        """
        Lazily initialise and return the retriever.

        Returns:
            UCBRetriever instance.
        """
        if self._retriever is None:
            from retrieval.retriever import UCBRetriever
            self._retriever = UCBRetriever()
            logger.info("✅ Retriever initialised")
        return self._retriever

    def _get_llm(self):
        """
        Lazily initialise and return the LLM client.

        Returns:
            UCBOllamaLLM instance.
        """
        if self._llm is None:
            from llm.ollama_llm import UCBOllamaLLM
            self._llm = UCBOllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
            logger.info("✅ LLM initialised")
        return self._llm

    def chat(
        self,
        query: str,
        session_id: str = "default",
        language: Optional[str] = None,
    ) -> dict:
        """
        Process a user query through the full RAG pipeline.

        Steps:
          1. Detect language (or use override)
          2. Load conversation history
          3. Rewrite query for self-containedness
          4. Retrieve relevant chunks (hybrid search + reranking)
          5. Generate grounded answer with Ollama
          6. Save exchange to memory
          7. Return structured response

        Args:
            query: User's question in any supported language.
            session_id: Unique session identifier for conversation memory.
            language: Language override ("english"/"bangla"/"banglish").
                      Auto-detected if None.

        Returns:
            Dict with keys:
              - answer: Generated text response
              - language: Language used
              - sources: List of cited source URLs
              - fallback: True if no good context was found
              - session_id: Echo of session_id
              - retrieved_count: Number of chunks retrieved
        """
        logger.info(f"💬 Chat | session={session_id} | query={query[:60]}")

        try:
            # Step 0: Language detection first so quick replies respect UI/user language
            detected_lang = language or detect_language(query)

            # Step 0a: Handle casual / greeting messages without RAG
            casual_reply = _casual_reply(query, detected_lang)
            if casual_reply:
                self._memory.add_exchange(session_id, query, casual_reply)
                return {
                    "answer": casual_reply,
                    "language": detected_lang,
                    "sources": [],
                    "fallback": False,
                    "session_id": session_id,
                    "retrieved_count": 0,
                }

            # Step 0b: Detect vague queries and ask for clarification
            if _is_vague(query):
                clarification = _VAGUE_CLARIFICATION.get(detected_lang, _VAGUE_CLARIFICATION["english"])
                self._memory.add_exchange(session_id, query, clarification)
                return {
                    "answer": clarification,
                    "language": detected_lang,
                    "sources": [],
                    "fallback": False,
                    "session_id": session_id,
                    "retrieved_count": 0,
                }
            logger.info(f"🌐 Detected language: {detected_lang}")

            # Step 2: Load conversation history
            history = self._memory.get_history(session_id)

            # Step 3: Correct typos, then rewrite follow-up queries
            corrected_query = correct_typos(query)
            effective_query = rewrite_query(corrected_query, history)

            # Step 4: Retrieve relevant chunks
            retriever = self._get_retriever()
            retrieved = retriever.retrieve(
                effective_query,
                top_k=TOP_K_RETRIEVAL,
                rerank_top_k=RERANK_TOP_K,
            )

            # Step 5: Generate answer
            llm = self._get_llm()
            result = llm.generate(
                question=query,          # use original query for natural answer
                retrieved_chunks=retrieved,
                language=detected_lang,
            )

            # Step 6: Save to memory
            self._memory.add_exchange(session_id, query, result["answer"])

            # Step 7: Deduplicate sources (normalise http → https)
            seen, unique_sources = set(), []
            for url in result["sources"]:
                canonical = url.replace("http://", "https://").rstrip("/")
                if canonical not in seen:
                    seen.add(canonical)
                    unique_sources.append(canonical)

            # Step 8: Build and return response
            response = {
                "answer": result["answer"],
                "language": result["language"],
                "sources": unique_sources,
                "fallback": result["fallback"],
                "session_id": session_id,
                "retrieved_count": len(retrieved),
            }
            logger.info(
                f"✅ Response ready | lang={detected_lang} | "
                f"sources={len(result['sources'])} | fallback={result['fallback']}"
            )
            return response

        except Exception as exc:
            logger.error(f"❌ Pipeline error: {exc}", exc_info=True)
            # Return a safe error response rather than crashing the API
            return {
                "answer": (
                    "I encountered an error processing your request. "
                    "Please try again or contact UCB Bank support."
                ),
                "language": language or "english",
                "sources": [],
                "fallback": True,
                "session_id": session_id,
                "retrieved_count": 0,
                "error": str(exc),
            }

    def health_check(self) -> dict:
        """
        Check the health of all pipeline components.

        Returns:
            Dict with health status of retriever, LLM, and memory.
        """
        status = {"pipeline": "ok", "components": {}}

        # Check retrieval availability. Prefer Qdrant, but fall back to the
        # local cached corpus when the vector store is unavailable.
        try:
            from qdrant_client import QdrantClient
            from config.settings import QDRANT_URL, QDRANT_COLLECTION
            client = QdrantClient(url=QDRANT_URL, timeout=5)
            info = client.get_collection(QDRANT_COLLECTION)
            status["components"]["retrieval"] = {
                "status": "qdrant",
                "points": info.points_count,
            }
        except Exception as e:
            if PROCESSED_DATA_PATH.exists():
                try:
                    import json
                    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    status["components"]["retrieval"] = {
                        "status": "local",
                        "chunks": len(chunks),
                        "path": str(PROCESSED_DATA_PATH),
                    }
                except Exception as local_exc:
                    status["components"]["retrieval"] = {
                        "status": "error",
                        "detail": str(local_exc),
                    }
                    status["pipeline"] = "degraded"
            else:
                status["components"]["retrieval"] = {"status": "error", "detail": str(e)}
                status["pipeline"] = "degraded"

        # Check Ollama
        try:
            llm = self._get_llm()
            ollama_ok = llm.health_check()
            status["components"]["ollama"] = {
                "status": "ok" if ollama_ok else "model_missing",
                "model": OLLAMA_MODEL,
            }
            if not ollama_ok:
                status["pipeline"] = "degraded"
        except Exception as e:
            status["components"]["ollama"] = {"status": "error", "detail": str(e)}
            status["pipeline"] = "degraded"

        return status


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = UCBRagPipeline()

    test_queries = [
        ("What are UCB Bank's loan products?", "english"),
        ("UCB ব্যাংকের সুদের হার কত?", "bangla"),
        ("UCB er loan nite ki ki lagbe?", "banglish"),
    ]

    for query, lang in test_queries:
        print(f"\n{'='*60}\nQuery: {query}")
        result = pipeline.chat(query, session_id="test_session", language=lang)
        print(f"Answer: {result['answer'][:200]}")
        print(f"Sources: {result['sources']}")
        print(f"Fallback: {result['fallback']}")
