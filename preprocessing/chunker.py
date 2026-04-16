"""
preprocessing/chunker.py
------------------------
Text cleaning, language detection, and semantic chunking pipeline.

Reads raw scraped JSON from data/raw/ucb_raw.json, cleans each page's text,
detects the language (English / Bangla / Banglish), splits into overlapping
chunks, and saves the result to data/processed/ucb_chunks.json.

Usage:
  python preprocessing/chunker.py
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap project root for imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.chunker")

# ---------------------------------------------------------------------------
# Unicode ranges for Bangla script detection
# ---------------------------------------------------------------------------
BANGLA_UNICODE_START = 0x0980
BANGLA_UNICODE_END = 0x09FF


def _bangla_char_ratio(text: str) -> float:
    """
    Calculate the fraction of characters in `text` that are Bangla script.

    Args:
        text: Input string.

    Returns:
        Float between 0.0 and 1.0 representing the proportion of Bangla chars.
    """
    if not text:
        return 0.0
    bangla_count = sum(
        1 for ch in text
        if BANGLA_UNICODE_START <= ord(ch) <= BANGLA_UNICODE_END
    )
    return bangla_count / len(text)


def detect_language(text: str) -> str:
    """
    Detect whether a text chunk is English, Bangla, or Banglish.

    Strategy:
    - If >30% of characters are Bangla unicode → "bangla"
    - If 5–30% Bangla characters (mixed with ASCII) → "banglish"
    - Otherwise use langdetect library as a fallback → "english" or "bangla"

    Args:
        text: Text string to classify.

    Returns:
        One of: "english", "bangla", "banglish"
    """
    ratio = _bangla_char_ratio(text)

    # Clearly Bangla (mostly Bangla script)
    if ratio > 0.30:
        return "bangla"

    # Mixed: some Bangla characters within mostly Latin text → Banglish
    if ratio > 0.05:
        return "banglish"

    # Use langdetect for purely Latin-script texts
    try:
        from langdetect import detect
        lang_code = detect(text)
        # langdetect returns "bn" for Bangla, "en" for English
        if lang_code == "bn":
            return "bangla"
        return "english"
    except Exception:
        # Default to English if detection fails (e.g. very short text)
        return "english"


# Common UCB website navigation phrases that appear at the top of every page
_UCB_NAV_PHRASES = {
    "products & services", "retail banking", "nrb banking", "sme & agri banking",
    "corporate banking", "agent banking", "know ucb", "news & events",
    "ucb taqwa", "ucb ayma", "offshore banking", "support", "career",
    "profile", "corporate information", "about us", "investor relations",
    "csr", "green banking", "right of info", "annual report",
    "sme banking", "agri banking", "islamic banking", "upay",
    "choose your credit card", "features", "terms & conditions",
    "reward program & partners", "safety tips", "card security",
    "documents required", "schedule of charges", "important information",
    "card offers", "debit card", "prepaid card", "sticker card",
    "business card", "credit card",
}


def strip_nav_menu(text: str) -> str:
    """
    Remove UCB website navigation menu lines from the start of page text.

    UCB pages begin with repeated nav items (short lines like "Retail Banking",
    "Products & Services") before the actual content. These dominate embeddings
    and make all pages look similar to the retriever.

    Strategy: Skip initial lines that are short nav items. Stop when we hit
    a line with real content (>= 8 words or contains sentence punctuation).

    Args:
        text: Cleaned page text.

    Returns:
        Text with leading nav menu stripped.
    """
    lines = text.split("\n")
    content_start = 0
    consecutive_content = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        word_count = len(stripped.split())
        is_nav = (
            stripped.lower() in _UCB_NAV_PHRASES
            or (word_count <= 5 and not any(c in stripped for c in ".?!:–—"))
        )

        if is_nav:
            consecutive_content = 0
        else:
            consecutive_content += 1
            if consecutive_content >= 2:
                # Two consecutive real content lines — content has started
                # Go back to the first of these two lines
                content_start = i - 1
                break
    else:
        # Never found content start — return as-is
        return text

    result = "\n".join(lines[content_start:]).strip()
    # Only use stripped version if meaningful content remains
    return result if len(result) > 100 else text


def clean_text(text: str) -> str:
    """
    Clean raw scraped text by removing noise, normalising whitespace,
    stripping HTML artifacts, and removing non-printable characters.

    Steps:
    1. Normalise unicode (NFC form)
    2. Remove HTML entities (e.g. &amp;, &nbsp;)
    3. Remove residual HTML tags (shouldn't be many after BS4, but safety net)
    4. Replace non-breaking spaces with regular spaces
    5. Collapse multiple whitespace / newlines
    6. Strip navigation menu from page start
    7. Strip leading / trailing whitespace

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Step 1: Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Step 2: Common HTML entities
    html_entities = {
        "&amp;": "&", "&nbsp;": " ", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'", "&#8211;": "–",
        "&#8212;": "—", "&#8216;": "'", "&#8217;": "'",
        "&#8220;": '"', "&#8221;": '"',
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Step 3: Strip any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 4: Non-breaking and other unusual spaces → regular space
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

    # Step 5: Collapse runs of whitespace (preserve single newlines as sentence breaks)
    text = re.sub(r"[ \t]+", " ", text)          # multiple spaces → one
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines → 2

    # Step 6: Strip navigation menu from the top of the page
    text = strip_nav_menu(text)

    # Step 7: Strip
    return text.strip()


def token_count(text: str) -> int:
    """
    Approximate token count by splitting on whitespace.

    This is a rough heuristic (≈ word count). For accurate BPE token counts
    you'd load the model's tokenizer, but that adds ~2 GB RAM for chunking alone.

    Args:
        text: Text to count tokens for.

    Returns:
        Integer token estimate.
    """
    return len(text.split())


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split `text` into overlapping chunks of approximately `chunk_size` tokens.

    Algorithm:
    - Split text into sentences (on . ? ! \n boundaries)
    - Accumulate sentences into a chunk until the token budget is exceeded
    - When full, save the chunk and slide back `chunk_overlap` tokens
      by retaining the last few sentences

    Args:
        text: Full cleaned text of a page.
        chunk_size: Target number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    if not text:
        return []

    # Split into sentence-like segments
    # Preserve segment delimiters by keeping them attached to the segment
    segments = re.split(r"(?<=[।.?!\n])\s*", text)
    segments = [s.strip() for s in segments if s.strip()]

    chunks: list[str] = []
    current_tokens: list[str] = []  # list of words in current chunk
    current_char_count = 0

    for segment in segments:
        seg_words = segment.split()

        # If a single segment is longer than chunk_size, forcibly split it
        if len(seg_words) > chunk_size:
            # Flush current buffer first
            if current_tokens:
                chunks.append(" ".join(current_tokens))
            # Hard-split the mega-segment
            for i in range(0, len(seg_words), chunk_size - chunk_overlap):
                hard_chunk = seg_words[i: i + chunk_size]
                chunks.append(" ".join(hard_chunk))
            # Reset overlap from end of mega-segment
            overlap_words = seg_words[-(chunk_overlap):]
            current_tokens = overlap_words
            continue

        # If adding this segment exceeds chunk_size, flush
        if len(current_tokens) + len(seg_words) > chunk_size:
            if current_tokens:
                chunks.append(" ".join(current_tokens))
            # Retain overlap from the end of the flushed chunk
            current_tokens = current_tokens[-chunk_overlap:] if chunk_overlap else []

        current_tokens.extend(seg_words)

    # Flush remaining content
    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks


def process_pages(pages: list[dict]) -> list[dict[str, Any]]:
    """
    Process a list of raw scraped pages into cleaned, chunked records.

    For each page:
    1. Clean the text
    2. Split into overlapping chunks
    3. Detect language of each chunk
    4. Attach metadata (url, title, chunk_index, language)

    Args:
        pages: List of raw page dicts from data/raw/ucb_raw.json.
               Each dict has keys: url, title, text, links.

    Returns:
        List of chunk dicts, each with keys:
          chunk_id, text, url, title, chunk_index, language
    """
    all_chunks: list[dict] = []
    chunk_id = 0
    seen_canonical: set[str] = set()  # deduplicate http vs https

    for page_idx, page in enumerate(pages):
        url = page.get("url", "")
        title = page.get("title", "")
        raw_text = page.get("text", "")

        # Deduplicate http:// and https:// versions of the same page
        canonical = url.replace("http://", "https://").rstrip("/")
        if canonical in seen_canonical:
            logger.debug(f"Skipping duplicate: {url}")
            continue
        seen_canonical.add(canonical)

        # Skip 404 and maintenance pages — they have identical text and poison embeddings
        if ("404 PAGE NOT FOUND" in raw_text or
                "This Page Is Under Maintenance" in raw_text or
                "Under Maintenance" in raw_text):
            logger.debug(f"Skipping junk page: {url}")
            continue

        # Step 1: clean text
        cleaned = clean_text(raw_text)
        if not cleaned:
            logger.debug(f"Skipping empty page: {url}")
            continue

        # Step 2: split into chunks
        chunks = split_into_chunks(cleaned)
        if not chunks:
            continue

        logger.info(
            f"📄 [{page_idx + 1}/{len(pages)}] {url} → {len(chunks)} chunks"
        )

        # Step 3 & 4: detect language and build chunk records
        for chunk_index, chunk_text in enumerate(chunks):
            language = detect_language(chunk_text)
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "url": url,
                "title": title,
                "chunk_index": chunk_index,
                "language": language,
            })
            chunk_id += 1

    return all_chunks


def load_raw_data(path: Path) -> list[dict]:
    """
    Load scraped page data from a JSON file.

    Args:
        path: Path to ucb_raw.json.

    Returns:
        List of raw page dicts.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run the crawler first: python scraper/crawler.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"📂 Loaded {len(data)} raw pages from {path}")
    return data


def save_chunks(chunks: list[dict], path: Path) -> None:
    """
    Save processed chunks to a JSON file.

    Args:
        chunks: List of chunk dicts to save.
        path: Output file path (e.g. data/processed/ucb_chunks.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved {len(chunks)} chunks to {path}")


def print_language_stats(chunks: list[dict]) -> None:
    """
    Log a breakdown of chunk counts per detected language.

    Args:
        chunks: List of processed chunk dicts.
    """
    from collections import Counter
    counts = Counter(c["language"] for c in chunks)
    total = len(chunks)
    logger.info("📊 Language breakdown:")
    for lang, count in sorted(counts.items()):
        pct = count / total * 100 if total else 0
        logger.info(f"   {lang:10s}: {count:5d} chunks ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the chunker pipeline.

    Loads raw scraped data, processes it into chunks, and saves output.
    """
    logger.info("=" * 60)
    logger.info("UCB Bank Chunker — Starting")
    logger.info(f"Input  : {RAW_DATA_PATH}")
    logger.info(f"Output : {PROCESSED_DATA_PATH}")
    logger.info(f"Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP} tokens")
    logger.info("=" * 60)

    try:
        pages = load_raw_data(RAW_DATA_PATH)
        chunks = process_pages(pages)

        if not chunks:
            logger.error("❌ No chunks produced. Check raw data content.")
            sys.exit(1)

        save_chunks(chunks, PROCESSED_DATA_PATH)
        print_language_stats(chunks)
        logger.info(f"🎉 Done! {len(chunks)} chunks ready for embedding.")

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Chunker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
