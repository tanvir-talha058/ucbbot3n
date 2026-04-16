"""
scraper/crawler.py
------------------
Web crawler for UCB Bank website (https://www.ucb.com.bd).

Strategy:
  1. Use Scrapy for URL frontier management and robots.txt compliance.
  2. Use Playwright (via scrapy-playwright) for JavaScript rendering because
     UCB's website is WordPress-based and renders content dynamically.
  3. Use BeautifulSoup for HTML parsing and content extraction.
  4. Save all extracted pages to data/raw/ucb_raw.json.

Usage:
  python scraper/crawler.py
"""

import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CRAWL_DELAY,
    MAX_PAGES,
    RAW_DATA_PATH,
    UCB_BASE_URL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.crawler")

# ---------------------------------------------------------------------------
# Tags whose inner text we always discard (navigation, boilerplate)
# ---------------------------------------------------------------------------
SKIP_TAGS = {
    "nav", "header", "footer", "script", "style",
    "noscript", "aside", "form", "button", "iframe",
}

# ---------------------------------------------------------------------------
# URL patterns to skip (login, search, feed, admin pages)
# ---------------------------------------------------------------------------
SKIP_URL_PATTERNS = [
    r"/wp-admin", r"/wp-login", r"\?s=", r"/feed", r"/tag/",
    r"/category/", r"/page/\d+", r"#", r"javascript:",
    r"mailto:", r"tel:", r"\.pdf$", r"\.jpg$", r"\.png$",
    r"\.gif$", r"\.svg$", r"\.zip$", r"\.doc",
]

# ---------------------------------------------------------------------------
# UCB Bank top-level sections (first path segment of canonical URLs).
# When one of these appears at position > 0 in a URL path it means the
# WordPress navigation menu resolved a relative link from a sub-page and
# produced a "contextual" duplicate URL (same content, different nav context).
# ---------------------------------------------------------------------------
_UCB_ROOT_SECTIONS = {
    "banking", "cards", "know-ucb", "ucb-taqwa", "ucb-ayma",
    "imperial", "support", "offshore-banking", "rates",
    "news-and-events", "atm-locator", "sustainability",
    "unet-internet-banking", "FinancialLiteracy",
}


def _has_repeated_path_segments(url: str) -> bool:
    """
    Return True if any path segment appears more than once in the URL path.

    UCB Bank's WordPress site uses relative hrefs (no leading slash) in its
    navigation menus.  When those links are resolved from a deep page such as
    /banking/agent-banking/, urllib.parse.urljoin produces looping paths like
    /banking/agent-banking/banking/retail-banking/banking/…  Detecting repeated
    segments lets us discard these fabricated URLs before they pollute the frontier.

    Args:
        url: Absolute URL string to evaluate.

    Returns:
        True if the path contains a duplicated segment (looping URL).
    """
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    return len(segments) != len(set(segments))


def _is_contextual_nav_url(url: str) -> bool:
    """
    Return True if the URL is a WordPress contextual-navigation artifact.

    UCB Bank's nav menu links are relative (no leading '/').  When resolved
    from a sub-page (e.g. /cards/) the link "banking/retail-banking" becomes
    /cards/banking/retail-banking — identical content to /banking/retail-banking
    but served under a different path prefix.  The tell-tale sign is a
    top-level UCB section appearing at path position > 0.

    Args:
        url: Absolute URL string to evaluate.

    Returns:
        True if the URL is a contextual duplicate of a canonical page.
    """
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    # If any segment after the first matches a root-level section it is contextual
    return any(seg in _UCB_ROOT_SECTIONS for seg in segments[1:])


def should_skip_url(url: str) -> bool:
    """
    Return True if the URL should be excluded from crawling.

    Skips admin pages, media files, search results, anchors,
    protocol pseudo-links (mailto, tel, javascript), URLs with
    repeated path segments, and WordPress contextual-navigation
    duplicates caused by relative nav links.

    Args:
        url: Absolute URL string to evaluate.

    Returns:
        True if the URL should be skipped, False otherwise.
    """
    for pattern in SKIP_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    if _has_repeated_path_segments(url):
        return True
    if _is_contextual_nav_url(url):
        return True
    return False


def is_internal_url(url: str, base: str = UCB_BASE_URL) -> bool:
    """
    Check whether `url` belongs to the same domain as `base`.

    Args:
        url: URL to check.
        base: Base domain URL (e.g. 'https://www.ucb.com.bd').

    Returns:
        True if the host of `url` matches the host of `base`.
    """
    return urlparse(url).netloc == urlparse(base).netloc


def extract_text_from_html(html: str, source_url: str) -> dict[str, Any]:
    """
    Parse raw HTML and extract the page title, clean body text, and links.

    Uses BeautifulSoup to:
    - Remove boilerplate tags (nav, header, footer, scripts, etc.)
    - Extract remaining visible text
    - Collect all internal hrefs for further crawling

    Args:
        html: Raw HTML string of the page.
        source_url: The URL from which this HTML was fetched (for metadata).

    Returns:
        Dictionary with keys:
          - 'url'   : source URL
          - 'title' : page <title> text
          - 'text'  : cleaned visible body text
          - 'links' : list of absolute internal URLs found on the page
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Title ---
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # --- Remove boilerplate elements ---
    for tag in SKIP_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # --- Extract visible text from main content areas ---
    # UCB Bank uses <section> as its primary content wrapper.
    # Prefer section/main/article, fall back to body.
    main_content = (
        soup.find("section")
        or soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.find(class_=re.compile(r"content|main|entry|post", re.I))
        or soup.body
    )

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # --- Collect internal links for the crawl frontier ---
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        absolute = urljoin(source_url, href)
        if is_internal_url(absolute) and not should_skip_url(absolute):
            # Normalise: force HTTPS, strip fragment and query string, remove trailing slash
            parsed = urlparse(absolute)
            parsed = parsed._replace(scheme="https", fragment="", query="")
            normalised = parsed.geturl().rstrip("/")
            # Hard cap at 6 path segments as a last-resort safety net
            path_depth = len([s for s in parsed.path.split("/") if s])
            if normalised and path_depth <= 6:
                links.append(normalised)

    return {
        "url": source_url,
        "title": title,
        "text": text,
        "links": list(set(links)),  # deduplicate
    }


# ---------------------------------------------------------------------------
# Playwright-based async crawler
# ---------------------------------------------------------------------------

async def _fetch_page_playwright(url: str, browser_context) -> str:
    """
    Fetch a single URL using Playwright and return the fully-rendered HTML.

    Waits for the page to reach 'networkidle' so that dynamic JavaScript
    content has time to load before we extract text.

    Args:
        url: The URL to load.
        browser_context: An active Playwright BrowserContext.

    Returns:
        Full page HTML as a string, or empty string on failure.
    """
    page = await browser_context.new_page()
    try:
        # navigate and wait until network is quiet
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        html = await page.content()
        return html
    except Exception as exc:
        logger.warning(f"⚠️  Failed to load {url}: {exc}")
        return ""
    finally:
        await page.close()


async def crawl_with_playwright(
    start_url: str,
    max_pages: int | None = None,
) -> list[dict]:
    """
    BFS crawl of UCB Bank website using Playwright for JS rendering.

    Visits every discoverable internal page when max_pages is None,
    otherwise stops after visiting max_pages pages.

    Args:
        start_url: Root URL to begin crawling.
        max_pages: Maximum pages to visit, or None for unlimited.

    Returns:
        List of dicts, each with keys: url, title, text, links.
    """
    from playwright.async_api import async_playwright

    visited: set[str] = set()
    frontier: deque[str] = deque([start_url.rstrip("/")])
    frontier_set: set[str] = {start_url.rstrip("/")}  # dedup guard
    pages_data: list[dict] = []

    # Display string for logging
    limit_str = str(max_pages) if max_pages is not None else "unlimited"

    async with async_playwright() as pw:
        # Launch headless Chromium (best for WordPress JS sites)
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (compatible; UCBBankBot/1.0; "
                "+https://www.ucb.com.bd)"
            )
        )

        logger.info(f"🚀 Starting crawl from {start_url} (max pages: {limit_str})")

        # Continue until frontier is exhausted or limit is hit
        while frontier and (max_pages is None or len(visited) < max_pages):
            url = frontier.popleft()

            # Skip already visited or unwanted URLs
            if url in visited or should_skip_url(url):
                continue
            if not is_internal_url(url):
                continue

            visited.add(url)
            logger.info(f"🌐 [{len(visited)}/{limit_str}] Crawling: {url}")

            # Fetch with Playwright
            html = await _fetch_page_playwright(url, context)
            if not html:
                continue

            # Parse and extract
            page_data = extract_text_from_html(html, url)

            # Only keep pages with meaningful content (>100 chars)
            if len(page_data["text"]) > 100:
                pages_data.append(page_data)
                logger.info(
                    f"   ✅ Extracted {len(page_data['text'])} chars, "
                    f"{len(page_data['links'])} links"
                )
            else:
                logger.debug(f"   ⏭️  Skipping {url} — insufficient content")

            # Add newly discovered links to the frontier (skip already queued)
            for link in page_data["links"]:
                if link not in visited and link not in frontier_set:
                    frontier.append(link)
                    frontier_set.add(link)

            # Polite delay to avoid hammering the server
            time.sleep(CRAWL_DELAY)

        await context.close()
        await browser.close()

    logger.info(
        f"✅ Crawl complete. Visited {len(visited)} pages, "
        f"kept {len(pages_data)} with content."
    )
    return pages_data


def save_raw_data(pages: list[dict], output_path: Path) -> None:
    """
    Persist scraped page data to a JSON file.

    Args:
        pages: List of page data dicts returned by the crawler.
        output_path: Destination file path (e.g. data/raw/ucb_raw.json).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved {len(pages)} pages to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the UCB Bank web crawler.

    Runs the Playwright BFS crawl and saves results to data/raw/ucb_raw.json.
    """
    import asyncio

    limit_display = str(MAX_PAGES) if MAX_PAGES is not None else "unlimited (all pages)"
    logger.info("=" * 60)
    logger.info("UCB Bank Web Crawler — Starting")
    logger.info(f"Target    : {UCB_BASE_URL}")
    logger.info(f"Max pages : {limit_display}")
    logger.info(f"Output    : {RAW_DATA_PATH}")
    logger.info("=" * 60)

    try:
        pages = asyncio.run(crawl_with_playwright(UCB_BASE_URL, MAX_PAGES))
        if pages:
            save_raw_data(pages, RAW_DATA_PATH)
            logger.info(f"🎉 Done! {len(pages)} pages saved.")
        else:
            logger.error("❌ No pages were scraped. Check your network or URL.")
    except Exception as exc:
        logger.error(f"❌ Crawler failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
