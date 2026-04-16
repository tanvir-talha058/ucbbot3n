"""
find_noise_chunks.py
--------------------
Scans all chunks, classifies noise by category,
saves them to data/noise_chunks.json without touching the originals.
"""

import json
import re
from pathlib import Path

CHUNKS_PATH  = "data/processed/ucb_chunks.json"
NOISE_OUTPUT = "data/noise_chunks.json"
CLEAN_PREVIEW= "data/noise_summary.txt"

# ── Noise rules — (label, reason, url_pattern or text_pattern) ──────────────

def classify_chunk(chunk: dict) -> str | None:
    """
    Return a noise label if chunk is noise, else None.
    Labels explain WHY the chunk is noise.
    """
    url  = chunk.get("url", "").lower()
    text = chunk.get("text", "")
    text_lower = text.lower()

    # 1. ATM locator — 46 chunks of ATM branch addresses
    if "/atm-locator" in url:
        return "ATM_LOCATOR — database of ATM addresses, not product info"

    # 2. Terminated agents list — 23 chunks
    if "list-of-agents-terminated" in url:
        return "TERMINATED_AGENTS — compliance list, not customer info"

    # 3. T&C legal boilerplate — 17 chunks
    if "/cards/terms-and-conditions" in url:
        return "CREDIT_CARD_TC — legal T&C boilerplate, poisons all card queries"

    # 4. Reward program T&C — 9 chunks
    if "/cards/reward-program-and-partners" in url:
        return "REWARD_TC — reward legal text + partner tables, not product info"

    # 5. Press releases — event news, not product info
    if "/news-and-events/press-release/" in url:
        return "PRESS_RELEASE — news event, no product information"

    # 6. Individual board/committee/management profile pages
    if any(x in url for x in [
        "/board-of-directors/",
        "/audit-committee/",
        "/executive-committee/",
        "/risk-management-committee/",
        "/shariah-supervisory-committee/",
        "/senior-management/",
    ]):
        # Skip the index pages (e.g. /board-of-directors without trailing name)
        # Only flag individual profile pages
        profile_indicators = [
            "akhter", "afroza", "asifuzzaman", "bashir", "hajee",
            "nurul", "syed", "mr-", "mrs-", "dr-", "professor",
            "roxana", "kanak", "bazal", "mohammed", "mohammad",
            "nabil", "arif", "habibur", "abul", "md-", "profile"
        ]
        if any(p in url for p in profile_indicators):
            return "PERSON_PROFILE — individual director/manager profile page"

    # 7. Financial literacy — generic money education, not UCB products
    if "/financialliteracy/" in url or "/learn-about-money/" in url:
        return "FINANCIAL_LITERACY — generic money tips, not UCB-specific info"

    # 8. News & events non-press pages
    if any(x in url for x in [
        "/news-and-events/photo-gallery",
        "/news-and-events/latest-tvc",
        "/news-and-events/newsletter",
        "/news-and-events/press-ad",
        "/news-and-events/downloads",
        "/news-and-events/bank-note-identification",
        "/news-and-events/national-mourning-day",
    ]):
        return "NEWS_MEDIA — photo gallery / TVC / newsletter, no product info"

    # 9. Career page
    if "/career/" in url:
        return "CAREER — job recruitment page, not product info"

    # 10. Citizen charter
    if "/citizencharter/" in url:
        return "CITIZEN_CHARTER — govt compliance page, not product info"

    # 11. Investor relations sub-pages (financial statements etc.)
    if "/investor-relations/" in url and any(x in url for x in [
        "financial-statements", "shareholding", "directors-declaration",
        "unclaimed-dividend", "right-share", "price-sensetive",
        "notice-to-shareholders", "directors-report", "credit-rating",
        "code-of-conduct-for-board", "report-on-corporate-governance",
    ]):
        return "INVESTOR_RELATIONS — financial/governance docs, not customer product info"

    # 12. Sudhachar (ethics/integrity page)
    if "/sudhachar" in url:
        return "SUDHACHAR — ethics/integrity admin page"

    # 13. Right of information admin pages
    if "/right-of-info/" in url:
        return "RIGHT_OF_INFO — govt RTI compliance page"

    # 14. Auction notice
    if "/auction-notice" in url:
        return "AUCTION_NOTICE — property auction notice"

    # 15. Chunks that are almost entirely navigation menu text
    # (many short lines that are just product category names)
    nav_keywords = [
        "retail banking", "nrb banking", "sme & agri banking",
        "corporate banking", "agent banking", "choose your credit card",
        "features", "terms & conditions", "reward program & partners",
        "safety tips", "card security", "documents required",
        "schedule of charges", "important information", "card offers",
    ]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 3:
        nav_line_count = sum(1 for l in lines if l.lower() in nav_keywords)
        nav_ratio = nav_line_count / len(lines)
        if nav_ratio > 0.4 and len(text) < 600:
            return "NAV_MENU_ONLY — chunk is mostly website navigation menu text"

    # 16. Very short chunks with no useful info (< 80 chars)
    if len(text.strip()) < 80:
        return "TOO_SHORT — chunk has less than 80 characters of content"

    return None  # not noise


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    noise_chunks = []
    clean_chunks_count = 0
    label_counts: dict[str, int] = {}

    for chunk in chunks:
        label = classify_chunk(chunk)
        if label:
            chunk_with_label = dict(chunk)
            chunk_with_label["noise_reason"] = label
            noise_chunks.append(chunk_with_label)
            # Count by category
            category = label.split(" —")[0]
            label_counts[category] = label_counts.get(category, 0) + 1
        else:
            clean_chunks_count += 1

    # Save noise chunks
    Path("data").mkdir(exist_ok=True)
    with open(NOISE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(noise_chunks, f, ensure_ascii=False, indent=2)

    # Save summary report
    lines = []
    lines.append("=" * 70)
    lines.append("NOISE CHUNK ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total chunks scanned : {len(chunks)}")
    lines.append(f"Noise chunks found   : {len(noise_chunks)}")
    lines.append(f"Clean chunks         : {clean_chunks_count}")
    lines.append(f"Noise percentage     : {len(noise_chunks)/len(chunks)*100:.1f}%")
    lines.append("")
    lines.append("BY CATEGORY:")
    lines.append(f"  {'Category':<30} {'Count':>5}")
    lines.append(f"  {'-'*40}")
    for cat, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:<30} {count:>5}")
    lines.append("")
    lines.append("SAMPLE NOISE CHUNKS (first 2 per category):")
    shown: dict[str, int] = {}
    for chunk in noise_chunks:
        cat = chunk["noise_reason"].split(" —")[0]
        if shown.get(cat, 0) >= 2:
            continue
        shown[cat] = shown.get(cat, 0) + 1
        lines.append(f"\n  [{cat}]")
        lines.append(f"  URL   : {chunk.get('url','')}")
        lines.append(f"  Text  : {chunk.get('text','')[:150].strip()}...")

    summary = "\n".join(lines)

    with open(CLEAN_PREVIEW, "w", encoding="utf-8") as f:
        f.write(summary)

    safe = summary.encode("ascii", errors="replace").decode("ascii")
    print(safe)
    print(f"\nNoise chunks saved to : {NOISE_OUTPUT}")
    print(f"Summary saved to      : {CLEAN_PREVIEW}")


if __name__ == "__main__":
    main()
