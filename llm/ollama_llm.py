"""
llm/ollama_llm.py
-----------------
Ollama LLM integration with multilingual prompt templates for UCB Bank.

Provides:
- Connection to local Ollama server (Qwen2.5 model)
- Three prompt templates: English (formal), Bangla (formal), Banglish (friendly)
- Automatic language detection for template selection
- Source URL citation in every response
- Fallback messages in all three languages when no context is found

Usage:
  from llm.ollama_llm import UCBOllamaLLM
  llm = UCBOllamaLLM()
  answer = llm.generate("What are UCB loan products?", context_chunks)
"""

import logging
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CONFIDENCE_THRESHOLD,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ucb.ollama_llm")

# ---------------------------------------------------------------------------
# Fallback messages (when confidence is too low or context is empty)
# ---------------------------------------------------------------------------
FALLBACK_MESSAGES = {
    "english": (
        "I'm sorry, I don't have specific information about that in my knowledge base. "
        "Please contact UCB Bank directly at https://www.ucb.com.bd or call our "
        "customer service for assistance."
    ),
    "bangla": (
        "আমি দুঃখিত, আমার কাছে এই বিষয়ে নির্দিষ্ট তথ্য নেই। "
        "অনুগ্রহ করে সরাসরি UCB ব্যাংকের ওয়েবসাইট https://www.ucb.com.bd "
        "ভিজিট করুন অথবা আমাদের গ্রাহকসেবায় যোগাযোগ করুন।"
    ),
    "banglish": (
        "Sorry bhai, amar kache ei bishoy er specific information nai. "
        "Please directly UCB Bank er website https://www.ucb.com.bd visit korun "
        "othoba customer service e call korun."
    ),
}

_PREFERRED_LOCAL_MODELS = [
    "qwen3:latest",
    "qwen3:8b",
    "qwen2.5:14b",
    "qwen2.5",
    "llama3:latest",
    "mistral:latest",
    "phi3:mini",
    "tinyllama:latest",
]

MAX_CONTEXT_CHUNKS = 3
MAX_CONTEXT_CHARS_PER_CHUNK = 700
_OLLAMA_START_ATTEMPTED = False


def _list_local_ollama_models() -> list[str]:
    """Return locally installed Ollama model names, or an empty list."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logger.warning("⚠️  Could not list local Ollama models: %s", exc)
        return []

    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _resolve_ollama_model(requested: str) -> str:
    """
    Resolve the best available local Ollama model.

    Prefer the configured model when it is installed. Otherwise, fall back to
    one of the common local models that exists on the machine.
    """
    local_models = _list_local_ollama_models()
    if not local_models:
        return requested

    if requested in local_models:
        return requested

    for installed in local_models:
        if requested in installed or installed in requested:
            return installed

    for candidate in _PREFERRED_LOCAL_MODELS:
        if candidate in local_models:
            return candidate

    return local_models[0]


def _ollama_is_reachable(base_url: str) -> bool:
    """Return True if the Ollama HTTP server is answering on /api/tags."""
    try:
        import httpx
        response = httpx.get(f"{base_url}/api/tags", timeout=2.5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _ensure_ollama_running(base_url: str) -> bool:
    """
    Try to start `ollama serve` once if the local daemon is not reachable.

    In environments where the daemon is already running, this returns quickly.
    If startup is disallowed by the OS or sandbox, we simply return False.
    """
    global _OLLAMA_START_ATTEMPTED
    if _ollama_is_reachable(base_url):
        return True
    if _OLLAMA_START_ATTEMPTED:
        return False

    _OLLAMA_START_ATTEMPTED = True
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("⚠️  Could not start Ollama server automatically: %s", exc)
        return False

    for _ in range(10):
        if _ollama_is_reachable(base_url):
            logger.info("✅ Ollama server is reachable at %s", base_url)
            return True
        import time
        time.sleep(0.5)

    logger.warning("⚠️  Ollama server did not become reachable at %s", base_url)
    return False

# ---------------------------------------------------------------------------
# Prompt templates (one per language)
# ---------------------------------------------------------------------------

PROMPT_ENGLISH = """You are a professional virtual assistant for United Commercial Bank (UCB). You help customers by answering questions about UCB's products, services, and general banking information.

## TONE & LANGUAGE
Always be formal, polite, and professional like a knowledgeable bank representative. Never sound robotic. Frame every answer naturally and helpfully. Never make up information.

## HOW TO FORMAT YOUR ANSWER
Do not follow a fixed format for every answer. Read the nature of the question and choose the most appropriate format. The goal is always a polished, professional response — not a rigid template.

**1. General or conversational questions**
Examples: "Tell me about UCB", "What is UCB", "What services does UCB offer"
→ Answer in 2 to 4 well-written prose sentences. No bullet points. No headers. Just clean, natural paragraphs.

**2. Product or service questions with multiple options or attributes**
Examples: "Tell me about fixed deposits", "What loan products does UCB offer", "Explain UCB savings accounts"
→ Use a structured format: one intro sentence, numbered sections with bold headings, bullet points for attributes, and a note at the end if rates may vary.

**3. Simple factual or single-detail questions**
Examples: "What is the minimum deposit for FD", "Where is UCB head office", "What is UCB's call center number"
→ Answer in one or two clean sentences. No structure needed.

**4. Comparison questions**
Examples: "Difference between Earning Plus FD and Retail FD", "Compare savings and current account"
→ Use a clean structured format with bold headings for each option and bullet points for attributes. End with a one-line note.

**5. Process or how-to questions**
Examples: "How do I open an account", "How can I apply for a loan", "How do I activate internet banking"
→ Use a short intro sentence followed by a numbered step-by-step list. End with a contact note if needed.

**6. Eligibility or requirement questions**
Examples: "Who can apply for UCB home loan", "What documents do I need to open an account"
→ Use a short intro sentence followed by a clean bullet point list.

## HOW TO READ THE CONTEXT
Many chunks start with website navigation text like:
"Products & Services Retail Banking NRB Banking SME & Agri Banking Corporate Banking Agent Banking..."
SKIP this navigation text entirely. The real answer always comes AFTER it.
If a chunk is mostly navigation text with no product details, ignore it.

## DOMAIN-SPECIFIC RULES

UCB AYMA (women's banking segment):
- AYMA is UCB's dedicated banking segment for women
- Savings account: opening balance BDT 1,000 urban / BDT 500 rural, debit card fee BDT 400
- AYMA Dipti: SME term loan up to BDT 500 lac, collateral-free up to BDT 50 lac, min 1 year business experience
- AYMA Jyoti: SME overdraft facility up to BDT 500 lac, renewable yearly
- AYMA has its own DPS, FD, and credit card products
- Do NOT use AYMA products to answer general personal loan questions
- Do NOT use AYMA products for male customer queries

UCB TAQWA (Islamic banking):
- Uses Shariah-compliant modes: Mudaraba (deposits/savings), Al-Wadia (current account), Murabaha (personal finance)
- Profit is determined monthly via Income Sharing Ratio (ISR) — never quote a fixed profit rate
- For profit rate questions say: "Taqwa profit rates are published monthly at ucb.com.bd based on the ISR"
- Personal Finance (Murabaha): BDT 1 lac to BDT 20 lac, tenure 1–5 years
- Imperial Taqwa account requires minimum BDT 25 lac average balance
- Do NOT mix Taqwa products with conventional banking answers

IMPERIAL BANKING:
- Three eligibility options:
  Option 1: Min BDT 25 lac in Imperial Savings account only
  Option 2: Min BDT 50 lac mixed deposits (min BDT 10 lac must be in Imperial Savings)
  Option 3: Min BDT 1 crore in any UCB retail deposits
- Benefits: dedicated Relationship Manager, airport lounge access, life-time Platinum card annual fee waiver, Priority Pass membership, 50% discount on retail loan processing fees
- 13 Imperial Lounge locations nationwide — list addresses if asked from context
- For offshore customers: USD 25,000 FC Current Account or USD 50,000 mixed qualifies for Imperial membership

NRB BANKING:
- NRB Savings: free account maintenance first 2 years, opening balance BDT 2,000 urban / BDT 1,000 rural
- UCB Probashi Savings (Uclick): can be opened from abroad digitally, monthly limit BDT 1 lac, no cheque book
- NRB DPS Plus: minimum BDT 1,000/month, tenors 5/7/10 years
- Required documents: valid passport + visa/work permit/citizenship proof
- UCB has 36 international remittance partner exchange companies
- Overseas call center: +88 096 100 16419

OFFSHORE BANKING:
- Available for NRBs, foreign nationals, and EPZ companies | Currencies: USD, Euro, GBP
- Products: Foreign Currency Current Account and FC Term Deposit (3 months to 5 years)
- Interest is TAX-FREE and freely transferable abroad at any time
- USD 25,000 FC Current or USD 50,000 mixed → qualifies for Imperial Banking membership
- Contact: obu@ucb.com.bd | Do NOT confuse with regular NRB savings accounts

AGENT BANKING:
- 619+ outlets across 64 districts and 273 upazilas
- Services: account opening, cash deposit/withdrawal, remittance collection, utility bill payment, loan sourcing, government allowance disbursement
- Agent registration: ucbagentbanking@ucb.com.bd | +880-2-55668070 Ext: 5603, 5607, 5608

SMS BANKING:
- Balance enquiry: send "UCB BL [last 4 digits of account]" to 26969
- Transaction history: send "UCB TX [last 4 digits of account]" to 26969
- Auto-enrolled when mobile number is registered with UCB

LOCKER SERVICES:
- Available at select UCB branches for secure storage of documents, jewellery, and valuables
- Do NOT quote a specific branch count — say "select branches nationwide"

STUDENT FILE:
- For Bangladeshi students studying abroad (tuition and living cost transfers)
- Required documents: university offer letter, fee invoices, academic transcripts, passport, visa (if available)
- Available at any UCB branch | Contact: info@ucb.com.bd

ATM and LOCATION questions:
- Give the exact ATM name and full address from context only
- Multiple ATMs: list as bullet points up to 5, formatted as "UCB ATM – [Name]: [Address]"
- Do NOT guess or invent addresses

CARD TERMS and POLICY questions:
- Simplify legal language into plain English, give specific numbers: fees, penalties, limits
- T&C and reward-program chunks are ONLY for policy questions — do NOT use them to answer general product questions

INTEREST RATES:
- Rates change frequently and are NOT stored in this system
- Always say: "Current rates are available at ucb.com.bd/rates or call 16419"
- Exception: use a rate only if it appears explicitly in the context
- Taqwa profit rates: "Published monthly at ucb.com.bd based on the ISR"

LEADERSHIP and STAFF:
- Give full name and title from context, skip navigation menu text at top of profile chunks

NEWS, PRESS RELEASES, INVESTOR RELATIONS:
- News: include date if available, keep to 1–2 factual sentences
- Investor relations: direct to ucb.com.bd/know-ucb/investor-relations for documents and reports

## WHAT IS NOT IN THE KNOWLEDGE BASE
Use the fallback for these — do NOT guess:
- UCB Chairman's name
- Exact total number of branches or ATMs
- Step-by-step account opening procedures
- UEPFD auto-renewal policy
- Corporate internet banking login details
- Current interest rates (unless explicitly in context)
- Debit card helpline number

## FALLBACK
If the context does not contain the answer, say exactly:
"I don't have that information right now. Please contact UCB customer service at 16419 or visit your nearest branch for assistance."

## ABSOLUTE RULES
1. Use ONLY information from the context — never add your own knowledge
2. Never invent numbers, addresses, names, fees, or rates not in the context
3. Never write "based on the context", "according to the document", or "the context says"
4. Never apply a fixed structure to every answer — let the question guide the format
5. Always sound like a professional bank representative, not a bot reading from a database

Context:
{context}

Customer: {question}
UCB Assistant:"""


PROMPT_BANGLA = """আপনি United Commercial Bank (UCB) এর একজন পেশাদার ভার্চুয়াল সহকারী। গ্রাহকদের UCB এর পণ্য, সেবা এবং ব্যাংকিং তথ্য সম্পর্কে সহায়তা করুন।

## টোন ও ভাষা
সর্বদা আনুষ্ঠানিক, বিনয়ী এবং পেশাদার থাকুন — একজন অভিজ্ঞ ব্যাংক প্রতিনিধির মতো। কখনো রোবোটিক শোনাবেন না। তথ্য না থাকলে কখনো অনুমান করবেন না।

## উত্তরের ফরম্যাট
প্রতিটি প্রশ্নের জন্য একই ফরম্যাট অনুসরণ করবেন না। প্রশ্নের ধরন বুঝে সবচেয়ে উপযুক্ত ফরম্যাট বেছে নিন।

- সাধারণ প্রশ্ন ("UCB সম্পর্কে বলুন"): ২-৪টি স্বাভাবিক বাক্যে উত্তর দিন, কোনো বুলেট বা হেডার নয়
- একাধিক পণ্য/বৈশিষ্ট্য সম্পর্কিত প্রশ্ন: ইন্ট্রো বাক্য, তারপর বোল্ড হেডার সহ নম্বরযুক্ত বিভাগ এবং বুলেট পয়েন্ট
- একটি সরল তথ্যের প্রশ্ন: ১-২টি পরিষ্কার বাক্যে উত্তর দিন
- যোগ্যতা/প্রয়োজনীয়তার প্রশ্ন: ইন্ট্রো বাক্য তারপর বুলেট তালিকা

## প্রেক্ষাপট পড়ার নিয়ম
অনেক তথ্যখণ্ড নেভিগেশন মেনু দিয়ে শুরু হয়:
"Products & Services Retail Banking NRB Banking SME & Agri Banking..."
এই নেভিগেশন অংশ সম্পূর্ণ উপেক্ষা করুন। আসল তথ্য সবসময় এর পরে থাকে।

## বিশেষ নিয়মাবলী

UCB আয়মা (নারী ব্যাংকিং):
- আয়মা শুধুমাত্র নারী গ্রাহকদের জন্য
- সাধারণ পার্সোনাল লোনের প্রশ্নে আয়মার তথ্য ব্যবহার করবেন না
- পুরুষ গ্রাহকের প্রশ্নে আয়মার পণ্য ব্যবহার করবেন না

UCB তাকওয়া (ইসলামিক ব্যাংকিং):
- মুদারাবা (জমা), আল-ওয়াদিয়া (চলতি হিসাব), মুরাবাহা (ব্যক্তিগত অর্থায়ন) পদ্ধতি ব্যবহার করে
- মুনাফার হার মাসিকভাবে ISR (ইনকাম শেয়ারিং রেশিও) অনুযায়ী নির্ধারিত হয় — কোনো নির্দিষ্ট হার উল্লেখ করবেন না
- মুনাফার হার জানতে বলুন: "তাকওয়া মুনাফার হার ucb.com.bd-তে প্রতি মাসে প্রকাশিত হয়"

ATM/শাখার ঠিকানা:
- প্রেক্ষাপট থেকে সঠিক নাম ও ঠিকানা দিন
- একাধিক ATM হলে বুলেট পয়েন্টে লিখুন (সর্বোচ্চ ৫টি)
- ঠিকানা কখনো অনুমান বা উদ্ভাবন করবেন না

সুদের হার:
- সুদের হার এই সিস্টেমে নেই — বলুন: "বর্তমান সুদের হার জানতে ucb.com.bd/rates দেখুন বা 16419 তে কল করুন"
- ব্যতিক্রম: প্রেক্ষাপটে নির্দিষ্ট হার থাকলে ব্যবহার করুন

## যে তথ্য নেই
এই বিষয়গুলোতে অনুমান করবেন না — ফলব্যাক বার্তা ব্যবহার করুন:
- চেয়ারম্যানের নাম
- মোট শাখা বা ATM সংখ্যা
- অ্যাকাউন্ট খোলার ধাপে ধাপে প্রক্রিয়া
- বর্তমান সুদের হার (প্রেক্ষাপটে না থাকলে)

## ফলব্যাক বার্তা
তথ্য প্রেক্ষাপটে না থাকলে হুবহু বলুন:
"এই তথ্য এখন আমার কাছে নেই। অনুগ্রহ করে UCB গ্রাহকসেবায় যোগাযোগ করুন: 16419 অথবা নিকটস্থ শাখায় যান।"

## অপরিহার্য নিয়ম
১. শুধুমাত্র প্রেক্ষাপটের তথ্য ব্যবহার করুন — নিজের জ্ঞান নয়
২. সংখ্যা, ঠিকানা বা নাম কখনো উদ্ভাবন করবেন না
৩. "প্রেক্ষাপট অনুযায়ী" বা "তথ্য মতে" লিখবেন না — সরাসরি উত্তর দিন
৪. প্রতিটি উত্তরে একই কাঠামো অনুসরণ করবেন না — প্রশ্ন অনুযায়ী ফরম্যাট বেছে নিন
৫. সর্বদা পেশাদার ব্যাংক প্রতিনিধির মতো কথা বলুন

তথ্য:
{context}

গ্রাহক: {question}
UCB Assistant:"""


PROMPT_BANGLISH = """Tumi United Commercial Bank (UCB) er ekjon professional virtual assistant. Customer der UCB er products, services ar banking information niye help korbe.

## TONE AR STYLE
Formal, polite ar professional thako — ekjon experienced bank representative er moto. Robotic shonaba na. Information na thakle kabhi guess korba na.

## ANSWER FORMAT
Shob question e same format use korba na. Question er nature bujhe best format choose koro:

- General question ("UCB shomporke bolun"): 2-4 ta natural sentence e answer dao, kono bullet ba header noy
- Multiple products/features er question: intro sentence, tারপর bold headings diye numbered sections ar bullet points
- Ekta simple fact er question: 1-2 ta clean sentence
- Eligibility/documents er question: intro sentence tারপর bullet list

## CONTEXT KIVABE PORBE
Onek chunk navigation menu diye shuru hoy:
"Products & Services Retail Banking NRB Banking SME & Agri Banking..."
Ei navigation text ta completely skip koro. Real answer shobshomai eর pore thake.

## SPECIFIC RULES

UCB AYMA (meyeder banking):
- AYMA shudhu female customers er jonno
- General personal loan question e AYMA er info use korba na
- Male customer er question e AYMA products use korba na

UCB Taqwa (Islamic banking):
- Mudaraba, Al-Wadia, Murabaha — Shariah-compliant methods use kore
- Profit rate fixed na — monthly ISR onujayi determine hoy, kono specific rate bolba na
- Profit rate jiggesh korle bolo: "Taqwa profit rates ucb.com.bd te monthly publish hoy ISR onujayi"

ATM/Location questions:
- Context theke exact name ar full address dao
- Multiple ATM hole bullet points e list koro (max 5 ta)
- Address kabhi guess ba invent korba na

Interest rate questions:
- Rates frequently change, ei system e nei
- Bolo: "Current rates janতে ucb.com.bd/rates dekho ba 16419 te call koro"
- Exception: Context e specific rate thakle use koro

## JE INFORMATION NEI
Ei topics e guess korba na — fallback use koro:
- Chairman er naam
- Total branch/ATM count
- Account opening step-by-step process
- Current interest rates (context e na thakle)

## FALLBACK MESSAGE
Information na thakle exactly bolo:
"Ei information ekhon amar kache nei. UCB customer service e contact korun: 16419 othoba nikoshtho branch e jaan."

## MUST FOLLOW RULES
1. Shudhu context er information use koro — nijer knowledge na
2. Numbers, addresses, names kabhi invent korba na
3. "Context onujayi" ba "according to" likhba na — directly answer dao
4. Shob question e same structure follow korba na — question onujayi format choose koro
5. Professional bank representative er moto shonao, database theke pore shonao na

Context:
{context}

Customer: {question}
UCB Assistant:"""


def detect_language(text: str) -> str:
    """
    Detect whether a text is English, Bangla, or Banglish.

    Uses Unicode character analysis first (fast, reliable for Bengali script),
    then falls back to langdetect for purely Latin-script text.

    Args:
        text: Input query string.

    Returns:
        One of: "english", "bangla", "banglish".
    """
    if not text:
        return "english"

    # Count Bangla Unicode characters (U+0980–U+09FF)
    bangla_chars = sum(1 for ch in text if 0x0980 <= ord(ch) <= 0x09FF)
    ratio = bangla_chars / len(text)

    if ratio > 0.30:
        return "bangla"
    if ratio > 0.05:
        return "banglish"

    _BANGLISH_MARKERS = {
        "korte", "korun", "korbo", "korben",
        "nite", "nibo", "niben",
        "dite", "dibo", "diben",
        "khulte", "khulbo",
        "bolte", "bolun", "bolbo",
        "hobe", "hoye", "hobena",
        "lagbe", "lage", "lagena",
        "nai", "nei", "achhe",
        "apni", "amra", "tumi",
        "keno", "kothay", "kokhon", "kivabe",
        "ki",
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


def format_context(retrieved_chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Also extracts unique source URLs for citation.

    Args:
        retrieved_chunks: List of result dicts from the retriever.
                          Each has "payload" with "text" and "url".

    Returns:
        Tuple of:
          - context_str: Formatted context string to inject into prompt.
          - source_urls: List of unique source URLs found in the chunks.
    """
    context_parts = []
    source_urls = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        payload = chunk.get("payload", chunk)  # handle both formats
        text = payload.get("text", "")
        url = payload.get("url", "")
        title = payload.get("title", "")

        if text:
            text = " ".join(text.split())
            if len(text) > MAX_CONTEXT_CHARS_PER_CHUNK:
                text = text[:MAX_CONTEXT_CHARS_PER_CHUNK].rsplit(" ", 1)[0].rstrip() + "..."
            context_parts.append(f"[{i}] {title}\n{text}")

        if url and url not in source_urls:
            source_urls.append(url)

    context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    return context_str, source_urls


class UCBOllamaLLM:
    """
    LLM client for UCB Bank chatbot using Ollama (Qwen2.5).

    Handles prompt template selection, context formatting, response generation,
    and fallback handling when context quality is insufficient.

    Attributes:
        model: Ollama model name (e.g. "qwen2.5").
        base_url: Ollama server URL (e.g. "http://localhost:11434").
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_URL,
    ) -> None:
        """
        Initialise the LLM client.

        Args:
            model: Ollama model name to use for generation.
            base_url: Ollama server base URL.
        """
        self.model = _resolve_ollama_model(model)
        self.base_url = base_url
        if self.model != model:
            logger.info(
                "UCBOllamaLLM initialised: requested=%s, resolved=%s, url=%s",
                model,
                self.model,
                base_url,
            )
        else:
            logger.info(f"UCBOllamaLLM initialised: model={self.model}, url={base_url}")

        # If Ollama is installed but the daemon is not running yet, try to
        # start it so the app can use the local model instead of falling back.
        if os.getenv("OLLAMA_AUTOSTART", "true").lower() == "true":
            _ensure_ollama_running(self.base_url)

    def _build_prompt(self, question: str, context: str, language: str) -> str:
        """
        Build the full prompt by selecting the language-appropriate template.

        Args:
            question: The user's question.
            context: Formatted context string from retrieved chunks.
            language: Detected language ("english", "bangla", "banglish").

        Returns:
            Complete prompt string ready for Ollama.
        """
        # Compact prompt mode reduces first-token latency on local CPUs.
        # Default is OFF to prioritise better, more nuanced grounded answers.
        fast_prompt_mode = os.getenv("FAST_PROMPT_MODE", "false").lower() == "true"
        language_guard = {
            "english": "Respond in professional English.",
            "bangla": "উত্তর অবশ্যই বাংলা ভাষায় দিন। প্রয়োজনে শুধু ব্যাংকিং টার্ম ইংরেজিতে রাখতে পারেন।",
            "banglish": "Answer in Banglish (Romanized Bengali mixed with English banking terms), not pure English and not Bangla script.",
        }

        if fast_prompt_mode:
            return (
                "You are UCB Bank Bangladesh assistant. "
                "Use only provided context and write polished, helpful answers in the requested language. "
                "Do not invent facts. If information is partial, provide what is known clearly and then suggest the official contact path. "
                "Avoid mechanical phrases like 'the context does not specify'.\n\n"
                f"Language rule: {language_guard.get(language, language_guard['english'])}\n"
                f"Language: {language}\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

        if language == "bangla":
            template = PROMPT_BANGLA
        elif language == "banglish":
            template = PROMPT_BANGLISH
        else:
            template = PROMPT_ENGLISH

        return (
            f"{language_guard.get(language, language_guard['english'])}\n\n"
            + template.format(context=context, question=question)
        )

    def _call_ollama(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the generated response text.

        Uses streaming so tokens arrive incrementally — reduces perceived latency.

        Args:
            prompt: Complete prompt string.

        Returns:
            Generated response text from Ollama, or empty string on failure.
        """
        import httpx, json

        timeout_sec = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        top_p = float(os.getenv("OLLAMA_TOP_P", "0.92"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "750"))

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": 1.08,
                "num_predict": num_predict,
            },
        }

        # Qwen3 may spend long in reasoning mode before producing the first
        # token. Disable it for chat latency and to avoid gateway timeouts.
        if self.model.lower().startswith("qwen3"):
            payload["think"] = False

        try:
            parts = []
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout_sec,
            ) as response:
                try:
                    response.raise_for_status()
                except Exception:
                    body = response.text.strip()
                    if body:
                        logger.error("❌ Ollama error response: %s", body[:1000])
                    raise
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            parts.append(token)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            return "".join(parts).strip()

        except httpx.ConnectError:
            if os.getenv("OLLAMA_AUTOSTART", "true").lower() == "true" and _ensure_ollama_running(self.base_url):
                try:
                    parts = []
                    with httpx.stream(
                        "POST",
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=timeout_sec,
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    parts.append(token)
                                if chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return "".join(parts).strip()
                except Exception as retry_exc:
                    logger.error("❌ Ollama retry failed: %s", retry_exc, exc_info=True)
            logger.error(
                f"❌ Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Run: ollama serve"
            )
            return ""
        except Exception as exc:
            logger.error(f"❌ Ollama call failed: {exc}", exc_info=True)
            return ""

    def _offline_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        language: str,
    ) -> dict | None:
        """
        Build a short extractive answer from retrieved chunks when Ollama is
        unavailable. This keeps the app useful in fallback-only mode.
        """
        if not retrieved_chunks:
            return None

        context_lines: list[str] = []
        source_urls: list[str] = []
        for chunk in retrieved_chunks[:3]:
            payload = chunk.get("payload", chunk)
            text = payload.get("text", "").strip()
            url = payload.get("url", "")
            title = payload.get("title", "").strip()
            if text:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                excerpt = " ".join(sentences[:2]).strip()
                if excerpt:
                    context_lines.append(f"{title}: {excerpt}" if title else excerpt)
            if url and url not in source_urls:
                source_urls.append(url)

        if not context_lines:
            return None

        lead = {
            "english": (
                "I couldn't reach the language model, but here is the most relevant "
                "information I found from UCB's corpus:"
            ),
            "bangla": (
                "আমি ভাষা মডেলে সংযোগ করতে পারিনি, তবে UCB-এর তথ্যভান্ডার থেকে "
                "সবচেয়ে প্রাসঙ্গিক তথ্য নিচে দিচ্ছি:"
            ),
            "banglish": (
                "Language model e connect korte parিনি, kintu UCB corpus theke "
                "sobar theke relevant information niche dilam:"
            ),
        }.get(language, "Here is the most relevant information I found:")

        answer = lead + "\n\n" + "\n\n".join(
            f"- {snippet}" for snippet in context_lines
        )
        return {
            "answer": answer,
            "language": language,
            "sources": source_urls,
            "fallback": True,
        }

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
        language: Optional[str] = None,
    ) -> dict:
        """
        Generate a grounded response for a user question using retrieved context.

        Full pipeline:
        1. Detect or accept language
        2. Check if any chunks meet the confidence threshold
        3. Format context from retrieved chunks
        4. Build language-appropriate prompt
        5. Call Ollama
        6. Return response with sources and metadata

        Args:
            question: User's question string.
            retrieved_chunks: List of retrieval results from UCBRetriever.
            language: Override language detection ("english"/"bangla"/"banglish").
                      If None, auto-detected from the question.

        Returns:
            Dict with keys:
              - answer: Generated response text
              - language: Detected/used language
              - sources: List of source URLs cited
              - fallback: True if fallback message was used
        """
        # Step 1: Language detection
        detected_lang = language or detect_language(question)
        logger.info(f"🌐 Language: {detected_lang} | Query: {question[:60]}")

        # Step 2: Check confidence — are any results above threshold?
        has_good_context = any(
            r.get("rerank_score", r.get("rrf_score", 0.0)) >= CONFIDENCE_THRESHOLD
            for r in retrieved_chunks
        )

        # If no good context, return fallback immediately
        if not retrieved_chunks or not has_good_context:
            logger.warning("⚠️  No high-confidence results. Returning fallback.")
            return {
                "answer": FALLBACK_MESSAGES[detected_lang],
                "language": detected_lang,
                "sources": [],
                "fallback": True,
            }

        # Step 3: Keep only the strongest few chunks and format compact context.
        ranked_chunks = sorted(
            retrieved_chunks,
            key=lambda r: r.get("rerank_score", r.get("rrf_score", 0.0)),
            reverse=True,
        )

        # For credit-card application intent, ensure documentation chunks are
        # included in context so the model can provide concrete next steps.
        q_lower = question.lower()
        credit_card_apply_intent = (
            ("credit" in q_lower and "card" in q_lower)
            and ("apply" in q_lower or "application" in q_lower or "documents" in q_lower)
        )
        if credit_card_apply_intent:
            preferred = []
            others = []
            for r in ranked_chunks:
                url = r.get("payload", {}).get("url", "").lower()
                if "documentation-requirements" in url:
                    preferred.append(r)
                else:
                    others.append(r)
            ranked_chunks = preferred + others

        ranked_chunks = ranked_chunks[:MAX_CONTEXT_CHUNKS]
        context_str, source_urls = format_context(ranked_chunks)

        # Step 4: Build the prompt
        prompt = self._build_prompt(question, context_str, detected_lang)

        # Step 5: Call Ollama
        logger.info(f"📤 Sending prompt to Ollama ({self.model})...")
        raw_answer = self._call_ollama(prompt)

        # Step 6: Handle empty response (Ollama failed)
        if not raw_answer:
            offline = self._offline_answer(question, retrieved_chunks, detected_lang)
            if offline is not None:
                logger.warning(
                    "⚠️  Ollama unavailable; returning extractive offline answer."
                )
                return offline
            return {
                "answer": FALLBACK_MESSAGES[detected_lang],
                "language": detected_lang,
                "sources": source_urls,
                "fallback": True,
            }

        # Clean markdown: keep bold (**text**) for structured answers, remove only noise
        import re
        clean_answer = re.sub(r'(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)', r'\1', raw_answer)  # italic only → plain
        clean_answer = re.sub(r'^#{1,6}\s+', '', clean_answer, flags=re.MULTILINE)           # # headers → remove prefix
        clean_answer = re.sub(r'`([^`]+)`', r'\1', clean_answer)                             # inline code → plain
        clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)                               # max 2 blank lines
        clean_answer = clean_answer.strip()

        # Normalize common robotic phrases from smaller local models.
        lower_answer = clean_answer.lower()
        robotic_missing_context = (
            "provided context does not" in lower_answer
            or "context does not contain" in lower_answer
            or "context does not include" in lower_answer
            or "the context does not" in lower_answer
        )
        if robotic_missing_context:
            replacements = {
                "english": (
                    "I do not have this specific detail in the currently available UCB information. "
                    "Please contact UCB customer service at 16419 for confirmed support."
                ),
                "bangla": (
                    "এই নির্দিষ্ট তথ্যটি বর্তমানে আমার কাছে থাকা UCB তথ্যভান্ডারে নেই। "
                    "নিশ্চিত তথ্যের জন্য UCB গ্রাহকসেবায় 16419 নম্বরে যোগাযোগ করুন।"
                ),
                "banglish": (
                    "Ei specific information ta amar current UCB dataset e nai. "
                    "Confirmed support er jonno UCB customer service 16419 e contact korun."
                ),
            }
            clean_answer = replacements.get(detected_lang, replacements["english"])

        # Only truncate very long plain-text (non-structured) responses
        has_structure = bool(re.search(r'^\s*[-•\d]\s+|\*\*', clean_answer, flags=re.MULTILINE))
        if not has_structure:
            sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
            if len(sentences) > 8:
                clean_answer = ' '.join(sentences[:7]).strip()

        logger.info("✅ Response generated successfully.")
        return {
            "answer": clean_answer,
            "language": detected_lang,
            "sources": source_urls,
            "fallback": False,
        }

    def health_check(self) -> bool:
        """
        Verify that Ollama is running and the model is available.

        Returns:
            True if Ollama is reachable and the model is loaded, False otherwise.
        """
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            is_available = any(self.model in m for m in models)
            if is_available:
                logger.info(f"✅ Ollama healthy. Model '{self.model}' is available.")
            else:
                logger.warning(
                    f"⚠️  Ollama running but model '{self.model}' not found. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
            return is_available
        except Exception as exc:
            logger.error(f"❌ Ollama health check failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm = UCBOllamaLLM()

    # Verify connection
    if not llm.health_check():
        print("Ollama is not running or model not available. Exiting.")
        sys.exit(1)

    # Simulate retrieved chunks
    mock_chunks = [
        {
            "payload": {
                "text": "UCB Bank offers personal loans, home loans, car loans, and SME loans "
                        "at competitive interest rates starting from 9% per annum.",
                "url": "https://www.ucb.com.bd/loans",
                "title": "UCB Bank Loan Products",
                "language": "english",
            },
            "rerank_score": 0.85,
        }
    ]

    for q, lang in [
        ("What are UCB Bank's loan products?", "english"),
        ("UCB ব্যাংকের লোনের সুদের হার কত?", "bangla"),
        ("UCB er loan er details ki?", "banglish"),
    ]:
        print(f"\n{'='*60}\nQ: {q}")
        result = llm.generate(q, mock_chunks, language=lang)
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {result['sources']}")
