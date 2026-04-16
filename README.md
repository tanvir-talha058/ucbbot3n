# UCB Bank RAG Chatbot

A production-grade, multilingual Retrieval-Augmented Generation (RAG) chatbot for **UCB Bank Bangladesh**, supporting English, Bangla, and Banglish. Embedded as a floating chat bubble on the bank's website. 100% free and open-source — no paid APIs.

---

## 1. Project Overview

This chatbot scrapes the UCB Bank website, indexes all content into a vector database, and answers customer questions using a locally-run LLM. It uses hybrid search (dense + BM25) with cross-encoder reranking for high-precision retrieval.

**Key capabilities:**
- Multilingual: English, Bangla (বাংলা), and Banglish (mixed)
- All computation runs locally — no cloud APIs needed
- Single `<script>` tag embed on any webpage
- Admin dashboard for monitoring and testing

---

## 2. Architecture Diagram

```
User Query (EN / BN / BL)
        │
        ▼
┌─────────────────┐
│  Language Detect│  (langdetect + Unicode heuristics)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Rewrite  │  (LangChain ConversationMemory — last 5 turns)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Hybrid Retrieval             │
│  Dense: multilingual-e5-large (GPU) │
│  Sparse: BM25 (rank-bm25)           │
│  Fusion: Reciprocal Rank Fusion     │
│  Store: Qdrant (Docker)             │
└────────┬─────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Re-Ranker      │  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation │  Qwen2.5 via Ollama (local)
│  Prompt by lang │  EN / BN / BL templates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │  → JSON response with sources
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chat Widget    │  Vanilla JS floating bubble (no deps)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Phoenix Monitor│  Arize Phoenix (local traces, http://localhost:6006)
└─────────────────┘
```

---

## 3. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12 | |
| CUDA | 12.6 | NVIDIA RTX 4070 or better |
| Docker Desktop | Latest | Must be running |
| Ollama | Latest | `ollama pull qwen2.5` |
| Tesseract | 5.3.3 | With Bengali language pack |
| RAM | 32 GB+ | For embeddings + LLM |
| VRAM | 16 GB | RTX 4070 recommended |

---

## 4. Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/Nafiyaahmed/-ucb-rag-chatbot.git
cd ucb-rag-chatbot

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Copy environment variables
cp .env.example .env

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install PyTorch with CUDA 12.1 GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install Playwright browser
playwright install chromium

# 7. Start Docker services (Qdrant + Phoenix)
docker-compose up -d

# 8. Pull Ollama model (if not already done)
ollama pull qwen2.5
```

---

## 5. How to Run Locally (Step by Step)

```bash
# Step 1: Start Docker services
docker-compose up -d

# Step 2: Wait ~15 seconds for Qdrant to initialise
# Verify: http://localhost:6333/dashboard

# Step 3: Scrape UCB Bank website
python scraper/crawler.py

# Step 4: Clean and chunk text
python preprocessing/chunker.py

# Step 5: Compute embeddings (GPU accelerated)
python embeddings/embedder.py

# Step 6: Index into Qdrant
python vectorstore/qdrant_store.py

# Step 7: Start the FastAPI server
uvicorn api.main:app --reload --port 8000

# Step 8: Open the test UI (in a new terminal or browser)
# Just double-click: ui/test.html
# Or open the admin panel: ui/admin.html
```

### Quick Start (single command)

If `docker-compose` is unavailable on your machine, use the helper script below.
It ensures Qdrant is running and then starts the FastAPI API.

```bash
./start_local.sh
```

Optional flags:

```bash
# only ensure Qdrant is up, then exit
./start_local.sh --qdrant-only

# start API without uvicorn auto-reload
./start_local.sh --no-reload
```

### Run on a remote VM over SSH

If your local machine has no GPU, run the full stack on the VM instead of locally.

```bash
ssh tanvir@172.23.10.207
cd /path/to/bot\ version\ 3n
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5
python app.py
```

For a headless SSH session, set `OPEN_BROWSER=false` in `.env` or export it before launching. If you want to access the VM from your local browser, use SSH port forwarding and open `http://127.0.0.1:8080` on your local machine.

```bash
# On your local machine, create the tunnel:
ssh -L 8080:127.0.0.1:8080 tanvir@172.23.10.207

# Then open this in your local browser:
http://127.0.0.1:8080
```

If port `8080` is already in use locally, pick another local port, for example `http://127.0.0.1:8090` with `ssh -L 8090:127.0.0.1:8080 tanvir@172.23.10.207`.

---

## 6. How to Test the UI Locally

1. Make sure `uvicorn api.main:app --reload --port 8000` is running.
2. Double-click `ui/test.html` — no server needed, opens directly in browser.
3. The chat widget will appear bottom-right.
4. Switch languages using the **EN / বাং / BL** buttons in the top-right.
5. Open `ui/admin.html` to see stats, health, and test queries.

**Test queries:**
- English: `What are UCB Bank's loan products?`
- Bangla: `UCB ব্যাংকের সুদের হার কত?`
- Banglish: `UCB er loan nite ki ki lagbe?`

---

## 7. How to Scrape and Update Data

To refresh the knowledge base after UCB Bank updates their website:

```bash
# Option 1: Run each step manually
python scraper/crawler.py
python preprocessing/chunker.py
python embeddings/embedder.py
python vectorstore/qdrant_store.py

# Option 2: Trigger via API (runs in background)
curl -X POST http://localhost:8000/scrape
```

---

## 8. How to Deploy on a Server

```bash
# On Ubuntu/Debian server:

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5

# Clone and setup
git clone https://github.com/Nafiyaahmed/-ucb-rag-chatbot.git
cd ucb-rag-chatbot
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set CORS_ORIGINS=https://www.ucb.com.bd, LOCAL_TEST_MODE=false

# Start services
docker-compose up -d

# Run the pipeline
python scraper/crawler.py && python preprocessing/chunker.py && \
python embeddings/embedder.py && python vectorstore/qdrant_store.py

# Start API (use gunicorn in production)
pip install gunicorn
gunicorn api.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with systemd / PM2 for auto-restart
```

---

## 9. How to Embed Widget on Website

Add a **single script tag** before `</body>` on any page:

```html
<!-- UCB Bank Chat Widget -->
<link rel="stylesheet" href="https://your-cdn.com/chatbot.css">
<script src="https://your-cdn.com/chatbot.js"></script>
```

Or host the files on your server:

```html
<link rel="stylesheet" href="/static/chatbot.css">
<script src="/static/chatbot.js"></script>
```

**To change the API URL**, edit the top of `widget/chatbot.js`:

```javascript
var CONFIG = {
  apiUrl: "https://api.ucb.com.bd",  // ← change this
  ...
};
```

---

## 10. API Documentation

The FastAPI server auto-generates interactive docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /health`
Returns health status of all services.

```json
{
  "status": "ok",
  "components": {
    "qdrant": {"status": "ok", "points": 1234},
    "ollama": {"status": "ok", "model": "qwen2.5"},
    "data": {"status": "ok"}
  },
  "version": "1.0.0"
}
```

#### `POST /chat`
Main chat endpoint.

**Request:**
```json
{
  "query": "What are UCB loan products?",
  "language": "english",
  "session_id": "user_abc123"
}
```

**Response:**
```json
{
  "answer": "UCB Bank offers personal loans, home loans...",
  "language": "english",
  "sources": ["https://www.ucb.com.bd/loans"],
  "fallback": false,
  "session_id": "user_abc123",
  "retrieved_count": 5
}
```

#### `GET /stats`
Collection statistics.

```json
{
  "total_chunks": 1500,
  "language_breakdown": {
    "english": {"count": 900, "percentage": 60.0},
    "bangla":  {"count": 500, "percentage": 33.3},
    "banglish": {"count": 100, "percentage": 6.7}
  },
  "collection_name": "ucb_bank",
  "qdrant_url": "http://localhost:6333"
}
```

#### `POST /scrape`
Trigger a background re-scrape.

```json
{
  "status": "queued",
  "message": "Re-scrape started in the background..."
}
```

---

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot connect to Qdrant` | Run `docker-compose up -d`, wait 15s |
| `Ollama model not found` | Run `ollama pull qwen2.5` |
| `CUDA not available` | Install CUDA 12.x drivers, reinstall torch with `--index-url https://download.pytorch.org/whl/cu121` |
| `playwright install` fails | Run `playwright install chromium --with-deps` |
| Widget shows "Cannot connect" | Ensure `uvicorn api.main:app --port 8000` is running |
| Empty chat responses | Check `/health` endpoint; verify Qdrant has data |
| Bengali text rendering badly | Ensure `Noto Sans Bengali` font is installed on the OS |
| Tesseract not found | Set `TESSERACT_PATH` in `.env` to correct path |

---

## 12. Project Structure

```
ucb-rag-chatbot/
├── config/settings.py          # All config from .env
├── scraper/crawler.py          # Playwright + BS4 web scraper
├── preprocessing/chunker.py    # Text cleaning, chunking, lang detection
├── embeddings/embedder.py      # multilingual-e5-large GPU embeddings
├── vectorstore/qdrant_store.py # Qdrant hybrid index (dense + BM25)
├── retrieval/retriever.py      # Hybrid search + CrossEncoder reranking
├── llm/ollama_llm.py           # Qwen2.5 with multilingual prompts
├── pipeline/rag_pipeline.py    # LangChain orchestration + memory
├── api/main.py                 # FastAPI server (4 endpoints)
├── widget/
│   ├── chatbot.js              # Vanilla JS floating bubble widget
│   └── chatbot.css             # UCB Bank themed styles
├── ui/
│   ├── test.html               # Local test page (no server needed)
│   └── admin.html              # Admin dashboard
├── monitoring/phoenix_monitor.py # Arize Phoenix tracing
├── tests/test_pipeline.py      # Unit + integration tests
├── data/
│   ├── raw/ucb_raw.json        # Scraped pages (git-ignored)
│   └── processed/ucb_chunks.json # Processed chunks (git-ignored)
├── docker-compose.yml          # Qdrant + Phoenix services
├── requirements.txt            # All Python dependencies
├── .env.example                # Template environment variables
├── .gitignore
└── README.md
```

---

*Built with open-source tools. No paid APIs. Runs entirely on local hardware.*
