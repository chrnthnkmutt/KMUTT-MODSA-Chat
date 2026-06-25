# MOD-SA RAG Starter

Python RAG system for the MOD-SA KMUTT Student Affairs chatbot.
Two decoupled sides that meet only at `chunks/`:

- **🟦 Data pipeline (`data/`)** — turns source documents into clean, chunked JSON
- **🟩 RAG app (`modsa_rag/`)** — indexes `chunks/` and answers questions with citations

Stack: FastAPI · LangChain · Chroma · OpenAI-compatible LLM/embeddings (Ollama by
default) · Typhoon OCR for scanned Thai PDFs.

## Setup

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # RAG app deps
pip install -r data/requirements.txt     # data pipeline deps (for the data team)
cp .env.example .env
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## Configure `.env`

Copy `.env.example` → `.env` and adjust. Defaults target a local Ollama:

```env
LLM_BASE_URL="http://localhost:11434/v1"
LLM_API_KEY="ollama"
LLM_MODEL="minimax-m3:cloud"

EMBEDDING_BASE_URL="http://localhost:11434/v1"
EMBEDDING_API_KEY="ollama"
EMBEDDING_MODEL="bge-m3:latest"

# RAG storage and ingestion
CHROMA_DIR="chroma_db"
CHROMA_COLLECTION="modsa_kmutt"
RAG_SOURCE_PATHS="chunks"
CHUNK_SIZE="1000"
CHUNK_OVERLAP="150"
RETRIEVAL_K="4"

# API
APP_HOST="127.0.0.1"
APP_PORT="8000"

# Data pipeline · Typhoon OCR (scanned PDFs only)
TYPHOON_OCR_API_KEY="your-typhoon-ocr-key"
```

Using Ollama? Pull the embedding model first: `ollama pull bge-m3`.

## Prepare knowledge (data side)

Put official KMUTT documents in `data/raw/<category>/`
(categories: `registration`, `fees`, `academic_rules`, `scholarship`,
`dormitory`, `academic_calendar`, `others`). Supported: `.pdf`, `.txt`, `.md`.

Then run the pipeline from the repo root:

```bash
python -m data.pipeline.triage      # see what each file needs
python -m data.pipeline.normalize   # -> data/processed/ (clean Markdown)
python -m data.pipeline.chunk       # -> chunks/ (JSON + metadata)
```

Edit `data/sources.json` to enrich citation metadata (title/department/url), then
re-run `chunk`. Full guide: `data/pipeline/README.md`.

## Run the app

### For MacOS and Linux

```bash
uvicorn modsa_rag.api:app --reload --host 127.0.0.1 --port 8000
```

### For Windows users

```bash
python -m uvicorn modsa_rag.api:app --reload --host 127.0.0.1 --port 8000
```

## Ask With Curl

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"วันไหว้ครูคือวันไหน"}' \
| jq
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "AcademicCalendar2026-2569TH2",
      "title": "ปฏิทินการศึกษา ประจำปีการศึกษา 2569",
      "department": "สำนักงานทะเบียนนักศึกษา",
      "page": 2
    }
  ]
}
```

## Useful Endpoints

- `GET /health` checks service status and the last ingestion result.
- `POST /ask` asks the RAG chatbot.
- `POST /reindex` forces a full rebuild of the Chroma collection.

## Project Layout

```text
data/                 🟦 DATA side
  pipeline/           triage / normalize / chunk / clean
  raw/                source documents (by category)
  processed/          generated clean Markdown
  sources.json        hand-filled citation metadata
  requirements.txt    data pipeline deps
chunks/               📦 handoff (data -> rag)
modsa_rag/            🟩 RAG side
  api.py              FastAPI endpoints
  config.py           environment-based settings
  ingest.py           chunks JSON loader + indexing
  rag.py              retrieval + prompting
chroma_db/            generated Chroma vector database
requirements.txt      RAG app deps
```

## Notes

- Do not commit `.env`, API keys, tokens, or private student data.
- Use official university sources for production knowledge.
- High-risk data (fees, dates, scholarship eligibility) must be human-verified
  against the original document before use.
- If the retrieved context is insufficient, the assistant is instructed to say it
  does not have enough information instead of guessing.
```
