# MOD-SA RAG Starter

Simple Python RAG architecture for the MOD-SA KMUTT Student Affairs chatbot.

The starter uses:

- FastAPI for a small HTTP API callable by `curl`
- LangChain for document loading, splitting, retrieval, and prompting
- Chroma DB as the persistent vector store
- OpenAI-compatible chat and embedding APIs with separate base URLs and API keys
- Automatic ingestion on server startup when source files change

## Setup

### For MacOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### For Windows users

```bash
python -m venv .venv
.venv/Script/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your LLM and embedding provider settings.

Both model providers are OpenAI-compatible:

```env
LLM_BASE_URL=https://your-chat-provider.example/v1
LLM_API_KEY=your-chat-key
LLM_MODEL=your-chat-model

EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=your-embedding-model

# RAG storage and ingestion
CHROMA_DIR="chroma_db"
CHROMA_COLLECTION="modsa_kmutt"
RAG_SOURCE_PATHS="knowledge,MODsa-proposal_students.pdf"
CHUNK_SIZE="1000"
CHUNK_OVERLAP="150"
RETRIEVAL_K="4"

# API
APP_HOST="127.0.0.1"
APP_PORT="8000"
```

## Add Knowledge Files

Put official KMUTT/student-affairs knowledge files in `knowledge/`.

Supported starter formats:

- `.pdf`
- `.txt`
- `.md`
- `.json`

By default the app scans both `knowledge/` and `MODsa-proposal_students.pdf`.

The app does not require a human to run a separate ingestion command every time. On startup and before each `/ask` request, it fingerprints source files and refreshes Chroma only when files were added, removed, or changed.

## Run

```bash
uvicorn modsa_rag.api:app --reload --host 127.0.0.1 --port 8000
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
      "source": "MODsa-proposal_students.pdf",
      "page": 1
    }
  ]
}
```

## Useful Endpoints

- `GET /health` checks service status.
- `POST /ask` asks the RAG chatbot.
- `POST /reindex` forces a full rebuild of the Chroma collection.

## Project Layout

```text
modsa_rag/
  api.py        FastAPI endpoints
  config.py     environment-based settings
  ingest.py     automatic source scanning and indexing
  rag.py        LangChain RAG chain
knowledge/      place source documents here
chroma_db/      generated Chroma vector database
```

## Notes

- Do not commit `.env`, API keys, tokens, or private student data.
- Use official university sources for production knowledge.
- If the retrieved context is insufficient, the assistant is instructed to say it does not have enough information instead of guessing.
