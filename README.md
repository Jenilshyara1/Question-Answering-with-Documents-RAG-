# Question Answering with Documents — RAG Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot. Upload documents, ask questions, get answers grounded in your content.

**Stack:** FastAPI · Streamlit · Qdrant (hybrid BM25 + dense search) · OpenAI-compatible LLM · FlashRank reranking · GLM-OCR (PDF parsing) · Langfuse observability

---

## Architecture

```
Streamlit UI  →  FastAPI backend  →  embedding-service  (all-MiniLM-l6-v2 + BM25)
                      │                     │
                      │                     └──  Qdrant  (vector store, hybrid search)
                      │
                      ├── [PDF upload]  →  pdf-parser  →  glm-ocr / vLLM  (GLM-OCR)
                      │
                      └──  OpenAI API  (or any OpenAI-compatible endpoint)
```

| Service | Port | Description |
|---|---|---|
| Streamlit UI | 8000 | Chat interface with document upload (`.txt` and `.pdf`) |
| FastAPI backend | 5858 | Orchestration, retrieval, LLM calls |
| Embedding service | 7000 | Dense + sparse embedding model server |
| Qdrant | 6333 | Persistent vector database |
| pdf-parser | 7001 | PDF → page images → OCR text (wraps GLM-OCR) |
| glm-ocr | 8080 | GLM-OCR model served via vLLM (OpenAI-compatible) |
| Langfuse *(optional)* | 3000 | Trace and observe LLM calls |

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An OpenAI API key **or** a local LLM endpoint (Ollama, vLLM, LM Studio)
- A GPU is recommended for the GLM-OCR service (CPU works but is ~10× slower)

---

## Quickstart

### 1. Clone and configure

```bash
git clone <repo-url>
cd Question-Answering-with-Documents-RAG-

cp .env.example .env
```

Open `.env` and set your values:

```env
OPENAI_API_KEY=sk-...           # required — your OpenAI key
OPENAI_BASE_URL=                # leave empty for OpenAI cloud
OPENAI_MODEL=gpt-4o-mini        # or any model your endpoint supports

# defaults work as-is for Docker:
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=documents
EMBEDDING_SERVICE_URL=http://embedding-service:7000

# PDF parsing (defaults work as-is for Docker):
PDF_PARSER_URL=http://pdf-parser:7001
GLM_OCR_URL=http://glm-ocr:8080

# leave empty to disable Langfuse:
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://langfuse-server:3000
```

### 2. Build and start

```bash
docker-compose up --build
```

> The first build downloads model weights for the embedding service and GLM-OCR — this takes several minutes. Subsequent starts are fast (weights are cached in Docker volumes).

Services start in dependency order automatically:
1. Qdrant starts and passes healthcheck
2. Embedding service loads models and passes healthcheck
3. GLM-OCR (vLLM) downloads weights and passes healthcheck
4. PDF-parser starts
5. FastAPI backend starts
6. Streamlit UI starts

### 3. Open the app

**http://localhost:8000**

Upload a `.txt` or `.pdf` file from the sidebar, then ask questions in the chat.

#### GPU support for GLM-OCR (recommended)

To enable GPU acceleration for the `glm-ocr` service, uncomment the `deploy` block in `docker-compose.yaml`:

```yaml
glm-ocr:
  # ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Using a local LLM (Ollama / vLLM / LM Studio)

Set these in `.env` before starting:

```env
OPENAI_API_KEY=ollama            # any non-empty string
OPENAI_BASE_URL=http://host.docker.internal:11434/v1   # adjust port to your server
OPENAI_MODEL=llama3.2            # model name as your server expects it
```

---

## Optional: Langfuse observability

Langfuse v3 runs Postgres, ClickHouse, MinIO, and Redis alongside the web UI and worker.

**1. Generate secure secrets and update `.env`:**
```bash
openssl rand -hex 32   # → LANGFUSE_ENCRYPTION_KEY
```
Set `LANGFUSE_SALT`, `NEXTAUTH_SECRET`, and the database passwords in `.env`. See the `# CHANGEME` comments in `docker-compose.langfuse.yml`.

**2. Start the full stack:**
```bash
docker-compose -f docker-compose.yaml -f docker-compose.langfuse.yml up --build
```

**3. Create your first account and project at http://localhost:3000**, then copy the project's API keys into `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**4. Restart the `api` service to pick up the keys:**
```bash
docker-compose restart api
```

Every LLM call is now traced automatically in the Langfuse UI.

---

## Running tests

No live services needed — all external dependencies are mocked.

```bash
pip install pytest pytest-asyncio
pytest test/ -v
```

---

## Useful commands

```bash
# Rebuild a single service after code changes
docker-compose up --build api

# Stream logs from a service
docker-compose logs -f api
docker-compose logs -f embedding-service
docker-compose logs -f pdf-parser
docker-compose logs -f glm-ocr

# Stop all containers
docker-compose down

# Stop and wipe all data (including Qdrant vectors)
docker-compose down -v
```

---

## API endpoints

**FastAPI backend** (`localhost:5858`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/create_embeddings` | Embed a text document → returns `doc_id` |
| `POST` | `/upload_pdf` | Parse + embed a PDF (multipart) → returns `doc_id` |
| `POST` | `/chat_response` | Single (non-streaming) answer |
| `POST` | `/chat_response_stream` | Streaming answer |

Pass `doc_id` on chat requests to search within a single document; omit to search all.

**Embedding service** (`localhost:7000`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Ready check (healthy = models loaded) |
| `POST` | `/embed/dense` | Dense float vectors |
| `POST` | `/embed/sparse` | Sparse BM25 vectors |

**PDF parser** (`localhost:7001`)

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Ready check |
| `POST` | `/parse` | `{"pdf_bytes": "<base64>", "filename": "..."}` | OCR a PDF → returns `{"text": "..."}` |
