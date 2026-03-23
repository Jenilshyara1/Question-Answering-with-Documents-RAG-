# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A RAG (Retrieval-Augmented Generation) chatbot that allows users to upload documents and ask questions about them. Uses hybrid BM25 + dense vector search (Qdrant), cross-encoder reranking (FlashRank), and an OpenAI-compatible LLM. Optionally instrumented with self-hosted Langfuse for observability.

## Architecture

```
Streamlit UI (port 8000)
    └── POST /create_embeddings  →  api (port 5858)
    └── POST /chat_response_stream  →  api
                                          ├── embedding-service (port 7000)  [dense + sparse vectors]
                                          ├── Qdrant (port 6333)             [vector store]
                                          └── OpenAI API / local LLM         [text generation]
```

**Services:**
| Service | Port | Description |
|---|---|---|
| `chatbot` | 8000 | Streamlit UI |
| `api` | 5858 | FastAPI backend — chat + embedding orchestration |
| `embedding-service` | 7000 | Dedicated model server: `all-MiniLM-l6-v2` (dense) + BM25 (sparse) |
| `qdrant` | 6333 | Vector database (hybrid dense + sparse) |
| `langfuse-server` | 3000 | Observability UI (optional) |

**Key modules:**
- `api.py` — FastAPI app; endpoints: `/chat_response`, `/chat_response_stream`, `/create_embeddings`, `/health`
- `streamlit_app.py` — Chat UI with sidebar file uploader and per-doc filtering
- `src/text_generation/query.py` — Async Qdrant hybrid search + FlashRank reranking
- `src/text_generation/chat_model.py` — `ChatOpenAI` wrapper (sync + async streaming)
- `src/text_generation/embedding_client.py` — HTTP clients for the embedding service
- `embedding_service/main.py` — FastAPI embedding server (loads models at startup)
- `setup_logger.py` — Logs to `logs/app.log` and console

---

## Prerequisites

- Docker + Docker Compose
- An OpenAI API key (or a local LLM endpoint, e.g. Ollama/vLLM/LM Studio)

---

## Setup

**1. Copy the env file and fill in your values:**
```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-...          # required
OPENAI_BASE_URL=               # leave empty for OpenAI cloud; set for local LLM
OPENAI_MODEL=gpt-4o-mini       # or any model your endpoint supports

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=documents

EMBEDDING_SERVICE_URL=http://embedding-service:7000

LANGFUSE_SECRET_KEY=           # leave empty to disable Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://langfuse-server:3000
```

---

## Running with Docker (primary method)

```bash
# Build and start all services (first build downloads embedding model weights — takes a few minutes)
docker-compose up --build

# Subsequent starts (no rebuild needed)
docker-compose up
```

Service startup order is enforced automatically:
1. `qdrant` starts and passes healthcheck
2. `embedding-service` starts, loads models, passes healthcheck
3. `api` starts (depends on both above)
4. `chatbot` starts

Open **http://localhost:8000** in your browser.

### With optional Langfuse observability

```bash
docker-compose -f docker-compose.yaml -f docker-compose.langfuse.yml up --build
```

Langfuse UI available at **http://localhost:3000**.

### Useful Docker commands

```bash
# Rebuild a single service after code changes
docker-compose up --build api

# View logs for a specific service
docker-compose logs -f api
docker-compose logs -f embedding-service

# Stop everything and remove containers
docker-compose down

# Stop and delete volumes (wipes Qdrant data)
docker-compose down -v
```

---

## Running locally (without Docker)

Requires Qdrant and the embedding service running separately, or pointing to remote instances.

**1. Start Qdrant (Docker only, or use Qdrant Cloud):**
```bash
docker run -p 6333:6333 qdrant/qdrant:v1.9.2
```

**2. Start the embedding service:**
```bash
pip install -r requirements_embedding.txt
uvicorn embedding_service.main:app --host 0.0.0.0 --port 7000
```

**3. Set environment variables for local development:**
```bash
export QDRANT_URL=http://localhost:6333
export EMBEDDING_SERVICE_URL=http://localhost:7000
export OPENAI_API_KEY=sk-...
```

**4. Start the FastAPI backend:**
```bash
pip install -r requirements_api.txt
uvicorn api:app --host 0.0.0.0 --port 5858
```

**5. Start the Streamlit frontend:**
```bash
pip install streamlit
streamlit run streamlit_app.py --server.port 8000
```

---

## Running Tests

Tests mock all external dependencies (Qdrant, embedding service, OpenAI) — no live services needed.

```bash
pip install pytest pytest-asyncio
pytest test/ -v

# Single test
pytest test/test_api.py::test_health -v
```

---

## API Reference

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness check |
| POST | `/chat_response` | `{"text": "...", "doc_id": "..."}` | Single response (non-streaming) |
| POST | `/chat_response_stream` | `{"text": "...", "doc_id": "..."}` | Streaming response |
| POST | `/create_embeddings` | `{"text": "...", "filename": "..."}` | Embed a document, returns `doc_id` |

`doc_id` is optional on chat endpoints — omit to search across all uploaded documents.

### Embedding service

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Ready check (healthy = models loaded) |
| POST | `/embed/dense` | `{"texts": [...]}` | Dense vectors (`list[list[float]]`) |
| POST | `/embed/sparse` | `{"texts": [...]}` | Sparse BM25 vectors |

---

## Dependency Files

| File | Used by |
|---|---|
| `requirements_api.txt` | `api` (FastAPI backend) |
| `requirements_embedding.txt` | `embedding-service` |
| `requirements_streamlit.txt` | `chatbot` (Streamlit frontend) |

---

## Notes

- `logs/` is gitignored; created at runtime by `setup_logger.py`.
- `mistral_model.py` and `llama_model.py` are legacy — not wired into the API.
- `test.py` at root is a standalone Qwen2-VL experiment — unrelated to the main pipeline.
- `mistral_model.py` and `llama_model.py` are legacy — not wired into the API.
