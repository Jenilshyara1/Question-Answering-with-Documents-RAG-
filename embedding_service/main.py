from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastembed.sparse.bm25 import Bm25
from pydantic import BaseModel


dense_model: HuggingFaceEmbeddings = None
sparse_model: Bm25 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dense_model, sparse_model
    dense_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    sparse_model = Bm25("Qdrant/bm25")
    yield


app = FastAPI(lifespan=lifespan)


class TextsRequest(BaseModel):
    texts: List[str]


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/embed/dense")
async def embed_dense(request: TextsRequest):
    embeddings = dense_model.embed_documents(request.texts)
    return JSONResponse({"embeddings": embeddings})


@app.post("/embed/sparse")
async def embed_sparse(request: TextsRequest):
    results = list(sparse_model.embed(request.texts))
    embeddings = [
        {"indices": r.indices.tolist(), "values": r.values.tolist()}
        for r in results
    ]
    return JSONResponse({"embeddings": embeddings})
