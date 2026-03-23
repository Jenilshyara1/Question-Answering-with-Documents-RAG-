import base64
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import httpx
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.text_generation.chat_model import Chatbot
from src.text_generation.query import Query
from setup_logger import init_logger

try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY"))
except ImportError:
    LANGFUSE_ENABLED = False

init_logger()
logger = logging.getLogger("app")

PDF_PARSER_URL = os.getenv("PDF_PARSER_URL", "http://pdf-parser:7001")

bot: Chatbot = None
query: Query = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, query
    bot = Chatbot()
    query = Query()
    await query.setup()
    logger.info("Model and vector store loaded successfully.")
    yield
    await query.async_client.close()
    await query.embeddings.close()
    await query.sparse_embeddings.close()


app = FastAPI(lifespan=lifespan)


def _get_langfuse_handler():
    if not LANGFUSE_ENABLED:
        return None
    return LangfuseCallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )


class Prompt(BaseModel):
    text: str
    doc_id: Optional[str] = None


class Text(BaseModel):
    text: str
    filename: Optional[str] = "unknown"


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/chat_response")
async def chat_response(prompt: Prompt):
    logger.info(f"prompt: {prompt.text}")
    handler = _get_langfuse_handler()
    try:
        context = await query.query_search(prompt.text, doc_id=prompt.doc_id)
        response = await bot.generate_response(prompt.text, context)
        return JSONResponse({"response": response})
    except Exception as e:
        logger.error(f"Error occurred while generating the response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response")
    finally:
        if handler:
            handler.flush()


@app.post("/chat_response_stream")
async def chat_response_stream(prompt: Prompt):
    logger.info(f"stream prompt: {prompt.text}")
    handler = _get_langfuse_handler()

    async def token_generator() -> AsyncIterator[str]:
        try:
            context = await query.query_search(prompt.text, doc_id=prompt.doc_id)
            async for token in bot.stream_response(prompt.text, context):
                yield token
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
        finally:
            if handler:
                handler.flush()

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post("/create_embeddings")
async def create_embeddings(text: Text):
    try:
        doc_id = await query.create_embeddings(text.text, text.filename)
        logger.info(f"Embeddings created for file '{text.filename}', doc_id={doc_id}")
        return JSONResponse({"create_embeddings": True, "doc_id": doc_id})
    except Exception as e:
        logger.error(f"Error occurred while creating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create embeddings")


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        async with httpx.AsyncClient(timeout=300.0) as client:
            parse_resp = await client.post(
                f"{PDF_PARSER_URL}/parse",
                json={"pdf_bytes": pdf_b64, "filename": file.filename},
            )
            parse_resp.raise_for_status()
        text = parse_resp.json()["text"]
        doc_id = await query.create_embeddings(text, file.filename)
        logger.info(f"PDF uploaded and embedded: '{file.filename}', doc_id={doc_id}")
        return JSONResponse({"create_embeddings": True, "doc_id": doc_id})
    except Exception as e:
        logger.error(f"Error processing PDF '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process PDF")
