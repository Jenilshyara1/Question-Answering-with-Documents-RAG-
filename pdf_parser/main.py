import base64
import logging
import os
from contextlib import asynccontextmanager

import fitz  # pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

GLM_OCR_URL = os.getenv("GLM_OCR_URL", "http://glm-ocr:8080")

logger = logging.getLogger("pdf_parser")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s: %(module)s: %(message)s",
)

glm_client: AsyncOpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global glm_client
    glm_client = AsyncOpenAI(base_url=f"{GLM_OCR_URL}/v1", api_key="dummy")
    logger.info(f"GLM-OCR client initialized (base_url={GLM_OCR_URL}/v1)")
    yield
    await glm_client.close()


app = FastAPI(lifespan=lifespan)


class ParseRequest(BaseModel):
    pdf_bytes: str  # base64-encoded PDF
    filename: str = "unknown.pdf"


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/parse")
async def parse_pdf(request: ParseRequest):
    logger.info(f"Parsing PDF: {request.filename}")
    try:
        raw = base64.b64decode(request.pdf_bytes)
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF '{request.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    page_texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()

        logger.info(f"OCR page {page_num + 1}/{len(doc)} of '{request.filename}'")
        try:
            response = await glm_client.chat.completions.create(
                model="zai-org/GLM-OCR",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }
                ],
                max_tokens=8192,
            )
            page_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"GLM-OCR failed on page {page_num + 1}: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"OCR failed on page {page_num + 1}")

        page_texts.append(page_text)

    doc.close()
    full_text = "\n\n".join(page_texts)
    logger.info(f"Successfully parsed '{request.filename}' ({len(doc)} pages)")
    return JSONResponse({"text": full_text})
