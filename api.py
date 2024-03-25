from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.text_generation.mistral_model import Chatbot
from src.text_generation.query import Query
from setup_logger import init_logger
import logging

init_logger()
logger=logging.getLogger("app")
app = FastAPI()

try:
    bot = Chatbot()
    query = Query()
    logger.info("Model loaded successfully..")
except Exception as e:
    logger.error(f"error occurred while loading the model:{e}",exc_info=True)

class Prompt(BaseModel):
    text: str

class Text(BaseModel):
    text: str

@app.post("/chat_response")
async def chat_response(prompt: Prompt):
    logger.info(f"prompt: {prompt.text}")
    try:
        context = query.query_search(query.db,prompt.text)
        response = bot.generate_response(prompt.text, context)
    except Exception as e:
        logger.error(f"error occurred while generating the response:{e}",exc_info=True)
    return JSONResponse({"response": response})

@app.post("/create_embeddings")
async def create_embeddings(text: Text):
    try:
        # logger.info(f"{text} {type(text)}")
        query.create_embeddings(text.text)
        logger.info("Embeddings created..")
    except Exception as e:
        logger.error(f"error occurred while creating embeddings:{e}",exc_info=True)
    return JSONResponse({"create_embeddings": True})
