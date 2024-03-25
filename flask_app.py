from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.text_generation.mistral_model import Chatbot
from setup_logger import init_logger
logger = init_logger("info","logs","app.log")
app = FastAPI()
try:
    bot = Chatbot()
    logger.info("Model loaded successfully..")
except Exception as e:
    logger.error(f"error occurred while loading the model:{e}",exc_info=True)

class Prompt(BaseModel):
    text: str
    context: str

@app.post("/chat_response")
async def chat_response(prompt: Prompt):
    logger.info(f"prompt: {prompt.text}")
    try:
        response = bot.generate_response(prompt.text, prompt.context)
    except Exception as e:
        logger.error(f"error occurred while generating the response:{e}",exc_info=True)
    return JSONResponse({"response": response})