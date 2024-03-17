from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mistral_model import Chatbot

app = FastAPI()
bot = Chatbot()

class Prompt(BaseModel):
    text: str
    context: str

@app.post("/chat_response")
async def chat_response(prompt: Prompt):
    print("prompt:", prompt)
    response = bot.generate_response(prompt.text, prompt.context)
    return JSONResponse({"response": response})
