import os
from typing import AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the provided context to answer the user's question. "
    "If the answer is not in the context, say so clearly but still try to help "
    "using your own knowledge, and indicate when you are doing so."
)


class Chatbot:
    def __init__(self) -> None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None

        kwargs = dict(model=model, api_key=api_key)
        if base_url:
            kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**kwargs)

    def _build_messages(self, prompt: str, context: str) -> list:
        user_content = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]

    async def generate_response(self, prompt: str, context: str) -> str:
        messages = self._build_messages(prompt, context)
        result = await self.llm.ainvoke(messages)
        return result.content

    async def stream_response(self, prompt: str, context: str) -> AsyncIterator[str]:
        messages = self._build_messages(prompt, context)
        async for chunk in self.llm.astream(messages):
            yield chunk.content
