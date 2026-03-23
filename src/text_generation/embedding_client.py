from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from qdrant_client.models import SparseVector


class RemoteDenseEmbeddings(Embeddings):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._async_client = httpx.AsyncClient(timeout=30.0)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self._async_client.post(
            f"{self.base_url}/embed/dense",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    async def aembed_query(self, text: str) -> List[float]:
        results = await self.aembed_documents([text])
        return results[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = httpx.post(
            f"{self.base_url}/embed/dense",
            json={"texts": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def close(self) -> None:
        await self._async_client.aclose()


class RemoteSparseEmbeddings:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._async_client = httpx.AsyncClient(timeout=30.0)

    async def aembed_documents(self, texts: List[str]) -> List[SparseVector]:
        response = await self._async_client.post(
            f"{self.base_url}/embed/sparse",
            json={"texts": texts},
        )
        response.raise_for_status()
        data = response.json()["embeddings"]
        return [SparseVector(indices=d["indices"], values=d["values"]) for d in data]

    async def aembed_query(self, text: str) -> SparseVector:
        results = await self.aembed_documents([text])
        return results[0]

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        response = httpx.post(
            f"{self.base_url}/embed/sparse",
            json={"texts": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()["embeddings"]
        return [SparseVector(indices=d["indices"], values=d["values"]) for d in data]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]

    async def close(self) -> None:
        await self._async_client.aclose()
