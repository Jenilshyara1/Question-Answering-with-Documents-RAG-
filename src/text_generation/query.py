import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from src.text_generation.embedding_client import RemoteDenseEmbeddings, RemoteSparseEmbeddings

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:7000")
EMBEDDING_DIM = 384


class Query:
    def __init__(self) -> None:
        self.embeddings = RemoteDenseEmbeddings(base_url=EMBEDDING_SERVICE_URL)
        self.sparse_embeddings = RemoteSparseEmbeddings(base_url=EMBEDDING_SERVICE_URL)
        self.async_client = AsyncQdrantClient(url=QDRANT_URL)
        self.vector_store = QdrantVectorStore(
            async_client=self.async_client,
            collection_name=QDRANT_COLLECTION,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        self.reranker = FlashrankRerank(top_n=5)

    async def setup(self) -> None:
        await self._ensure_collection()

    async def _ensure_collection(self) -> None:
        result = await self.async_client.get_collections()
        existing = [c.name for c in result.collections]
        if QDRANT_COLLECTION not in existing:
            await self.async_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={"dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
            )

    async def create_embeddings(self, text: str, filename: str = "unknown") -> str:
        doc_id = str(uuid.uuid4())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        docs = splitter.create_documents([text])
        for doc in docs:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["filename"] = filename
        await self.vector_store.aadd_documents(docs)
        return doc_id

    async def query_search(self, prompt: str, doc_id: str = None) -> str:
        search_kwargs = {"k": 10}
        if doc_id:
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            )
        retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.vector_store.as_retriever(search_kwargs=search_kwargs),
        )
        docs = await retriever.ainvoke(prompt)
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
