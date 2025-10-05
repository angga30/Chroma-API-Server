from typing import Optional
from config import settings
from service.chromadb import ChromaDBService
from service.pinecone_service import PineconeService


_singleton_cache = {}


def get_rag_service(rag_server: Optional[str] = None, embedding_model: Optional[str] = None):
    """Return a RAG backend service instance based on configuration or parameter."""
    backend = (rag_server or settings.RAG_SERVER).lower()
    key = f"{backend}:{embedding_model or ''}"
    if key in _singleton_cache:
        return _singleton_cache[key]

    if backend == "pinecone":
        svc = PineconeService(embedding_model=embedding_model)
    else:
        svc = ChromaDBService(embedding_model=embedding_model)

    _singleton_cache[key] = svc
    return svc