import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from config import settings
from openai import OpenAI


class PineconeService:
    """Service wrapper for Pinecone to match the ChromaDBService interface."""

    def __init__(self, embedding_model: Optional[str] = None):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY) if settings.PINECONE_API_KEY else None
        self.embedding_model = embedding_model or settings.OPENAI_MODEL_NAME
        self._index_cache: Dict[str, Any] = {}

    def _index_name(self, name: str) -> str:
        prefix = settings.PINECONE_INDEX_PREFIX
        return f"{prefix}-{name}" if prefix else name

    def _get_or_create_index(self, name: str):
        if not self.pc:
            raise Exception("Pinecone API key not configured")
        index_name = self._index_name(name)
        if index_name in self._index_cache:
            return self._index_cache[index_name]

        if not self.pc.has_index(index_name):
            self.pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.embedding_model,
                    "field_map":{"text": "content"}
                }
            )

        index = self.pc.Index(index_name)
        self._index_cache[index_name] = index
        return index

    def get_or_create_collection(self, name: str):
        return self._get_or_create_index(name)

    def add_documents(self, collection_name: str, chunks: List[Dict[str, Any]]) -> List[str]:
        index = self._get_or_create_index(collection_name)
        ids = [chunk["id"] for chunk in chunks]
        docs = [chunk["content"] for chunk in chunks]
        metas = [chunk["metadata"] for chunk in chunks]

        vectors = []
        for i, vid in enumerate(ids):
            meta = metas[i].copy() if metas[i] else {}
            meta["content"] = docs[i]
            vectors.append({"id": vid, "content": docs[i], "metadata": json.dumps(meta)})

        index.upsert_records("default-namespace", vectors)
        return ids

    def update_documents(self, collection_name: str, doc_ids: List[str], chunks: List[Dict[str, Any]]) -> int:
        index = self._get_or_create_index(collection_name)
        # Delete by filter (parent_document_id)
        for doc_id in doc_ids:
            try:
                index.delete(filter={"parent_document_id": doc_id})
            except Exception:
                pass
        # Add new chunks
        self.add_documents(collection_name, chunks)
        return len(chunks)

    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> None:
        index = self._get_or_create_index(collection_name)
        index.delete(ids=doc_ids)

    def search_similarity(
        self,
        collection_name: str,
        query: str,
        n_results: int = settings.DEFAULT_SEARCH_RESULTS,
        threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List]:
        start_ts = time.time()
        print(
            f"[Pinecone] search start at {datetime.now().isoformat()} "
            f"collection='{collection_name}' query='{query}' n_results={n_results}"
        )
        index = self._get_or_create_index(collection_name)
        duration_ms = (time.time() - start_ts) * 1000
        print(
            f"[Pinecone] get or create collection in {duration_ms:.2f} ms, "
        )
        res = index.search(
            namespace="default-namespace", 
            query={
                "inputs": {"text": query}, 
                "top_k": n_results
            },
            fields=["content", "metadata"]
        )
        print(res)
        duration_ms = (time.time() - start_ts) * 1000
        print(
            f"[Pinecone] search to rag in {duration_ms:.2f} ms, "
            f"hits={len(res['result']['hits'])}"
        )
        filtered_results = {"ids": [], "distances": [], "metadatas": [], "documents": []}
        for match in res["result"]["hits"]:
            filtered_results["ids"].append(match["_id"])
            filtered_results["distances"].append(match["_score"])
            meta = match["fields"].get("metadata") or {}
            filtered_results["metadatas"].append(json.loads(meta))
            filtered_results["documents"].append(match["fields"].get("content", ""))
        duration_ms = (time.time() - start_ts) * 1000
        print(
            f"[Pinecone] search finished in {duration_ms:.2f} ms, "
            f"hits={len(filtered_results['ids'])}"
        )
        return filtered_results

    def list_collections(self) -> List[str]:
        if not self.pc:
            return []
        return [idx.name for idx in self.pc.list_indexes()]

    def delete_collection(self, name: str) -> None:
        if not self.pc:
            return
        index_name = self._index_name(name)
        self.pc.delete_index(index_name)