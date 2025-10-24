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
        self._index_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry_hours = 2

    def _index_name(self, name: str) -> str:
        name = name.replace("_", "-")
        prefix = settings.PINECONE_INDEX_PREFIX
        return f"{prefix}-{name}" if prefix else name

    def _is_cache_expired(self, index_name: str) -> bool:
        """Check if cached index has expired (older than 2 hours)."""
        if index_name not in self._index_cache:
            return True
        
        cache_entry = self._index_cache[index_name]
        cached_time = cache_entry.get('timestamp', 0)
        current_time = time.time()
        expiry_seconds = self._cache_expiry_hours * 3600  # Convert hours to seconds
        
        return (current_time - cached_time) > expiry_seconds

    def _clear_expired_cache(self, index_name: str) -> None:
        """Remove expired cache entry."""
        if index_name in self._index_cache:
            del self._index_cache[index_name]

    def _get_or_create_index(self, name: str):
        if not self.pc:
            raise Exception("Pinecone API key not configured")
        index_name = self._index_name(name)
        
        # Check if cache exists and is not expired
        if index_name in self._index_cache and not self._is_cache_expired(index_name):
            return self._index_cache[index_name]['index']
        
        # Clear expired cache if it exists
        if self._is_cache_expired(index_name):
            self._clear_expired_cache(index_name)

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
        # Store index with timestamp for expiry tracking
        self._index_cache[index_name] = {
            'index': index,
            'timestamp': time.time()
        }
        return index

    def get_or_create_collection(self, name: str):
        return self._get_or_create_index(name)

    def flatten_metadata(self, metadata):
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                metadata[key] = json.dumps(value)
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
        print("flatten_metadata:", metadata)
        return metadata

    def add_documents(self, collection_name: str, chunks: List[Dict[str, Any]]) -> List[str]:
        index = self._get_or_create_index(collection_name)
        ids = [chunk["id"] for chunk in chunks]
        docs = [chunk["content"] for chunk in chunks]
        metas = [chunk["metadata"] for chunk in chunks]

        vectors = []
        for i, vid in enumerate(ids):
            meta = metas[i].copy() if metas[i] else {}
            meta["content"] = docs[i]
            vectors.append({"id": vid, "content": docs[i], **self.flatten_metadata(meta)})
        
        index.upsert_records("default-namespace", vectors)
        return ids

    def delete_by_parent_id(self, collection_name: str, parent_id: str) -> None:
        index = self._get_or_create_index(collection_name)
        index.delete(
            namespace='example-namespace',
            filter={
                "parent_document_id": {"$eq": parent_id}
            }
        )

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
        duration_ms = (time.time() - start_ts) * 1000
        print(
            f"[Pinecone] search to rag in {duration_ms:.2f} ms, "
            f"hits={len(res['result']['hits'])}"
        )
        filtered_results = {"ids": [], "distances": [], "metadatas": [], "documents": []}
        for match in res["result"]["hits"]:
            filtered_results["ids"].append(match["_id"])
            filtered_results["distances"].append(match["_score"])
            metas = {}

            for key, value in match["fields"].items():
                if key == "content":
                    continue
                metas[key] = value

            filtered_results["metadatas"].append(metas)
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

    def get_documents_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any],
        limit: int = 100
    ) -> Dict[str, List]:
        """
        Retrieve documents from Pinecone based on metadata filter.
        
        Args:
            collection_name: Name of the collection to search in
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            limit: Maximum number of documents to return (default: 100)
            
        Returns:
            Dictionary with keys: ids, metadatas, documents
        """
        start_ts = time.time()
        print(
            f"[Pinecone] get_documents_by_metadata start at {datetime.now().isoformat()} "
            f"collection='{collection_name}' filter={metadata_filter} limit={limit}"
        )
        
        index = self._get_or_create_index(collection_name)
        
        try:
            # Use Pinecone's query with metadata filter
            # Note: This uses a dummy vector query with metadata filter
            # Since Pinecone requires a vector for querying, we'll use a zero vector
            # and rely on the metadata filter to get the desired results
            res = index.query(
                namespace="default-namespace",
                vector=[0.0] * 1536,  # Dummy vector (adjust dimensions as needed)
                top_k=limit,
                include_metadata=True,
                include_values=False,
                filter=metadata_filter
            )
            
            duration_ms = (time.time() - start_ts) * 1000
            print(
                f"[Pinecone] metadata query completed in {duration_ms:.2f} ms, "
                f"matches={len(res.matches)}"
            )
            
            # Format results to match expected structure
            filtered_results = {"ids": [], "metadatas": [], "documents": []}
            
            for match in res.matches:
                filtered_results["ids"].append(match.id)
                filtered_results["metadatas"].append(match.metadata or {})
                # Extract content from metadata if available
                content = match.metadata.get("content", "") if match.metadata else ""
                filtered_results["documents"].append(content)
            
            duration_ms = (time.time() - start_ts) * 1000
            print(
                f"[Pinecone] get_documents_by_metadata finished in {duration_ms:.2f} ms, "
                f"results={len(filtered_results['ids'])}"
            )
            
            return filtered_results
            
        except Exception as e:
            print(f"[Pinecone] Error in get_documents_by_metadata: {str(e)}")
            return {"ids": [], "metadatas": [], "documents": []}

        