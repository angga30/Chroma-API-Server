from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from config import settings

class ChromaDBService:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False
            )
        )
        self.embedding_function = None
        if settings.OPENAI_API_KEY:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL_NAME
            )
    
    def get_or_create_collection(self, name: str):
        """Get an existing collection or create a new one if it doesn't exist."""
        try:
            if name not in ["help-article"]:
                return self.client.get_collection(
                    name=name,
                    embedding_function=self.embedding_function
                )
            else:
                return self.client.get_collection(
                    name=name,
                )
        except:
            return self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, collection_name: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add document chunks to a collection."""
        collection = self.get_or_create_collection(collection_name)
        
        chunk_ids = [chunk["id"] for chunk in chunks]
        chunk_contents = [chunk["content"] for chunk in chunks]
        chunk_metadatas = [chunk["metadata"] for chunk in chunks]
        
        collection.upsert(
            documents=chunk_contents,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        return chunk_ids
    
    def update_documents(self, collection_name: str, doc_ids: List[str], chunks: List[Dict[str, Any]]) -> int:
        """Update documents in a collection."""
        collection = self.get_or_create_collection(collection_name)
        
        # Delete existing chunks for these documents
        for doc_id in doc_ids:
            try:
                results = collection.query(
                    query_texts=[""],
                    where={"parent_document_id": doc_id},
                    n_results=100
                )
                if results and results['ids'] and results['ids'][0]:
                    collection.delete(ids=results['ids'][0])
            except Exception as e:
                print(f"Error finding chunks for document {doc_id}: {str(e)}")
        
        # Add new chunks
        chunk_ids = [chunk["id"] for chunk in chunks]
        chunk_contents = [chunk["content"] for chunk in chunks]
        chunk_metadatas = [chunk["metadata"] for chunk in chunks]
        
        collection.add(
            documents=chunk_contents,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        return len(chunks)
    
    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> None:
        """Delete documents from a collection."""
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=doc_ids)
    
    def search_similarity(
        self,
        collection_name: str,
        query: str,
        n_results: int = settings.DEFAULT_SEARCH_RESULTS,
        threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD
    ) -> Dict[str, List]:
        """Search for similar documents in a collection."""
        collection = self.get_or_create_collection(collection_name)
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        filtered_results = {
            'ids': [],
            'distances': [],
            'metadatas': [],
            'documents': []
        }
        
        for i, distance in enumerate(results['distances'][0]):
            # if 1 - distance >= threshold:
            if True:
                filtered_results['ids'].append(results['ids'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])
                filtered_results['metadatas'].append(results['metadatas'][0][i])
                filtered_results['documents'].append(results['documents'][0][i])
        
        return filtered_results
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.list_collections()
        return [col.name for col in collections]

# Create global service instance
chroma_service = ChromaDBService()