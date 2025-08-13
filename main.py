from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import uuid

app = FastAPI()

# Initialize ChromaDB client
client = chromadb.HttpClient(host='chroma', port=8000)

class Document(BaseModel):
    uid: str
    content: str
    metadata: dict = Field(default_factory=dict)

class BatchDocumentRequest(BaseModel):
    documents: List[Document]
    collection_name: str

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    n_results: int = 5
    threshold: float = 0.5

def get_or_create_collection(name: str):
    try:
        return client.get_collection(name=name)
    except:
        return client.create_collection(name=name)

@app.post("/api/add_documents")
async def add_documents(request: BatchDocumentRequest):
    try:
        collection = get_or_create_collection(request.collection_name)
        print(request.documents)
        # Generate unique IDs for each document
        doc_ids = [doc.uid for doc in request.documents]
        
        # Prepare batch data
        documents = [doc.content for doc in request.documents]
        metadatas = [doc.metadata for doc in request.documents]
        
        # Add documents to ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        return {"ids": doc_ids, "message": "Documents added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update_documents/{collection_name}")
async def update_documents(collection_name: str, documents: List[Document], doc_ids: List[str]):
    try:
        if len(documents) != len(doc_ids):
            raise HTTPException(status_code=400, detail="Number of documents must match number of IDs")
            
        collection = get_or_create_collection(collection_name)
        
        # Update documents in ChromaDB
        collection.update(
            ids=doc_ids,
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        
        return {"message": "Documents updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete_documents/{collection_name}")
async def delete_documents(collection_name: str, doc_ids: List[str]):
    try:
        collection = get_or_create_collection(collection_name)
        
        # Delete documents from ChromaDB
        collection.delete(ids=doc_ids)
        
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search_similarity")
async def search_similarity(request: SearchRequest):
    try:
        collection = get_or_create_collection(request.collection_name)
        
        # Search for similar documents
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results
        )

        # Filter results based on similarity threshold
        filtered_results = {
            'ids': [],
            'distances': [],
            'metadatas': [],
            'documents': []
        }
        
        # Only include results that meet the similarity threshold
        for i, distance in enumerate(results['distances'][0]):
            if 1 - distance >= request.threshold:  # Convert distance to similarity score
                filtered_results['ids'].append(results['ids'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])
                filtered_results['metadatas'].append(results['metadatas'][0][i])
                filtered_results['documents'].append(results['documents'][0][i])

        return filtered_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    try:
        collections = client.list_collections()
        return {"collections": [col.name for col in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)