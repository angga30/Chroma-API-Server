from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
import uuid
from chunkers import SmartChunker

app = FastAPI()

# Initialize ChromaDB client
client = chromadb.HttpClient(host='chroma', port=8000)

class Document(BaseModel):
    uid: str
    content: str
    metadata: dict = Field(default_factory=dict)
    content_type: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    @validator('content_type', pre=True, always=True)
    def verify_content_type(cls, content_type, values):
        # If content_type is provided, use it
        if content_type is not None:
            return content_type
        
        # If content is available, detect content_type from it
        if 'content' in values and values['content']:
            smart_chunker = SmartChunker()
            detected_type = smart_chunker.detect_content_type(values['content'])
            return detected_type
        
        # Default to text if no content is available
        return "text"

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
        
        # Initialize smart chunker
        smart_chunker = SmartChunker()
        
        # Process each document with smart chunking
        all_chunks = []
        all_chunk_ids = []
        all_chunk_contents = []
        all_chunk_metadatas = []
        
        for doc in request.documents:
            # Apply smart chunking
            chunks = smart_chunker.chunk(
                content=doc.content,
                content_type=doc.content_type,
                chunk_size=doc.chunk_size,
                chunk_overlap=doc.chunk_overlap
            )
            
            # Process each chunk
            for chunk in chunks:
                # Generate a unique ID for each chunk
                chunk_id = f"{doc.uid}-{chunk['metadata']['chunk_id']}"
                
                # Combine document metadata with chunk metadata
                combined_metadata = doc.metadata.copy()
                combined_metadata.update(chunk['metadata'])
                combined_metadata['parent_document_id'] = doc.uid
                
                all_chunks.append({
                    "id": chunk_id,
                    "content": chunk['content'],
                    "metadata": combined_metadata
                })
                print(all_chunks)
                all_chunk_ids.append(chunk_id)
                all_chunk_contents.append(chunk['content'])
                all_chunk_metadatas.append(combined_metadata)
        
        # Add chunks to ChromaDB
        if all_chunks:
            collection.upsert(
                documents=all_chunk_contents,
                metadatas=all_chunk_metadatas,
                ids=all_chunk_ids
            )
        
        return {"ids": all_chunk_ids, "message": f"Documents processed and {len(all_chunks)} chunks added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update_documents/{collection_name}")
async def update_documents(collection_name: str, documents: List[Document], doc_ids: List[str]):
    try:
        if len(documents) != len(doc_ids):
            raise HTTPException(status_code=400, detail="Number of documents must match number of IDs")
            
        collection = get_or_create_collection(collection_name)
        
        # First, delete existing chunks for these documents
        for doc_id in doc_ids:
            # Find all chunks with this parent document ID
            try:
                results = collection.query(
                    query_texts=["dummy query"],  # We're not actually searching by content
                    where={"parent_document_id": doc_id},
                    n_results=100  # Get all chunks for this document
                )
                
                if results and results['ids'] and results['ids'][0]:
                    # Delete all chunks for this document
                    collection.delete(ids=results['ids'][0])
            except Exception as e:
                # If query fails, it might be because the document doesn't exist or has no chunks
                print(f"Error finding chunks for document {doc_id}: {str(e)}")
        
        # Initialize smart chunker
        smart_chunker = SmartChunker()
        
        # Process each document with smart chunking
        all_chunks = []
        all_chunk_ids = []
        all_chunk_contents = []
        all_chunk_metadatas = []
        
        for i, doc in enumerate(documents):
            # Apply smart chunking
            chunks = smart_chunker.chunk(
                content=doc.content,
                content_type=doc.content_type,
                chunk_size=doc.chunk_size,
                chunk_overlap=doc.chunk_overlap
            )
            
            # Process each chunk
            for chunk in chunks:
                # Generate a unique ID for each chunk
                chunk_id = f"{doc_ids[i]}-{chunk['metadata']['chunk_id']}"
                
                # Combine document metadata with chunk metadata
                combined_metadata = doc.metadata.copy()
                combined_metadata.update(chunk['metadata'])
                combined_metadata['parent_document_id'] = doc_ids[i]
                
                all_chunks.append({
                    "id": chunk_id,
                    "content": chunk['content'],
                    "metadata": combined_metadata
                })
                
                all_chunk_ids.append(chunk_id)
                all_chunk_contents.append(chunk['content'])
                all_chunk_metadatas.append(combined_metadata)
        
        # Add new chunks to ChromaDB
        if all_chunks:
            collection.add(
                documents=all_chunk_contents,
                metadatas=all_chunk_metadatas,
                ids=all_chunk_ids
            )
        
        return {"message": f"Documents updated successfully with {len(all_chunks)} new chunks"}
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