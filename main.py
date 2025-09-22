from fastapi import FastAPI, HTTPException
from typing import List
from models.document import Document, BatchDocumentRequest, SearchRequest
from service.chromadb import ChromaDBService
from chunkers import SmartChunker

app = FastAPI()

@app.post("/api/add_documents")
async def add_documents(request: BatchDocumentRequest):
    try:
        # Initialize smart chunker
        chroma_service = ChromaDBService(request.embedding_model)
        smart_chunker = SmartChunker()
        
        # Process each document with smart chunking
        all_chunks = []
        
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
        
        # Add chunks to ChromaDB
        if all_chunks:
            chunk_ids = chroma_service.add_documents(request.collection_name, all_chunks)
            return {"ids": chunk_ids, "message": f"Documents processed and {len(all_chunks)} chunks added successfully"}
        
        return {"ids": [], "message": "No documents to process"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/update_documents/{collection_name}")
async def update_documents(collection_name: str, documents: List[Document], doc_ids: List[str]):
    try:
        if len(documents) != len(doc_ids):
            raise HTTPException(status_code=400, detail="Number of documents must match number of IDs")
        
        # Initialize smart chunker
        smart_chunker = SmartChunker()
        
        # Process each document with smart chunking
        all_chunks = []
        
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
        
        # Update documents in ChromaDB
        chunks_updated = chroma_service.update_documents(collection_name, doc_ids, all_chunks)
        return {"message": f"Documents updated successfully with {chunks_updated} new chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete_documents/{collection_name}")
async def delete_documents(collection_name: str, doc_ids: List[str]):
    try:
        chroma_service.delete_documents(collection_name, doc_ids)
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search_similarity")
async def search_similarity(request: SearchRequest):
    try:
        chroma_service = ChromaDBService(request.embedding_model)
        results = chroma_service.search_similarity(
            collection_name=request.collection_name,
            query=request.query,
            n_results=request.n_results,
            threshold=request.threshold
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    try:
        collections = chroma_service.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)