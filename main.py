from fastapi import FastAPI, HTTPException
from typing import List
from models.document import Document, BatchDocumentRequest, SearchRequest
from service.rag_factory import get_rag_service
from chunkers import SmartChunker

app = FastAPI()

@app.post("/api/add_documents")
async def add_documents(request: BatchDocumentRequest):
    try:
        # Initialize backend service and smart chunker
        # Validate request size (400 KB limit)
        request_size = len(request.json().encode('utf-8'))
        if request_size > 400 * 1024:
            raise HTTPException(status_code=413, detail="Request payload too large. Maximum allowed size is 400 KB.")
            
        service = get_rag_service(request.rag_server, request.embedding_model)
        smart_chunker = SmartChunker()
        
        # Process each document with smart chunking
        all_chunks = []
        
        for doc in request.documents:
            # Delete existing chunks for this parent document before re-inserting
            try:
                deleted = service.delete_by_parent_id(request.collection_name, doc.uid)
                print(f"Deleted related chunks for parent_document_id={doc.uid}: {deleted}")
            except Exception as de:
                print(f"Warning: failed to delete related chunks for parent_document_id={doc.uid}: {de}")
            
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
                print(combined_metadata)
                all_chunks.append({
                    "id": chunk_id,
                    "content": chunk['content'],
                    "metadata": combined_metadata
                })
        
        # Add chunks to selected RAG backend
        if all_chunks:
            chunk_ids = service.add_documents(request.collection_name, all_chunks)
            return {"ids": chunk_ids, "message": f"Documents processed and {len(all_chunks)} chunks added successfully"}
        
        return {"ids": [], "message": "No documents to process"}
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
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
        
        # Update documents in selected RAG backend
        service = get_rag_service(None, None)
        chunks_updated = service.update_documents(collection_name, doc_ids, all_chunks)
        return {"message": f"Documents updated successfully with {chunks_updated} new chunks"}
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete_documents/{collection_name}")
async def delete_documents(collection_name: str, doc_ids: List[str]):
    try:
        service = get_rag_service(None, None)
        service.delete_documents(collection_name, doc_ids)
        return {"message": "Documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search_similarity")
async def search_similarity(request: SearchRequest):
    try:
        service = get_rag_service(request.rag_server, request.embedding_model)
        results = service.search_similarity(
            collection_name=request.collection_name,
            query=request.query,
            n_results=request.n_results,
            threshold=request.threshold
        )
        return results
    except Exception as e:
        print(f"Error processing search similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    try:
        service = get_rag_service(None, None)
        collections = service.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)