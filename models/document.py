from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from chunkers import SmartChunker
from config import settings

class Document(BaseModel):
    uid: str
    content: str
    metadata: dict = Field(default_factory=dict)
    content_type: Optional[str] = None
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE
    chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP
    
    @validator('content_type', pre=True, always=True)
    def verify_content_type(cls, content_type, values):
        if content_type is not None:
            return content_type
        
        if 'content' in values and values['content']:
            smart_chunker = SmartChunker()
            detected_type = smart_chunker.detect_content_type(values['content'])
            return detected_type
        
        return "text"

class BatchDocumentRequest(BaseModel):
    documents: List[Document]
    collection_name: str
    embedding_model: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    n_results: int = settings.DEFAULT_SEARCH_RESULTS
    threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD
    embedding_model: Optional[str] = None