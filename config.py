from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "text-embedding-3-small"
    
    # Application Configuration
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    DEFAULT_SEARCH_RESULTS: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.2
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()