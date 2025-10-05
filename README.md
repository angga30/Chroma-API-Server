# RAG Server with FastAPI (ChromaDB/Pinecone)

A REST API service built with FastAPI and ChromaDB for document management and similarity search, supporting batch operations and multiple collections.

## Features

- Batch operations for adding/updating/deleting documents
- Dynamic collection management
- Cross-collection similarity search
- List available collections
- Filter search results based on similarity threshold
- Smart content chunking based on content type (text, HTML, code)
- Automatic content type detection and verification
- Environment-based configuration management
- CLI tool for easy ChromaDB management
- Selectable RAG backend: ChromaDB or Pinecone

## System Requirements

- Python 3.9 or higher
- Docker and Docker Compose (for container deployment)

## Installation

### Using Docker (Recommended)

1. Clone repository:
```bash
git clone <repository-url>
cd RAGServer
```

2. Configure environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file and set your configuration values, especially:
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings
- Other optional configurations as needed

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

Services will be available at:
- FastAPI: `http://localhost:8002`
- ChromaDB: `http://localhost:8001`

### Local Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file with your configuration values.

4. Run server:
```bash
python main.py
```

## Configuration

The application can be configured using environment variables or a `.env` file:

### Required Configuration
- `OPENAI_API_KEY`: Your OpenAI API key for generating embeddings

### Optional Configuration
- `CHROMA_HOST`: ChromaDB host (default: "localhost")
- `CHROMA_PORT`: ChromaDB port (default: 8000)
- `OPENAI_MODEL_NAME`: OpenAI embedding model (default: "text-embedding-3-small")
- `OPENAI_EMBEDDING_DIM`: Embedding dimension when diperlukan backend lain (opsional)
- `DEFAULT_CHUNK_SIZE`: Default document chunk size (default: 1000)
- `DEFAULT_CHUNK_OVERLAP`: Default chunk overlap size (default: 200)
- `DEFAULT_SEARCH_RESULTS`: Default number of search results (default: 5)
- `DEFAULT_SIMILARITY_THRESHOLD`: Default similarity threshold (default: 0.5)
- `RAG_SERVER`: Pilih backend RAG default (`"chroma"` atau `"pinecone"`)

#### Pinecone Configuration (opsional, diperlukan jika menggunakan Pinecone)
- `PINECONE_API_KEY`: API key Pinecone
- `PINECONE_CLOUD`: Cloud Pinecone (contoh: `"aws"`)
- `PINECONE_REGION`: Region Pinecone (contoh: `"us-west-2"`)
- `PINECONE_INDEX_PREFIX`: Prefix nama index (contoh: `"rag-server"`)

## CLI Usage

The application includes a CLI tool for managing RAG backend directly from the command line.

### Available Commands

1. List Collections:
```bash
# Default backend (mengikuti konfigurasi `RAG_SERVER`)
python cli.py list-collections

# Pilih backend secara eksplisit
python cli.py list-collections --rag chroma
python cli.py list-collections --rag pinecone
```

2. List Documents in a Collection:
```bash
# List all documents (default limit: 10)
python cli.py list-documents my_collection --rag chroma

# List with custom limit
python cli.py list-documents my_collection --limit 20 --rag chroma

# List with metadata filter
python cli.py list-documents my_collection --where '{"category": "technology"}' --rag chroma

# Catatan: list-documents belum didukung untuk Pinecone; gunakan perintah search.
```

3. Search Documents:
```bash
# Basic search
python cli.py search my_collection "search query" --rag chroma
python cli.py search my_collection "search query" --rag pinecone

# Search with custom parameters
python cli.py search my_collection "search query" --n-results 10 --threshold 0.7 --rag chroma
```

4. Delete Documents:
```bash
# Delete specific documents
python cli.py delete my_collection doc_id1 doc_id2 --rag chroma
python cli.py delete my_collection doc_id1 doc_id2 --rag pinecone

# Delete entire collection
python cli.py delete-collection my_collection --rag chroma
python cli.py delete-collection my_collection --rag pinecone
```

### CLI Help

For detailed information about each command:
```bash
# General help
python cli.py --help

# Command-specific help
python cli.py list-documents --help
python cli.py search --help
```

## API Usage

### Examples with curl

1. Add Documents (Batch):
```bash
curl -X POST http://localhost:8002/api/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "documents": [
      {
        "content": "FastAPI is a modern Python framework for building APIs",
        "metadata": {"category": "technology"},
        "content_type": "text"  # Optional: will be auto-detected if not provided
      },
      {
        "content": "<h1>ChromaDB</h1><p>ChromaDB is a powerful vector database</p>",
        "metadata": {"category": "database"},
        "content_type": "html"  # Optional: will be auto-detected if not provided
      },
      {
        "content": "def hello_world():\n    print('Hello, world!')",
        "metadata": {"category": "code"}  # Content type will be auto-detected as code
      }
    ]
  }'
```

2. Update Documents (Batch):
```bash
curl -X PUT "http://localhost:8002/api/update_documents/articles?doc_ids=[\"doc1\",\"doc2\"]" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "content": "FastAPI is a fast and modern Python framework",
      "metadata": {"category": "technology", "updated": true}
    },
    {
      "content": "ChromaDB is an open-source vector database",
      "metadata": {"category": "database", "updated": true}
    }
  ]'
```

3. Delete Documents (Batch):
```bash
curl -X DELETE "http://localhost:8002/api/delete_documents/articles?doc_ids=[\"doc1\",\"doc2\"]"
```

4. Similarity Search:
```bash
curl -X POST http://localhost:8002/api/search_similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python framework for API",
    "collection_name": "articles",
    "n_results": 5,
    "threshold": 0.7
  }'
```

5. List Collections:
```bash
curl http://localhost:8002/api/collections
```

### Backend Switching (API)

Semua endpoint mendukung pemilihan backend melalui parameter `rag_server` per-request. Jika tidak diberikan, server mengikuti konfigurasi default `RAG_SERVER`.

1. Add Documents ke Pinecone:
```bash
curl -X POST http://localhost:8002/api/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "rag_server": "pinecone",
    "documents": [
      { "content": "Contoh konten", "metadata": {"category": "test"} }
    ]
  }'
```

2. Similarity Search di Pinecone:
```bash
curl -X POST http://localhost:8002/api/search_similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python framework for API",
    "collection_name": "articles",
    "rag_server": "pinecone",
    "n_results": 5
  }'
```

3. List Collections Pinecone:
```bash
curl "http://localhost:8002/api/collections?rag_server=pinecone"
```

### Examples with Python Requests

```python
import requests

BASE_URL = "http://localhost:8002"

# Add documents
def add_documents():
    response = requests.post(
        f"{BASE_URL}/api/add_documents",
        json={
            "collection_name": "articles",
            "rag_server": "pinecone",  # pilih backend per request
            "documents": [
                {
                    "content": "FastAPI is a modern Python framework for building APIs",
                    "metadata": {"category": "technology"},
                    "content_type": "text"  # Optional: will be auto-detected if not provided
                },
                {
                    "content": "<h1>ChromaDB</h1><p>ChromaDB is a powerful vector database</p>",
                    "metadata": {"category": "database"},
                    "content_type": "html"  # Optional: will be auto-detected if not provided
                },
                {
                    "content": "def hello_world():\n    print('Hello, world!')",
                    "metadata": {"category": "code"}  # Content type will be auto-detected as code
                }
            ]
        }
    )
    return response.json()

# Update documents
def update_documents(doc_ids):
    response = requests.put(
        f"{BASE_URL}/api/update_documents/articles",
        params={"doc_ids": doc_ids},
        json=[
            {
                "content": "FastAPI is a fast and modern Python framework",
                "metadata": {"category": "technology", "updated": True}
            },
            {
                "content": "ChromaDB is an open-source vector database",
                "metadata": {"category": "database", "updated": True}
            }
        ]
    )
    return response.json()

# Delete documents
def delete_documents(doc_ids):
    response = requests.delete(
        f"{BASE_URL}/api/delete_documents/articles",
        params={"doc_ids": doc_ids}
    )
    return response.json()

# Similarity search
def search_documents(query):
    response = requests.post(
        f"{BASE_URL}/api/search_similarity",
        json={
            "query": query,
            "collection_name": "articles",
            "rag_server": "pinecone",  # atau "chroma"
            "n_results": 5,
            "threshold": 0.7
        }
    )
    return response.json()

# List collections
def list_collections():
    response = requests.get(f"{BASE_URL}/api/collections")
    return response.json()

# Usage example
if __name__ == "__main__":
    # Add documents
    result = add_documents()
    doc_ids = result["ids"]
    
    # Update documents
    update_documents(doc_ids)
    
    # Search documents
    search_results = search_documents("Python framework")
    print(search_results)
    
    # Delete documents
    delete_documents(doc_ids)
```

## API Documentation

After the server is running, you can access:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── service/
│   ├── chromadb.py        # ChromaDB service implementation
│   ├── pinecone_service.py # Pinecone service implementation
│   └── rag_factory.py     # Backend selector (Chroma/Pinecone)
├── models/
│   └── document.py        # Pydantic models for request/response
├── chunkers.py            # Content chunking implementations
├── cli.py                 # CLI tool for RAG backend management
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Docker build configuration
└── .env.example          # Environment variables template
```