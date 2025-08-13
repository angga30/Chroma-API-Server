# RAG Server with FastAPI and ChromaDB

A REST API service built with FastAPI and ChromaDB for document management and similarity search, supporting batch operations and multiple collections.

## Features

- Batch operations for adding/updating/deleting documents
- Dynamic collection management
- Cross-collection similarity search
- List available collections
- Filter search results based on similarity threshold

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

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

Services will be available at:
- FastAPI: `http://localhost:8000`
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

3. Run server:
```bash
python main.py
```

## API Usage

### Examples with curl

1. Add Documents (Batch):
```bash
curl -X POST http://localhost:8000/api/add_documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "articles",
    "documents": [
      {
        "content": "FastAPI is a modern Python framework for building APIs",
        "metadata": {"category": "technology"}
      },
      {
        "content": "ChromaDB is a powerful vector database",
        "metadata": {"category": "database"}
      }
    ]
  }'
```

2. Update Documents (Batch):
```bash
curl -X PUT "http://localhost:8000/api/update_documents/articles?doc_ids=[\"doc1\",\"doc2\"]" \
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
curl -X DELETE "http://localhost:8000/api/delete_documents/articles?doc_ids=[\"doc1\",\"doc2\"]"
```

4. Similarity Search:
```bash
curl -X POST http://localhost:8000/api/search_similarity \
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
curl http://localhost:8000/api/collections
```

### Examples with Python Requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Add documents
def add_documents():
    response = requests.post(
        f"{BASE_URL}/api/add_documents",
        json={
            "collection_name": "articles",
            "documents": [
                {
                    "content": "FastAPI is a modern Python framework for building APIs",
                    "metadata": {"category": "technology"}
                },
                {
                    "content": "ChromaDB is a powerful vector database",
                    "metadata": {"category": "database"}
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
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`