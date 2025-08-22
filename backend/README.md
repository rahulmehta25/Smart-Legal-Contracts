# Arbitration RAG API

A RAG (Retrieval-Augmented Generation) system for detecting arbitration clauses in Terms of Use documents using FastAPI, LangChain, ChromaDB, and sentence transformers.

## Features

- **Document Processing**: Upload and process legal documents with optimized chunking for legal text
- **Arbitration Detection**: Advanced detection of arbitration clauses using semantic search and keyword matching
- **Vector Search**: ChromaDB-powered similarity search with sentence transformers
- **RESTful API**: Complete FastAPI application with OpenAPI documentation
- **User Authentication**: JWT-based authentication system
- **Analysis History**: Track and compare analysis results over time
- **Legal Text Optimization**: Specialized text processing for legal documents

## Architecture

```
├── app/
│   ├── api/                 # API endpoints
│   │   ├── documents.py     # Document management endpoints
│   │   ├── analysis.py      # Analysis endpoints
│   │   └── users.py         # User authentication endpoints
│   ├── core/                # Core configuration
│   │   ├── config.py        # Application settings
│   │   └── security.py      # Security utilities
│   ├── db/                  # Database layer
│   │   ├── database.py      # SQLite database setup
│   │   └── vector_store.py  # ChromaDB vector store
│   ├── models/              # Data models
│   │   ├── document.py      # Document models
│   │   ├── analysis.py      # Analysis models
│   │   └── user.py          # User models
│   ├── rag/                 # RAG pipeline components
│   │   ├── text_processor.py # Legal text processing
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── retriever.py     # Arbitration clause retrieval
│   │   └── pipeline.py      # Complete RAG pipeline
│   ├── services/            # Business logic
│   │   ├── document_service.py
│   │   ├── analysis_service.py
│   │   └── user_service.py
│   └── main.py              # FastAPI application
├── requirements.txt         # Python dependencies
└── .env.example            # Environment configuration template
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd arbitration-rag-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Initialize database**
   ```bash
   python -c "from app.db.database import init_db; init_db()"
   ```

## Running the Application

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload and process a document
- `GET /api/v1/documents/` - List documents
- `GET /api/v1/documents/{id}` - Get document details
- `GET /api/v1/documents/{id}/chunks` - Get document chunks
- `GET /api/v1/documents/search/` - Search documents
- `DELETE /api/v1/documents/{id}` - Delete document

### Analysis
- `POST /api/v1/analysis/analyze` - Analyze document for arbitration clauses
- `POST /api/v1/analysis/quick-analyze` - Quick text analysis
- `GET /api/v1/analysis/` - List analyses
- `GET /api/v1/analysis/{id}` - Get analysis details
- `GET /api/v1/analysis/document/{id}` - Get document analyses

### Users
- `POST /api/v1/users/register` - Register new user
- `POST /api/v1/users/login` - User login
- `GET /api/v1/users/me` - Get current user info
- `PUT /api/v1/users/me` - Update user info

## Usage Examples

### Upload and Analyze Document
```python
import requests

# Upload document
files = {'file': open('terms_of_use.txt', 'rb')}
response = requests.post('http://localhost:8000/api/v1/documents/upload', files=files)
document_id = response.json()['document_id']

# Analyze for arbitration clauses
analysis_request = {
    "document_id": document_id,
    "force_reanalysis": False
}
response = requests.post('http://localhost:8000/api/v1/analysis/analyze', json=analysis_request)
analysis = response.json()

print(f"Has arbitration clause: {analysis['has_arbitration_clause']}")
print(f"Confidence: {analysis['confidence_score']:.2f}")
```

### Quick Text Analysis
```python
import requests

quick_request = {
    "text": "All disputes shall be resolved through binding arbitration...",
    "include_context": True
}
response = requests.post('http://localhost:8000/api/v1/analysis/quick-analyze', json=quick_request)
result = response.json()

print(f"Arbitration detected: {result['has_arbitration_clause']}")
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `DATABASE_URL`: Database connection string
- `VECTOR_STORE_PATH`: ChromaDB storage path
- `SECRET_KEY`: JWT secret key (change in production)
- `CHUNK_SIZE`: Text chunk size for processing
- `EMBEDDING_MODEL`: Sentence transformer model name

### RAG Pipeline Settings

- **Chunk Size**: 1000 characters (optimized for legal text)
- **Chunk Overlap**: 100 characters
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Store**: ChromaDB with cosine similarity
- **Confidence Threshold**: 0.4 minimum for arbitration detection

## Legal Text Processing

The system includes specialized components for legal document processing:

### Text Processor
- Legal section detection
- Arbitration keyword identification
- Sentence-based chunking with legal context preservation

### Arbitration Detection
- Binding arbitration clauses
- Class action waivers
- Jury trial waivers
- Mandatory arbitration requirements
- Alternative dispute resolution provisions

### Scoring System
- Semantic similarity scores
- Keyword match scoring
- Legal signal strength assessment
- Position-based relevance scoring

## Performance

- **Document Processing**: ~1-2 seconds per 10,000 characters
- **Analysis Speed**: ~500ms for typical terms of service
- **Vector Search**: Sub-second similarity search
- **Concurrent Requests**: Supports multiple simultaneous uploads/analyses

## Development

### Adding New Arbitration Patterns
1. Update keyword lists in `rag/text_processor.py`
2. Modify scoring weights in `rag/retriever.py`
3. Add new clause types in `models/analysis.py`

### Database Migrations
For schema changes, manually update the models and run:
```python
from app.db.database import reset_database
reset_database()  # Caution: This drops all data
```

### Testing
```bash
pytest tests/  # Run tests (when test suite is added)
```

## Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use PostgreSQL instead of SQLite for production
- Set strong `SECRET_KEY` and secure JWT settings
- Configure CORS origins for your frontend domain
- Set up proper logging and monitoring
- Use reverse proxy (nginx) for static files and SSL

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For questions or issues, please create an issue in the repository.