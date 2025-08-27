# PDF Processing System

## Overview

A comprehensive, production-ready PDF processing system that provides:

- **Advanced PDF text extraction** with native parsing and OCR fallback
- **Multi-format document support** (PDF, DOCX, TXT, HTML, RTF, ODT)
- **Intelligent text preprocessing** with cleaning and structure detection
- **Scalable file storage** with S3-compatible backends
- **Asynchronous batch processing** using Celery task queues
- **RESTful API endpoints** for easy integration

## Features

### 🔍 PDF Text Extraction
- **Native text extraction** using pdfplumber and PyPDF2
- **OCR capabilities** with pytesseract for scanned/image-based PDFs
- **Multi-page processing** with per-page metadata
- **Layout preservation** and table extraction
- **Error recovery** with automatic OCR fallback
- **Configurable quality levels** (low, medium, high, ultra)

### 📄 Multi-Format Document Processing
- **PDF** - Native text + OCR fallback
- **Microsoft Word** - DOCX with structure preservation
- **Plain Text** - With encoding detection and normalization
- **HTML** - Structure-aware parsing with BeautifulSoup
- **RTF** - Basic Rich Text Format support
- **OpenDocument** - ODT file processing

### 🔧 Text Preprocessing
- **Language detection** with confidence scoring
- **Text cleaning** and OCR artifact removal
- **Structure detection** (headings, paragraphs, lists, tables)
- **Content type classification** (legal, technical, academic, etc.)
- **Quality assessment** with readability scoring
- **Statistical analysis** (word count, lexical diversity, etc.)

### 💾 File Storage
- **Multiple backends**: Local filesystem, AWS S3, MinIO
- **Secure file handling** with validation and scanning
- **Thumbnail generation** for images and PDFs
- **File versioning** and deduplication
- **Metadata storage** with comprehensive indexing
- **Secure URL generation** with expiration

### ⚡ Batch Processing
- **Celery-based** asynchronous task queues
- **Progress tracking** with real-time updates
- **Error recovery** and retry mechanisms
- **Result caching** and storage
- **Performance monitoring** with execution metrics

### 🌐 API Endpoints
- **RESTful design** with comprehensive error handling
- **File upload** with validation and security
- **Text extraction** with preprocessing options
- **Document preview** with thumbnails and metadata
- **Batch operations** for multiple files
- **Job monitoring** with status and progress tracking

## Installation

### Prerequisites

```bash
# System dependencies for OCR
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install poppler-utils  # For pdf2image

# For image processing
sudo apt-get install libmagic1

# Redis for Celery (if using batch processing)
sudo apt-get install redis-server
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `PyPDF2>=3.0.1` - PDF parsing
- `pdfplumber>=0.10.0` - Advanced PDF processing
- `pytesseract>=0.3.10` - OCR capabilities
- `python-docx>=1.0.0` - Word document processing
- `beautifulsoup4>=4.12.0` - HTML parsing
- `celery>=5.3.0` - Asynchronous task processing
- `redis>=5.0.0` - Task queue backend
- `boto3>=1.28.0` - AWS S3 integration
- `Pillow>=10.0.0` - Image processing

## Quick Start

### Basic Usage

```python
from app.services.pdf_service import extract_pdf_text
from app.services.document_processor import process_document
from app.services.preprocessing import preprocess_text

# Extract text from PDF
result = extract_pdf_text('document.pdf')
print(f"Extracted {len(result.text)} characters")

# Process any document type
doc_result = process_document('document.docx', 'my_document.docx')
print(f"Found {len(doc_result.structured_content)} sections")

# Preprocess and analyze text
prep_result = preprocess_text(result.text)
print(f"Language: {prep_result.detected_language}")
print(f"Quality: {prep_result.text_quality.value}")
```

### Storage Operations

```python
from app.services.storage_service import create_storage_service, StorageBackend

# Create storage service
storage = create_storage_service(
    backend=StorageBackend.LOCAL,
    local_base_path="/path/to/storage"
)

# Upload file
with open('document.pdf', 'rb') as f:
    file_content = f.read()

upload_result = storage.upload_file(
    file_data=file_content,
    filename='document.pdf',
    generate_thumbnails=True
)

file_id = upload_result.file_id
print(f"Uploaded as {file_id}")

# Retrieve file
content = storage.get_file(file_id)
metadata = storage.get_metadata(file_id)
```

### Batch Processing

```python
from app.services.batch_processor import get_batch_processor, BatchJobConfig

# Submit batch job
processor = get_batch_processor()
config = BatchJobConfig(max_retries=3, timeout=1800)

job_id = processor.submit_job(
    'batch_process_documents',
    config,
    file_list=[
        {'file_path': 'doc1.pdf', 'filename': 'document1.pdf'},
        {'file_path': 'doc2.docx', 'filename': 'document2.docx'}
    ]
)

# Monitor progress
status = processor.get_job_status(job_id)
print(f"Status: {status.status.value}")
print(f"Progress: {status.progress.percentage}%")
```

## API Usage

### Upload PDF

```bash
curl -X POST "http://localhost:8000/api/upload/pdf" \
  -F "file=@document.pdf" \
  -F "extract_text=true" \
  -F "generate_thumbnails=true" \
  -F "ocr_quality=medium"
```

Response:
```json
{
  "file_id": "abc123-def456",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "file_size": 1024000,
  "secure_url": "https://storage.example.com/files/...",
  "thumbnails": {
    "small": "/api/thumbnails/abc123/small",
    "medium": "/api/thumbnails/abc123/medium"
  },
  "message": "File uploaded successfully"
}
```

### Get Document Text

```bash
curl "http://localhost:8000/api/documents/abc123-def456/text?include_preprocessing=true"
```

Response:
```json
{
  "file_id": "abc123-def456",
  "text": "Extracted document text content...",
  "word_count": 1250,
  "character_count": 8500,
  "language": "en",
  "extraction_method": "native",
  "processing_time": 2.34,
  "quality_score": "good"
}
```

### Batch Upload

```bash
curl -X POST "http://localhost:8000/api/batch/upload" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "process_async=true"
```

Response:
```json
{
  "batch_id": "batch-789xyz",
  "total_files": 2,
  "job_id": "job-456def",
  "status_url": "/api/jobs/job-456def/status",
  "estimated_completion_time": "~2 minutes",
  "message": "Uploaded 2 files successfully. Processing job job-456def started."
}
```

### Check Job Status

```bash
curl "http://localhost:8000/api/jobs/job-456def/status"
```

Response:
```json
{
  "job_id": "job-456def",
  "status": "processing",
  "progress": {
    "current": 1,
    "total": 2,
    "percentage": 50.0,
    "stage": "processing",
    "message": "Processing file 1 of 2: doc1.pdf"
  },
  "started_at": "2024-01-15T10:30:00Z"
}
```

## Configuration

### Storage Configuration

```python
from app.services.storage_service import StorageConfig, StorageBackend

# Local storage
local_config = StorageConfig(
    backend=StorageBackend.LOCAL,
    local_base_path="/var/lib/documents",
    enable_thumbnails=True,
    max_file_size=100 * 1024 * 1024  # 100MB
)

# S3 storage
s3_config = StorageConfig(
    backend=StorageBackend.S3,
    s3_bucket="my-documents",
    s3_region="us-east-1",
    s3_access_key="AKIAIOSFODNN7EXAMPLE",
    s3_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    enable_thumbnails=True
)

# MinIO storage
minio_config = StorageConfig(
    backend=StorageBackend.MINIO,
    minio_endpoint="localhost:9000",
    minio_access_key="minioadmin",
    minio_secret_key="minioadmin",
    s3_bucket="documents",
    minio_secure=False
)
```

### Batch Processing Configuration

```python
from app.services.batch_processor import BatchJobConfig

config = BatchJobConfig(
    max_retries=3,
    retry_delay=60,      # seconds
    timeout=3600,        # 1 hour
    priority=5,          # 1-10 scale
    expires=86400,       # 24 hours
    store_results=True
)
```

### OCR Quality Settings

```python
from app.services.pdf_service import PDFProcessor, PDFQuality

# Different quality levels for speed vs accuracy tradeoff
processor_fast = PDFProcessor(quality=PDFQuality.LOW)      # 150 DPI
processor_balanced = PDFProcessor(quality=PDFQuality.MEDIUM)  # 200 DPI  
processor_high = PDFProcessor(quality=PDFQuality.HIGH)     # 300 DPI
processor_ultra = PDFProcessor(quality=PDFQuality.ULTRA)   # 600 DPI
```

## Performance

### Benchmarks

Typical performance on modern hardware:

| Operation | Small (1KB) | Medium (100KB) | Large (10MB) |
|-----------|-------------|----------------|--------------|
| PDF Native Extract | 0.05s | 0.2s | 2.5s |
| PDF OCR Extract | 0.5s | 3.2s | 45s |
| DOCX Processing | 0.1s | 0.8s | 8s |
| Text Preprocessing | 0.01s | 0.05s | 0.5s |
| Storage Upload | 0.02s | 0.1s | 1.2s |

### Optimization Tips

1. **Use native extraction when possible** - 10-100x faster than OCR
2. **Choose appropriate OCR quality** - LOW for speed, HIGH for accuracy  
3. **Enable async processing** for large files or batches
4. **Use local storage** for development, S3 for production
5. **Configure Celery workers** based on CPU cores and memory

### Scaling

- **Horizontal scaling**: Deploy multiple Celery workers
- **Storage scaling**: Use S3 or distributed storage systems
- **Database scaling**: Use Redis Cluster for job queues
- **API scaling**: Deploy multiple API instances behind load balancer

## Testing

### Run Integration Tests

```bash
# Basic functionality test
python backend/test_pdf_integration.py

# Performance benchmarks
python -m app.services.pdf_processing_example benchmark

# Full demo
python -m app.services.pdf_processing_example
```

### Test API Endpoints

```bash
# Start the API server
python -m app.main

# Run API tests (in another terminal)
curl http://localhost:8000/health
curl http://localhost:8000/api/health
```

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Processing      │    │   Storage       │
│                 │    │                  │    │                 │
│ • FastAPI       │───▶│ • PDF Service    │───▶│ • Local FS      │
│ • File Upload   │    │ • Doc Processor  │    │ • AWS S3        │
│ • Job Status    │    │ • Preprocessor   │    │ • MinIO         │
│ • Thumbnails    │    │ • OCR Engine     │    │ • Thumbnails    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       
         ▼                       ▼                       
┌─────────────────┐    ┌──────────────────┐              
│  Task Queue     │    │   Monitoring     │              
│                 │    │                  │              
│ • Celery        │    │ • Progress       │              
│ • Redis Broker  │    │ • Error Logs     │              
│ • Job Results   │    │ • Performance    │              
│ • Retry Logic   │    │ • Health Checks  │              
└─────────────────┘    └──────────────────┘              
```

### Data Flow

1. **File Upload** → API validates and stores file
2. **Processing Job** → Celery task extracts text and metadata  
3. **Text Analysis** → Preprocessing cleans and analyzes content
4. **Results Storage** → Processed data stored with metadata
5. **API Response** → Client receives results or job status

## Error Handling

The system includes comprehensive error handling:

- **File validation** with security scanning
- **OCR fallback** when native extraction fails
- **Retry mechanisms** for transient failures
- **Graceful degradation** with partial results
- **Detailed logging** for debugging
- **User-friendly error messages**

## Security

- **File type validation** with content verification
- **Malware scanning** patterns for suspicious content
- **Secure file storage** with access controls
- **Input sanitization** for all user data
- **Rate limiting** and resource controls
- **Audit logging** for compliance

## Monitoring

- **Health check endpoints** for service status
- **Performance metrics** collection
- **Error rate monitoring** with alerts
- **Resource usage tracking**
- **Job queue statistics**
- **Storage utilization metrics**

## Troubleshooting

### Common Issues

1. **OCR not working**
   ```bash
   # Check tesseract installation
   tesseract --version
   
   # Install language packs
   sudo apt-get install tesseract-ocr-eng
   ```

2. **Redis connection failed**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Start Redis
   sudo systemctl start redis-server
   ```

3. **Storage permissions**
   ```bash
   # Check directory permissions
   ls -la /path/to/storage
   
   # Fix permissions
   chmod 755 /path/to/storage
   ```

4. **Memory issues with large files**
   - Increase worker memory limits
   - Use streaming processing for large files
   - Enable file chunking

### Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Service-specific loggers
logging.getLogger('app.services.pdf_service').setLevel(logging.DEBUG)
logging.getLogger('app.services.batch_processor').setLevel(logging.INFO)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the integration tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the integration tests for usage examples
- File an issue with detailed error logs and system information