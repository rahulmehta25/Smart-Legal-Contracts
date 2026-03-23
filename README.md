# Smart Legal Contracts

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688?logo=fastapi&logoColor=white)
![Legal BERT](https://img.shields.io/badge/Legal%20BERT-NLP-orange?logo=huggingface&logoColor=white)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.2+-yellow?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/Next.js-14+-black?logo=next.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

AI-powered legal document analysis for arbitration clause detection using RAG (Retrieval-Augmented Generation). Upload legal documents and instantly detect arbitration clauses, jury waivers, and class action waivers with 85%+ accuracy.

**Built by Rahul Mehta** | Research presented at Harvard NCRC 2025

## Overview

Smart Legal Contracts analyzes legal documents to identify potentially binding arbitration provisions that consumers often unknowingly agree to. The system extracts and classifies:

- **Binding Arbitration Clauses**: Mandatory arbitration requirements that waive court access
- **Jury Trial Waivers**: Provisions eliminating the right to a jury trial
- **Class Action Waivers**: Terms preventing collective legal action
- **Dispute Resolution Terms**: Alternative dispute resolution requirements

Each detected clause includes a confidence score, enforceability assessment, and plain-language explanation of the legal implications.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│                    Next.js 14 + Tailwind + Radix UI                         │
│                         (Vercel / localhost:3000)                           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Backend                                 │
│                         (Cloud Run / localhost:8000)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Document Upload  │  Text Extraction  │  Analysis API  │  Batch Processing  │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────┐
│  Text Processor  │          │   Hybrid Retrieval   │        │   ML Pipeline    │
│                  │          │                      │        │                  │
│ • Preprocessing  │    ┌─────┤ • Keyword Search     │        │ • Legal BERT     │
│ • Section detect │    │     │   (30% weight)       │        │ • Classifiers    │
│ • Legal patterns │    │     │ • Semantic Search    │        │ • Threshold opt  │
└────────┬─────────┘    │     │   (70% weight)       │        └────────┬─────────┘
         │              │     └──────────┬───────────┘                 │
         │              │                │                             │
         ▼              ▼                ▼                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Confidence Scoring                                 │
│         Pattern matching + Semantic similarity + Classification fusion       │
└──────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────┐
│     SQLite       │          │  MemoryVectorStore   │        │   Risk Report    │
│   (Documents)    │          │   (Embeddings)       │        │   Generation     │
└──────────────────┘          └──────────────────────┘        └──────────────────┘
```

### Pipeline Flow

1. **Document Upload**: PDF, DOCX, or TXT files are uploaded and text is extracted
2. **Preprocessing**: Legal text is cleaned, sections are detected, and the document is chunked
3. **Hybrid Retrieval**: Combines keyword matching (for explicit terms like "binding arbitration") with semantic search (for paraphrased or obscure clauses)
4. **Classification**: Legal BERT classifies candidate chunks as arbitration-related or not
5. **Confidence Scoring**: Multi-signal fusion produces final confidence scores
6. **Risk Assessment**: Generates plain-language explanations and enforceability ratings

## Why This Architecture

| Design Choice | Rationale |
|--------------|-----------|
| **Hybrid Retrieval (30% keyword / 70% semantic)** | Keywords catch explicit mentions ("binding arbitration"); semantic search catches paraphrased or legally equivalent language that keywords would miss |
| **Legal BERT (nlpaueb/legal-bert-base-uncased)** | Pre-trained on legal corpora, outperforms general-purpose transformers on legal NLP tasks by 8-12% on benchmark datasets |
| **Lazy-loaded ML Models** | sentence-transformers and Legal BERT are loaded on first request, not at startup. Critical for Cloud Run cold starts where memory spikes cause OOM kills |
| **SQLite + MemoryVectorStore** | Zero-config deployment for demos. No external database dependencies. Swap to PostgreSQL + Qdrant for production scale |
| **FastAPI + Pydantic** | Type-safe API with automatic OpenAPI docs. Async support for I/O-bound embedding operations |

## Features

### Clause Detection
- Arbitration clauses with binding/non-binding classification
- Jury trial waivers
- Class action waivers
- Dispute resolution provisions
- Governing law and jurisdiction terms

### Risk Assessment
- Enforceability scoring (0-100)
- Plain-language explanations
- Consumer rights impact analysis
- Comparison to industry standard terms

### Batch Processing
- Upload multiple documents for parallel analysis
- Export results as JSON or CSV
- Aggregate statistics across document sets

### Document Comparison
- Side-by-side clause comparison between documents
- Highlight differences in arbitration terms
- Track changes across contract versions

## Research Context

This project originated from research on consumer contract fairness presented at:

**Harvard Negotiation & Conflict Resolution Collaboratory (NCRC) 2025**
- Research on arbitration clause prevalence in consumer agreements
- Analysis of linguistic patterns in enforceable vs. unenforceable clauses
- Development of detection methodology achieving 85%+ accuracy

**Georgia Tech AI Research**
- Application of transformer models to legal NLP
- Hybrid retrieval optimization for legal document search
- Confidence calibration for high-stakes classification

## Tech Stack

**Backend**
- Python 3.11+
- FastAPI + Uvicorn
- sentence-transformers (all-MiniLM-L6-v2)
- Legal BERT (nlpaueb/legal-bert-base-uncased)
- scikit-learn (ensemble classifiers)
- SQLAlchemy + SQLite
- MLflow (experiment tracking)

**Frontend**
- Next.js 14
- React 18 + TypeScript
- Tailwind CSS + Radix UI
- TanStack Query
- Recharts (visualizations)

**Infrastructure**
- GCP Cloud Run (backend)
- Vercel (frontend)
- Docker + Docker Compose

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/rahulmehta25/Smart-Legal-Contracts.git
cd Smart-Legal-Contracts

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start backend
uvicorn app.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

Access the application:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Docker

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## API Reference

### Health Check

```http
GET /health
```

Returns service status and loaded components.

### Upload Document

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <document.pdf>
```

Returns document ID for subsequent analysis.

### Analyze Document

```http
POST /api/v1/analysis/{document_id}
Content-Type: application/json

{
  "analysis_type": "arbitration",
  "include_explanations": true
}
```

Returns detected clauses with confidence scores.

### Search Documents

```http
POST /api/v1/documents/search
Content-Type: application/json

{
  "query": "binding arbitration waiver",
  "limit": 10
}
```

Semantic search across indexed documents.

### Batch Analysis

```http
POST /api/v1/batch/analyze
Content-Type: multipart/form-data

files: [<doc1.pdf>, <doc2.pdf>, ...]
```

Analyze multiple documents in parallel.

Full API documentation available at `/docs` when running the backend.

## Deployment

### Cloud Run (GCP)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/smart-legal-contracts

# Deploy
gcloud run deploy smart-legal-contracts \
  --image gcr.io/PROJECT_ID/smart-legal-contracts \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 2
```

### Vercel (Frontend)

```bash
cd frontend
vercel --prod
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | SQLite path or PostgreSQL URL | No (defaults to SQLite) |
| `CORS_ORIGINS` | Allowed origins for CORS | No (defaults to *) |
| `LOG_LEVEL` | Logging verbosity | No (defaults to INFO) |
| `ENABLE_DOCS` | Enable Swagger UI | No (defaults to true) |

## License

MIT License. See [LICENSE](LICENSE) for details.
