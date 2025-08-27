# 🚀 Arbitration Clause Detector - Quick Start Demo

## Prerequisites
- Python 3.8+
- Node.js 16+
- Docker & Docker Compose
- 4GB RAM minimum

## One-Command Setup

Run this single command to get everything working:

```bash
cd /Users/rahulmehta/Desktop/Test/demo
./setup-and-run.sh
```

## Manual Setup (if automatic fails)

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn python-multipart aiofiles sqlalchemy pydantic redis
pip install sentence-transformers numpy scikit-learn
python app/main.py
```

### 2. Frontend Setup (in new terminal)
```bash
cd frontend
npm init -y
npm install next react react-dom typescript @types/react @types/node
npm install tailwindcss autoprefixer postcss
npm run dev
```

### 3. Access the Demo
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Test the System

### 1. Via Web Interface
1. Open http://localhost:3000
2. Click "Try Demo" or "Upload Document"
3. Upload a Terms of Service document
4. View the analysis results

### 2. Via API
```bash
# Test health check
curl http://localhost:8000/health

# Analyze text
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Any disputes arising from this agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.",
    "language": "en"
  }'
```

### 3. Sample Documents
Find test documents in `/demo/sample-data/documents/`:
- `uber_tos.txt` - Uber Terms with arbitration
- `spotify_tos.txt` - Spotify Terms with arbitration
- `github_tos.txt` - GitHub Terms (no arbitration)

## Features to Try

1. **Upload Document**: Drag and drop a PDF or paste text
2. **View Analysis**: See detected clauses with confidence scores
3. **Multi-language**: Try documents in different languages
4. **Export Results**: Download analysis as JSON or PDF
5. **API Integration**: Use the REST API for programmatic access

## Troubleshooting

### Port Already in Use
```bash
# Kill processes on ports
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

### Dependencies Missing
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### Database Issues
```bash
# Reset database
rm -f backend/app.db
python backend/scripts/init_db.py
```

## What's Working

✅ Document text extraction  
✅ Arbitration clause detection  
✅ Confidence scoring  
✅ Web interface  
✅ REST API  
✅ Basic database storage  

## What's In Progress

🔄 Vector similarity search (needs ChromaDB setup)  
🔄 User authentication  
🔄 Real-time collaboration  
🔄 Advanced ML models  

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- API docs: http://localhost:8000/docs
- Debug mode: Set `DEBUG=true` in `.env`