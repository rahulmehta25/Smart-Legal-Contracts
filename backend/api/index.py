"""
Vercel serverless function entry point for FastAPI backend
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path to import app module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables for Vercel
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("DATABASE_URL", "sqlite:///./arbitration.db")
os.environ.setdefault("SECRET_KEY", "vercel-production-secret-key-change-in-prod")
os.environ.setdefault("ENABLE_DOCS", "false")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Create lightweight FastAPI app for Vercel
app = FastAPI(
    title="Arbitration RAG API",
    description="RAG system for detecting arbitration clauses in Terms of Use documents",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://smart-legal-contracts.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Arbitration RAG API",
        "description": "RAG system for detecting arbitration clauses in Terms of Use documents",
        "version": "1.0.0",
        "status": "running",
        "environment": "vercel_serverless"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arbitration-rag-api",
        "version": "1.0.0",
        "environment": "vercel"
    }

# API overview endpoint
@app.get("/api/v1")
async def api_overview():
    """API v1 overview"""
    return {
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/"
        },
        "status": "serverless_ready"
    }

# Simple analysis endpoint for testing
@app.post("/api/v1/analyze")
async def analyze_text(data: dict):
    """Simple text analysis endpoint"""
    text = data.get("text", "")
    
    # Basic arbitration keyword detection
    arbitration_keywords = [
        "arbitration", "arbitrate", "arbitrator", "binding arbitration",
        "dispute resolution", "mediation", "adr", "alternative dispute resolution"
    ]
    
    found_keywords = [keyword for keyword in arbitration_keywords if keyword.lower() in text.lower()]
    
    return {
        "text": text,
        "has_arbitration_clause": len(found_keywords) > 0,
        "found_keywords": found_keywords,
        "confidence": min(len(found_keywords) * 0.3, 1.0),
        "method": "keyword_based"
    }

# Export for Vercel
handler = app