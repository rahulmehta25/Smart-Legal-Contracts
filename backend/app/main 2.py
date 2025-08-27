from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import os
from loguru import logger

from app.db.database import init_db
from app.db.vector_store import init_vector_store
from app.api import documents_router, analysis_router, users_router
from app.api.pdf_api import router as pdf_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application
    """
    # Startup
    logger.info("Starting Arbitration RAG API...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize vector store
        init_vector_store()
        logger.info("Vector store initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Arbitration RAG API...")


# Create FastAPI application
app = FastAPI(
    title="Arbitration RAG API",
    description="RAG system for detecting arbitration clauses in Terms of Use documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include API routes
app.include_router(documents_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(pdf_router)  # PDF processing routes


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Arbitration RAG API",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Arbitration RAG API",
        "description": "RAG system for detecting arbitration clauses in Terms of Use documents",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# API overview endpoint
@app.get("/api/v1")
async def api_overview():
    """
    API v1 overview
    """
    return {
        "version": "1.0.0",
        "endpoints": {
            "documents": "/api/v1/documents",
            "analysis": "/api/v1/analysis", 
            "users": "/api/v1/users",
            "pdf_processing": "/api/upload/pdf",
            "batch_processing": "/api/batch/upload",
            "job_status": "/api/jobs/{id}/status"
        },
        "features": [
            "Document upload and processing",
            "Arbitration clause detection",
            "Vector similarity search",
            "User authentication",
            "Analysis history and statistics",
            "PDF text extraction with OCR",
            "Multi-format document processing",
            "Batch file processing",
            "Asynchronous job processing",
            "File storage and retrieval"
        ]
    }


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )