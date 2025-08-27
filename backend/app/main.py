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
from app.api.health import router as health_router
from app.api.advanced_rag_api import router as advanced_rag_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging, RequestLoggingMiddleware
from app.core.error_handlers import (
    api_error_handler,
    http_exception_handler,
    validation_error_handler,
    database_error_handler,
    general_exception_handler,
    APIError,
    DatabaseConnectionError
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application
    """
    # Startup
    # Configure logging first
    configure_logging()
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


# Get settings
settings = get_settings()

# Create FastAPI application with conditional docs
app = FastAPI(
    title="Arbitration RAG API",
    description="RAG system for detecting arbitration clauses in Terms of Use documents",
    version="1.0.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Comprehensive exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(SQLAlchemyError, database_error_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Include API routes
app.include_router(health_router)  # Health checks at root level
app.include_router(documents_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(advanced_rag_router)  # Advanced RAG endpoints




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
            "users": "/api/v1/users"
        },
        "features": [
            "Document upload and processing",
            "Arbitration clause detection",
            "Vector similarity search",
            "User authentication",
            "Analysis history and statistics"
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