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
from app.websocket.server import websocket_server, create_websocket_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application
    """
    # Startup
    logger.info("Starting Arbitration RAG API with WebSocket support...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize vector store
        init_vector_store()
        logger.info("Vector store initialized successfully")
        
        # Start WebSocket background tasks
        await websocket_server.start_background_tasks()
        logger.info("WebSocket server background tasks started")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Arbitration RAG API...")
    
    # Shutdown WebSocket server
    try:
        await websocket_server.shutdown()
        logger.info("WebSocket server shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down WebSocket server: {e}")


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
# Configure allowed origins based on environment
allowed_origins = [
    "http://localhost:3000",  # Local development
    "http://localhost:3001",  # Alternative local port
    "http://localhost:5173",  # Vite default port
    "http://localhost:5174",  # Vite alternative port
    "https://arbitration-frontend.vercel.app",  # Your Vercel deployment
    "https://arbitration-frontend.netlify.app",  # Netlify deployment
    "https://arbitration-detector.com",  # Production domain
]

# Add any custom origins from environment variable
custom_origins = os.getenv("CORS_ORIGINS", "").split(",")
if custom_origins and custom_origins[0]:
    allowed_origins.extend(custom_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
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


# Setup WebSocket routes
websocket_server.setup_routes(app)
logger.info("WebSocket routes configured")

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
            "job_status": "/api/jobs/{id}/status",
            "websocket": "/ws",
            "websocket_stats": "/api/websocket/stats",
            "websocket_rooms": "/api/websocket/rooms"
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
            "File storage and retrieval",
            "Real-time WebSocket connections",
            "Live analysis progress tracking",
            "Collaborative document annotation",
            "User presence indicators",
            "Real-time notifications",
            "Document sharing and collaboration",
            "Cursor tracking and synchronization"
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