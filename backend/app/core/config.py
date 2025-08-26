from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application settings and configuration
    """
    
    # Application
    app_name: str = "Arbitration RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    
    # Environment detection
    environment: str = os.getenv("ENVIRONMENT", "production")
    
    # Database - Production ready
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/arbitration_db"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Vector Store
    vector_store_path: str = "./chroma_db"
    vector_collection_name: str = "arbitration_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Security - Production secure defaults
    secret_key: str = os.getenv("SECRET_KEY", "change-this-in-production-please")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # RAG Pipeline
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_retrieval_results: int = 20
    min_confidence_threshold: float = 0.4
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: str = "text/plain,application/pdf,text/markdown"
    upload_directory: str = "./uploads"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # CORS
    allowed_origins: List[str] = [
        "https://smart-legal-contracts.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ] if os.getenv("ENVIRONMENT", "development") == "production" else ["*"]
    allow_credentials: bool = True
    
    # Rate Limiting - Production settings
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Redis configuration for rate limiting and caching
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_max_connections: int = 10
    
    # Model Configuration - Production optimized
    device: str = os.getenv("MODEL_DEVICE", "cpu")  # "auto", "cpu", or "cuda"
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))  # Smaller for production
    
    # Health check configuration
    health_check_timeout: int = 30
    
    # Production optimization flags
    enable_docs: bool = os.getenv("ENABLE_DOCS", "false").lower() == "true"
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance with caching
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings
    """
    return Settings()


settings = get_settings()


# Helper function to get allowed file types as a list
def get_allowed_file_types() -> List[str]:
    """Convert comma-separated file types string to list"""
    return [ft.strip() for ft in settings.allowed_file_types.split(",") if ft.strip()]


# Legacy function for backward compatibility
def get_app_settings() -> Settings:
    """
    Get application settings (legacy function)
    """
    return get_settings()


# Environment-specific configurations
def get_database_url() -> str:
    """
    Get database URL with environment-specific handling
    """
    url = settings.database_url
    
    # Ensure directory exists for SQLite
    if url.startswith("sqlite"):
        db_path = url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    return url


def get_upload_directory() -> str:
    """
    Get upload directory and ensure it exists
    """
    upload_dir = settings.upload_directory
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def get_vector_store_path() -> str:
    """
    Get vector store path and ensure directory exists
    """
    vector_path = settings.vector_store_path
    if not os.path.exists(vector_path):
        os.makedirs(vector_path, exist_ok=True)
    return vector_path