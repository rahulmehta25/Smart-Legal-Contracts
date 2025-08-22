from pydantic_settings import BaseSettings
from typing import Optional, List
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
    port: int = 8000
    
    # Database
    database_url: str = "sqlite:///./arbitration_rag.db"
    
    # Vector Store
    vector_store_path: str = "./chroma_db"
    vector_collection_name: str = "arbitration_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # RAG Pipeline
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_retrieval_results: int = 20
    min_confidence_threshold: float = 0.4
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["text/plain", "application/pdf", "text/markdown"]
    upload_directory: str = "./uploads"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # CORS
    allowed_origins: List[str] = ["*"]
    allow_credentials: bool = True
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Model Configuration
    device: str = "auto"  # "auto", "cpu", or "cuda"
    batch_size: int = 32
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings
    """
    return settings


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