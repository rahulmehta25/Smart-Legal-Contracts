from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./arbitration_rag.db")

# Create engine with specific SQLite configurations
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 20
        },
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables
    """
    from app.models.document import Base as DocumentBase
    from app.models.analysis import Base as AnalysisBase
    from app.models.user import Base as UserBase
    
    # Import all models to ensure they are registered
    from app.models import Document, DocumentChunk, ArbitrationAnalysis, ArbitrationClause, User
    
    # Create all tables
    DocumentBase.metadata.create_all(bind=engine)
    AnalysisBase.metadata.create_all(bind=engine)
    UserBase.metadata.create_all(bind=engine)


def create_tables():
    """
    Create database tables if they don't exist
    """
    init_db()


def drop_tables():
    """
    Drop all tables (useful for testing)
    """
    from app.models.document import Base as DocumentBase
    from app.models.analysis import Base as AnalysisBase
    from app.models.user import Base as UserBase
    
    DocumentBase.metadata.drop_all(bind=engine)
    AnalysisBase.metadata.drop_all(bind=engine)
    UserBase.metadata.drop_all(bind=engine)


def reset_database():
    """
    Reset database by dropping and recreating all tables
    """
    drop_tables()
    create_tables()