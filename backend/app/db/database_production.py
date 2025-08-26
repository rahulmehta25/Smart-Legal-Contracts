from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
import os
from typing import Generator
from loguru import logger
from app.core.config import get_settings

# Get settings for database configuration
settings = get_settings()

# Create engine with production-optimized configurations
if settings.database_url.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_engine(
        settings.database_url,
        connect_args={
            "check_same_thread": False,
            "timeout": 30
        },
        poolclass=StaticPool,
        echo=False,
        pool_pre_ping=True  # Verify connections before use
    )
elif settings.database_url.startswith("postgresql"):
    # PostgreSQL configuration for production
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        pool_pre_ping=True,  # Verify connections before use
        echo=False,
        connect_args={
            "connect_timeout": 10,
            "application_name": "arbitration-rag-api"
        }
    )
else:
    # Generic configuration
    engine = create_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True
    )


# Add connection event listeners for better error handling
@event.listens_for(engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """Handle new database connections"""
    logger.info("New database connection established")
    

@event.listens_for(engine, "close")
def receive_close(dbapi_connection, connection_record):
    """Handle database connection closures"""
    logger.debug("Database connection closed")


@event.listens_for(engine, "invalidate")
def receive_invalidate(dbapi_connection, connection_record, exception):
    """Handle database connection invalidation"""
    logger.warning(f"Database connection invalidated: {exception}")


# Connection health check
def check_database_connection():
    """Check if database connection is healthy"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session with error handling
    """
    db = SessionLocal()
    try:
        # Test connection before yielding
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables with error handling
    """
    try:
        from app.models.document import Base as DocumentBase
        from app.models.analysis import Base as AnalysisBase
        from app.models.user import Base as UserBase
        
        # Import all models to ensure they are registered
        from app.models import Document, DocumentChunk, ArbitrationAnalysis, ArbitrationClause, User
        
        # Create all tables
        DocumentBase.metadata.create_all(bind=engine)
        AnalysisBase.metadata.create_all(bind=engine)
        UserBase.metadata.create_all(bind=engine)
        
        logger.info("Database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise


def create_tables():
    """
    Create database tables if they don't exist
    """
    init_db()


def drop_tables():
    """
    Drop all tables (useful for testing)
    """
    try:
        from app.models.document import Base as DocumentBase
        from app.models.analysis import Base as AnalysisBase
        from app.models.user import Base as UserBase
        
        DocumentBase.metadata.drop_all(bind=engine)
        AnalysisBase.metadata.drop_all(bind=engine)
        UserBase.metadata.drop_all(bind=engine)
        
        logger.info("Database tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


def reset_database():
    """
    Reset database by dropping and recreating all tables
    """
    drop_tables()
    create_tables()


# Connection pool monitoring
def get_connection_pool_status():
    """Get current connection pool status for monitoring"""
    if hasattr(engine.pool, 'size'):
        return {
            "pool_size": engine.pool.size(),
            "checked_in": engine.pool.checkedin(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
            "invalid": engine.pool.invalid()
        }
    return {"status": "pool_info_unavailable"}