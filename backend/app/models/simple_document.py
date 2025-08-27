"""
Simple document model for SQLite testing
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .simple_base import BaseModel, AuditMixin


class Document(BaseModel, AuditMixin):
    """Simple document model for testing"""
    
    __tablename__ = 'documents'
    
    # User information
    user_id = Column(Integer, nullable=True, index=True)  # Simplified for testing
    
    # File information
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(50), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    
    # Processing status
    upload_date = Column(DateTime, default=func.now(), nullable=False, index=True)
    last_processed = Column(DateTime, nullable=True)
    processing_status = Column(String(50), default='pending', nullable=False, index=True)
    processing_progress = Column(Integer, default=0, nullable=False)
    
    # Document metadata
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0, nullable=False)
    language = Column(String(10), default='en', nullable=False, index=True)
    document_type = Column(String(100), nullable=True, index=True)
    document_metadata = Column(JSON, default='{}', nullable=False)
    
    # Version control
    version = Column(Integer, default=1, nullable=False)
    parent_document_id = Column(Integer, ForeignKey('documents.id'), nullable=True)
    
    # Access control
    is_public = Column(Boolean, default=False, nullable=False)
    retention_until = Column(DateTime, nullable=True)
    
    # Relationships - commented out to avoid circular imports
    # chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    # analyses = relationship("Analysis", back_populates="document", cascade="all, delete-orphan")
    
    @property
    def is_processed(self) -> bool:
        return self.processing_status == 'completed'


class Chunk(BaseModel):
    """Simple chunk model for testing"""
    
    __tablename__ = 'chunks'
    
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    chunk_hash = Column(String(64), nullable=False)
    page_number = Column(Integer, nullable=True)
    section_title = Column(Text, nullable=True)
    embedding_model = Column(String(100), nullable=False, default='sentence-transformers')
    
    # Relationships - commented out to avoid circular imports
    # document = relationship("Document", back_populates="chunks")