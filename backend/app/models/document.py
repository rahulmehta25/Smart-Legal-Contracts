from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

Base = declarative_base()


class Document(Base):
    """SQLAlchemy model for documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text/plain")
    file_size = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    is_processed = Column(Boolean, default=False)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    analyses = relationship("ArbitrationAnalysis", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """SQLAlchemy model for document chunks"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=True)
    
    # Vector embedding metadata
    embedding_id = Column(String(100), nullable=True)  # ID in vector store
    
    # Relationships
    document = relationship("Document", back_populates="chunks")


# Pydantic models for API
class DocumentBase(BaseModel):
    filename: str
    content_type: Optional[str] = "text/plain"


class DocumentCreate(DocumentBase):
    content: str


class DocumentResponse(DocumentBase):
    id: int
    file_size: Optional[int]
    uploaded_at: datetime
    processed_at: Optional[datetime]
    is_processed: bool
    
    class Config:
        from_attributes = True


class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    token_count: Optional[int]
    
    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    filename: str
    chunks_created: int