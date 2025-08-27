"""
Pydantic schemas for document-related operations
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class DocumentBase(BaseModel):
    """Base document schema with common fields"""
    filename: str = Field(..., max_length=255, description="Document filename")
    document_type: Optional[str] = Field(None, max_length=100, description="Type of document")
    language: str = Field("en", max_length=10, description="Document language")
    tags: Optional[List[str]] = Field(None, description="Document tags")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document"""
    content: Optional[str] = Field(None, description="Document text content")
    content_type: str = Field("text/plain", description="MIME type of the document")
    file_size: Optional[int] = Field(None, gt=0, description="File size in bytes")
    is_public: bool = Field(False, description="Whether document is public")
    

class DocumentUpdate(BaseModel):
    """Schema for updating an existing document"""
    filename: Optional[str] = Field(None, max_length=255)
    document_type: Optional[str] = Field(None, max_length=100)
    language: Optional[str] = Field(None, max_length=10)
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class DocumentResponse(DocumentBase):
    """Schema for document responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: Optional[int] = None
    organization_id: Optional[int] = None
    file_path: str
    file_type: str
    file_size: int
    content_hash: str
    upload_date: datetime
    last_processed: Optional[datetime] = None
    processing_status: str
    processing_progress: int = 0
    total_pages: Optional[int] = None
    total_chunks: int = 0
    version: int = 1
    is_public: bool = False
    retention_until: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    """Schema for document upload responses"""
    message: str
    document_id: int
    filename: str
    chunks_created: int = 0
    processing_status: str = "pending"


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    document_id: int
    chunk_index: int
    content: str
    content_length: int
    chunk_hash: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding_model: str
    created_at: datetime


class DocumentSearchRequest(BaseModel):
    """Schema for document search requests"""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=100)
    document_type: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None


class DocumentSearchResponse(BaseModel):
    """Schema for document search responses"""
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    processing_time_ms: Optional[int] = None


class DocumentStatistics(BaseModel):
    """Schema for document statistics"""
    total_documents: int
    total_chunks: int
    documents_by_type: Dict[str, int]
    documents_by_language: Dict[str, int]
    documents_by_status: Dict[str, int]
    recent_uploads: int
    storage_used_bytes: int