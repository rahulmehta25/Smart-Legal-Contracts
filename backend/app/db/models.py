"""
SQLAlchemy models for Arbitration Detection RAG System
Optimized for both read-heavy workloads and efficient writes
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, LargeBinary, JSON, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from datetime import datetime, timedelta
import hashlib
import json
from typing import List, Dict, Any, Optional

Base = declarative_base()


class Document(Base):
    """Document model with optimized indexing for fast retrieval"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)
    upload_date = Column(DateTime, default=func.now())
    last_processed = Column(DateTime)
    processing_status = Column(String(50), default='pending')
    total_pages = Column(Integer)
    total_chunks = Column(Integer, default=0)
    metadata = Column(SQLiteJSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes for optimization
    __table_args__ = (
        Index('idx_documents_status', 'processing_status'),
        Index('idx_documents_hash', 'content_hash'),
        Index('idx_documents_type', 'file_type'),
        Index('idx_documents_upload_date', 'upload_date'),
        Index('idx_documents_filename', 'filename'),
        CheckConstraint('file_size > 0', name='check_file_size_positive'),
        CheckConstraint("processing_status IN ('pending', 'processing', 'completed', 'failed')", 
                       name='check_processing_status')
    )
    
    @validates('file_type')
    def validate_file_type(self, key, value):
        allowed_types = ['pdf', 'docx', 'txt', 'doc', 'rtf']
        if value.lower() not in allowed_types:
            raise ValueError(f"File type {value} not supported. Allowed: {allowed_types}")
        return value.lower()
    
    @property
    def is_processed(self) -> bool:
        return self.processing_status == 'completed'
    
    @property
    def detection_count(self) -> int:
        return len(self.detections) if self.detections else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'processing_status': self.processing_status,
            'total_pages': self.total_pages,
            'total_chunks': self.total_chunks,
            'detection_count': self.detection_count,
            'metadata': self.metadata
        }


class Chunk(Base):
    """Document chunk model optimized for vector similarity search"""
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    chunk_hash = Column(String(64), nullable=False)
    page_number = Column(Integer)
    section_title = Column(Text)
    embedding_vector = Column(LargeBinary)  # Binary storage for efficiency
    embedding_model = Column(String(100), nullable=False, default='text-embedding-ada-002')
    similarity_threshold = Column(Float, default=0.7)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    detections = relationship("Detection", back_populates="chunk", cascade="all, delete-orphan")
    
    # Optimized indexes for vector search and retrieval
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_hash', 'chunk_hash'),
        Index('idx_chunks_length', 'content_length'),
        Index('idx_chunks_page', 'page_number'),
        Index('idx_chunks_doc_index', 'document_id', 'chunk_index'),  # Composite index
        CheckConstraint('content_length > 0', name='check_content_length_positive'),
        CheckConstraint('chunk_index >= 0', name='check_chunk_index_non_negative'),
        CheckConstraint('similarity_threshold >= 0 AND similarity_threshold <= 1', 
                       name='check_similarity_threshold_range')
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.content and not self.content_length:
            self.content_length = len(self.content)
        if self.content and not self.chunk_hash:
            self.chunk_hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate SHA-256 hash of content for deduplication"""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
    
    def set_embedding(self, embedding: List[float]) -> None:
        """Store embedding as binary data for efficiency"""
        import numpy as np
        self.embedding_vector = np.array(embedding, dtype=np.float32).tobytes()
    
    def get_embedding(self) -> Optional[List[float]]:
        """Retrieve embedding from binary storage"""
        if not self.embedding_vector:
            return None
        import numpy as np
        return np.frombuffer(self.embedding_vector, dtype=np.float32).tolist()
    
    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0 if self.detections else False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'content_length': self.content_length,
            'page_number': self.page_number,
            'section_title': self.section_title,
            'has_embedding': self.embedding_vector is not None,
            'has_detections': self.has_detections
        }


class Detection(Base):
    """Arbitration clause detection results with confidence scoring"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey('chunks.id'), nullable=False)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    detection_type = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    pattern_id = Column(Integer, ForeignKey('patterns.id'))
    matched_text = Column(Text, nullable=False)
    context_before = Column(Text)
    context_after = Column(Text)
    start_position = Column(Integer)
    end_position = Column(Integer)
    page_number = Column(Integer)
    detection_method = Column(String(100), nullable=False)
    model_version = Column(String(50))
    is_validated = Column(Boolean, default=False)
    validation_score = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    chunk = relationship("Chunk", back_populates="detections")
    document = relationship("Document", back_populates="detections")
    pattern = relationship("Pattern", back_populates="detections")
    
    # Optimized indexes for fast filtering and retrieval
    __table_args__ = (
        Index('idx_detections_chunk_id', 'chunk_id'),
        Index('idx_detections_document_id', 'document_id'),
        Index('idx_detections_type', 'detection_type'),
        Index('idx_detections_confidence', 'confidence_score'),
        Index('idx_detections_validated', 'is_validated'),
        Index('idx_detections_created', 'created_at'),
        # Composite indexes for common query patterns
        Index('idx_detections_doc_type', 'document_id', 'detection_type'),
        Index('idx_detections_type_confidence', 'detection_type', 'confidence_score'),
        Index('idx_detections_pattern_confidence', 'pattern_id', 'confidence_score'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', 
                       name='check_confidence_score_range'),
        CheckConstraint("detection_method IN ('rule_based', 'ml_model', 'hybrid')", 
                       name='check_detection_method'),
        CheckConstraint('start_position >= 0', name='check_start_position_non_negative'),
        CheckConstraint('end_position >= start_position', name='check_end_position_valid')
    )
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, value):
        if not (0 <= value <= 1):
            raise ValueError("Confidence score must be between 0 and 1")
        return value
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence_score >= 0.8
    
    @property
    def context_snippet(self) -> str:
        """Get a context snippet around the matched text"""
        before = self.context_before[-100:] if self.context_before else ""
        after = self.context_after[:100] if self.context_after else ""
        return f"{before}{self.matched_text}{after}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'detection_type': self.detection_type,
            'confidence_score': self.confidence_score,
            'matched_text': self.matched_text,
            'context_snippet': self.context_snippet,
            'page_number': self.page_number,
            'detection_method': self.detection_method,
            'is_validated': self.is_validated,
            'is_high_confidence': self.is_high_confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Pattern(Base):
    """Arbitration clause patterns for matching and detection"""
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_name = Column(String(200), nullable=False)
    pattern_text = Column(Text, nullable=False)
    pattern_type = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)
    language = Column(String(10), default='en')
    effectiveness_score = Column(Float, default=0.5)
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(100), default='system')
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    detections = relationship("Detection", back_populates="pattern")
    
    # Indexes for pattern matching optimization
    __table_args__ = (
        Index('idx_patterns_type', 'pattern_type'),
        Index('idx_patterns_category', 'category'),
        Index('idx_patterns_active', 'is_active'),
        Index('idx_patterns_effectiveness', 'effectiveness_score'),
        Index('idx_patterns_usage', 'usage_count'),
        CheckConstraint('effectiveness_score >= 0 AND effectiveness_score <= 1', 
                       name='check_effectiveness_score_range'),
        CheckConstraint("pattern_type IN ('regex', 'keyword', 'semantic')", 
                       name='check_pattern_type'),
        CheckConstraint('usage_count >= 0', name='check_usage_count_non_negative')
    )
    
    @validates('effectiveness_score')
    def validate_effectiveness_score(self, key, value):
        if not (0 <= value <= 1):
            raise ValueError("Effectiveness score must be between 0 and 1")
        return value
    
    @property
    def is_effective(self) -> bool:
        return self.effectiveness_score >= 0.7
    
    def increment_usage(self) -> None:
        """Increment usage count and update last used timestamp"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pattern_name': self.pattern_name,
            'pattern_type': self.pattern_type,
            'category': self.category,
            'language': self.language,
            'effectiveness_score': self.effectiveness_score,
            'usage_count': self.usage_count,
            'is_active': self.is_active,
            'is_effective': self.is_effective,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class QueryCache(Base):
    """Cache table for frequently accessed query results"""
    __tablename__ = 'query_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(255), nullable=False, unique=True)
    query_hash = Column(String(64), nullable=False)
    result_data = Column(SQLiteJSON, nullable=False)
    result_count = Column(Integer, default=0)
    access_count = Column(Integer, default=1)
    ttl = Column(Integer, nullable=False)  # Time to live in seconds
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    
    # Indexes for cache optimization
    __table_args__ = (
        Index('idx_cache_key', 'cache_key'),
        Index('idx_cache_expires', 'expires_at'),
        Index('idx_cache_access_count', 'access_count'),
        CheckConstraint('ttl > 0', name='check_ttl_positive'),
        CheckConstraint('access_count > 0', name='check_access_count_positive')
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.ttl and not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)
    
    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at
    
    def increment_access(self) -> None:
        """Increment access count"""
        self.access_count += 1
    
    @classmethod
    def generate_cache_key(cls, query_type: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from query type and parameters"""
        params_str = json.dumps(params, sort_keys=True)
        return f"{query_type}:{hashlib.md5(params_str.encode()).hexdigest()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cache_key': self.cache_key,
            'result_count': self.result_count,
            'access_count': self.access_count,
            'ttl': self.ttl,
            'is_expired': self.is_expired,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


# Performance optimization utilities
class DatabaseOptimizer:
    """Utility class for database performance optimization"""
    
    @staticmethod
    def get_table_stats(session) -> Dict[str, Any]:
        """Get table statistics for performance monitoring"""
        stats = {}
        
        # Document statistics
        doc_count = session.query(Document).count()
        processed_docs = session.query(Document).filter(
            Document.processing_status == 'completed'
        ).count()
        
        # Chunk statistics
        chunk_count = session.query(Chunk).count()
        chunks_with_embeddings = session.query(Chunk).filter(
            Chunk.embedding_vector.isnot(None)
        ).count()
        
        # Detection statistics
        detection_count = session.query(Detection).count()
        high_confidence_detections = session.query(Detection).filter(
            Detection.confidence_score >= 0.8
        ).count()
        
        # Pattern statistics
        active_patterns = session.query(Pattern).filter(Pattern.is_active == True).count()
        
        stats = {
            'documents': {
                'total': doc_count,
                'processed': processed_docs,
                'processing_rate': processed_docs / doc_count if doc_count > 0 else 0
            },
            'chunks': {
                'total': chunk_count,
                'with_embeddings': chunks_with_embeddings,
                'embedding_rate': chunks_with_embeddings / chunk_count if chunk_count > 0 else 0
            },
            'detections': {
                'total': detection_count,
                'high_confidence': high_confidence_detections,
                'confidence_rate': high_confidence_detections / detection_count if detection_count > 0 else 0
            },
            'patterns': {
                'active': active_patterns
            }
        }
        
        return stats
    
    @staticmethod
    def cleanup_expired_cache(session) -> int:
        """Clean up expired cache entries"""
        expired_count = session.query(QueryCache).filter(
            QueryCache.expires_at < func.now()
        ).delete()
        session.commit()
        return expired_count