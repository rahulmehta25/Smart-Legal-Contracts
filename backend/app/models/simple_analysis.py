"""
Simple analysis model for SQLite testing
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .simple_base import BaseModel, AuditMixin


class Analysis(BaseModel, AuditMixin):
    """Simple analysis model for testing"""
    
    __tablename__ = 'analyses'
    
    # References
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Analysis type and status
    analysis_type = Column(String(100), default='arbitration_detection', nullable=False)
    status = Column(String(50), default='pending', nullable=False, index=True)
    
    # Results
    overall_score = Column(Float, nullable=True)
    risk_level = Column(String(50), nullable=True, index=True)
    
    # Content
    summary = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    findings = Column(JSON, default='[]', nullable=False)
    analysis_metadata = Column(JSON, default='{}', nullable=False)
    
    # Processing info
    model_version = Column(String(50), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships - commented out to avoid circular imports
    # document = relationship("Document", back_populates="analyses")
    # user = relationship("User", back_populates="analyses")
    # detections = relationship("Detection", back_populates="analysis", cascade="all, delete-orphan")
    
    @property
    def is_completed(self) -> bool:
        return self.status == 'completed'
    
    @property
    def has_high_risk(self) -> bool:
        return self.risk_level in ['high', 'critical']


class Detection(BaseModel):
    """Simple detection model for individual findings"""
    
    __tablename__ = 'detections'
    
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=False, index=True)
    
    # Detection details
    detection_type = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    text_content = Column(Text, nullable=False)
    
    # Position in document
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(255), nullable=True)
    
    # Additional info
    explanation = Column(Text, nullable=True)
    detection_metadata = Column(JSON, default='{}', nullable=False)
    
    # Relationships - commented out to avoid circular imports
    # analysis = relationship("Analysis", back_populates="detections")