from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

Base = declarative_base()


class ArbitrationAnalysis(Base):
    """SQLAlchemy model for arbitration analysis results"""
    __tablename__ = "arbitration_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Analysis results
    has_arbitration_clause = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    analysis_summary = Column(Text, nullable=True)
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String(50), default="1.0")
    processing_time_ms = Column(Integer, nullable=True)
    
    # Store additional analysis data as JSON
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="analyses")
    clauses = relationship("ArbitrationClause", back_populates="analysis", cascade="all, delete-orphan")


class ArbitrationClause(Base):
    """SQLAlchemy model for individual arbitration clauses found"""
    __tablename__ = "arbitration_clauses"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("arbitration_analyses.id"), nullable=False)
    
    # Clause details
    clause_text = Column(Text, nullable=False)
    clause_type = Column(String(100), nullable=True)  # e.g., "mandatory", "optional", "class_action_waiver"
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    
    # Scoring
    relevance_score = Column(Float, nullable=True)  # How relevant this clause is
    severity_score = Column(Float, nullable=True)   # How restrictive the clause is
    
    # Context
    surrounding_context = Column(Text, nullable=True)
    section_title = Column(String(200), nullable=True)
    
    # Relationships
    analysis = relationship("ArbitrationAnalysis", back_populates="clauses")


# Enums for API responses
class ClauseType(str, Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional" 
    CLASS_ACTION_WAIVER = "class_action_waiver"
    BINDING_ARBITRATION = "binding_arbitration"
    DISPUTE_RESOLUTION = "dispute_resolution"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Pydantic models for API
class ArbitrationClauseResponse(BaseModel):
    id: int
    clause_text: str
    clause_type: Optional[str]
    start_position: Optional[int]
    end_position: Optional[int]
    relevance_score: Optional[float]
    severity_score: Optional[float]
    surrounding_context: Optional[str]
    section_title: Optional[str]
    
    class Config:
        from_attributes = True


class ArbitrationAnalysisResponse(BaseModel):
    id: int
    document_id: int
    has_arbitration_clause: bool
    confidence_score: Optional[float]
    analysis_summary: Optional[str]
    analyzed_at: datetime
    analysis_version: str
    processing_time_ms: Optional[int]
    clauses: List[ArbitrationClauseResponse] = []
    
    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    document_id: int
    force_reanalysis: bool = False


class QuickAnalysisRequest(BaseModel):
    text: str
    include_context: bool = True


class QuickAnalysisResponse(BaseModel):
    has_arbitration_clause: bool
    confidence_score: float
    clauses_found: List[Dict[str, Any]]
    summary: str
    processing_time_ms: int