"""
Pydantic schemas for analysis-related operations
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
# import uuid


class AnalysisBase(BaseModel):
    """Base analysis schema"""
    analysis_type: str = Field("arbitration_detection", description="Type of analysis")


class AnalysisRequest(AnalysisBase):
    """Schema for analysis requests"""
    document_id: int = Field(..., description="ID of document to analyze")
    analysis_options: Optional[Dict[str, Any]] = Field(None, description="Analysis configuration options")


class QuickAnalysisRequest(BaseModel):
    """Schema for quick text analysis requests"""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to analyze")
    options: Optional[Dict[str, Any]] = Field(None, description="Analysis options")


class ArbitrationClauseResponse(BaseModel):
    """Schema for individual arbitration clause findings"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    clause_text: str
    clause_type: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    start_position: int
    end_position: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    explanation: Optional[str] = None


class AnalysisResponse(AnalysisBase):
    """Schema for analysis responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    document_id: int
    user_id: int
    status: str
    overall_score: Optional[float] = None
    risk_level: Optional[str] = None
    summary: Optional[str] = None
    recommendations: Optional[str] = None
    findings: List[Dict[str, Any]] = []
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ArbitrationAnalysisResponse(BaseModel):
    """Schema for arbitration-specific analysis responses"""
    id: int
    document_id: int
    has_arbitration_clause: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    analysis_summary: Optional[str] = None
    analyzed_at: datetime
    analysis_version: str = "1.0"
    processing_time_ms: Optional[int] = None
    clauses: List[ArbitrationClauseResponse] = []


class QuickAnalysisResponse(BaseModel):
    """Schema for quick analysis responses"""
    has_arbitration_clause: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    clauses_found: List[Dict[str, Any]] = []
    summary: Optional[str] = None
    processing_time_ms: Optional[int] = None


class AnalysisStatistics(BaseModel):
    """Schema for analysis statistics"""
    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    average_confidence_score: float
    analyses_by_risk_level: Dict[str, int]
    analyses_by_type: Dict[str, int]
    recent_analyses: int