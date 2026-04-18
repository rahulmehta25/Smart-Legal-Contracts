"""
Pydantic v2 schemas for analysis-related operations

Enhanced with:
- Field validators
- OpenAPI examples
- Batch analysis support
- Confidence level enums
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from enum import Enum


class ClauseType(str, Enum):
    """Types of arbitration clauses"""
    MANDATORY_BINDING = "mandatory_binding_arbitration"
    BINDING = "binding_arbitration"
    MANDATORY = "mandatory_arbitration"
    OPTIONAL = "optional_arbitration"
    CLASS_ACTION_WAIVER = "class_action_waiver"
    JURY_WAIVER = "jury_waiver"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GENERAL = "general_arbitration"


class ConfidenceLevel(str, Enum):
    """Confidence level classifications"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    NONE = "none"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AnalysisStatus(str, Enum):
    """Analysis processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisBase(BaseModel):
    """Base analysis schema"""
    analysis_type: str = Field(
        "arbitration_detection",
        description="Type of analysis to perform"
    )


class AnalysisRequest(AnalysisBase):
    """Request schema for document analysis"""
    document_id: int = Field(
        ...,
        gt=0,
        description="ID of document to analyze",
        json_schema_extra={"example": 1}
    )
    force_reanalysis: bool = Field(
        False,
        description="Force re-analysis even if results exist"
    )
    min_confidence_threshold: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold"
    )
    include_context: bool = Field(
        True,
        description="Include surrounding context for detected clauses"
    )
    analysis_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional analysis configuration options"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": 1,
                "force_reanalysis": False,
                "min_confidence_threshold": 0.4,
                "include_context": True
            }
        }
    )


class QuickAnalysisRequest(BaseModel):
    """Request schema for quick text analysis without persistence"""
    text: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Text content to analyze for arbitration clauses"
    )
    include_context: bool = Field(
        True,
        description="Include analysis context and details"
    )
    min_confidence_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for clause detection"
    )

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Ensure text has meaningful content"""
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("Text must contain at least 10 characters of content")
        return stripped

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Any disputes arising from this agreement shall be resolved through binding arbitration...",
                "include_context": True,
                "min_confidence_threshold": 0.4
            }
        }
    )


class BatchAnalysisItem(BaseModel):
    """Single item in a batch analysis request"""
    document_id: Optional[int] = Field(
        None,
        gt=0,
        description="Document ID (mutually exclusive with text)"
    )
    text: Optional[str] = Field(
        None,
        max_length=100000,
        description="Raw text to analyze (mutually exclusive with document_id)"
    )
    reference_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Client-provided reference ID for tracking"
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        """Ensure either document_id or text is provided, not both"""
        if self.document_id is None and self.text is None:
            raise ValueError("Either document_id or text must be provided")
        if self.document_id is not None and self.text is not None:
            raise ValueError("Provide either document_id or text, not both")
        return self


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch document analysis"""
    items: List[BatchAnalysisItem] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of documents or texts to analyze"
    )
    min_confidence_threshold: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for all analyses"
    )
    include_context: bool = Field(
        True,
        description="Include context in results"
    )
    parallel: bool = Field(
        True,
        description="Process items in parallel for faster results"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"document_id": 1, "reference_id": "contract-001"},
                    {"document_id": 2, "reference_id": "contract-002"},
                    {"text": "Arbitration clause text...", "reference_id": "inline-001"}
                ],
                "min_confidence_threshold": 0.4,
                "include_context": True,
                "parallel": True
            }
        }
    )


class ArbitrationClauseResponse(BaseModel):
    """Response schema for detected arbitration clause"""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = Field(None, description="Clause ID in database")
    clause_text: str = Field(..., description="Full text of the detected clause")
    clause_type: ClauseType = Field(
        ClauseType.GENERAL,
        description="Classification of the clause type"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0-1.0)"
    )
    relevance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Relevance to arbitration query"
    )
    severity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Restrictiveness/severity of the clause"
    )
    start_position: Optional[int] = Field(
        None,
        ge=0,
        description="Start character position in document"
    )
    end_position: Optional[int] = Field(
        None,
        ge=0,
        description="End character position in document"
    )
    page_number: Optional[int] = Field(
        None,
        ge=1,
        description="Page number where clause appears"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section or heading containing the clause"
    )
    surrounding_context: Optional[str] = Field(
        None,
        description="Text surrounding the clause for context"
    )
    signals: Optional[Dict[str, bool]] = Field(
        None,
        description="Detected arbitration signals",
        json_schema_extra={
            "example": {
                "binding_arbitration": True,
                "class_action_waiver": True,
                "jury_waiver": False
            }
        }
    )
    explanation: Optional[str] = Field(
        None,
        description="Human-readable explanation of the finding"
    )


class AnalysisResponse(AnalysisBase):
    """Generic analysis response schema"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Analysis ID")
    document_id: int = Field(..., description="Analyzed document ID")
    user_id: Optional[int] = Field(None, description="User who requested analysis")
    status: AnalysisStatus = Field(..., description="Analysis status")
    overall_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall analysis score"
    )
    risk_level: Optional[RiskLevel] = Field(None, description="Risk assessment level")
    summary: Optional[str] = Field(None, description="Analysis summary")
    recommendations: Optional[str] = Field(None, description="Recommendations based on analysis")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed findings"
    )
    model_version: Optional[str] = Field(None, description="ML model version used")
    processing_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Processing time in milliseconds"
    )
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ArbitrationAnalysisResponse(BaseModel):
    """Response schema for arbitration-specific analysis"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Analysis ID")
    document_id: int = Field(..., description="Analyzed document ID")
    has_arbitration_clause: bool = Field(
        ...,
        description="Whether arbitration clauses were detected"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in detection"
    )
    confidence_level: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Categorical confidence level"
    )
    risk_level: Optional[RiskLevel] = Field(
        None,
        description="Risk assessment based on clause severity"
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Human-readable analysis summary"
    )
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    analysis_version: str = Field("1.0", description="Analysis algorithm version")
    processing_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Processing time in milliseconds"
    )
    clauses: List[ArbitrationClauseResponse] = Field(
        default_factory=list,
        description="Detected arbitration clauses"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional analysis metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "document_id": 1,
                "has_arbitration_clause": True,
                "confidence_score": 0.87,
                "confidence_level": "high",
                "risk_level": "medium",
                "analysis_summary": "Document contains binding arbitration clause with class action waiver.",
                "analyzed_at": "2024-01-15T10:30:00Z",
                "analysis_version": "1.0",
                "processing_time_ms": 245,
                "clauses": [
                    {
                        "clause_text": "Any disputes shall be resolved through binding arbitration...",
                        "clause_type": "binding_arbitration",
                        "confidence_score": 0.92
                    }
                ]
            }
        }
    )


class QuickAnalysisResponse(BaseModel):
    """Response schema for quick text analysis"""
    has_arbitration_clause: bool = Field(
        ...,
        description="Whether arbitration content was detected"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Categorical confidence level"
    )
    clauses_found: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected clauses with details"
    )
    summary: Optional[str] = Field(
        None,
        description="Analysis summary"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    signals_detected: Optional[Dict[str, bool]] = Field(
        None,
        description="Arbitration signals found in text"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_arbitration_clause": True,
                "confidence_score": 0.85,
                "confidence_level": "high",
                "clauses_found": [
                    {
                        "text": "All disputes shall be resolved by binding arbitration...",
                        "type": "binding_arbitration",
                        "score": 0.92
                    }
                ],
                "summary": "Text contains binding arbitration requirements.",
                "processing_time_ms": 45,
                "signals_detected": {
                    "binding_arbitration": True,
                    "mandatory_arbitration": True,
                    "class_action_waiver": False
                }
            }
        }
    )


class BatchAnalysisItemResult(BaseModel):
    """Result for a single item in batch analysis"""
    reference_id: Optional[str] = Field(None, description="Client-provided reference ID")
    document_id: Optional[int] = Field(None, description="Document ID if applicable")
    success: bool = Field(..., description="Whether analysis succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[ArbitrationAnalysisResponse] = Field(
        None,
        description="Analysis result if successful"
    )


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis"""
    total: int = Field(..., ge=0, description="Total items in batch")
    successful: int = Field(..., ge=0, description="Successfully analyzed items")
    failed: int = Field(..., ge=0, description="Failed analyses")
    results: List[BatchAnalysisItemResult] = Field(
        ...,
        description="Individual analysis results"
    )
    total_processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Total processing time"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 3,
                "successful": 2,
                "failed": 1,
                "results": [
                    {
                        "reference_id": "contract-001",
                        "document_id": 1,
                        "success": True,
                        "result": {"has_arbitration_clause": True, "confidence_score": 0.85}
                    }
                ],
                "total_processing_time_ms": 1250
            }
        }
    )


class AnalysisStatistics(BaseModel):
    """Statistics about analyses in the system"""
    total_analyses: int = Field(..., ge=0, description="Total number of analyses")
    completed_analyses: int = Field(..., ge=0, description="Completed analyses")
    failed_analyses: int = Field(..., ge=0, description="Failed analyses")
    with_arbitration_clause: int = Field(
        ...,
        ge=0,
        description="Analyses that detected arbitration clauses"
    )
    without_arbitration_clause: int = Field(
        ...,
        ge=0,
        description="Analyses without detected clauses"
    )
    average_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    average_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average processing time"
    )
    confidence_distribution: Dict[str, int] = Field(
        ...,
        description="Distribution by confidence level"
    )
    clause_type_distribution: Dict[str, int] = Field(
        ...,
        description="Distribution by clause type"
    )
    analyses_by_day: Optional[Dict[str, int]] = Field(
        None,
        description="Analysis counts by day"
    )