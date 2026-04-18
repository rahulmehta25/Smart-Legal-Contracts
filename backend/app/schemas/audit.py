"""
Pydantic v2 schemas for audit trail and history tracking

Provides:
- Analysis history records
- Audit log entries
- User activity tracking
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class AuditAction(str, Enum):
    """Types of auditable actions"""
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_UPDATE = "document_update"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    BATCH_ANALYSIS = "batch_analysis"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    EXPORT_DATA = "export_data"
    SETTINGS_CHANGED = "settings_changed"


class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Audit log entry ID")
    timestamp: datetime = Field(..., description="When the action occurred")
    action: AuditAction = Field(..., description="Type of action performed")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    user_email: Optional[str] = Field(None, description="User email")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

    # Resource information
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of affected resource")
    resource_name: Optional[str] = Field(None, description="Name of affected resource")

    # Action details
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional action details"
    )
    changes: Optional[Dict[str, Any]] = Field(
        None,
        description="Before/after values for changes"
    )

    # Result
    success: bool = Field(True, description="Whether action succeeded")
    error_message: Optional[str] = Field(None, description="Error if action failed")
    duration_ms: Optional[int] = Field(None, description="Action duration in ms")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "timestamp": "2024-01-15T10:30:00Z",
                "action": "analysis_complete",
                "user_id": "user_123",
                "user_email": "user@example.com",
                "ip_address": "192.168.1.1",
                "request_id": "req_abc123",
                "resource_type": "document",
                "resource_id": "42",
                "resource_name": "contract.pdf",
                "details": {
                    "has_arbitration_clause": True,
                    "confidence_score": 0.87
                },
                "success": True,
                "duration_ms": 245
            }
        }
    )


class AuditLogCreate(BaseModel):
    """Schema for creating audit log entries"""
    action: AuditAction = Field(..., description="Type of action")
    resource_type: str = Field(..., max_length=50, description="Resource type")
    resource_id: Optional[str] = Field(None, max_length=100)
    resource_name: Optional[str] = Field(None, max_length=255)
    details: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None


class AuditLogQuery(BaseModel):
    """Query parameters for audit log search"""
    start_date: Optional[datetime] = Field(
        None,
        description="Filter entries from this date"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Filter entries until this date"
    )
    action: Optional[AuditAction] = Field(
        None,
        description="Filter by action type"
    )
    user_id: Optional[str] = Field(
        None,
        description="Filter by user ID"
    )
    resource_type: Optional[str] = Field(
        None,
        description="Filter by resource type"
    )
    resource_id: Optional[str] = Field(
        None,
        description="Filter by resource ID"
    )
    success_only: Optional[bool] = Field(
        None,
        description="Filter by success status"
    )
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=500, description="Items per page")

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v, info):
        """Ensure end_date is after start_date"""
        if v is not None and info.data.get("start_date") is not None:
            if v < info.data["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v


class AuditLogResponse(BaseModel):
    """Paginated audit log response"""
    entries: List[AuditLogEntry] = Field(..., description="Audit log entries")
    total: int = Field(..., ge=0, description="Total matching entries")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entries": [
                    {
                        "id": 1,
                        "timestamp": "2024-01-15T10:30:00Z",
                        "action": "analysis_complete",
                        "resource_type": "document",
                        "resource_id": "42",
                        "success": True
                    }
                ],
                "total": 100,
                "page": 1,
                "page_size": 50,
                "total_pages": 2,
                "has_next": True,
                "has_previous": False
            }
        }
    )


class AnalysisHistoryEntry(BaseModel):
    """Historical record of an analysis"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Analysis record ID")
    document_id: int = Field(..., description="Analyzed document ID")
    document_name: Optional[str] = Field(None, description="Document filename")
    user_id: Optional[str] = Field(None, description="User who ran analysis")
    user_email: Optional[str] = Field(None, description="User email")

    # Analysis results
    has_arbitration_clause: bool = Field(
        ...,
        description="Whether arbitration was detected"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    clauses_count: int = Field(
        0,
        ge=0,
        description="Number of clauses found"
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Summary of findings"
    )

    # Metadata
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    analysis_version: str = Field("1.0", description="Algorithm version")
    processing_time_ms: Optional[int] = Field(None, description="Processing time")

    # Request context
    request_id: Optional[str] = Field(None, description="Request ID")
    ip_address: Optional[str] = Field(None, description="Client IP")
    source: str = Field(
        "api",
        description="Analysis source (api, batch, scheduled)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "document_id": 42,
                "document_name": "contract.pdf",
                "user_id": "user_123",
                "has_arbitration_clause": True,
                "confidence_score": 0.87,
                "clauses_count": 2,
                "analysis_summary": "Found binding arbitration with class action waiver",
                "analyzed_at": "2024-01-15T10:30:00Z",
                "analysis_version": "1.0",
                "processing_time_ms": 245,
                "source": "api"
            }
        }
    )


class AnalysisHistoryQuery(BaseModel):
    """Query parameters for analysis history"""
    document_id: Optional[int] = Field(None, description="Filter by document")
    user_id: Optional[str] = Field(None, description="Filter by user")
    has_arbitration: Optional[bool] = Field(
        None,
        description="Filter by detection result"
    )
    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score"
    )
    start_date: Optional[datetime] = Field(None, description="From date")
    end_date: Optional[datetime] = Field(None, description="To date")
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=500)


class AnalysisHistoryResponse(BaseModel):
    """Paginated analysis history response"""
    entries: List[AnalysisHistoryEntry] = Field(..., description="History entries")
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1)
    total_pages: int = Field(..., ge=0)
    has_next: bool = Field(...)
    has_previous: bool = Field(...)

    # Aggregated stats
    total_with_arbitration: int = Field(
        0,
        description="Total analyses with arbitration detected"
    )
    average_confidence: float = Field(
        0.0,
        description="Average confidence score"
    )
