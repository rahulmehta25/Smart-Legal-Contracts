"""
Base Pydantic v2 Schema Models

Foundation schemas with:
- Common field definitions
- Pagination support
- Standard error/success responses
- Timestamp mixins
- OpenAPI documentation helpers
"""

from datetime import datetime
from typing import Generic, TypeVar, List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


# Generic type for paginated responses
T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common configuration"""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
        json_schema_extra={
            "examples": []
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )


class PaginationParams(BaseModel):
    """Pagination query parameters"""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper"""
    items: List[T] = Field(default_factory=list, description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5,
                "has_next": True,
                "has_previous": False
            }
        }
    )


class ErrorCode(str, Enum):
    """Standard error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    BAD_REQUEST = "BAD_REQUEST"
    DOCUMENT_PROCESSING_ERROR = "DOCUMENT_PROCESSING_ERROR"
    ANALYSIS_ERROR = "ANALYSIS_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"


class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any] = Field(
        ...,
        description="Error details",
        json_schema_extra={
            "example": {
                "message": "Resource not found",
                "status_code": 404,
                "error_code": "NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_abc123"
            }
        }
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "message": "Document not found",
                    "status_code": 404,
                    "error_code": "NOT_FOUND",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    )


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"id": 1}
            }
        }
    )


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component"""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component status")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    last_check: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Comprehensive health check response"""
    status: HealthStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    components: Dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Individual component health"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "components": {
                    "database": {
                        "name": "database",
                        "status": "healthy",
                        "response_time_ms": 5.2
                    },
                    "qdrant": {
                        "name": "qdrant",
                        "status": "healthy",
                        "response_time_ms": 12.5
                    }
                }
            }
        }
    )


class RequestMetadata(BaseModel):
    """Request metadata for logging and tracing"""
    request_id: str = Field(..., description="Unique request identifier")
    user_id: Optional[str] = Field(None, description="User ID if authenticated")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BulkOperationResult(BaseModel):
    """Result of bulk operations"""
    total: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors for failed items"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 10,
                "successful": 8,
                "failed": 2,
                "errors": [
                    {"field": "document_id", "message": "Document 5 not found"},
                    {"field": "document_id", "message": "Document 7 is locked"}
                ]
            }
        }
    )
