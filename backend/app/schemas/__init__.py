"""
Pydantic v2 schemas for request/response models

Comprehensive schema definitions with:
- Field validators and constraints
- OpenAPI examples
- Serialization configuration
- Batch operation support
- Audit trail schemas
"""

from .base import (
    BaseSchema,
    PaginatedResponse,
    PaginationParams,
    ErrorResponse,
    ErrorDetail,
    ErrorCode,
    SuccessResponse,
    TimestampMixin,
    HealthStatus,
    ComponentHealth,
    HealthResponse,
    RequestMetadata,
    BulkOperationResult,
)
from .document import (
    DocumentCreate, DocumentResponse, DocumentUploadResponse,
    DocumentChunkResponse, DocumentUpdate, DocumentSearchRequest,
    DocumentSearchResponse, DocumentStatistics
)
from .analysis import (
    AnalysisRequest, AnalysisResponse, QuickAnalysisRequest,
    QuickAnalysisResponse, ArbitrationAnalysisResponse,
    ArbitrationClauseResponse, BatchAnalysisRequest,
    BatchAnalysisResponse, BatchAnalysisItem, BatchAnalysisItemResult,
    AnalysisStatistics, ClauseType, ConfidenceLevel, RiskLevel,
    AnalysisStatus
)
from .audit import (
    AuditAction, AuditLogEntry, AuditLogCreate, AuditLogQuery,
    AuditLogResponse, AnalysisHistoryEntry, AnalysisHistoryQuery,
    AnalysisHistoryResponse
)
from .user import (
    UserCreate, UserResponse, UserLogin, Token, UserUpdate
)

__all__ = [
    # Base schemas
    "BaseSchema",
    "PaginatedResponse",
    "PaginationParams",
    "ErrorResponse",
    "ErrorDetail",
    "ErrorCode",
    "SuccessResponse",
    "TimestampMixin",
    "HealthStatus",
    "ComponentHealth",
    "HealthResponse",
    "RequestMetadata",
    "BulkOperationResult",
    # Document schemas
    "DocumentCreate",
    "DocumentResponse",
    "DocumentUploadResponse",
    "DocumentChunkResponse",
    "DocumentUpdate",
    "DocumentSearchRequest",
    "DocumentSearchResponse",
    "DocumentStatistics",
    # Analysis schemas
    "AnalysisRequest",
    "AnalysisResponse",
    "QuickAnalysisRequest",
    "QuickAnalysisResponse",
    "ArbitrationAnalysisResponse",
    "ArbitrationClauseResponse",
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "BatchAnalysisItem",
    "BatchAnalysisItemResult",
    "AnalysisStatistics",
    "ClauseType",
    "ConfidenceLevel",
    "RiskLevel",
    "AnalysisStatus",
    # Audit schemas
    "AuditAction",
    "AuditLogEntry",
    "AuditLogCreate",
    "AuditLogQuery",
    "AuditLogResponse",
    "AnalysisHistoryEntry",
    "AnalysisHistoryQuery",
    "AnalysisHistoryResponse",
    # User schemas
    "UserCreate",
    "UserResponse",
    "UserLogin",
    "Token",
    "UserUpdate",
]