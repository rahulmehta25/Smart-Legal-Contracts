"""
Comprehensive Exception Classes for Arbitration RAG API

Provides:
- Domain-specific exceptions
- Standardized error codes
- HTTP status mapping
- Detailed error context
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
)


class ErrorCode(str, Enum):
    """Standardized error codes for API responses"""

    # Generic errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    BAD_REQUEST = "BAD_REQUEST"

    # Authentication/Authorization
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_API_KEY = "INVALID_API_KEY"
    FORBIDDEN = "FORBIDDEN"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    ANALYSIS_NOT_FOUND = "ANALYSIS_NOT_FOUND"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    CONFLICT = "CONFLICT"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"

    # Document processing
    DOCUMENT_PROCESSING_ERROR = "DOCUMENT_PROCESSING_ERROR"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    DOCUMENT_EXTRACTION_FAILED = "DOCUMENT_EXTRACTION_FAILED"
    DOCUMENT_NOT_PROCESSED = "DOCUMENT_NOT_PROCESSED"

    # Analysis errors
    ANALYSIS_ERROR = "ANALYSIS_ERROR"
    ANALYSIS_TIMEOUT = "ANALYSIS_TIMEOUT"
    INSUFFICIENT_CONTENT = "INSUFFICIENT_CONTENT"
    BATCH_ANALYSIS_PARTIAL_FAILURE = "BATCH_ANALYSIS_PARTIAL_FAILURE"

    # Vector store errors
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"

    # Database errors
    DATABASE_ERROR = "DATABASE_ERROR"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_INTEGRITY_ERROR = "DATABASE_INTEGRITY_ERROR"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class BaseAPIException(Exception):
    """Base exception for all API errors"""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.headers = headers or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for response"""
        error_dict = {
            "message": self.message,
            "error_code": self.error_code.value,
            "status_code": self.status_code,
        }
        if self.details:
            error_dict["details"] = self.details
        return error_dict

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict(),
            headers=self.headers if self.headers else None,
        )


# Authentication/Authorization Exceptions


class AuthenticationError(BaseAPIException):
    """Base authentication error"""

    def __init__(
        self,
        message: str = "Authentication required",
        error_code: ErrorCode = ErrorCode.UNAUTHORIZED,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=HTTP_401_UNAUTHORIZED,
            details=details,
            headers={"WWW-Authenticate": "Bearer"},
        )


class InvalidTokenError(AuthenticationError):
    """Invalid or malformed token"""

    def __init__(
        self,
        message: str = "Invalid or malformed authentication token",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_TOKEN,
            details=details,
        )


class TokenExpiredError(AuthenticationError):
    """Token has expired"""

    def __init__(
        self,
        message: str = "Authentication token has expired",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TOKEN_EXPIRED,
            details=details,
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid API key"""

    def __init__(
        self,
        message: str = "Invalid or revoked API key",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_API_KEY,
            details=details,
        )


class ForbiddenError(BaseAPIException):
    """Access forbidden"""

    def __init__(
        self,
        message: str = "Access forbidden",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            status_code=HTTP_403_FORBIDDEN,
            details=details,
        )


class InsufficientPermissionsError(ForbiddenError):
    """User lacks required permissions"""

    def __init__(
        self,
        required_permissions: List[str],
        message: str = "Insufficient permissions for this action",
    ):
        super().__init__(
            message=message,
            details={"required_permissions": required_permissions},
        )
        self.error_code = ErrorCode.INSUFFICIENT_PERMISSIONS


# Resource Exceptions


class NotFoundError(BaseAPIException):
    """Resource not found"""

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        message: Optional[str] = None,
    ):
        msg = message or f"{resource_type} not found"
        if resource_id:
            msg = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            message=msg,
            error_code=ErrorCode.NOT_FOUND,
            status_code=HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class DocumentNotFoundError(NotFoundError):
    """Document not found"""

    def __init__(self, document_id: int):
        super().__init__(
            resource_type="Document",
            resource_id=str(document_id),
        )
        self.error_code = ErrorCode.DOCUMENT_NOT_FOUND


class AnalysisNotFoundError(NotFoundError):
    """Analysis not found"""

    def __init__(self, analysis_id: int):
        super().__init__(
            resource_type="Analysis",
            resource_id=str(analysis_id),
        )
        self.error_code = ErrorCode.ANALYSIS_NOT_FOUND


class ConflictError(BaseAPIException):
    """Resource conflict"""

    def __init__(
        self,
        message: str = "Resource conflict",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFLICT,
            status_code=HTTP_409_CONFLICT,
            details=details,
        )


class DuplicateResourceError(ConflictError):
    """Duplicate resource error"""

    def __init__(
        self,
        resource_type: str,
        identifier: str,
    ):
        super().__init__(
            message=f"{resource_type} with this identifier already exists",
            details={"resource_type": resource_type, "identifier": identifier},
        )
        self.error_code = ErrorCode.DUPLICATE_RESOURCE


# Document Processing Exceptions


class DocumentProcessingError(BaseAPIException):
    """Document processing failed"""

    def __init__(
        self,
        message: str = "Document processing failed",
        document_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        extra_details = details or {}
        if document_id:
            extra_details["document_id"] = document_id
        super().__init__(
            message=message,
            error_code=ErrorCode.DOCUMENT_PROCESSING_ERROR,
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            details=extra_details,
        )


class DocumentTooLargeError(DocumentProcessingError):
    """Document exceeds size limit"""

    def __init__(
        self,
        file_size: int,
        max_size: int,
    ):
        super().__init__(
            message=f"Document size ({file_size} bytes) exceeds maximum allowed ({max_size} bytes)",
            details={"file_size": file_size, "max_size": max_size},
        )
        self.error_code = ErrorCode.DOCUMENT_TOO_LARGE
        self.status_code = HTTP_400_BAD_REQUEST


class UnsupportedFileTypeError(DocumentProcessingError):
    """Unsupported file type"""

    def __init__(
        self,
        file_type: str,
        supported_types: List[str],
    ):
        super().__init__(
            message=f"File type '{file_type}' is not supported",
            details={"file_type": file_type, "supported_types": supported_types},
        )
        self.error_code = ErrorCode.UNSUPPORTED_FILE_TYPE
        self.status_code = HTTP_400_BAD_REQUEST


class DocumentExtractionError(DocumentProcessingError):
    """Failed to extract content from document"""

    def __init__(
        self,
        document_id: int,
        reason: str,
    ):
        super().__init__(
            message=f"Failed to extract content from document: {reason}",
            document_id=document_id,
            details={"reason": reason},
        )
        self.error_code = ErrorCode.DOCUMENT_EXTRACTION_FAILED


class DocumentNotProcessedError(DocumentProcessingError):
    """Document not yet processed"""

    def __init__(self, document_id: int):
        super().__init__(
            message=f"Document {document_id} has not been processed yet",
            document_id=document_id,
        )
        self.error_code = ErrorCode.DOCUMENT_NOT_PROCESSED


# Analysis Exceptions


class AnalysisError(BaseAPIException):
    """Analysis operation failed"""

    def __init__(
        self,
        message: str = "Analysis failed",
        document_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        extra_details = details or {}
        if document_id:
            extra_details["document_id"] = document_id
        super().__init__(
            message=message,
            error_code=ErrorCode.ANALYSIS_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            details=extra_details,
        )


class AnalysisTimeoutError(AnalysisError):
    """Analysis timed out"""

    def __init__(
        self,
        document_id: int,
        timeout_seconds: int,
    ):
        super().__init__(
            message=f"Analysis timed out after {timeout_seconds} seconds",
            document_id=document_id,
            details={"timeout_seconds": timeout_seconds},
        )
        self.error_code = ErrorCode.ANALYSIS_TIMEOUT
        self.status_code = HTTP_504_GATEWAY_TIMEOUT if hasattr(self, 'status_code') else HTTP_500_INTERNAL_SERVER_ERROR


class InsufficientContentError(AnalysisError):
    """Document has insufficient content for analysis"""

    def __init__(
        self,
        document_id: int,
        content_length: int,
        min_required: int,
    ):
        super().__init__(
            message="Document has insufficient content for meaningful analysis",
            document_id=document_id,
            details={
                "content_length": content_length,
                "min_required": min_required,
            },
        )
        self.error_code = ErrorCode.INSUFFICIENT_CONTENT
        self.status_code = HTTP_422_UNPROCESSABLE_ENTITY


# Vector Store Exceptions


class VectorStoreError(BaseAPIException):
    """Vector store operation failed"""

    def __init__(
        self,
        message: str = "Vector store operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_STORE_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class VectorStoreUnavailableError(VectorStoreError):
    """Vector store is unavailable"""

    def __init__(
        self,
        message: str = "Vector store service is unavailable",
    ):
        super().__init__(message=message)
        self.error_code = ErrorCode.VECTOR_STORE_UNAVAILABLE
        self.status_code = HTTP_503_SERVICE_UNAVAILABLE


class EmbeddingError(VectorStoreError):
    """Embedding generation failed"""

    def __init__(
        self,
        message: str = "Failed to generate embeddings",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, details=details)
        self.error_code = ErrorCode.EMBEDDING_ERROR


# Database Exceptions


class DatabaseError(BaseAPIException):
    """Database operation failed"""

    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DatabaseConnectionError(DatabaseError):
    """Database connection failed"""

    def __init__(
        self,
        message: str = "Failed to connect to database",
    ):
        super().__init__(message=message)
        self.error_code = ErrorCode.DATABASE_CONNECTION_ERROR
        self.status_code = HTTP_503_SERVICE_UNAVAILABLE


class DatabaseIntegrityError(DatabaseError):
    """Database integrity constraint violation"""

    def __init__(
        self,
        message: str = "Database integrity constraint violated",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, details=details)
        self.error_code = ErrorCode.DATABASE_INTEGRITY_ERROR
        self.status_code = HTTP_409_CONFLICT


# Rate Limiting Exceptions


class RateLimitExceededError(BaseAPIException):
    """Rate limit exceeded"""

    def __init__(
        self,
        limit_type: str = "global",
        retry_after: Optional[int] = None,
        reset_time: Optional[int] = None,
    ):
        super().__init__(
            message=f"Rate limit exceeded for {limit_type}",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            details={
                "limit_type": limit_type,
                "retry_after": retry_after,
                "reset_time": reset_time,
            },
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )


class QuotaExceededError(BaseAPIException):
    """Usage quota exceeded"""

    def __init__(
        self,
        quota_type: str,
        current_usage: int,
        quota_limit: int,
    ):
        super().__init__(
            message=f"Usage quota exceeded for {quota_type}",
            error_code=ErrorCode.QUOTA_EXCEEDED,
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            details={
                "quota_type": quota_type,
                "current_usage": current_usage,
                "quota_limit": quota_limit,
            },
        )


# Validation Exceptions


class ValidationError(BaseAPIException):
    """Input validation failed"""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": errors} if errors else None,
        )


# Service Exceptions


class ExternalServiceError(BaseAPIException):
    """External service call failed"""

    def __init__(
        self,
        service_name: str,
        message: str = "External service call failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        extra_details = details or {}
        extra_details["service"] = service_name
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=HTTP_502_BAD_GATEWAY,
            details=extra_details,
        )


class ServiceUnavailableError(BaseAPIException):
    """Service temporarily unavailable"""

    def __init__(
        self,
        service_name: str = "service",
        retry_after: Optional[int] = None,
    ):
        super().__init__(
            message=f"{service_name} is temporarily unavailable",
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service_name, "retry_after": retry_after},
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )
