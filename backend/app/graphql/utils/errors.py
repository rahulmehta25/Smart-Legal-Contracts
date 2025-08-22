"""
GraphQL Error Handling and Custom Exceptions
"""

import logging
from typing import List, Dict, Any, Optional, Union
from graphql import GraphQLError, GraphQLFormattedError
from graphql.error import format_error
from strawberry.types import Info
from enum import Enum


class ErrorCode(Enum):
    """Standard error codes for the API"""
    
    # Authentication & Authorization
    UNAUTHENTICATED = "UNAUTHENTICATED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    
    # Validation
    INVALID_INPUT = "INVALID_INPUT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    FIELD_TOO_LONG = "FIELD_TOO_LONG"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Business Logic
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Query Complexity
    QUERY_TOO_COMPLEX = "QUERY_TOO_COMPLEX"
    QUERY_TOO_DEEP = "QUERY_TOO_DEEP"
    TIMEOUT = "TIMEOUT"
    
    # Processing
    PROCESSING_ERROR = "PROCESSING_ERROR"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    
    # System
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"


class GraphQLAPIError(Exception):
    """Base exception for GraphQL API errors"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        extensions: Optional[Dict[str, Any]] = None,
        path: Optional[List[Union[str, int]]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.extensions = extensions or {}
        self.path = path
    
    def to_graphql_error(self) -> GraphQLError:
        """Convert to GraphQL error"""
        extensions = {
            "code": self.code.value,
            **self.extensions
        }
        
        return GraphQLError(
            message=self.message,
            extensions=extensions,
            path=self.path
        )


class AuthenticationError(GraphQLAPIError):
    """Authentication required error"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            code=ErrorCode.UNAUTHENTICATED,
            extensions={"requires_auth": True}
        )


class AuthorizationError(GraphQLAPIError):
    """Authorization/permission error"""
    
    def __init__(self, message: str = "Insufficient permissions", required_role: Optional[str] = None):
        extensions = {}
        if required_role:
            extensions["required_role"] = required_role
        
        super().__init__(
            message=message,
            code=ErrorCode.UNAUTHORIZED,
            extensions=extensions
        )


class ValidationError(GraphQLAPIError):
    """Input validation error"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        extensions = {}
        if field:
            extensions["field"] = field
        if value is not None:
            extensions["invalid_value"] = str(value)
        
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            extensions=extensions
        )


class ResourceNotFoundError(GraphQLAPIError):
    """Resource not found error"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' not found",
            code=ErrorCode.RESOURCE_NOT_FOUND,
            extensions={
                "resource_type": resource_type,
                "resource_id": resource_id
            }
        )


class ResourceExistsError(GraphQLAPIError):
    """Resource already exists error"""
    
    def __init__(self, resource_type: str, identifier: str):
        super().__init__(
            message=f"{resource_type} with identifier '{identifier}' already exists",
            code=ErrorCode.RESOURCE_ALREADY_EXISTS,
            extensions={
                "resource_type": resource_type,
                "identifier": identifier
            }
        )


class RateLimitError(GraphQLAPIError):
    """Rate limit exceeded error"""
    
    def __init__(self, message: str, retry_after: int, limit_type: str = "general"):
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            extensions={
                "retry_after": retry_after,
                "limit_type": limit_type
            }
        )


class QueryComplexityError(GraphQLAPIError):
    """Query complexity error"""
    
    def __init__(self, message: str, complexity: int, max_complexity: int):
        super().__init__(
            message=message,
            code=ErrorCode.QUERY_TOO_COMPLEX,
            extensions={
                "complexity": complexity,
                "max_complexity": max_complexity
            }
        )


class ProcessingError(GraphQLAPIError):
    """Processing/analysis error"""
    
    def __init__(self, message: str, processing_type: str = "general"):
        super().__init__(
            message=message,
            code=ErrorCode.PROCESSING_ERROR,
            extensions={
                "processing_type": processing_type
            }
        )


class ErrorLogger:
    """Centralized error logging"""
    
    def __init__(self):
        self.logger = logging.getLogger("graphql_errors")
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with context"""
        context = context or {}
        
        if isinstance(error, GraphQLAPIError):
            self.logger.warning(
                f"GraphQL API Error: {error.code.value} - {error.message}",
                extra={
                    "error_code": error.code.value,
                    "extensions": error.extensions,
                    "path": error.path,
                    **context
                }
            )
        else:
            self.logger.error(
                f"Unexpected error: {str(error)}",
                extra={
                    "error_type": type(error).__name__,
                    **context
                },
                exc_info=True
            )


class ErrorHandler:
    """GraphQL error handler with formatting and logging"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = ErrorLogger()
    
    def format_error(self, error: GraphQLError, info: Optional[Info] = None) -> GraphQLFormattedError:
        """Format GraphQL error for response"""
        context = {}
        if info:
            context.update({
                "operation": getattr(info.operation, 'operation', None),
                "field_name": info.field_name,
                "path": info.path,
            })
        
        # Log the error
        if hasattr(error, 'original_error') and error.original_error:
            self.logger.log_error(error.original_error, context)
        else:
            self.logger.log_error(error, context)
        
        # Format error for response
        formatted = format_error(error)
        
        # Add additional context in debug mode
        if self.debug and hasattr(error, 'original_error') and error.original_error:
            formatted['extensions'] = formatted.get('extensions', {})
            formatted['extensions']['debug'] = {
                'exception': type(error.original_error).__name__,
                'trace': str(error.original_error) if not isinstance(error.original_error, GraphQLAPIError) else None
            }
        
        return formatted
    
    def handle_field_error(self, error: Exception, info: Info) -> GraphQLError:
        """Handle field-level errors"""
        if isinstance(error, GraphQLAPIError):
            return error.to_graphql_error()
        
        # Convert common exceptions to GraphQL errors
        if isinstance(error, PermissionError):
            api_error = AuthorizationError("Permission denied")
        elif isinstance(error, ValueError):
            api_error = ValidationError(str(error))
        elif isinstance(error, KeyError):
            api_error = ResourceNotFoundError("Resource", str(error))
        else:
            # Generic internal error
            api_error = GraphQLAPIError(
                message="An internal error occurred" if not self.debug else str(error),
                code=ErrorCode.INTERNAL_ERROR
            )
        
        return api_error.to_graphql_error()


# Error utilities
def create_user_friendly_error(error: Exception) -> GraphQLAPIError:
    """Create user-friendly error from exception"""
    if isinstance(error, GraphQLAPIError):
        return error
    
    error_type = type(error).__name__
    
    error_mappings = {
        "ValidationError": lambda e: ValidationError(str(e)),
        "PermissionError": lambda e: AuthorizationError(str(e)),
        "FileNotFoundError": lambda e: ResourceNotFoundError("File", str(e)),
        "ConnectionError": lambda e: GraphQLAPIError("Service temporarily unavailable", ErrorCode.SERVICE_UNAVAILABLE),
        "TimeoutError": lambda e: GraphQLAPIError("Request timeout", ErrorCode.TIMEOUT),
    }
    
    if error_type in error_mappings:
        return error_mappings[error_type](error)
    
    return GraphQLAPIError(
        message="An unexpected error occurred",
        code=ErrorCode.INTERNAL_ERROR
    )


def validate_field_access(info: Info, field_name: str, required_role: Optional[str] = None) -> bool:
    """Validate field access permissions"""
    from .auth import get_current_user, authorize_field
    
    try:
        # Check if user is authenticated for protected fields
        user = get_current_user(info)
        
        # Check field-level authorization
        if not authorize_field(info, field_name, info.parent_type):
            return False
        
        # Check role-based access
        if required_role and user:
            user_role = getattr(user, 'role', 'USER')
            if user_role != required_role and user_role != 'ADMIN':
                return False
        
        return True
        
    except Exception:
        return False


def require_field_auth(required_role: Optional[str] = None):
    """Decorator for field-level authorization"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Find Info argument
            info = None
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            if not info:
                raise AuthenticationError("No context available")
            
            # Validate field access
            field_name = info.field_name
            if not validate_field_access(info, field_name, required_role):
                if required_role:
                    raise AuthorizationError(
                        f"Role '{required_role}' required to access field '{field_name}'",
                        required_role=required_role
                    )
                else:
                    raise AuthorizationError(f"Access denied to field '{field_name}'")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Error response helpers
def create_error_response(
    success: bool = False,
    message: str = "",
    errors: Optional[List[str]] = None,
    data: Optional[Any] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": success,
        "message": message,
        "errors": errors or [],
        "data": data
    }


def handle_mutation_errors(func):
    """Decorator to handle mutation errors consistently"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except GraphQLAPIError as e:
            return create_error_response(
                success=False,
                message=e.message,
                errors=[e.message]
            )
        except Exception as e:
            api_error = create_user_friendly_error(e)
            return create_error_response(
                success=False,
                message=api_error.message,
                errors=[api_error.message]
            )
    
    return wrapper


# Field masking for sensitive data
class FieldMasker:
    """Mask sensitive fields based on user permissions"""
    
    SENSITIVE_FIELDS = {
        "User": ["hashedPassword", "email"],
        "Document": ["metadata"],
        "ArbitrationAnalysis": ["metadata"],
    }
    
    @classmethod
    def mask_field(cls, obj_type: str, field_name: str, value: Any, user_role: str) -> Any:
        """Mask field value if user doesn't have permission"""
        if obj_type in cls.SENSITIVE_FIELDS and field_name in cls.SENSITIVE_FIELDS[obj_type]:
            if user_role not in ["ADMIN", "ANALYST"]:
                if field_name == "email":
                    return cls._mask_email(value)
                elif field_name in ["metadata"]:
                    return None
                elif field_name == "hashedPassword":
                    return None
        
        return value
    
    @staticmethod
    def _mask_email(email: str) -> str:
        """Mask email address"""
        if not email or "@" not in email:
            return email
        
        local, domain = email.split("@", 1)
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"


# Global error handler instance
error_handler = ErrorHandler(debug=False)  # Set to True in development


# Custom GraphQL error formatter
def custom_format_error(error: GraphQLError) -> Dict[str, Any]:
    """Custom error formatter for GraphQL responses"""
    formatted = error_handler.format_error(error)
    
    # Add timestamp
    import datetime
    formatted['timestamp'] = datetime.datetime.utcnow().isoformat()
    
    # Add request ID if available
    if hasattr(error, 'extensions') and 'request_id' in error.extensions:
        formatted['request_id'] = error.extensions['request_id']
    
    return formatted