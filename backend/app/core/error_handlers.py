"""
Comprehensive error handling and recovery mechanisms for production
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import ValidationError
import sys
import traceback
from datetime import datetime
from typing import Union
from loguru import logger

from app.core.config import get_settings


class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message: str, status_code: int = 500, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class DatabaseConnectionError(APIError):
    """Database connection specific error"""
    def __init__(self, message: str = "Database connection failed"):
        super().__init__(message, status_code=503, error_code="DB_CONNECTION_ERROR")


class ValidationException(APIError):
    """Validation error"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")
        self.details = details


class RateLimitException(APIError):
    """Rate limiting error"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, error_code="RATE_LIMIT_EXCEEDED")


def create_error_response(
    status_code: int, 
    message: str, 
    error_code: str = None,
    details: dict = None,
    request_id: str = None
) -> JSONResponse:
    """Create standardized error response"""
    settings = get_settings()
    
    error_response = {
        "error": {
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat(),
        }
    }
    
    if error_code:
        error_response["error"]["error_code"] = error_code
        
    if details and settings.environment != "production":
        error_response["error"]["details"] = details
        
    if request_id:
        error_response["error"]["request_id"] = request_id
        
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


# Exception handlers
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom API errors"""
    logger.error(f"API Error: {exc.message} (Status: {exc.status_code})")
    
    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_code=exc.error_code,
        request_id=getattr(request.state, 'request_id', None)
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.detail} (Status: {exc.status_code})")
    
    return create_error_response(
        status_code=exc.status_code,
        message=exc.detail or "An error occurred",
        error_code="HTTP_ERROR",
        request_id=getattr(request.state, 'request_id', None)
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors"""
    logger.warning(f"Validation Error: {exc.errors()}")
    
    # Format validation errors for better readability
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return create_error_response(
        status_code=422,
        message="Validation failed",
        error_code="VALIDATION_ERROR",
        details={"validation_errors": formatted_errors},
        request_id=getattr(request.state, 'request_id', None)
    )


async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle database errors with recovery attempts"""
    settings = get_settings()
    
    # Log the full error for debugging
    logger.error(f"Database Error: {str(exc)}")
    
    # Determine error type and appropriate response
    if isinstance(exc, OperationalError):
        # Database connection issues
        message = "Database temporarily unavailable. Please try again later."
        error_code = "DB_CONNECTION_ERROR"
        status_code = 503
    elif isinstance(exc, IntegrityError):
        # Data integrity violations
        message = "Data integrity error. Please check your input."
        error_code = "DB_INTEGRITY_ERROR"
        status_code = 400
    else:
        # Generic database error
        message = "Database error occurred"
        error_code = "DB_ERROR"
        status_code = 500
    
    # In development, include more details
    details = None
    if settings.environment == "development":
        details = {"database_error": str(exc)}
    
    return create_error_response(
        status_code=status_code,
        message=message,
        error_code=error_code,
        details=details,
        request_id=getattr(request.state, 'request_id', None)
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions"""
    settings = get_settings()
    
    # Generate unique error ID for tracking
    error_id = f"ERR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(exc)}"
    
    # Log the full traceback
    logger.error(
        f"Unhandled Exception [{error_id}]: {str(exc)}\n"
        f"Request: {request.method} {request.url}\n"
        f"Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
    )
    
    # Determine if it's a known error type
    message = "An internal error occurred"
    status_code = 500
    error_code = "INTERNAL_ERROR"
    
    # Don't expose internal errors in production
    details = None
    if settings.environment == "development":
        details = {
            "error_id": error_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
    
    return create_error_response(
        status_code=status_code,
        message=message,
        error_code=error_code,
        details=details,
        request_id=getattr(request.state, 'request_id', None)
    )


# Circuit breaker for external services
class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise APIError("Service temporarily unavailable", 503, "SERVICE_UNAVAILABLE")
        
        try:
            result = await func(*args, **kwargs) if hasattr(func, '__await__') else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# Global circuit breakers for external services
database_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
vector_store_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


# Retry mechanism with exponential backoff
async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff"""
    import asyncio
    
    for attempt in range(max_retries):
        try:
            if hasattr(func, '__await__'):
                return await func()
            else:
                return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Max retries exceeded for function {func.__name__}")
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Retry attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
            await asyncio.sleep(delay)


# Health check with error recovery
async def health_check_with_recovery():
    """Perform health check with automatic recovery attempts"""
    from app.db.database_production import check_database_connection
    
    checks = {
        "database": False,
        "vector_store": False,
        "memory": False
    }
    
    # Database health check with circuit breaker
    try:
        checks["database"] = await database_circuit_breaker.call(check_database_connection)
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = False
    
    # Memory health check
    try:
        import psutil
        memory = psutil.virtual_memory()
        checks["memory"] = memory.percent < 90  # Healthy if less than 90% used
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        checks["memory"] = False
    
    # Vector store health check
    try:
        from app.core.config import get_settings
        settings = get_settings()
        import os
        checks["vector_store"] = os.path.exists(settings.vector_store_path)
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        checks["vector_store"] = False
    
    return checks