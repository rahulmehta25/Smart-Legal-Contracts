"""
Production-ready logging configuration
"""
import sys
import os
from loguru import logger
from datetime import datetime
from app.core.config import get_settings


def configure_logging():
    """
    Configure logging for production deployment
    """
    settings = get_settings()
    
    # Remove default logger
    logger.remove()
    
    # Console logging with appropriate level
    logger.add(
        sys.stdout,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True if settings.environment == "development" else False,
        backtrace=True,
        diagnose=True if settings.environment == "development" else False
    )
    
    # File logging if specified
    if settings.log_file:
        logger.add(
            settings.log_file,
            format=settings.log_format,
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="gz",
            backtrace=True,
            diagnose=False  # Don't include sensitive data in file logs
        )
    
    # Structured logging for production
    if settings.environment == "production":
        # Add JSON structured logging for better parsing by log aggregators
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            level=settings.log_level,
            serialize=True,  # Output as JSON
            backtrace=False,
            diagnose=False
        )
    
    # Add error-specific logging
    error_log_path = f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    
    logger.add(
        error_log_path,
        format=settings.log_format,
        level="ERROR",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging configured for {settings.environment} environment")


def setup_request_logging():
    """
    Configure request/response logging middleware
    """
    import logging
    from fastapi import Request
    
    # Configure uvicorn access logging
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()
    
    # Create custom request logger
    async def log_request(request: Request):
        start_time = datetime.now()
        
        # Log incoming request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": start_time.isoformat()
            }
        )
        
        return start_time
    
    return log_request


class RequestLoggingMiddleware:
    """
    Middleware for logging requests and responses
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = datetime.now()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Extract request info from scope
                    method = scope["method"]
                    path = scope["path"]
                    client_ip = scope["client"][0] if scope.get("client") else "unknown"
                    
                    # Log response
                    log_level = "ERROR" if status_code >= 400 else "INFO"
                    logger.log(
                        log_level,
                        f"Response: {status_code} {method} {path} - {processing_time:.3f}s",
                        extra={
                            "method": method,
                            "path": path,
                            "status_code": status_code,
                            "processing_time": processing_time,
                            "client_ip": client_ip
                        }
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


def get_logger(name: str = None):
    """
    Get a logger instance for a specific module
    """
    if name:
        return logger.bind(name=name)
    return logger