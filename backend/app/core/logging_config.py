"""
Production-ready logging configuration with structured JSON logging and correlation IDs
"""
import sys
import os
import uuid
import json
from contextvars import ContextVar
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime
from app.core.config import get_settings


# Context variable for request correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one"""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


def json_formatter(record: Dict[str, Any]) -> str:
    """Custom JSON formatter for structured logging"""
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "correlation_id": get_correlation_id(),
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra fields if present
    if record.get("extra"):
        for key, value in record["extra"].items():
            if key not in log_entry:
                log_entry[key] = value

    # Add exception info if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
        }

    return json.dumps(log_entry)


def configure_logging():
    """
    Configure logging for production deployment with structured JSON output
    """
    settings = get_settings()

    # Remove default logger
    logger.remove()

    # Determine log format based on environment
    log_format = os.getenv('LOG_FORMAT', 'text')

    if settings.environment == "production" or log_format == 'json':
        # JSON structured logging for production
        logger.add(
            sys.stdout,
            format="{message}",
            level=settings.log_level,
            serialize=False,
            backtrace=False,
            diagnose=False,
            filter=lambda record: True,
        )

        # Patch to use JSON formatter
        logger.configure(
            patcher=lambda record: record.update(
                message=json_formatter(record)
            )
        )
    else:
        # Human-readable format for development
        dev_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>[{extra[correlation_id]}]</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=dev_format,
            level=settings.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )

    # File logging for all environments
    log_dir = os.getenv('LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Application log file
    logger.add(
        f"{log_dir}/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=False
    )

    # Error-specific log file
    logger.add(
        f"{log_dir}/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="gz",
        backtrace=True,
        diagnose=True
    )

    # Set default extra context
    logger.configure(extra={"correlation_id": lambda: get_correlation_id()})

    logger.info(f"Logging configured for {settings.environment} environment")


def setup_request_logging():
    """
    Configure request/response logging middleware
    """
    import logging

    # Configure uvicorn access logging
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()


class RequestLoggingMiddleware:
    """
    Middleware for logging requests and responses with correlation IDs
    """

    # Paths to exclude from detailed logging
    EXCLUDED_PATHS = {'/health', '/metrics', '/favicon.ico'}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        # Skip logging for health/metrics endpoints
        if path in self.EXCLUDED_PATHS:
            await self.app(scope, receive, send)
            return

        start_time = datetime.now()

        # Extract or generate correlation ID from headers
        headers = dict(scope.get("headers", []))
        correlation_id = headers.get(b"x-correlation-id", b"").decode() or None
        correlation_id = set_correlation_id(correlation_id)

        # Extract request info
        method = scope["method"]
        client_ip = scope["client"][0] if scope.get("client") else "unknown"
        user_agent = headers.get(b"user-agent", b"").decode()

        # Log incoming request
        logger.info(
            f"Request started: {method} {path}",
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent[:100] if user_agent else None,
            correlation_id=correlation_id,
        )

        status_code = 500  # Default if something goes wrong

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]

                # Add correlation ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            logger.exception(
                f"Request failed: {method} {path}",
                method=method,
                path=path,
                error=str(e),
                correlation_id=correlation_id,
            )
            raise
        finally:
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log response
            log_level = "ERROR" if status_code >= 500 else "WARNING" if status_code >= 400 else "INFO"
            logger.log(
                log_level,
                f"Request completed: {status_code} {method} {path} ({processing_time:.3f}s)",
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=round(processing_time * 1000, 2),
                client_ip=client_ip,
                correlation_id=correlation_id,
            )


def get_logger(name: str = None):
    """
    Get a logger instance for a specific module with correlation ID context
    """
    if name:
        return logger.bind(name=name, correlation_id=get_correlation_id())
    return logger.bind(correlation_id=get_correlation_id())


def log_with_context(**context):
    """
    Create a logger with additional context
    """
    return logger.bind(correlation_id=get_correlation_id(), **context)