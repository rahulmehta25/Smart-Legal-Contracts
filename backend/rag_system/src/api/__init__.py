"""
API endpoints and route handlers.

Provides REST API endpoints for document analysis, user management,
and system operations.
"""

from .routes import *
from .middleware import *
from .schemas import *
from .dependencies import *

__all__ = [
    "routes",
    "middleware",
    "schemas",
    "dependencies"
]