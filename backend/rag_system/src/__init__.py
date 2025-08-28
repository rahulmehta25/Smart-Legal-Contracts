"""
Source code for the RAG Arbitration Detection System.

This package contains the core modules for document processing,
model management, comparison engines, and API endpoints.
"""

from .core import *
from .models import *
from .document import *
from .comparison import *
from .database import *
from .explainability import *
from .api import *

__all__ = [
    "core",
    "models", 
    "document",
    "comparison",
    "database",
    "explainability",
    "api"
]