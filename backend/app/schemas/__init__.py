"""
Pydantic schemas for request/response models
"""

from .document import (
    DocumentCreate, DocumentResponse, DocumentUploadResponse, 
    DocumentChunkResponse, DocumentUpdate
)
from .analysis import (
    AnalysisRequest, AnalysisResponse, QuickAnalysisRequest,
    QuickAnalysisResponse, ArbitrationAnalysisResponse,
    ArbitrationClauseResponse
)
from .user import (
    UserCreate, UserResponse, UserLogin, Token, UserUpdate
)

__all__ = [
    # Document schemas
    "DocumentCreate", "DocumentResponse", "DocumentUploadResponse", 
    "DocumentChunkResponse", "DocumentUpdate",
    # Analysis schemas
    "AnalysisRequest", "AnalysisResponse", "QuickAnalysisRequest",
    "QuickAnalysisResponse", "ArbitrationAnalysisResponse",
    "ArbitrationClauseResponse",
    # User schemas
    "UserCreate", "UserResponse", "UserLogin", "Token", "UserUpdate"
]