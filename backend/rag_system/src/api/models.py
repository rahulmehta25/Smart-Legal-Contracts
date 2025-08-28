from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DocumentType(str, Enum):
    LEGAL = "legal"
    TECHNICAL = "technical"
    FINANCIAL = "financial"

class DocumentUploadRequest(BaseModel):
    document_type: DocumentType
    metadata: Optional[dict] = None

class DetectionResult(BaseModel):
    document_id: str
    similarity_score: float
    detected_patterns: List[str]
    risk_level: str

class ComparisonRequest(BaseModel):
    document1_id: str
    document2_id: str

class ComparisonResult(BaseModel):
    similarity_percentage: float
    key_differences: List[str]
    recommendation: str

class BatchProcessingRequest(BaseModel):
    document_ids: List[str]
    processing_type: str = Field(description="Type of batch processing to perform")

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None