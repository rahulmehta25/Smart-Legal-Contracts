"""
Interface for clause classification service.

Defines the contract for detecting and classifying all types of
legal clauses in contracts and documents.
"""

from abc import abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class ClauseType(str, Enum):
    """Standard legal clause types."""
    ARBITRATION = "arbitration"
    INDEMNIFICATION = "indemnification"
    LIABILITY_LIMITATION = "liability_limitation"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    CONFIDENTIALITY = "confidentiality"
    FORCE_MAJEURE = "force_majeure"
    IP_ASSIGNMENT = "ip_assignment"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    WARRANTY = "warranty"
    PAYMENT_TERMS = "payment_terms"
    DATA_PROTECTION = "data_protection"
    AUDIT_RIGHTS = "audit_rights"
    INSURANCE = "insurance"
    ASSIGNMENT = "assignment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    AMENDMENT = "amendment"
    NOTICE = "notice"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedClause:
    """A classified clause with metadata."""
    text: str
    clause_type: ClauseType
    confidence: float
    start_position: int
    end_position: int
    sub_type: Optional[str] = None
    key_terms: List[str] = field(default_factory=list)
    party_obligations: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "clause_type": self.clause_type.value,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "sub_type": self.sub_type,
            "key_terms": self.key_terms,
            "party_obligations": self.party_obligations,
            "risk_indicators": self.risk_indicators,
        }


@dataclass
class ClassificationResult:
    """Result of clause classification for a document."""
    document_id: str
    total_clauses: int
    clauses: List[ClassifiedClause]
    clause_distribution: dict
    processing_time_ms: float
    model_used: str

    def get_clauses_by_type(self, clause_type: ClauseType) -> List[ClassifiedClause]:
        """Get all clauses of a specific type."""
        return [c for c in self.clauses if c.clause_type == clause_type]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "total_clauses": self.total_clauses,
            "clauses": [c.to_dict() for c in self.clauses],
            "clause_distribution": self.clause_distribution,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class ClauseClassificationInterface(BaseAIService):
    """
    Interface for clause classification services.

    Implementations detect and classify all types of legal clauses
    in contracts, extracting key information about each clause.
    """

    @abstractmethod
    async def classify_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        target_clause_types: Optional[List[ClauseType]] = None,
        min_confidence: float = 0.5,
    ) -> ClassificationResult:
        """
        Classify all clauses in a document.

        Args:
            document_text: Full text of the document to analyze
            document_id: Optional identifier for the document
            target_clause_types: If provided, only detect these clause types
            min_confidence: Minimum confidence threshold for classification

        Returns:
            ClassificationResult with all classified clauses

        Raises:
            ValueError: If document_text is empty
            RuntimeError: If classification fails
        """
        pass

    @abstractmethod
    async def classify_clause(
        self,
        clause_text: str,
        context: Optional[str] = None,
    ) -> ClassifiedClause:
        """
        Classify a single clause.

        Args:
            clause_text: Text of the clause to classify
            context: Optional surrounding context for better classification

        Returns:
            ClassifiedClause with classification details
        """
        pass

    @abstractmethod
    async def extract_clauses(
        self,
        document_text: str,
    ) -> List[str]:
        """
        Extract individual clause segments from a document.

        Args:
            document_text: Full document text

        Returns:
            List of clause text segments
        """
        pass
