"""
Interface for document comparison service.

Defines the contract for semantic diff between contract versions,
highlighting changes and their risk implications.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class ChangeType(str, Enum):
    """Types of changes detected."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    MOVED = "moved"
    SEMANTIC_CHANGE = "semantic_change"
    FORMATTING_ONLY = "formatting_only"


class ChangeImpact(str, Enum):
    """Impact level of changes."""
    CRITICAL = "critical"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    COSMETIC = "cosmetic"


class ChangeDomain(str, Enum):
    """Domain affected by the change."""
    LEGAL = "legal"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    ADMINISTRATIVE = "administrative"


@dataclass
class DocumentChange:
    """A detected change between documents."""
    change_type: ChangeType
    impact: ChangeImpact
    domain: ChangeDomain
    original_text: Optional[str]
    modified_text: Optional[str]
    original_location: Optional[str]
    modified_location: Optional[str]
    semantic_description: str
    risk_assessment: str
    risk_score: float  # 0.0 - 1.0
    affected_parties: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "change_type": self.change_type.value,
            "impact": self.impact.value,
            "domain": self.domain.value,
            "original_text": self.original_text,
            "modified_text": self.modified_text,
            "original_location": self.original_location,
            "modified_location": self.modified_location,
            "semantic_description": self.semantic_description,
            "risk_assessment": self.risk_assessment,
            "risk_score": self.risk_score,
            "affected_parties": self.affected_parties,
            "recommendation": self.recommendation,
        }


@dataclass
class ClauseComparison:
    """Comparison of a specific clause between versions."""
    clause_type: str
    original_text: Optional[str]
    modified_text: Optional[str]
    status: str  # "unchanged", "modified", "added", "removed"
    changes: List[DocumentChange]
    overall_impact: ChangeImpact
    summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "clause_type": self.clause_type,
            "original_text": self.original_text,
            "modified_text": self.modified_text,
            "status": self.status,
            "changes": [c.to_dict() for c in self.changes],
            "overall_impact": self.overall_impact.value,
            "summary": self.summary,
        }


@dataclass
class SimilarityMetrics:
    """Similarity metrics between documents."""
    text_similarity: float  # Lexical similarity
    semantic_similarity: float  # Meaning similarity
    structural_similarity: float  # Document structure similarity
    clause_overlap: float  # Percentage of matching clauses
    word_count_original: int
    word_count_modified: int
    change_percentage: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "text_similarity": self.text_similarity,
            "semantic_similarity": self.semantic_similarity,
            "structural_similarity": self.structural_similarity,
            "clause_overlap": self.clause_overlap,
            "word_count_original": self.word_count_original,
            "word_count_modified": self.word_count_modified,
            "change_percentage": self.change_percentage,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two documents."""
    original_document_id: str
    modified_document_id: str
    similarity_metrics: SimilarityMetrics
    total_changes: int
    changes: List[DocumentChange]
    clause_comparisons: List[ClauseComparison]
    impact_summary: Dict[ChangeImpact, int]
    domain_summary: Dict[ChangeDomain, int]
    overall_risk_delta: float  # Positive = increased risk
    executive_summary: str
    key_changes: List[str]
    negotiation_points: List[str]
    processing_time_ms: float
    model_used: str

    def get_critical_changes(self) -> List[DocumentChange]:
        """Get all critical changes."""
        return [c for c in self.changes if c.impact == ChangeImpact.CRITICAL]

    def get_changes_by_domain(self, domain: ChangeDomain) -> List[DocumentChange]:
        """Get changes for a specific domain."""
        return [c for c in self.changes if c.domain == domain]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "original_document_id": self.original_document_id,
            "modified_document_id": self.modified_document_id,
            "similarity_metrics": self.similarity_metrics.to_dict(),
            "total_changes": self.total_changes,
            "changes": [c.to_dict() for c in self.changes],
            "clause_comparisons": [c.to_dict() for c in self.clause_comparisons],
            "impact_summary": {k.value: v for k, v in self.impact_summary.items()},
            "domain_summary": {k.value: v for k, v in self.domain_summary.items()},
            "overall_risk_delta": self.overall_risk_delta,
            "executive_summary": self.executive_summary,
            "key_changes": self.key_changes,
            "negotiation_points": self.negotiation_points,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class DocumentComparisonInterface(BaseAIService):
    """
    Interface for document comparison services.

    Implementations provide semantic diff between contract versions,
    identifying changes and assessing their risk implications.
    """

    @abstractmethod
    async def compare_documents(
        self,
        original_text: str,
        modified_text: str,
        original_id: Optional[str] = None,
        modified_id: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        party_perspective: str = "client",
    ) -> ComparisonResult:
        """
        Compare two versions of a document.

        Args:
            original_text: Original document text
            modified_text: Modified document text
            original_id: Optional identifier for original
            modified_id: Optional identifier for modified
            focus_areas: Specific areas to focus on
            party_perspective: Perspective for risk assessment

        Returns:
            ComparisonResult with detailed analysis
        """
        pass

    @abstractmethod
    async def compare_clauses(
        self,
        original_clause: str,
        modified_clause: str,
        clause_type: Optional[str] = None,
    ) -> ClauseComparison:
        """
        Compare two versions of a specific clause.

        Args:
            original_clause: Original clause text
            modified_clause: Modified clause text
            clause_type: Type of clause if known

        Returns:
            ClauseComparison with changes
        """
        pass

    @abstractmethod
    async def calculate_similarity(
        self,
        text1: str,
        text2: str,
    ) -> SimilarityMetrics:
        """
        Calculate similarity metrics between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            SimilarityMetrics
        """
        pass

    @abstractmethod
    async def generate_redline(
        self,
        comparison_result: ComparisonResult,
        format: str = "markdown",
    ) -> str:
        """
        Generate a redline document showing changes.

        Args:
            comparison_result: Comparison result
            format: Output format ("markdown", "html", "text")

        Returns:
            Formatted redline document
        """
        pass

    @abstractmethod
    async def assess_change_risk(
        self,
        change: DocumentChange,
        full_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assess the risk implications of a specific change.

        Args:
            change: Change to assess
            full_context: Full document context

        Returns:
            Risk assessment details
        """
        pass
