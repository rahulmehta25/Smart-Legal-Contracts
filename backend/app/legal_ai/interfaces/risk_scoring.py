"""
Interface for risk scoring service.

Defines the contract for assigning risk scores to contracts
and flagging unfavorable terms.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class RiskLevel(str, Enum):
    """Risk level categories."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class RiskCategory(str, Enum):
    """Categories of contract risk."""
    LEGAL = "legal"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    REPUTATIONAL = "reputational"
    STRATEGIC = "strategic"


@dataclass
class RiskFactor:
    """Individual risk factor identified in a contract."""
    name: str
    description: str
    category: RiskCategory
    severity: RiskLevel
    score: float  # 0.0 - 1.0
    clause_text: str
    clause_location: Optional[str] = None
    mitigation_suggestions: List[str] = field(default_factory=list)
    precedent_info: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "score": self.score,
            "clause_text": self.clause_text,
            "clause_location": self.clause_location,
            "mitigation_suggestions": self.mitigation_suggestions,
            "precedent_info": self.precedent_info,
        }


@dataclass
class UnfavorableTerm:
    """An identified unfavorable contract term."""
    term_text: str
    issue_description: str
    risk_level: RiskLevel
    affected_party: str  # "client", "counterparty", "both"
    recommended_alternative: Optional[str] = None
    negotiation_priority: int = 5  # 1 = highest priority
    market_standard_deviation: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "term_text": self.term_text,
            "issue_description": self.issue_description,
            "risk_level": self.risk_level.value,
            "affected_party": self.affected_party,
            "recommended_alternative": self.recommended_alternative,
            "negotiation_priority": self.negotiation_priority,
            "market_standard_deviation": self.market_standard_deviation,
        }


@dataclass
class RiskScoreResult:
    """Complete risk scoring result for a contract."""
    document_id: str
    overall_score: float  # 0.0 (lowest risk) - 1.0 (highest risk)
    overall_level: RiskLevel
    category_scores: Dict[RiskCategory, float]
    risk_factors: List[RiskFactor]
    unfavorable_terms: List[UnfavorableTerm]
    executive_summary: str
    risk_distribution: Dict[str, int]
    recommendations: List[str]
    processing_time_ms: float
    model_used: str

    def get_critical_risks(self) -> List[RiskFactor]:
        """Get all critical risk factors."""
        return [r for r in self.risk_factors if r.severity == RiskLevel.CRITICAL]

    def get_high_priority_terms(self, max_priority: int = 3) -> List[UnfavorableTerm]:
        """Get high priority unfavorable terms."""
        return [t for t in self.unfavorable_terms if t.negotiation_priority <= max_priority]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "category_scores": {k.value: v for k, v in self.category_scores.items()},
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "unfavorable_terms": [t.to_dict() for t in self.unfavorable_terms],
            "executive_summary": self.executive_summary,
            "risk_distribution": self.risk_distribution,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class RiskScoringInterface(BaseAIService):
    """
    Interface for risk scoring services.

    Implementations analyze contracts for risks, assign scores,
    and identify unfavorable terms with mitigation recommendations.
    """

    @abstractmethod
    async def score_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        party_perspective: str = "client",
        industry: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> RiskScoreResult:
        """
        Calculate risk scores for an entire document.

        Args:
            document_text: Full contract text
            document_id: Optional identifier
            party_perspective: Whose perspective to score from
            industry: Industry context for benchmarking
            contract_type: Type of contract (e.g., "SaaS", "employment")

        Returns:
            RiskScoreResult with comprehensive risk analysis
        """
        pass

    @abstractmethod
    async def score_clause(
        self,
        clause_text: str,
        clause_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> RiskFactor:
        """
        Score risk for a single clause.

        Args:
            clause_text: Text of the clause
            clause_type: Type of clause if known
            context: Surrounding document context

        Returns:
            RiskFactor for the clause
        """
        pass

    @abstractmethod
    async def identify_unfavorable_terms(
        self,
        document_text: str,
        party_perspective: str = "client",
    ) -> List[UnfavorableTerm]:
        """
        Identify all unfavorable terms in a document.

        Args:
            document_text: Full contract text
            party_perspective: Perspective for favorability assessment

        Returns:
            List of identified unfavorable terms
        """
        pass

    @abstractmethod
    async def generate_risk_report(
        self,
        risk_result: RiskScoreResult,
        include_mitigation: bool = True,
    ) -> str:
        """
        Generate a human-readable risk report.

        Args:
            risk_result: Risk scoring result to report on
            include_mitigation: Whether to include mitigation recommendations

        Returns:
            Formatted risk report as markdown string
        """
        pass
