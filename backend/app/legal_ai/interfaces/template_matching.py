"""
Interface for template matching service.

Defines the contract for comparing uploaded contracts against
standard templates and flagging deviations.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class DeviationType(str, Enum):
    """Types of deviations from template."""
    MISSING_CLAUSE = "missing_clause"
    ADDITIONAL_CLAUSE = "additional_clause"
    MODIFIED_LANGUAGE = "modified_language"
    DIFFERENT_TERMS = "different_terms"
    STRUCTURAL_DIFFERENCE = "structural_difference"
    SEMANTIC_DEVIATION = "semantic_deviation"


class DeviationRisk(str, Enum):
    """Risk level of template deviations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ACCEPTABLE = "acceptable"


@dataclass
class Template:
    """A contract template for comparison."""
    template_id: str
    name: str
    version: str
    contract_type: str
    industry: Optional[str]
    description: str
    template_text: str
    required_clauses: List[str]
    optional_clauses: List[str]
    variable_fields: List[str]
    last_updated: Optional[str] = None
    jurisdiction: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "version": self.version,
            "contract_type": self.contract_type,
            "industry": self.industry,
            "description": self.description,
            "template_text": self.template_text[:500] + "..." if len(self.template_text) > 500 else self.template_text,
            "required_clauses": self.required_clauses,
            "optional_clauses": self.optional_clauses,
            "variable_fields": self.variable_fields,
            "last_updated": self.last_updated,
            "jurisdiction": self.jurisdiction,
        }


@dataclass
class TemplateDeviation:
    """A detected deviation from template."""
    deviation_type: DeviationType
    risk_level: DeviationRisk
    template_section: Optional[str]
    document_section: Optional[str]
    template_text: Optional[str]
    document_text: Optional[str]
    description: str
    impact_analysis: str
    approval_required: bool
    suggested_action: str
    confidence: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "deviation_type": self.deviation_type.value,
            "risk_level": self.risk_level.value,
            "template_section": self.template_section,
            "document_section": self.document_section,
            "template_text": self.template_text,
            "document_text": self.document_text,
            "description": self.description,
            "impact_analysis": self.impact_analysis,
            "approval_required": self.approval_required,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
        }


@dataclass
class ClauseAlignment:
    """Alignment between template and document clause."""
    template_clause_name: str
    template_clause_text: str
    document_clause_text: Optional[str]
    alignment_score: float  # 0.0 - 1.0
    status: str  # "matched", "modified", "missing", "extra"
    deviations: List[TemplateDeviation]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "template_clause_name": self.template_clause_name,
            "template_clause_text": self.template_clause_text,
            "document_clause_text": self.document_clause_text,
            "alignment_score": self.alignment_score,
            "status": self.status,
            "deviations": [d.to_dict() for d in self.deviations],
        }


@dataclass
class TemplateMatchResult:
    """Result of matching a document against a template."""
    document_id: str
    template: Template
    overall_match_score: float  # 0.0 - 1.0
    deviations: List[TemplateDeviation]
    clause_alignments: List[ClauseAlignment]
    missing_required_clauses: List[str]
    additional_clauses: List[str]
    high_risk_deviations: int
    requires_approval: bool
    approval_reasons: List[str]
    summary: str
    recommendations: List[str]
    processing_time_ms: float
    model_used: str

    def get_deviations_by_risk(self, risk: DeviationRisk) -> List[TemplateDeviation]:
        """Get deviations by risk level."""
        return [d for d in self.deviations if d.risk_level == risk]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "template": self.template.to_dict(),
            "overall_match_score": self.overall_match_score,
            "deviations": [d.to_dict() for d in self.deviations],
            "clause_alignments": [c.to_dict() for c in self.clause_alignments],
            "missing_required_clauses": self.missing_required_clauses,
            "additional_clauses": self.additional_clauses,
            "high_risk_deviations": self.high_risk_deviations,
            "requires_approval": self.requires_approval,
            "approval_reasons": self.approval_reasons,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class TemplateMatchingInterface(BaseAIService):
    """
    Interface for template matching services.

    Implementations compare documents against standard templates,
    identifying deviations and assessing their implications.
    """

    @abstractmethod
    async def match_template(
        self,
        document_text: str,
        template: Template,
        document_id: Optional[str] = None,
        strict_mode: bool = False,
    ) -> TemplateMatchResult:
        """
        Match a document against a template.

        Args:
            document_text: Document to analyze
            template: Template to compare against
            document_id: Optional identifier
            strict_mode: If True, flag all deviations as requiring review

        Returns:
            TemplateMatchResult with analysis
        """
        pass

    @abstractmethod
    async def find_best_template(
        self,
        document_text: str,
        templates: List[Template],
    ) -> tuple[Template, float]:
        """
        Find the best matching template for a document.

        Args:
            document_text: Document to match
            templates: Available templates

        Returns:
            Tuple of (best template, match score)
        """
        pass

    @abstractmethod
    async def align_clauses(
        self,
        document_text: str,
        template: Template,
    ) -> List[ClauseAlignment]:
        """
        Align document clauses with template clauses.

        Args:
            document_text: Document text
            template: Template to align with

        Returns:
            List of clause alignments
        """
        pass

    @abstractmethod
    async def generate_deviation_report(
        self,
        result: TemplateMatchResult,
        include_recommendations: bool = True,
    ) -> str:
        """
        Generate a deviation report.

        Args:
            result: Template match result
            include_recommendations: Include remediation recommendations

        Returns:
            Formatted deviation report
        """
        pass

    @abstractmethod
    async def suggest_template_conformance(
        self,
        deviation: TemplateDeviation,
        template: Template,
    ) -> str:
        """
        Suggest how to make document conform to template.

        Args:
            deviation: Deviation to address
            template: Target template

        Returns:
            Suggested conforming language
        """
        pass

    @abstractmethod
    async def register_template(
        self,
        template: Template,
    ) -> str:
        """
        Register a new template for matching.

        Args:
            template: Template to register

        Returns:
            Registered template ID
        """
        pass
