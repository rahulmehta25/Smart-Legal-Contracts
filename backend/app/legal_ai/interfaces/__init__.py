"""
Service Interfaces for Legal AI Services.

All AI services implement these abstract base classes to ensure consistent
behavior and allow easy swapping of implementations.
"""

from app.legal_ai.interfaces.base import BaseAIService
from app.legal_ai.interfaces.clause_classification import ClauseClassificationInterface
from app.legal_ai.interfaces.risk_scoring import RiskScoringInterface
from app.legal_ai.interfaces.summarization import SummarizationInterface
from app.legal_ai.interfaces.compliance_checking import ComplianceCheckingInterface
from app.legal_ai.interfaces.term_extraction import TermExtractionInterface
from app.legal_ai.interfaces.document_comparison import DocumentComparisonInterface
from app.legal_ai.interfaces.template_matching import TemplateMatchingInterface
from app.legal_ai.interfaces.natural_language_query import NaturalLanguageQueryInterface

__all__ = [
    "BaseAIService",
    "ClauseClassificationInterface",
    "RiskScoringInterface",
    "SummarizationInterface",
    "ComplianceCheckingInterface",
    "TermExtractionInterface",
    "DocumentComparisonInterface",
    "TemplateMatchingInterface",
    "NaturalLanguageQueryInterface",
]
