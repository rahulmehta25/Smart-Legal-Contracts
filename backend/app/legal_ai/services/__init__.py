"""
Legal AI Services Implementation.

Concrete implementations of all legal AI service interfaces.
"""

from app.legal_ai.services.clause_classification_service import ClauseClassificationService
from app.legal_ai.services.risk_scoring_service import RiskScoringService
from app.legal_ai.services.summarization_service import ContractSummarizationService
from app.legal_ai.services.compliance_checking_service import ComplianceCheckingService
from app.legal_ai.services.term_extraction_service import TermExtractionService
from app.legal_ai.services.document_comparison_service import DocumentComparisonService
from app.legal_ai.services.template_matching_service import TemplateMatchingService
from app.legal_ai.services.natural_language_query_service import NaturalLanguageQueryService

__all__ = [
    "ClauseClassificationService",
    "RiskScoringService",
    "ContractSummarizationService",
    "ComplianceCheckingService",
    "TermExtractionService",
    "DocumentComparisonService",
    "TemplateMatchingService",
    "NaturalLanguageQueryService",
]
