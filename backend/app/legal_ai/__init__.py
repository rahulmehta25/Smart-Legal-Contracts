"""
Legal AI Services Module.

This module provides comprehensive AI-powered legal document analysis capabilities
including clause classification, risk scoring, compliance checking, summarization,
term extraction, document comparison, template matching, and natural language queries.
"""

from app.legal_ai.interfaces import (
    BaseAIService,
    ClauseClassificationInterface,
    RiskScoringInterface,
    SummarizationInterface,
    ComplianceCheckingInterface,
    TermExtractionInterface,
    DocumentComparisonInterface,
    TemplateMatchingInterface,
    NaturalLanguageQueryInterface,
)
from app.legal_ai.providers import AIProviderConfig, get_ai_provider
from app.legal_ai.services import (
    ClauseClassificationService,
    RiskScoringService,
    ContractSummarizationService,
    ComplianceCheckingService,
    TermExtractionService,
    DocumentComparisonService,
    TemplateMatchingService,
    NaturalLanguageQueryService,
)

__all__ = [
    # Interfaces
    "BaseAIService",
    "ClauseClassificationInterface",
    "RiskScoringInterface",
    "SummarizationInterface",
    "ComplianceCheckingInterface",
    "TermExtractionInterface",
    "DocumentComparisonInterface",
    "TemplateMatchingInterface",
    "NaturalLanguageQueryInterface",
    # Providers
    "AIProviderConfig",
    "get_ai_provider",
    # Services
    "ClauseClassificationService",
    "RiskScoringService",
    "ContractSummarizationService",
    "ComplianceCheckingService",
    "TermExtractionService",
    "DocumentComparisonService",
    "TemplateMatchingService",
    "NaturalLanguageQueryService",
]
