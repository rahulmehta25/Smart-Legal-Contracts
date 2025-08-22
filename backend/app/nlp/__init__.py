"""
NLP module for multi-language arbitration clause detection.

This module provides comprehensive natural language processing capabilities
for detecting arbitration clauses across multiple languages and legal systems.
"""

from .multilingual import MultilingualProcessor
from .legal_translator import LegalTranslator
from .language_models import LanguageModelManager

__all__ = [
    'MultilingualProcessor',
    'LegalTranslator', 
    'LanguageModelManager'
]