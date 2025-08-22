"""
Advanced AI-Powered Legal Research Assistant

This module provides comprehensive legal research capabilities including:
- Case law search with AI similarity matching
- Statute interpretation and analysis
- Legal citation parsing and validation
- Brief generation and argument construction
- Knowledge graph integration for legal concepts
- Document analysis and contract review
- Research workflow management
"""

from .case_law_search import CaseLawSearchEngine
from .statute_analyzer import StatuteAnalyzer
from .citation_parser import CitationParser
from .shepardizer import Shepardizer
from .brief_generator import BriefGenerator
from .argument_builder import ArgumentBuilder

__all__ = [
    'CaseLawSearchEngine',
    'StatuteAnalyzer', 
    'CitationParser',
    'Shepardizer',
    'BriefGenerator',
    'ArgumentBuilder'
]

__version__ = '1.0.0'