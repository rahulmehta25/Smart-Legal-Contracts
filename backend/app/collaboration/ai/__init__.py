"""
AI-powered collaboration features.

This module provides AI-enhanced collaboration capabilities including:
- AI meeting assistant for summaries and action items
- Smart suggestions during document review
- Automated content generation
- Sentiment analysis of discussions
- Translation for international teams
- Context-aware recommendations
"""

from .meeting_assistant import MeetingAssistant
from .smart_suggestions import SmartSuggestionEngine
from .sentiment_analyzer import SentimentAnalyzer
from .translation_service import TranslationService
from .recommendation_engine import RecommendationEngine

__all__ = [
    'MeetingAssistant',
    'SmartSuggestionEngine', 
    'SentimentAnalyzer',
    'TranslationService',
    'RecommendationEngine'
]