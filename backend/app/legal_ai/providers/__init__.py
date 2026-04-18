"""
AI Provider Configuration Module.

Provides unified access to multiple AI providers (OpenAI, Anthropic, Google Vertex)
with configurable model selection per service.
"""

from app.legal_ai.providers.config import (
    AIProviderConfig,
    ServiceConfig,
    get_ai_provider,
    get_service_config,
    configure_service,
)
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import BaseProvider

__all__ = [
    "AIProviderConfig",
    "ServiceConfig",
    "get_ai_provider",
    "get_service_config",
    "configure_service",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "VertexProvider",
]
