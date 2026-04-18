"""
AI Provider Configuration.

Centralized configuration for AI providers and per-service model selection.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class AIProviderType(str, Enum):
    """Supported AI provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX = "vertex"


@dataclass
class ServiceConfig:
    """Configuration for a specific service."""
    provider: AIProviderType
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60
    fallback_provider: Optional[AIProviderType] = None
    fallback_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "fallback_provider": self.fallback_provider.value if self.fallback_provider else None,
            "fallback_model": self.fallback_model,
        }


@dataclass
class AIProviderConfig:
    """
    Global AI provider configuration.

    Manages API keys, default settings, and per-service configurations.
    """

    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    vertex_project_id: Optional[str] = field(
        default_factory=lambda: os.getenv("VERTEX_PROJECT_ID")
    )
    vertex_location: str = field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1")
    )

    # Default provider
    default_provider: AIProviderType = AIProviderType.OPENAI

    # Default models per provider
    default_models: Dict[AIProviderType, str] = field(default_factory=lambda: {
        AIProviderType.OPENAI: "gpt-4o",
        AIProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
        AIProviderType.VERTEX: "gemini-1.5-pro",
    })

    # Per-service configurations
    service_configs: Dict[str, ServiceConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default service configurations."""
        if not self.service_configs:
            self._set_default_service_configs()

    def _set_default_service_configs(self):
        """Set default configurations for all services."""
        default_config = ServiceConfig(
            provider=self.default_provider,
            model=self.default_models[self.default_provider],
        )

        services = [
            "clause_classification",
            "risk_scoring",
            "summarization",
            "compliance_checking",
            "term_extraction",
            "document_comparison",
            "template_matching",
            "natural_language_query",
        ]

        for service in services:
            self.service_configs[service] = ServiceConfig(
                provider=default_config.provider,
                model=default_config.model,
                temperature=default_config.temperature,
                max_tokens=default_config.max_tokens,
                timeout=default_config.timeout,
            )

        # Override specific services with optimized settings
        # Claude excels at nuanced legal analysis
        if self.anthropic_api_key:
            self.service_configs["risk_scoring"] = ServiceConfig(
                provider=AIProviderType.ANTHROPIC,
                model="claude-sonnet-4-20250514",
                temperature=0.0,
                max_tokens=8192,
            )
            self.service_configs["compliance_checking"] = ServiceConfig(
                provider=AIProviderType.ANTHROPIC,
                model="claude-sonnet-4-20250514",
                temperature=0.0,
                max_tokens=8192,
            )
            self.service_configs["natural_language_query"] = ServiceConfig(
                provider=AIProviderType.ANTHROPIC,
                model="claude-sonnet-4-20250514",
                temperature=0.1,
                max_tokens=4096,
            )

    def get_service_config(self, service_name: str) -> ServiceConfig:
        """
        Get configuration for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceConfig for the service
        """
        if service_name not in self.service_configs:
            logger.warning(f"No config for service '{service_name}', using default")
            return ServiceConfig(
                provider=self.default_provider,
                model=self.default_models[self.default_provider],
            )
        return self.service_configs[service_name]

    def configure_service(
        self,
        service_name: str,
        provider: AIProviderType,
        model: str,
        **kwargs,
    ) -> None:
        """
        Configure a specific service.

        Args:
            service_name: Name of the service
            provider: AI provider to use
            model: Model identifier
            **kwargs: Additional configuration options
        """
        self.service_configs[service_name] = ServiceConfig(
            provider=provider,
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 60),
            fallback_provider=kwargs.get("fallback_provider"),
            fallback_model=kwargs.get("fallback_model"),
        )
        logger.info(f"Configured service '{service_name}' with {provider.value}/{model}")

    def validate(self) -> Dict[str, bool]:
        """
        Validate configuration and check API key availability.

        Returns:
            Dictionary of provider -> is_configured status
        """
        return {
            AIProviderType.OPENAI.value: bool(self.openai_api_key),
            AIProviderType.ANTHROPIC.value: bool(self.anthropic_api_key),
            AIProviderType.VERTEX.value: bool(self.vertex_project_id),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without sensitive keys)."""
        return {
            "default_provider": self.default_provider.value,
            "default_models": {k.value: v for k, v in self.default_models.items()},
            "service_configs": {
                k: v.to_dict() for k, v in self.service_configs.items()
            },
            "providers_configured": self.validate(),
        }


# Global configuration instance
_config: Optional[AIProviderConfig] = None


def get_ai_provider_config() -> AIProviderConfig:
    """
    Get the global AI provider configuration.

    Returns:
        AIProviderConfig instance
    """
    global _config
    if _config is None:
        _config = AIProviderConfig()
    return _config


def get_ai_provider(service_name: str) -> tuple[AIProviderType, str]:
    """
    Get the provider and model for a service.

    Args:
        service_name: Name of the service

    Returns:
        Tuple of (provider_type, model_name)
    """
    config = get_ai_provider_config()
    service_config = config.get_service_config(service_name)
    return service_config.provider, service_config.model


def get_service_config(service_name: str) -> ServiceConfig:
    """
    Get the full configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        ServiceConfig for the service
    """
    config = get_ai_provider_config()
    return config.get_service_config(service_name)


def configure_service(
    service_name: str,
    provider: AIProviderType,
    model: str,
    **kwargs,
) -> None:
    """
    Configure a specific service.

    Args:
        service_name: Name of the service
        provider: AI provider to use
        model: Model identifier
        **kwargs: Additional options
    """
    config = get_ai_provider_config()
    config.configure_service(service_name, provider, model, **kwargs)
