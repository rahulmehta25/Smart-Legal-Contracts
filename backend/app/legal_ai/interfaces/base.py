"""
Base interface for all AI services.

Provides common functionality including provider configuration,
health checking, and async context management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX = "vertex"
    LOCAL = "local"


@dataclass
class ServiceHealth:
    """Health status for a service."""
    healthy: bool
    provider: AIProvider
    model: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseAIService(ABC):
    """
    Abstract base class for all AI-powered legal analysis services.

    All concrete service implementations must inherit from this class
    and implement the required abstract methods. This ensures consistent
    behavior across all services and enables easy provider swapping.

    Attributes:
        provider: The AI provider to use (OpenAI, Anthropic, Vertex, etc.)
        model: The specific model identifier
        temperature: Model temperature for generation
        max_tokens: Maximum tokens for responses
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize the base AI service.

        Args:
            provider: AI provider to use
            model: Model identifier (provider-specific defaults if None)
            temperature: Generation temperature (0.0 = deterministic)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Optional[Any] = None

    def _get_default_model(self) -> str:
        """Get the default model for the configured provider."""
        defaults = {
            AIProvider.OPENAI: "gpt-4o",
            AIProvider.ANTHROPIC: "claude-sonnet-4-20250514",
            AIProvider.VERTEX: "gemini-1.5-pro",
            AIProvider.LOCAL: "llama-3-70b",
        }
        return defaults.get(self.provider, "gpt-4o")

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the service and establish provider connection.

        Must be called before using the service. Sets up the AI client
        and validates credentials.

        Raises:
            ConnectionError: If provider connection fails
            ValueError: If credentials are invalid
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the service and release resources.

        Should be called when the service is no longer needed.
        """
        pass

    @abstractmethod
    async def health_check(self) -> ServiceHealth:
        """
        Check the health status of the service.

        Returns:
            ServiceHealth object with status details
        """
        pass

    async def __aenter__(self) -> "BaseAIService":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def _log_request(self, operation: str, **kwargs) -> None:
        """Log an API request for debugging."""
        logger.debug(
            f"{self.__class__.__name__}.{operation} called",
            extra={"provider": self.provider.value, "model": self.model, **kwargs}
        )

    def _log_error(self, operation: str, error: Exception) -> None:
        """Log an API error."""
        logger.error(
            f"{self.__class__.__name__}.{operation} failed: {error}",
            extra={"provider": self.provider.value, "model": self.model}
        )
