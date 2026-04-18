"""
Base provider interface for AI providers.

Defines the common interface that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResponse:
    """Response from a completion request."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Optional[Any] = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]


class BaseProvider(ABC):
    """
    Abstract base class for AI providers.

    All provider implementations must implement these methods to ensure
    consistent behavior across different AI services.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[Any] = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of supported model identifiers."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider client.

        Must be called before making requests.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the provider and release resources.
        """
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """
        Generate a completion from messages.

        Args:
            messages: List of conversation messages
            model: Model to use (default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            response_format: Response format specification

        Returns:
            CompletionResponse with generated text
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion.

        Args:
            messages: List of conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Yields:
            Content chunks as they're generated
        """
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.

        Args:
            texts: Texts to embed
            model: Embedding model to use

        Returns:
            EmbeddingResponse with vectors
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health and connectivity.

        Returns:
            Health status dictionary
        """
        pass

    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
