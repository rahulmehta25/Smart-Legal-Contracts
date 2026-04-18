"""
OpenAI Provider Implementation.

Implements the BaseProvider interface for OpenAI's API.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import os
import logging
import json

from app.legal_ai.providers.base_provider import (
    BaseProvider,
    Message,
    CompletionResponse,
    EmbeddingResponse,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider implementation.

    Supports GPT-4, GPT-4o, GPT-3.5, and embedding models.
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ]

    EMBEDDING_MODELS = [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        organization: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            organization: Optional organization ID
        """
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        """Return supported models."""
        return self.SUPPORTED_MODELS

    @property
    def default_model(self) -> str:
        """Return default model."""
        return "gpt-4o"

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI

            kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.organization:
                kwargs["organization"] = self.organization

            self._client = AsyncOpenAI(**kwargs)
            logger.info("OpenAI provider initialized")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("OpenAI provider shutdown")

    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Generate a completion from messages."""
        if not self._client:
            await self.initialize()

        model = model or self.default_model

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop:
            kwargs["stop"] = stop

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self._client.chat.completions.create(**kwargs)

            return CompletionResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise

    async def complete_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion."""
        if not self._client:
            await self.initialize()

        model = model or self.default_model
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if stop:
            kwargs["stop"] = stop

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming completion failed: {e}")
            raise

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        if not self._client:
            await self.initialize()

        model = model or "text-embedding-3-small"

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=texts,
            )

            return EmbeddingResponse(
                embeddings=[item.embedding for item in response.data],
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            if not self._client:
                await self.initialize()

            # Make a minimal request to verify connectivity
            response = await self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model_tested": "gpt-3.5-turbo",
                "response_received": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "error": str(e),
            }

    async def complete_json(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generate a JSON completion.

        Args:
            messages: List of messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON response
        """
        response = await self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {response.content[:200]}")
