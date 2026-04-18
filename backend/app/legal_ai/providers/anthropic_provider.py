"""
Anthropic Provider Implementation.

Implements the BaseProvider interface for Anthropic's Claude API.
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


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude API provider implementation.

    Supports Claude 3.5, Claude 3, and Claude 2 models.
    """

    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"

    @property
    def supported_models(self) -> List[str]:
        """Return supported models."""
        return self.SUPPORTED_MODELS

    @property
    def default_model(self) -> str:
        """Return default model."""
        return "claude-sonnet-4-20250514"

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**kwargs)
            logger.info("Anthropic provider initialized")
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Anthropic provider shutdown")

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

        # Extract system message if present
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                user_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            kwargs["system"] = system_message

        if stop:
            kwargs["stop_sequences"] = stop

        try:
            response = await self._client.messages.create(**kwargs)

            content = ""
            if response.content:
                content = response.content[0].text if response.content[0].type == "text" else ""

            return CompletionResponse(
                content=content,
                model=response.model,
                finish_reason=response.stop_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
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

        # Extract system message
        system_message = None
        user_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                user_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            kwargs["system"] = system_message

        if stop:
            kwargs["stop_sequences"] = stop

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming completion failed: {e}")
            raise

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.

        Note: Anthropic doesn't have a native embedding API.
        This uses a workaround or raises NotImplementedError.
        """
        raise NotImplementedError(
            "Anthropic does not provide embedding models. "
            "Use OpenAI or another provider for embeddings."
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            if not self._client:
                await self.initialize()

            # Make a minimal request to verify connectivity
            response = await self._client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model_tested": "claude-3-haiku-20240307",
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

        Instructs Claude to respond with valid JSON.

        Args:
            messages: List of messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON response
        """
        # Add instruction for JSON output
        enhanced_messages = list(messages)
        if enhanced_messages and enhanced_messages[-1].role == "user":
            enhanced_messages[-1] = Message(
                role="user",
                content=enhanced_messages[-1].content + "\n\nRespond with valid JSON only.",
            )

        response = await self.complete(
            messages=enhanced_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Clean up response if needed
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {content[:200]}")
