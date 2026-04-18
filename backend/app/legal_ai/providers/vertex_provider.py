"""
Google Vertex AI Provider Implementation.

Implements the BaseProvider interface for Google's Vertex AI / Gemini API.
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


class VertexProvider(BaseProvider):
    """
    Google Vertex AI provider implementation.

    Supports Gemini and PaLM models.
    """

    SUPPORTED_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    EMBEDDING_MODELS = [
        "text-embedding-004",
        "textembedding-gecko@003",
    ]

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize Vertex AI provider.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location
            api_key: Optional Gemini API key (for direct API access)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_LOCATION", "us-central1")
        self._use_vertex = bool(self.project_id)

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "vertex"

    @property
    def supported_models(self) -> List[str]:
        """Return supported models."""
        return self.SUPPORTED_MODELS

    @property
    def default_model(self) -> str:
        """Return default model."""
        return "gemini-1.5-pro"

    async def initialize(self) -> None:
        """Initialize the Vertex AI / Gemini client."""
        try:
            if self._use_vertex:
                # Use Vertex AI SDK
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.project_id, location=self.location)
                self._client = GenerativeModel(self.default_model)
                self._client_type = "vertex"
            else:
                # Use Gemini API directly
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
                self._client_type = "gemini"

            logger.info(f"Vertex/Gemini provider initialized (type: {self._client_type})")
        except ImportError as e:
            if "vertexai" in str(e):
                raise ImportError(
                    "google-cloud-aiplatform package required. "
                    "Install with: pip install google-cloud-aiplatform"
                )
            else:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Vertex/Gemini provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._client = None
        logger.info("Vertex/Gemini provider shutdown")

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

        # Convert messages to Gemini format
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })

        try:
            if self._client_type == "gemini":
                import google.generativeai as genai

                gen_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction,
                )

                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if stop:
                    generation_config["stop_sequences"] = stop

                response = await gen_model.generate_content_async(
                    contents,
                    generation_config=generation_config,
                )

                content = response.text if response.text else ""

                return CompletionResponse(
                    content=content,
                    model=model,
                    finish_reason=response.candidates[0].finish_reason.name if response.candidates else "stop",
                    usage={
                        "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                        "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                        "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                    },
                    raw_response=response,
                )
            else:
                # Vertex AI path
                from vertexai.generative_models import GenerativeModel, GenerationConfig

                gen_model = GenerativeModel(
                    model,
                    system_instruction=system_instruction,
                )

                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop,
                )

                response = await gen_model.generate_content_async(
                    contents,
                    generation_config=config,
                )

                content = response.text if response.text else ""

                return CompletionResponse(
                    content=content,
                    model=model,
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    raw_response=response,
                )

        except Exception as e:
            logger.error(f"Vertex/Gemini completion failed: {e}")
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

        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })

        try:
            if self._client_type == "gemini":
                import google.generativeai as genai

                gen_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction,
                )

                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if stop:
                    generation_config["stop_sequences"] = stop

                response = await gen_model.generate_content_async(
                    contents,
                    generation_config=generation_config,
                    stream=True,
                )

                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                # Vertex AI streaming
                from vertexai.generative_models import GenerativeModel, GenerationConfig

                gen_model = GenerativeModel(
                    model,
                    system_instruction=system_instruction,
                )

                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop,
                )

                response = await gen_model.generate_content_async(
                    contents,
                    generation_config=config,
                    stream=True,
                )

                async for chunk in response:
                    if chunk.text:
                        yield chunk.text

        except Exception as e:
            logger.error(f"Vertex/Gemini streaming completion failed: {e}")
            raise

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        if not self._client:
            await self.initialize()

        model = model or "text-embedding-004"

        try:
            if self._client_type == "gemini":
                import google.generativeai as genai

                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model=f"models/{model}",
                        content=text,
                    )
                    embeddings.append(result["embedding"])

                return EmbeddingResponse(
                    embeddings=embeddings,
                    model=model,
                    usage={"total_tokens": 0},
                )
            else:
                from vertexai.language_models import TextEmbeddingModel

                embed_model = TextEmbeddingModel.from_pretrained(model)
                embeddings_result = embed_model.get_embeddings(texts)

                return EmbeddingResponse(
                    embeddings=[e.values for e in embeddings_result],
                    model=model,
                    usage={"total_tokens": 0},
                )

        except Exception as e:
            logger.error(f"Vertex/Gemini embedding failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            if not self._client:
                await self.initialize()

            # Make a minimal request
            response = await self.complete(
                messages=[Message(role="user", content="ping")],
                model="gemini-1.5-flash",
                max_tokens=5,
            )

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "client_type": self._client_type,
                "model_tested": "gemini-1.5-flash",
                "response_received": bool(response.content),
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
        # Add instruction for JSON output
        enhanced_messages = list(messages)
        if enhanced_messages and enhanced_messages[-1].role == "user":
            enhanced_messages[-1] = Message(
                role="user",
                content=enhanced_messages[-1].content + "\n\nRespond with valid JSON only, no markdown formatting.",
            )

        response = await self.complete(
            messages=enhanced_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Clean up response
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
