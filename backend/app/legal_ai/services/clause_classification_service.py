"""
Clause Classification Service Implementation.

Detects and classifies all types of legal clauses in contracts.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.clause_classification import (
    ClauseClassificationInterface,
    ClauseType,
    ClassifiedClause,
    ClassificationResult,
)
from app.legal_ai.providers.config import get_service_config, AIProviderType
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

CLAUSE_CLASSIFICATION_PROMPT = """You are an expert legal document analyzer specializing in contract clause classification.

Analyze the following legal document and identify all distinct clauses. For each clause, classify it into one of these categories:
- arbitration: Arbitration and binding dispute resolution clauses
- indemnification: Indemnification and hold harmless provisions
- liability_limitation: Limitation of liability clauses
- termination: Termination and cancellation provisions
- non_compete: Non-compete and non-solicitation clauses
- confidentiality: Confidentiality and NDA provisions
- force_majeure: Force majeure and act of God clauses
- ip_assignment: Intellectual property assignment and licensing
- governing_law: Governing law and jurisdiction clauses
- dispute_resolution: General dispute resolution procedures
- warranty: Warranty and representation clauses
- payment_terms: Payment, pricing, and billing terms
- data_protection: Data protection and privacy provisions
- audit_rights: Audit and inspection rights
- insurance: Insurance requirements
- assignment: Assignment and transfer restrictions
- severability: Severability provisions
- entire_agreement: Entire agreement/integration clauses
- amendment: Amendment and modification provisions
- notice: Notice requirements
- unknown: Cannot be classified

For each clause found, provide:
1. The exact text of the clause
2. The clause type from the list above
3. Confidence score (0.0 to 1.0)
4. Key terms within the clause
5. Any party obligations mentioned
6. Risk indicators (if any)

Respond with a JSON array of classified clauses."""

SINGLE_CLAUSE_PROMPT = """Classify the following legal clause into one of these categories:
arbitration, indemnification, liability_limitation, termination, non_compete, confidentiality,
force_majeure, ip_assignment, governing_law, dispute_resolution, warranty, payment_terms,
data_protection, audit_rights, insurance, assignment, severability, entire_agreement,
amendment, notice, unknown

Also extract:
- Key terms
- Party obligations
- Risk indicators

Respond with JSON only."""


class ClauseClassificationService(ClauseClassificationInterface):
    """
    Service for detecting and classifying legal clauses.

    Uses AI models to identify clause boundaries and classify them
    into standard legal categories with confidence scores.
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """
        Initialize clause classification service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None

        # Load configuration
        config = get_service_config("clause_classification")
        if model is None:
            self.model = config.model
            self.provider = AIProvider(config.provider.value)

    async def initialize(self) -> None:
        """Initialize the AI provider client."""
        if self.provider == AIProvider.OPENAI:
            self._provider_client = OpenAIProvider(timeout=self.timeout)
        elif self.provider == AIProvider.ANTHROPIC:
            self._provider_client = AnthropicProvider(timeout=self.timeout)
        elif self.provider == AIProvider.VERTEX:
            self._provider_client = VertexProvider(timeout=self.timeout)
        else:
            self._provider_client = OpenAIProvider(timeout=self.timeout)

        await self._provider_client.initialize()
        logger.info(f"ClauseClassificationService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("ClauseClassificationService shutdown")

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            start = time.time()
            health = await self._provider_client.health_check()
            latency = (time.time() - start) * 1000

            return ServiceHealth(
                healthy=health.get("status") == "healthy",
                provider=self.provider,
                model=self.model,
                latency_ms=latency,
                details=health,
            )
        except Exception as e:
            return ServiceHealth(
                healthy=False,
                provider=self.provider,
                model=self.model,
                error=str(e),
            )

    async def classify_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        target_clause_types: Optional[List[ClauseType]] = None,
        min_confidence: float = 0.5,
    ) -> ClassificationResult:
        """
        Classify all clauses in a document.

        Args:
            document_text: Full document text
            document_id: Optional identifier
            target_clause_types: Filter to these types only
            min_confidence: Minimum confidence threshold

        Returns:
            ClassificationResult with all classified clauses
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("classify_document", document_length=len(document_text))

        try:
            # Build prompt with optional filtering
            prompt = CLAUSE_CLASSIFICATION_PROMPT
            if target_clause_types:
                types_str = ", ".join([t.value for t in target_clause_types])
                prompt += f"\n\nFocus only on these clause types: {types_str}"

            prompt += f"\n\nDocument to analyze:\n\n{document_text}"

            messages = [
                Message(role="system", content="You are a legal document analysis expert. Always respond with valid JSON."),
                Message(role="user", content=prompt),
            ]

            # Get response from AI
            if hasattr(self._provider_client, 'complete_json'):
                response_data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_data = self._parse_json_response(response.content)

            # Parse clauses from response
            clauses = self._parse_clauses(response_data, document_text, min_confidence)

            # Filter by target types if specified
            if target_clause_types:
                clauses = [c for c in clauses if c.clause_type in target_clause_types]

            # Calculate distribution
            distribution = self._calculate_distribution(clauses)

            processing_time = (time.time() - start_time) * 1000

            return ClassificationResult(
                document_id=doc_id,
                total_clauses=len(clauses),
                clauses=clauses,
                clause_distribution=distribution,
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("classify_document", e)
            raise RuntimeError(f"Classification failed: {e}")

    async def classify_clause(
        self,
        clause_text: str,
        context: Optional[str] = None,
    ) -> ClassifiedClause:
        """
        Classify a single clause.

        Args:
            clause_text: Text of the clause
            context: Surrounding context

        Returns:
            ClassifiedClause with classification
        """
        prompt = SINGLE_CLAUSE_PROMPT
        if context:
            prompt += f"\n\nContext:\n{context[:500]}"
        prompt += f"\n\nClause to classify:\n{clause_text}"

        messages = [
            Message(role="system", content="You are a legal expert. Respond with valid JSON only."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                data = self._parse_json_response(response.content)

            return ClassifiedClause(
                text=clause_text,
                clause_type=self._parse_clause_type(data.get("clause_type", "unknown")),
                confidence=float(data.get("confidence", 0.8)),
                start_position=0,
                end_position=len(clause_text),
                sub_type=data.get("sub_type"),
                key_terms=data.get("key_terms", []),
                party_obligations=data.get("party_obligations", []),
                risk_indicators=data.get("risk_indicators", []),
            )

        except Exception as e:
            self._log_error("classify_clause", e)
            return ClassifiedClause(
                text=clause_text,
                clause_type=ClauseType.UNKNOWN,
                confidence=0.0,
                start_position=0,
                end_position=len(clause_text),
            )

    async def extract_clauses(
        self,
        document_text: str,
    ) -> List[str]:
        """
        Extract individual clause segments from a document.

        Args:
            document_text: Full document text

        Returns:
            List of clause text segments
        """
        prompt = """Extract all distinct legal clauses from the following document.
Return each clause as a separate item. Include the full text of each clause.

Document:
""" + document_text + """

Respond with a JSON array of clause texts."""

        messages = [
            Message(role="system", content="You are a legal document parser. Respond with valid JSON array."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                data = self._parse_json_response(response.content)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "clauses" in data:
                return data["clauses"]
            else:
                return []

        except Exception as e:
            self._log_error("extract_clauses", e)
            # Fallback: split by double newlines
            return [p.strip() for p in document_text.split("\n\n") if len(p.strip()) > 50]

    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON from response content."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

    def _parse_clause_type(self, type_str: str) -> ClauseType:
        """Parse clause type string to enum."""
        try:
            return ClauseType(type_str.lower().strip())
        except ValueError:
            return ClauseType.UNKNOWN

    def _parse_clauses(
        self,
        data: Any,
        document_text: str,
        min_confidence: float,
    ) -> List[ClassifiedClause]:
        """Parse classified clauses from response data."""
        clauses = []

        # Handle different response structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("clauses", data.get("classified_clauses", []))
        else:
            return clauses

        for item in items:
            if not isinstance(item, dict):
                continue

            confidence = float(item.get("confidence", 0.8))
            if confidence < min_confidence:
                continue

            text = item.get("text", item.get("clause_text", ""))
            if not text:
                continue

            # Find position in document
            start_pos = document_text.find(text[:100])
            end_pos = start_pos + len(text) if start_pos >= 0 else len(text)

            clause = ClassifiedClause(
                text=text,
                clause_type=self._parse_clause_type(
                    item.get("clause_type", item.get("type", "unknown"))
                ),
                confidence=confidence,
                start_position=max(0, start_pos),
                end_position=end_pos,
                sub_type=item.get("sub_type"),
                key_terms=item.get("key_terms", []),
                party_obligations=item.get("party_obligations", item.get("obligations", [])),
                risk_indicators=item.get("risk_indicators", []),
            )
            clauses.append(clause)

        return clauses

    def _calculate_distribution(self, clauses: List[ClassifiedClause]) -> Dict[str, int]:
        """Calculate clause type distribution."""
        distribution: Dict[str, int] = {}
        for clause in clauses:
            key = clause.clause_type.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution
