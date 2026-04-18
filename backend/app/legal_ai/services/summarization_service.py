"""
Contract Summarization Service Implementation.

Generates executive summaries of contracts with structured data extraction.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any
from datetime import date, datetime

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.summarization import (
    SummarizationInterface,
    ObligationType,
    KeyTerm,
    Obligation,
    ImportantDate,
    ContractSummary,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """You are an expert legal analyst specializing in contract summarization.

Generate a comprehensive executive summary of the following contract. Extract:

1. **Basic Information:**
   - Contract title/type
   - Parties involved (names and roles)
   - Effective date
   - Expiration date
   - Total contract value (if specified)

2. **Key Terms:** (name, description, value if applicable, importance level)
   Extract the most important terms defining the agreement.

3. **Obligations:** For each party, list their obligations including:
   - Description
   - Type (payment, delivery, performance, reporting, compliance, notification, maintenance, confidentiality, other)
   - Deadline if specified
   - Whether recurring
   - Penalty for breach

4. **Important Dates:** All significant dates including:
   - Effective date
   - Expiration date
   - Renewal dates
   - Payment due dates
   - Notice deadlines
   - Review periods

5. **Other Key Elements:**
   - Renewal terms
   - Termination conditions
   - Governing law
   - Dispute resolution mechanism
   - Key risks

6. **Action Items:** Recommended next steps based on the contract.

Summary length preference: {summary_length}
Focus areas: {focus_areas}

Contract text:
{document_text}

Respond with a JSON object containing all the above elements."""


class ContractSummarizationService(SummarizationInterface):
    """
    Service for generating executive summaries of contracts.

    Extracts structured information and generates human-readable
    summaries highlighting key terms, obligations, and deadlines.
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
        Initialize summarization service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None

        config = get_service_config("summarization")
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
        logger.info(f"ContractSummarizationService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("ContractSummarizationService shutdown")

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

    async def summarize_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        summary_length: str = "medium",
        focus_areas: Optional[List[str]] = None,
    ) -> ContractSummary:
        """
        Generate an executive summary of a contract.

        Args:
            document_text: Full contract text
            document_id: Optional identifier
            summary_length: "brief", "medium", or "detailed"
            focus_areas: Specific areas to emphasize

        Returns:
            ContractSummary with structured information
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("summarize_document", document_length=len(document_text))

        try:
            prompt = SUMMARIZATION_PROMPT.format(
                summary_length=summary_length,
                focus_areas=", ".join(focus_areas) if focus_areas else "all aspects",
                document_text=document_text,
            )

            messages = [
                Message(
                    role="system",
                    content="You are an expert legal analyst. Provide clear, accurate contract summaries. Respond with valid JSON."
                ),
                Message(role="user", content=prompt),
            ]

            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
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
                data = self._parse_json_response(response.content)

            # Parse all components
            key_terms = self._parse_key_terms(data.get("key_terms", []))
            obligations = self._parse_obligations(data.get("obligations", []))
            important_dates = self._parse_dates(data.get("important_dates", []))

            processing_time = (time.time() - start_time) * 1000

            return ContractSummary(
                document_id=doc_id,
                title=data.get("title", data.get("contract_type", "Untitled Contract")),
                executive_summary=data.get("executive_summary", data.get("summary", "")),
                contract_type=data.get("contract_type", "Unknown"),
                parties=data.get("parties", []),
                effective_date=self._parse_date(data.get("effective_date")),
                expiration_date=self._parse_date(data.get("expiration_date")),
                total_value=data.get("total_value", data.get("contract_value")),
                key_terms=key_terms,
                obligations=obligations,
                important_dates=important_dates,
                renewal_terms=data.get("renewal_terms"),
                termination_conditions=data.get("termination_conditions", []),
                governing_law=data.get("governing_law"),
                dispute_resolution=data.get("dispute_resolution"),
                key_risks=data.get("key_risks", []),
                action_items=data.get("action_items", []),
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("summarize_document", e)
            raise RuntimeError(f"Summarization failed: {e}")

    async def extract_key_terms(
        self,
        document_text: str,
    ) -> List[KeyTerm]:
        """Extract key terms from a contract."""
        prompt = f"""Extract all key terms from the following contract.
For each term, provide: term_name, description, value (if applicable), section_reference, importance (high/medium/low).

Contract:
{document_text}

Respond with a JSON array of key terms."""

        messages = [
            Message(role="system", content="You are a contract analysis expert. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
                data = self._parse_json_response(response.content)

            items = data if isinstance(data, list) else data.get("key_terms", [])
            return self._parse_key_terms(items)

        except Exception as e:
            self._log_error("extract_key_terms", e)
            return []

    async def extract_obligations(
        self,
        document_text: str,
        party_filter: Optional[str] = None,
    ) -> List[Obligation]:
        """Extract contractual obligations."""
        filter_text = f" for {party_filter}" if party_filter else ""
        prompt = f"""Extract all contractual obligations{filter_text} from the following contract.
For each obligation, provide: description, obligated_party, obligation_type, deadline, recurring, recurrence_pattern, penalty_for_breach, section_reference.

Obligation types: payment, delivery, performance, reporting, compliance, notification, maintenance, confidentiality, other

Contract:
{document_text}

Respond with a JSON array of obligations."""

        messages = [
            Message(role="system", content="You are a contract analysis expert. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                )
                data = self._parse_json_response(response.content)

            items = data if isinstance(data, list) else data.get("obligations", [])
            return self._parse_obligations(items)

        except Exception as e:
            self._log_error("extract_obligations", e)
            return []

    async def extract_dates(
        self,
        document_text: str,
        reference_date: Optional[date] = None,
    ) -> List[ImportantDate]:
        """Extract important dates from a contract."""
        ref_date = reference_date or date.today()

        prompt = f"""Extract all important dates from the following contract.
For each date, provide: date_value (YYYY-MM-DD format), description, date_type (effective, expiration, deadline, renewal, payment, notice, review, other), action_required.

Calculate days_until based on reference date: {ref_date.isoformat()}

Contract:
{document_text}

Respond with a JSON array of dates."""

        messages = [
            Message(role="system", content="You are a contract analysis expert. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("dates", data.get("important_dates", []))
            return self._parse_dates(items, ref_date)

        except Exception as e:
            self._log_error("extract_dates", e)
            return []

    async def generate_action_items(
        self,
        summary: ContractSummary,
        role: str = "legal_counsel",
    ) -> List[str]:
        """Generate action items based on contract summary."""
        summary_text = f"""Contract: {summary.title}
Parties: {', '.join(summary.parties)}
Effective: {summary.effective_date}
Expiration: {summary.expiration_date}
Key Risks: {', '.join(summary.key_risks)}
Obligations count: {len(summary.obligations)}
Upcoming deadlines: {len([d for d in summary.important_dates if d.days_until and d.days_until <= 30])}"""

        prompt = f"""Based on the following contract summary, generate action items for a {role}.

{summary_text}

Consider:
- Upcoming deadlines that need attention
- Risk mitigation steps
- Compliance requirements
- Negotiation opportunities
- Documentation needs

Respond with a JSON array of action items (strings)."""

        messages = [
            Message(role="system", content="You are a legal advisor. Provide practical, actionable recommendations. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=0.1,
                    max_tokens=2048,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=0.1,
                    max_tokens=2048,
                )
                data = self._parse_json_response(response.content)

            if isinstance(data, list):
                return data
            return data.get("action_items", [])

        except Exception as e:
            self._log_error("generate_action_items", e)
            return []

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

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None
        try:
            # Try various formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _parse_key_terms(self, terms_data: List[Dict]) -> List[KeyTerm]:
        """Parse key terms from response data."""
        terms = []
        for item in terms_data:
            if not isinstance(item, dict):
                continue
            terms.append(KeyTerm(
                term_name=item.get("term_name", item.get("name", "")),
                description=item.get("description", ""),
                value=item.get("value"),
                section_reference=item.get("section_reference", item.get("section")),
                importance=item.get("importance", "medium"),
            ))
        return terms

    def _parse_obligations(self, obligations_data: List[Dict]) -> List[Obligation]:
        """Parse obligations from response data."""
        obligations = []
        for item in obligations_data:
            if not isinstance(item, dict):
                continue

            try:
                ob_type = ObligationType(item.get("obligation_type", item.get("type", "other")).lower())
            except ValueError:
                ob_type = ObligationType.OTHER

            obligations.append(Obligation(
                description=item.get("description", ""),
                obligated_party=item.get("obligated_party", item.get("party", "")),
                obligation_type=ob_type,
                deadline=self._parse_date(item.get("deadline")),
                recurring=item.get("recurring", False),
                recurrence_pattern=item.get("recurrence_pattern"),
                penalty_for_breach=item.get("penalty_for_breach", item.get("penalty")),
                section_reference=item.get("section_reference", item.get("section")),
            ))
        return obligations

    def _parse_dates(self, dates_data: List[Dict], reference: Optional[date] = None) -> List[ImportantDate]:
        """Parse important dates from response data."""
        dates = []
        ref = reference or date.today()

        for item in dates_data:
            if not isinstance(item, dict):
                continue

            date_value = self._parse_date(item.get("date_value", item.get("date")))
            days_until = None
            if date_value:
                days_until = (date_value - ref).days

            dates.append(ImportantDate(
                date_value=date_value or date.today(),
                description=item.get("description", ""),
                date_type=item.get("date_type", item.get("type", "other")),
                action_required=item.get("action_required"),
                days_until=days_until,
            ))
        return dates
