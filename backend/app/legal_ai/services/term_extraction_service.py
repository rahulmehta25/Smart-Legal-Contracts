"""
Term Extraction Service Implementation.

Extracts structured data from contracts including parties, dates, amounts, and obligations.
"""

import time
import uuid
import logging
import json
import re
from typing import List, Optional, Dict, Any
from datetime import date, datetime

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.term_extraction import (
    TermExtractionInterface,
    EntityType,
    Party,
    MonetaryAmount,
    ContractDate,
    Deadline,
    ContractObligation,
    DefinedTerm,
    TermExtractionResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert legal document parser specializing in data extraction.

Extract all structured information from the following contract:

1. **Parties:** For each party, extract:
   - name: Full legal name
   - role: Their role (e.g., "Buyer", "Seller", "Licensor", "Service Provider")
   - entity_type: "corporation", "llc", "individual", "partnership", etc.
   - jurisdiction: State/country of incorporation if mentioned
   - address: Full address if provided
   - representative: Name of signing representative if mentioned

2. **Dates:** All dates mentioned including:
   - date_value: In YYYY-MM-DD format (null if relative)
   - date_string: Original text
   - date_type: "effective", "expiration", "renewal", "deadline", "notice", "payment", "other"
   - description: What the date represents
   - is_calculated: true if it's relative (e.g., "30 days after signing")
   - calculation_basis: The basis if calculated

3. **Monetary Amounts:** All financial values:
   - value: Numeric amount
   - currency: Currency code (USD, EUR, etc.)
   - description: What the amount is for
   - payment_type: "one-time", "recurring", "cap", "penalty", "other"
   - frequency: "monthly", "annually", "quarterly", etc. if recurring
   - conditions: Any conditions on the payment

4. **Deadlines:** Action deadlines:
   - description: What must be done
   - due_date: In YYYY-MM-DD if specific
   - due_date_text: Original text
   - responsible_party: Who must act
   - action_required: Specific action needed
   - penalty_for_missing: Consequences if missed
   - is_recurring: true/false
   - recurrence_pattern: If recurring

5. **Obligations:** Contractual duties:
   - description: Full description
   - obligated_party: Who has the obligation
   - obligation_type: "delivery", "payment", "performance", "reporting", etc.
   - trigger_condition: What triggers the obligation
   - deadline: When it must be fulfilled
   - deliverable: What must be delivered
   - standard_of_performance: Quality standard
   - consequences_of_breach: Penalties/remedies

6. **Defined Terms:** Terms with specific definitions:
   - term: The defined term
   - definition: Its definition
   - section_defined: Where defined
   - usage_count: Approximate uses in document

Contract text:
{document_text}

Respond with a JSON object containing: parties, dates, amounts, deadlines, obligations, defined_terms."""


class TermExtractionService(TermExtractionInterface):
    """
    Service for extracting structured data from contracts.

    Identifies and extracts parties, dates, amounts, deadlines,
    obligations, and defined terms into structured formats.
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
        Initialize term extraction service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None

        config = get_service_config("term_extraction")
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
        logger.info(f"TermExtractionService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("TermExtractionService shutdown")

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

    async def extract_all(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None,
    ) -> TermExtractionResult:
        """
        Extract all terms from a document.

        Args:
            document_text: Full document text
            document_id: Optional identifier
            entity_types: Specific types to extract (all if None)

        Returns:
            TermExtractionResult with all extracted data
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("extract_all", document_length=len(document_text))

        try:
            prompt = EXTRACTION_PROMPT.format(document_text=document_text)

            messages = [
                Message(
                    role="system",
                    content="You are an expert legal document parser. Extract information accurately and completely. Respond with valid JSON."
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

            # Parse all entity types
            parties = self._parse_parties(data.get("parties", []))
            dates = self._parse_dates(data.get("dates", []))
            amounts = self._parse_amounts(data.get("amounts", []))
            deadlines = self._parse_deadlines(data.get("deadlines", []))
            obligations = self._parse_obligations(data.get("obligations", []))
            defined_terms = self._parse_defined_terms(data.get("defined_terms", []))

            # Generate summaries
            contract_value_summary = self._generate_value_summary(amounts)
            timeline_summary = self._generate_timeline_summary(dates, deadlines)

            processing_time = (time.time() - start_time) * 1000

            return TermExtractionResult(
                document_id=doc_id,
                parties=parties,
                dates=dates,
                amounts=amounts,
                deadlines=deadlines,
                obligations=obligations,
                defined_terms=defined_terms,
                contract_value_summary=contract_value_summary,
                timeline_summary=timeline_summary,
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("extract_all", e)
            raise RuntimeError(f"Term extraction failed: {e}")

    async def extract_parties(
        self,
        document_text: str,
    ) -> List[Party]:
        """Extract all parties from a contract."""
        prompt = f"""Extract all parties from this contract with their details:
- name: Full legal name
- role: Their contractual role
- entity_type: Type of entity
- jurisdiction, address, representative if available

Contract:
{document_text}

Respond with a JSON array of parties."""

        messages = [
            Message(role="system", content="You are a legal document parser. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("parties", [])
            return self._parse_parties(items)

        except Exception as e:
            self._log_error("extract_parties", e)
            return []

    async def extract_amounts(
        self,
        document_text: str,
    ) -> List[MonetaryAmount]:
        """Extract all monetary amounts."""
        prompt = f"""Extract all monetary amounts from this contract:
- value: Numeric amount
- currency: Currency code
- description: What it's for
- payment_type: one-time, recurring, cap, etc.
- frequency if recurring
- conditions if any

Contract:
{document_text}

Respond with a JSON array of amounts."""

        messages = [
            Message(role="system", content="You are a financial document parser. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("amounts", [])
            return self._parse_amounts(items)

        except Exception as e:
            self._log_error("extract_amounts", e)
            return []

    async def extract_dates(
        self,
        document_text: str,
        reference_date: Optional[date] = None,
    ) -> List[ContractDate]:
        """Extract all dates from a contract."""
        ref = reference_date or date.today()

        prompt = f"""Extract all dates from this contract:
- date_value: YYYY-MM-DD format (null if relative)
- date_string: Original text
- date_type: effective, expiration, deadline, renewal, etc.
- description: What the date represents
- is_calculated: true if relative
- calculation_basis if applicable

Reference date for relative calculations: {ref.isoformat()}

Contract:
{document_text}

Respond with a JSON array of dates."""

        messages = [
            Message(role="system", content="You are a legal document parser. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("dates", [])
            return self._parse_dates(items)

        except Exception as e:
            self._log_error("extract_dates", e)
            return []

    async def extract_deadlines(
        self,
        document_text: str,
    ) -> List[Deadline]:
        """Extract all deadlines from a contract."""
        prompt = f"""Extract all deadlines from this contract:
- description: What must be done
- due_date: YYYY-MM-DD if specific
- due_date_text: Original text
- responsible_party: Who must act
- action_required: Specific action
- penalty_for_missing: Consequences
- is_recurring: true/false
- recurrence_pattern if recurring

Contract:
{document_text}

Respond with a JSON array of deadlines."""

        messages = [
            Message(role="system", content="You are a legal document parser. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("deadlines", [])
            return self._parse_deadlines(items)

        except Exception as e:
            self._log_error("extract_deadlines", e)
            return []

    async def extract_defined_terms(
        self,
        document_text: str,
    ) -> List[DefinedTerm]:
        """Extract defined terms and their definitions."""
        prompt = f"""Extract all defined terms from this contract (terms that are explicitly defined):
- term: The defined term
- definition: Its definition
- section_defined: Where it's defined
- usage_count: Approximate number of uses

Look for patterns like:
- "Term" means...
- "Term" shall mean...
- As used herein, "Term"...

Contract:
{document_text}

Respond with a JSON array of defined terms."""

        messages = [
            Message(role="system", content="You are a legal document parser. Respond with valid JSON."),
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

            items = data if isinstance(data, list) else data.get("defined_terms", [])
            return self._parse_defined_terms(items)

        except Exception as e:
            self._log_error("extract_defined_terms", e)
            return []

    async def generate_term_sheet(
        self,
        extraction_result: TermExtractionResult,
    ) -> str:
        """Generate a term sheet from extraction results."""
        sheet = [
            "# Contract Term Sheet",
            "",
            "## Parties",
        ]

        for party in extraction_result.parties:
            sheet.append(f"- **{party.role}:** {party.name} ({party.entity_type})")
            if party.jurisdiction:
                sheet.append(f"  - Jurisdiction: {party.jurisdiction}")
            if party.address:
                sheet.append(f"  - Address: {party.address}")

        if extraction_result.contract_value_summary:
            sheet.extend([
                "",
                "## Financial Terms",
                extraction_result.contract_value_summary,
            ])

        if extraction_result.amounts:
            sheet.append("")
            for amount in extraction_result.amounts:
                freq = f" ({amount.frequency})" if amount.frequency else ""
                sheet.append(f"- {amount.description}: {amount.currency} {amount.value:,.2f}{freq}")

        sheet.extend([
            "",
            "## Key Dates",
        ])

        for d in extraction_result.dates:
            date_str = d.date_value.isoformat() if d.date_value else d.date_string
            sheet.append(f"- **{d.date_type.capitalize()}:** {date_str} - {d.description}")

        if extraction_result.deadlines:
            sheet.extend([
                "",
                "## Deadlines",
            ])
            for dl in extraction_result.deadlines:
                recurring = " (recurring)" if dl.is_recurring else ""
                sheet.append(f"- {dl.due_date_text}: {dl.description} - {dl.responsible_party}{recurring}")

        if extraction_result.obligations:
            sheet.extend([
                "",
                "## Key Obligations",
            ])
            for ob in extraction_result.obligations:
                sheet.append(f"- **{ob.obligated_party}:** {ob.description}")

        if extraction_result.defined_terms:
            sheet.extend([
                "",
                "## Defined Terms",
            ])
            for term in extraction_result.defined_terms[:10]:  # Top 10
                sheet.append(f"- **{term.term}:** {term.definition[:100]}...")

        return "\n".join(sheet)

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

    def _parse_date_value(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str:
            return None
        try:
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _parse_parties(self, data: List[Dict]) -> List[Party]:
        """Parse parties from response data."""
        parties = []
        for item in data:
            if not isinstance(item, dict):
                continue
            parties.append(Party(
                name=item.get("name", ""),
                role=item.get("role", "Party"),
                entity_type=item.get("entity_type", "unknown"),
                jurisdiction=item.get("jurisdiction"),
                address=item.get("address"),
                contact_info=item.get("contact_info"),
                representative=item.get("representative"),
            ))
        return parties

    def _parse_dates(self, data: List[Dict]) -> List[ContractDate]:
        """Parse dates from response data."""
        dates = []
        for item in data:
            if not isinstance(item, dict):
                continue
            dates.append(ContractDate(
                date_value=self._parse_date_value(item.get("date_value")),
                date_string=item.get("date_string", ""),
                date_type=item.get("date_type", "other"),
                description=item.get("description", ""),
                is_calculated=item.get("is_calculated", False),
                calculation_basis=item.get("calculation_basis"),
            ))
        return dates

    def _parse_amounts(self, data: List[Dict]) -> List[MonetaryAmount]:
        """Parse monetary amounts from response data."""
        amounts = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                value = float(str(item.get("value", 0)).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                value = 0.0

            amounts.append(MonetaryAmount(
                value=value,
                currency=item.get("currency", "USD"),
                description=item.get("description", ""),
                payment_type=item.get("payment_type"),
                frequency=item.get("frequency"),
                conditions=item.get("conditions"),
                section_reference=item.get("section_reference"),
            ))
        return amounts

    def _parse_deadlines(self, data: List[Dict]) -> List[Deadline]:
        """Parse deadlines from response data."""
        deadlines = []
        for item in data:
            if not isinstance(item, dict):
                continue
            deadlines.append(Deadline(
                description=item.get("description", ""),
                due_date=self._parse_date_value(item.get("due_date")),
                due_date_text=item.get("due_date_text", ""),
                responsible_party=item.get("responsible_party", ""),
                action_required=item.get("action_required", ""),
                penalty_for_missing=item.get("penalty_for_missing"),
                is_recurring=item.get("is_recurring", False),
                recurrence_pattern=item.get("recurrence_pattern"),
            ))
        return deadlines

    def _parse_obligations(self, data: List[Dict]) -> List[ContractObligation]:
        """Parse obligations from response data."""
        obligations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            obligations.append(ContractObligation(
                description=item.get("description", ""),
                obligated_party=item.get("obligated_party", ""),
                obligation_type=item.get("obligation_type", "other"),
                trigger_condition=item.get("trigger_condition"),
                deadline=item.get("deadline"),
                deliverable=item.get("deliverable"),
                standard_of_performance=item.get("standard_of_performance"),
                consequences_of_breach=item.get("consequences_of_breach"),
            ))
        return obligations

    def _parse_defined_terms(self, data: List[Dict]) -> List[DefinedTerm]:
        """Parse defined terms from response data."""
        terms = []
        for item in data:
            if not isinstance(item, dict):
                continue
            terms.append(DefinedTerm(
                term=item.get("term", ""),
                definition=item.get("definition", ""),
                section_defined=item.get("section_defined"),
                usage_count=item.get("usage_count", 0),
                related_terms=item.get("related_terms", []),
            ))
        return terms

    def _generate_value_summary(self, amounts: List[MonetaryAmount]) -> Optional[str]:
        """Generate contract value summary."""
        if not amounts:
            return None

        total_by_currency: Dict[str, float] = {}
        for amount in amounts:
            total_by_currency[amount.currency] = total_by_currency.get(amount.currency, 0) + amount.value

        parts = [f"{currency} {value:,.2f}" for currency, value in total_by_currency.items()]
        return f"Total contract value: {', '.join(parts)}"

    def _generate_timeline_summary(
        self,
        dates: List[ContractDate],
        deadlines: List[Deadline],
    ) -> Optional[str]:
        """Generate timeline summary."""
        effective = next((d for d in dates if d.date_type == "effective"), None)
        expiration = next((d for d in dates if d.date_type == "expiration"), None)
        upcoming = [d for d in deadlines if d.due_date and (d.due_date - date.today()).days <= 30]

        parts = []
        if effective and effective.date_value:
            parts.append(f"Effective: {effective.date_value.isoformat()}")
        if expiration and expiration.date_value:
            parts.append(f"Expires: {expiration.date_value.isoformat()}")
        if upcoming:
            parts.append(f"Upcoming deadlines (30 days): {len(upcoming)}")

        return ". ".join(parts) if parts else None
