"""
Template Matching Service Implementation.

Compares contracts against standard templates and flags deviations.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import difflib

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.template_matching import (
    TemplateMatchingInterface,
    DeviationType,
    DeviationRisk,
    Template,
    TemplateDeviation,
    ClauseAlignment,
    TemplateMatchResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

# Built-in template database
TEMPLATE_DB: Dict[str, Template] = {}


class TemplateMatchingService(TemplateMatchingInterface):
    """
    Service for matching documents against templates.

    Identifies deviations from standard templates and assesses risk.
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
        Initialize template matching service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None
        self._templates: Dict[str, Template] = TEMPLATE_DB.copy()

        config = get_service_config("template_matching")
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
        logger.info(f"TemplateMatchingService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("TemplateMatchingService shutdown")

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

    async def match_template(
        self,
        document_text: str,
        template: Template,
        document_id: Optional[str] = None,
        strict_mode: bool = False,
    ) -> TemplateMatchResult:
        """
        Match a document against a template.

        Args:
            document_text: Document to analyze
            template: Template to compare against
            document_id: Optional identifier
            strict_mode: Flag all deviations if True

        Returns:
            TemplateMatchResult with analysis
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("match_template", template_id=template.template_id)

        try:
            # Get clause alignments
            clause_alignments = await self.align_clauses(document_text, template)

            # Analyze deviations
            prompt = f"""Compare this document against the standard template and identify all deviations.

TEMPLATE NAME: {template.name}
TEMPLATE TYPE: {template.contract_type}
REQUIRED CLAUSES: {', '.join(template.required_clauses)}
OPTIONAL CLAUSES: {', '.join(template.optional_clauses)}

TEMPLATE TEXT:
{template.template_text}

DOCUMENT TEXT:
{document_text}

For each deviation, provide:
1. deviation_type: "missing_clause", "additional_clause", "modified_language", "different_terms", "structural_difference", "semantic_deviation"
2. risk_level: "high", "medium", "low", "acceptable"
3. template_section: Which template section
4. document_section: Which document section
5. template_text: Expected text
6. document_text: Actual text
7. description: What the deviation is
8. impact_analysis: Business/legal impact
9. approval_required: true/false
10. suggested_action: What to do
11. confidence: 0-1

Also identify:
- Missing required clauses
- Additional non-template clauses
- Overall recommendations

Respond with JSON containing: deviations, missing_required_clauses, additional_clauses, summary, recommendations."""

            messages = [
                Message(
                    role="system",
                    content="You are a contract template compliance expert. Identify all deviations. Respond with valid JSON."
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

            # Parse deviations
            deviations = self._parse_deviations(data.get("deviations", []))

            # In strict mode, elevate all deviations
            if strict_mode:
                for dev in deviations:
                    dev.approval_required = True
                    if dev.risk_level == DeviationRisk.ACCEPTABLE:
                        dev.risk_level = DeviationRisk.LOW

            # Calculate match score
            total_clauses = len(template.required_clauses) + len(template.optional_clauses)
            matched = sum(1 for a in clause_alignments if a.status == "matched")
            match_score = matched / max(total_clauses, 1)

            # Count high risk deviations
            high_risk = len([d for d in deviations if d.risk_level == DeviationRisk.HIGH])

            # Determine if approval needed
            requires_approval = high_risk > 0 or any(d.approval_required for d in deviations)
            approval_reasons = [d.description for d in deviations if d.approval_required]

            processing_time = (time.time() - start_time) * 1000

            return TemplateMatchResult(
                document_id=doc_id,
                template=template,
                overall_match_score=match_score,
                deviations=deviations,
                clause_alignments=clause_alignments,
                missing_required_clauses=data.get("missing_required_clauses", []),
                additional_clauses=data.get("additional_clauses", []),
                high_risk_deviations=high_risk,
                requires_approval=requires_approval,
                approval_reasons=approval_reasons,
                summary=data.get("summary", ""),
                recommendations=data.get("recommendations", []),
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("match_template", e)
            raise RuntimeError(f"Template matching failed: {e}")

    async def find_best_template(
        self,
        document_text: str,
        templates: List[Template],
    ) -> Tuple[Template, float]:
        """Find the best matching template for a document."""
        if not templates:
            raise ValueError("No templates provided")

        best_template = templates[0]
        best_score = 0.0

        for template in templates:
            # Quick similarity check
            similarity = difflib.SequenceMatcher(
                None,
                document_text.lower(),
                template.template_text.lower()
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_template = template

        return best_template, best_score

    async def align_clauses(
        self,
        document_text: str,
        template: Template,
    ) -> List[ClauseAlignment]:
        """Align document clauses with template clauses."""
        prompt = f"""Align the clauses in this document with the template clauses.

Template clauses: {', '.join(template.required_clauses + template.optional_clauses)}

Document text:
{document_text[:8000]}

For each template clause, find if it exists in the document and how closely it matches.

Respond with JSON array containing for each clause:
- template_clause_name
- template_clause_text (brief)
- document_clause_text (if found)
- alignment_score (0-1)
- status: "matched", "modified", "missing", "extra"
- deviations: array of specific differences"""

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

            alignments = []
            items = data if isinstance(data, list) else data.get("alignments", [])

            for item in items:
                if not isinstance(item, dict):
                    continue

                deviations = []
                for dev_data in item.get("deviations", []):
                    if isinstance(dev_data, dict):
                        deviations.append(TemplateDeviation(
                            deviation_type=self._parse_deviation_type(dev_data.get("deviation_type", "modified_language")),
                            risk_level=self._parse_risk_level(dev_data.get("risk_level", "low")),
                            template_section=dev_data.get("template_section"),
                            document_section=dev_data.get("document_section"),
                            template_text=dev_data.get("template_text"),
                            document_text=dev_data.get("document_text"),
                            description=dev_data.get("description", ""),
                            impact_analysis=dev_data.get("impact_analysis", ""),
                            approval_required=dev_data.get("approval_required", False),
                            suggested_action=dev_data.get("suggested_action", ""),
                            confidence=float(dev_data.get("confidence", 0.8)),
                        ))

                alignments.append(ClauseAlignment(
                    template_clause_name=item.get("template_clause_name", ""),
                    template_clause_text=item.get("template_clause_text", ""),
                    document_clause_text=item.get("document_clause_text"),
                    alignment_score=float(item.get("alignment_score", 0.0)),
                    status=item.get("status", "missing"),
                    deviations=deviations,
                ))

            return alignments

        except Exception as e:
            self._log_error("align_clauses", e)
            return []

    async def generate_deviation_report(
        self,
        result: TemplateMatchResult,
        include_recommendations: bool = True,
    ) -> str:
        """Generate a deviation report."""
        report = [
            "# Template Deviation Report",
            "",
            f"**Document ID:** {result.document_id}",
            f"**Template:** {result.template.name} v{result.template.version}",
            f"**Match Score:** {result.overall_match_score:.1%}",
            f"**Requires Approval:** {'Yes' if result.requires_approval else 'No'}",
            "",
            "## Summary",
            result.summary,
            "",
        ]

        if result.missing_required_clauses:
            report.extend([
                "## Missing Required Clauses",
                "",
            ])
            for clause in result.missing_required_clauses:
                report.append(f"- **{clause}** - REQUIRED but not found")
            report.append("")

        if result.high_risk_deviations > 0:
            report.extend([
                "## High Risk Deviations",
                "",
            ])
            for dev in result.deviations:
                if dev.risk_level == DeviationRisk.HIGH:
                    report.extend([
                        f"### {dev.description}",
                        f"**Type:** {dev.deviation_type.value}",
                        f"**Impact:** {dev.impact_analysis}",
                        "",
                        f"*Template:* {dev.template_text or 'N/A'}",
                        f"*Document:* {dev.document_text or 'N/A'}",
                        "",
                        f"**Action:** {dev.suggested_action}",
                        "",
                    ])

        if result.deviations:
            report.extend([
                "## All Deviations",
                "",
            ])
            for i, dev in enumerate(result.deviations, 1):
                report.append(f"{i}. [{dev.risk_level.value.upper()}] {dev.description}")
            report.append("")

        if include_recommendations and result.recommendations:
            report.extend([
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")

        return "\n".join(report)

    async def suggest_template_conformance(
        self,
        deviation: TemplateDeviation,
        template: Template,
    ) -> str:
        """Suggest how to make document conform to template."""
        prompt = f"""Suggest replacement language to make this document conform to the template.

Template Name: {template.name}
Template Type: {template.contract_type}

Deviation:
- Type: {deviation.deviation_type.value}
- Current (non-conforming): {deviation.document_text}
- Expected (template): {deviation.template_text}
- Issue: {deviation.description}

Provide conforming language that:
1. Matches the template intent
2. Is legally sound
3. Uses clear language

Respond with just the suggested replacement text."""

        messages = [
            Message(role="system", content="You are a contract drafting expert."),
            Message(role="user", content=prompt),
        ]

        try:
            response = await self._provider_client.complete(
                messages=messages,
                model=self.model,
                temperature=0.1,
                max_tokens=1024,
            )
            return response.content.strip()

        except Exception as e:
            self._log_error("suggest_template_conformance", e)
            return ""

    async def register_template(
        self,
        template: Template,
    ) -> str:
        """Register a new template for matching."""
        # Generate ID if not provided
        if not template.template_id:
            template.template_id = str(uuid.uuid4())

        self._templates[template.template_id] = template
        logger.info(f"Registered template: {template.template_id} - {template.name}")

        return template.template_id

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

    def _parse_deviation_type(self, type_str: str) -> DeviationType:
        """Parse deviation type from string."""
        try:
            return DeviationType(type_str.lower().replace("-", "_").replace(" ", "_"))
        except ValueError:
            return DeviationType.MODIFIED_LANGUAGE

    def _parse_risk_level(self, risk_str: str) -> DeviationRisk:
        """Parse risk level from string."""
        try:
            return DeviationRisk(risk_str.lower())
        except ValueError:
            return DeviationRisk.MEDIUM

    def _parse_deviations(self, deviations_data: List[Dict]) -> List[TemplateDeviation]:
        """Parse deviations from response data."""
        deviations = []
        for item in deviations_data:
            if not isinstance(item, dict):
                continue

            deviations.append(TemplateDeviation(
                deviation_type=self._parse_deviation_type(item.get("deviation_type", "modified_language")),
                risk_level=self._parse_risk_level(item.get("risk_level", "medium")),
                template_section=item.get("template_section"),
                document_section=item.get("document_section"),
                template_text=item.get("template_text"),
                document_text=item.get("document_text"),
                description=item.get("description", ""),
                impact_analysis=item.get("impact_analysis", ""),
                approval_required=item.get("approval_required", False),
                suggested_action=item.get("suggested_action", ""),
                confidence=float(item.get("confidence", 0.8)),
            ))

        return deviations
