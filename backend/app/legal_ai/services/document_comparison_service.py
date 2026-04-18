"""
Document Comparison Service Implementation.

Provides semantic diff between contract versions with risk analysis.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any
from collections import defaultdict
import difflib

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.document_comparison import (
    DocumentComparisonInterface,
    ChangeType,
    ChangeImpact,
    ChangeDomain,
    DocumentChange,
    ClauseComparison,
    SimilarityMetrics,
    ComparisonResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """You are an expert legal document analyst specializing in contract comparison and redlining.

Compare the following two versions of a contract and identify all significant changes.

For each change, provide:
1. change_type: "addition", "deletion", "modification", "moved", "semantic_change", "formatting_only"
2. impact: "critical", "significant", "moderate", "minor", "cosmetic"
3. domain: "legal", "financial", "operational", "technical", "administrative"
4. original_text: Text from original (null if addition)
5. modified_text: Text from modified (null if deletion)
6. original_location: Section/clause in original
7. modified_location: Section/clause in modified
8. semantic_description: What the change means in plain language
9. risk_assessment: How this change affects risk profile
10. risk_score: 0.0-1.0 (higher = more risk)
11. affected_parties: Which parties are affected
12. recommendation: What action to take

Also provide:
- An executive summary of all changes
- Key negotiation points arising from changes
- Overall risk delta (positive = increased risk)

Perspective: {party_perspective}
Focus areas: {focus_areas}

ORIGINAL DOCUMENT:
{original_text}

MODIFIED DOCUMENT:
{modified_text}

Respond with a JSON object containing: changes, executive_summary, key_changes, negotiation_points, overall_risk_delta."""


class DocumentComparisonService(DocumentComparisonInterface):
    """
    Service for comparing contract versions.

    Provides semantic diff, risk analysis, and redline generation.
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 180,
    ):
        """
        Initialize document comparison service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None

        config = get_service_config("document_comparison")
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
        logger.info(f"DocumentComparisonService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("DocumentComparisonService shutdown")

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

    async def compare_documents(
        self,
        original_text: str,
        modified_text: str,
        original_id: Optional[str] = None,
        modified_id: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        party_perspective: str = "client",
    ) -> ComparisonResult:
        """
        Compare two versions of a document.

        Args:
            original_text: Original document text
            modified_text: Modified document text
            original_id: Optional identifier for original
            modified_id: Optional identifier for modified
            focus_areas: Areas to focus on
            party_perspective: Perspective for risk assessment

        Returns:
            ComparisonResult with detailed analysis
        """
        if not original_text.strip() or not modified_text.strip():
            raise ValueError("Both documents must have content")

        start_time = time.time()
        orig_id = original_id or str(uuid.uuid4())
        mod_id = modified_id or str(uuid.uuid4())

        self._log_request("compare_documents", orig_len=len(original_text), mod_len=len(modified_text))

        try:
            # Calculate similarity metrics first (fast, local operation)
            similarity = await self.calculate_similarity(original_text, modified_text)

            # Get AI analysis of changes
            prompt = COMPARISON_PROMPT.format(
                party_perspective=party_perspective,
                focus_areas=", ".join(focus_areas) if focus_areas else "all aspects",
                original_text=original_text,
                modified_text=modified_text,
            )

            messages = [
                Message(
                    role="system",
                    content="You are an expert contract comparison analyst. Identify all meaningful changes. Respond with valid JSON."
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

            # Parse changes
            changes = self._parse_changes(data.get("changes", []))

            # Calculate impact distribution
            impact_summary = self._calculate_impact_distribution(changes)
            domain_summary = self._calculate_domain_distribution(changes)

            # Get clause comparisons
            clause_comparisons = await self._generate_clause_comparisons(
                original_text, modified_text, changes
            )

            processing_time = (time.time() - start_time) * 1000

            return ComparisonResult(
                original_document_id=orig_id,
                modified_document_id=mod_id,
                similarity_metrics=similarity,
                total_changes=len(changes),
                changes=changes,
                clause_comparisons=clause_comparisons,
                impact_summary=impact_summary,
                domain_summary=domain_summary,
                overall_risk_delta=float(data.get("overall_risk_delta", 0.0)),
                executive_summary=data.get("executive_summary", ""),
                key_changes=data.get("key_changes", []),
                negotiation_points=data.get("negotiation_points", []),
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("compare_documents", e)
            raise RuntimeError(f"Document comparison failed: {e}")

    async def compare_clauses(
        self,
        original_clause: str,
        modified_clause: str,
        clause_type: Optional[str] = None,
    ) -> ClauseComparison:
        """Compare two versions of a specific clause."""
        prompt = f"""Compare these two versions of a {clause_type or 'contract'} clause:

ORIGINAL:
{original_clause}

MODIFIED:
{modified_clause}

Identify:
1. What changed (semantically)
2. Impact of the change
3. Who benefits from the change
4. Risk implications

Respond with JSON containing: changes (array), overall_impact, summary."""

        messages = [
            Message(role="system", content="You are a contract comparison expert. Respond with valid JSON."),
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

            changes = self._parse_changes(data.get("changes", []))

            # Determine status
            if not changes:
                status = "unchanged"
            elif all(c.change_type == ChangeType.FORMATTING_ONLY for c in changes):
                status = "formatting_only"
            else:
                status = "modified"

            impact_str = data.get("overall_impact", "moderate")
            try:
                overall_impact = ChangeImpact(impact_str.lower())
            except ValueError:
                overall_impact = ChangeImpact.MODERATE

            return ClauseComparison(
                clause_type=clause_type or "unknown",
                original_text=original_clause,
                modified_text=modified_clause,
                status=status,
                changes=changes,
                overall_impact=overall_impact,
                summary=data.get("summary", ""),
            )

        except Exception as e:
            self._log_error("compare_clauses", e)
            return ClauseComparison(
                clause_type=clause_type or "unknown",
                original_text=original_clause,
                modified_text=modified_clause,
                status="error",
                changes=[],
                overall_impact=ChangeImpact.MODERATE,
                summary=f"Comparison error: {e}",
            )

    async def calculate_similarity(
        self,
        text1: str,
        text2: str,
    ) -> SimilarityMetrics:
        """Calculate similarity metrics between two texts."""
        # Text similarity using difflib
        seq_matcher = difflib.SequenceMatcher(None, text1, text2)
        text_similarity = seq_matcher.ratio()

        # Word-level analysis
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)

        # Calculate change percentage
        word_count_original = len(text1.split())
        word_count_modified = len(text2.split())
        word_diff = abs(word_count_original - word_count_modified)
        change_percentage = word_diff / max(word_count_original, 1)

        # For semantic similarity, we'd ideally use embeddings
        # For now, use text similarity as proxy
        semantic_similarity = text_similarity * 0.7 + word_overlap * 0.3

        return SimilarityMetrics(
            text_similarity=text_similarity,
            semantic_similarity=semantic_similarity,
            structural_similarity=text_similarity,  # Simplified
            clause_overlap=word_overlap,
            word_count_original=word_count_original,
            word_count_modified=word_count_modified,
            change_percentage=change_percentage,
        )

    async def generate_redline(
        self,
        comparison_result: ComparisonResult,
        format: str = "markdown",
    ) -> str:
        """Generate a redline document showing changes."""
        if format == "markdown":
            return self._generate_markdown_redline(comparison_result)
        elif format == "html":
            return self._generate_html_redline(comparison_result)
        else:
            return self._generate_text_redline(comparison_result)

    async def assess_change_risk(
        self,
        change: DocumentChange,
        full_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess the risk implications of a specific change."""
        prompt = f"""Assess the risk implications of this contract change:

Change Type: {change.change_type.value}
Domain: {change.domain.value}
Original: {change.original_text or 'N/A'}
Modified: {change.modified_text or 'N/A'}
Description: {change.semantic_description}

{f"Context: {full_context[:1000]}" if full_context else ""}

Provide:
1. Risk score (0-1)
2. Risk factors
3. Affected stakeholders
4. Mitigation options
5. Negotiation strategy

Respond with JSON."""

        messages = [
            Message(role="system", content="You are a legal risk analyst. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                data = self._parse_json_response(response.content)

            return data

        except Exception as e:
            self._log_error("assess_change_risk", e)
            return {"error": str(e), "risk_score": change.risk_score}

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

    def _parse_changes(self, changes_data: List[Dict]) -> List[DocumentChange]:
        """Parse changes from response data."""
        changes = []
        for item in changes_data:
            if not isinstance(item, dict):
                continue

            try:
                change_type = ChangeType(item.get("change_type", "modification").lower())
            except ValueError:
                change_type = ChangeType.MODIFICATION

            try:
                impact = ChangeImpact(item.get("impact", "moderate").lower())
            except ValueError:
                impact = ChangeImpact.MODERATE

            try:
                domain = ChangeDomain(item.get("domain", "legal").lower())
            except ValueError:
                domain = ChangeDomain.LEGAL

            changes.append(DocumentChange(
                change_type=change_type,
                impact=impact,
                domain=domain,
                original_text=item.get("original_text"),
                modified_text=item.get("modified_text"),
                original_location=item.get("original_location"),
                modified_location=item.get("modified_location"),
                semantic_description=item.get("semantic_description", ""),
                risk_assessment=item.get("risk_assessment", ""),
                risk_score=float(item.get("risk_score", 0.5)),
                affected_parties=item.get("affected_parties", []),
                recommendation=item.get("recommendation"),
            ))

        return changes

    def _calculate_impact_distribution(self, changes: List[DocumentChange]) -> Dict[ChangeImpact, int]:
        """Calculate distribution of change impacts."""
        distribution: Dict[ChangeImpact, int] = defaultdict(int)
        for change in changes:
            distribution[change.impact] += 1
        return dict(distribution)

    def _calculate_domain_distribution(self, changes: List[DocumentChange]) -> Dict[ChangeDomain, int]:
        """Calculate distribution of change domains."""
        distribution: Dict[ChangeDomain, int] = defaultdict(int)
        for change in changes:
            distribution[change.domain] += 1
        return dict(distribution)

    async def _generate_clause_comparisons(
        self,
        original_text: str,
        modified_text: str,
        changes: List[DocumentChange],
    ) -> List[ClauseComparison]:
        """Generate clause-level comparisons."""
        # Group changes by clause location
        clause_changes: Dict[str, List[DocumentChange]] = defaultdict(list)
        for change in changes:
            loc = change.original_location or change.modified_location or "Unknown"
            clause_changes[loc].append(change)

        comparisons = []
        for clause_loc, clause_change_list in clause_changes.items():
            # Determine overall status and impact
            if all(c.change_type == ChangeType.ADDITION for c in clause_change_list):
                status = "added"
            elif all(c.change_type == ChangeType.DELETION for c in clause_change_list):
                status = "removed"
            else:
                status = "modified"

            # Get highest impact
            impacts = [c.impact for c in clause_change_list]
            impact_order = [ChangeImpact.CRITICAL, ChangeImpact.SIGNIFICANT, ChangeImpact.MODERATE, ChangeImpact.MINOR, ChangeImpact.COSMETIC]
            overall_impact = min(impacts, key=lambda x: impact_order.index(x) if x in impact_order else 5)

            comparisons.append(ClauseComparison(
                clause_type=clause_loc,
                original_text=clause_change_list[0].original_text,
                modified_text=clause_change_list[0].modified_text,
                status=status,
                changes=clause_change_list,
                overall_impact=overall_impact,
                summary=f"{len(clause_change_list)} change(s) in {clause_loc}",
            ))

        return comparisons

    def _generate_markdown_redline(self, result: ComparisonResult) -> str:
        """Generate markdown redline."""
        lines = [
            "# Contract Redline",
            "",
            f"**Original:** {result.original_document_id}",
            f"**Modified:** {result.modified_document_id}",
            f"**Similarity:** {result.similarity_metrics.text_similarity:.1%}",
            "",
            "## Executive Summary",
            result.executive_summary,
            "",
            "## Changes",
        ]

        for i, change in enumerate(result.changes, 1):
            impact_icon = {"critical": "!!!", "significant": "!!", "moderate": "!", "minor": ".", "cosmetic": ""}
            icon = impact_icon.get(change.impact.value, "")

            lines.append(f"\n### {i}. [{change.impact.value.upper()}]{icon} {change.domain.value.capitalize()}")
            lines.append(f"*{change.semantic_description}*")

            if change.original_text:
                lines.append(f"\n~~{change.original_text[:200]}...~~")
            if change.modified_text:
                lines.append(f"\n**{change.modified_text[:200]}...**")

            if change.recommendation:
                lines.append(f"\n> Recommendation: {change.recommendation}")

        return "\n".join(lines)

    def _generate_html_redline(self, result: ComparisonResult) -> str:
        """Generate HTML redline."""
        html = [
            "<html><head><style>",
            ".deleted { background-color: #ffcccc; text-decoration: line-through; }",
            ".added { background-color: #ccffcc; }",
            ".critical { border-left: 4px solid red; padding-left: 10px; }",
            ".significant { border-left: 4px solid orange; padding-left: 10px; }",
            "</style></head><body>",
            f"<h1>Contract Redline</h1>",
            f"<p>Similarity: {result.similarity_metrics.text_similarity:.1%}</p>",
            f"<h2>Summary</h2><p>{result.executive_summary}</p>",
            "<h2>Changes</h2>",
        ]

        for change in result.changes:
            css_class = change.impact.value
            html.append(f'<div class="{css_class}">')
            html.append(f'<strong>[{change.impact.value.upper()}] {change.domain.value}</strong>')
            html.append(f'<p>{change.semantic_description}</p>')
            if change.original_text:
                html.append(f'<span class="deleted">{change.original_text[:200]}</span>')
            if change.modified_text:
                html.append(f'<span class="added">{change.modified_text[:200]}</span>')
            html.append('</div>')

        html.append("</body></html>")
        return "\n".join(html)

    def _generate_text_redline(self, result: ComparisonResult) -> str:
        """Generate plain text redline."""
        lines = [
            "CONTRACT REDLINE",
            "=" * 50,
            f"Similarity: {result.similarity_metrics.text_similarity:.1%}",
            "",
            "SUMMARY:",
            result.executive_summary,
            "",
            "CHANGES:",
        ]

        for i, change in enumerate(result.changes, 1):
            lines.append(f"\n{i}. [{change.impact.value.upper()}] {change.domain.value}")
            lines.append(f"   {change.semantic_description}")
            if change.original_text:
                lines.append(f"   - OLD: {change.original_text[:100]}...")
            if change.modified_text:
                lines.append(f"   + NEW: {change.modified_text[:100]}...")

        return "\n".join(lines)
