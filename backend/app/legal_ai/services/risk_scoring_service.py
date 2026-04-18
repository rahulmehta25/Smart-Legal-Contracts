"""
Risk Scoring Service Implementation.

Assigns risk scores to contracts and identifies unfavorable terms.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any
from collections import defaultdict

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.risk_scoring import (
    RiskScoringInterface,
    RiskLevel,
    RiskCategory,
    RiskFactor,
    UnfavorableTerm,
    RiskScoreResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

RISK_SCORING_PROMPT = """You are an expert legal risk analyst specializing in contract analysis.

Analyze the following contract from the perspective of {party_perspective} and identify all risk factors.

For each risk factor, provide:
1. Name of the risk
2. Description
3. Category: legal, financial, operational, compliance, reputational, or strategic
4. Severity: critical, high, medium, low, or minimal
5. Score (0.0 to 1.0, where 1.0 is highest risk)
6. The specific clause text creating this risk
7. Location in document (section name if available)
8. Mitigation suggestions
9. Any relevant precedent information

Also identify unfavorable terms with:
1. The term text
2. Why it's unfavorable
3. Risk level
4. Which party it affects
5. Recommended alternative language
6. Negotiation priority (1-10, 1 being highest)
7. How it deviates from market standard

Provide an executive summary and overall recommendations.

Industry context: {industry}
Contract type: {contract_type}

Contract text:
{document_text}

Respond with a JSON object containing: risk_factors, unfavorable_terms, overall_score, executive_summary, recommendations"""


class RiskScoringService(RiskScoringInterface):
    """
    Service for comprehensive contract risk scoring.

    Analyzes contracts to identify risks, score them by severity,
    and provide mitigation recommendations.
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.ANTHROPIC,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """
        Initialize risk scoring service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None

        config = get_service_config("risk_scoring")
        if model is None:
            self.model = config.model
            self.provider = AIProvider(config.provider.value)

    async def initialize(self) -> None:
        """Initialize the AI provider client."""
        if self.provider == AIProvider.ANTHROPIC:
            self._provider_client = AnthropicProvider(timeout=self.timeout)
        elif self.provider == AIProvider.OPENAI:
            self._provider_client = OpenAIProvider(timeout=self.timeout)
        elif self.provider == AIProvider.VERTEX:
            self._provider_client = VertexProvider(timeout=self.timeout)
        else:
            self._provider_client = AnthropicProvider(timeout=self.timeout)

        await self._provider_client.initialize()
        logger.info(f"RiskScoringService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("RiskScoringService shutdown")

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

    async def score_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        party_perspective: str = "client",
        industry: Optional[str] = None,
        contract_type: Optional[str] = None,
    ) -> RiskScoreResult:
        """
        Calculate comprehensive risk scores for a document.

        Args:
            document_text: Full contract text
            document_id: Optional identifier
            party_perspective: Perspective for scoring
            industry: Industry context
            contract_type: Type of contract

        Returns:
            RiskScoreResult with full analysis
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("score_document", document_length=len(document_text))

        try:
            prompt = RISK_SCORING_PROMPT.format(
                party_perspective=party_perspective,
                industry=industry or "general",
                contract_type=contract_type or "standard contract",
                document_text=document_text,
            )

            messages = [
                Message(
                    role="system",
                    content="You are an expert legal risk analyst. Provide thorough, actionable risk analysis. Respond with valid JSON."
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

            # Parse risk factors
            risk_factors = self._parse_risk_factors(data.get("risk_factors", []))

            # Parse unfavorable terms
            unfavorable_terms = self._parse_unfavorable_terms(data.get("unfavorable_terms", []))

            # Calculate scores
            overall_score = float(data.get("overall_score", self._calculate_overall_score(risk_factors)))
            category_scores = self._calculate_category_scores(risk_factors)
            risk_distribution = self._calculate_risk_distribution(risk_factors)

            # Determine overall level
            overall_level = self._score_to_level(overall_score)

            processing_time = (time.time() - start_time) * 1000

            return RiskScoreResult(
                document_id=doc_id,
                overall_score=overall_score,
                overall_level=overall_level,
                category_scores=category_scores,
                risk_factors=risk_factors,
                unfavorable_terms=unfavorable_terms,
                executive_summary=data.get("executive_summary", ""),
                risk_distribution=risk_distribution,
                recommendations=data.get("recommendations", []),
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("score_document", e)
            raise RuntimeError(f"Risk scoring failed: {e}")

    async def score_clause(
        self,
        clause_text: str,
        clause_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> RiskFactor:
        """Score risk for a single clause."""
        prompt = f"""Analyze the following contract clause for risk:

Clause type: {clause_type or 'unknown'}
Context: {context[:500] if context else 'None provided'}

Clause:
{clause_text}

Provide a risk assessment with: name, description, category, severity, score, mitigation_suggestions, and precedent_info.
Respond with JSON."""

        messages = [
            Message(role="system", content="You are a legal risk expert. Respond with valid JSON."),
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

            return RiskFactor(
                name=data.get("name", "Unspecified Risk"),
                description=data.get("description", ""),
                category=self._parse_category(data.get("category", "legal")),
                severity=self._parse_severity(data.get("severity", "medium")),
                score=float(data.get("score", 0.5)),
                clause_text=clause_text,
                clause_location=data.get("location"),
                mitigation_suggestions=data.get("mitigation_suggestions", []),
                precedent_info=data.get("precedent_info"),
            )
        except Exception as e:
            self._log_error("score_clause", e)
            return RiskFactor(
                name="Analysis Error",
                description=str(e),
                category=RiskCategory.LEGAL,
                severity=RiskLevel.MEDIUM,
                score=0.5,
                clause_text=clause_text,
            )

    async def identify_unfavorable_terms(
        self,
        document_text: str,
        party_perspective: str = "client",
    ) -> List[UnfavorableTerm]:
        """Identify all unfavorable terms in a document."""
        prompt = f"""From the perspective of {party_perspective}, identify all unfavorable terms in this contract.

For each unfavorable term, provide:
1. term_text: The exact language
2. issue_description: Why it's unfavorable
3. risk_level: critical, high, medium, or low
4. affected_party: client, counterparty, or both
5. recommended_alternative: Suggested replacement language
6. negotiation_priority: 1-10 (1 is highest priority)
7. market_standard_deviation: How it differs from standard

Contract:
{document_text}

Respond with a JSON array of unfavorable terms."""

        messages = [
            Message(role="system", content="You are a contract negotiation expert. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
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

            return self._parse_unfavorable_terms(data if isinstance(data, list) else data.get("unfavorable_terms", []))

        except Exception as e:
            self._log_error("identify_unfavorable_terms", e)
            return []

    async def generate_risk_report(
        self,
        risk_result: RiskScoreResult,
        include_mitigation: bool = True,
    ) -> str:
        """Generate a human-readable risk report."""
        report = [
            "# Contract Risk Assessment Report",
            "",
            "## Executive Summary",
            risk_result.executive_summary or "No summary available.",
            "",
            f"**Overall Risk Score:** {risk_result.overall_score:.2f}/1.0 ({risk_result.overall_level.value.upper()})",
            "",
            "## Risk Distribution",
        ]

        for level, count in risk_result.risk_distribution.items():
            report.append(f"- {level.capitalize()}: {count} risks")

        report.extend([
            "",
            "## Category Breakdown",
        ])

        for category, score in risk_result.category_scores.items():
            report.append(f"- {category.value.capitalize()}: {score:.2f}")

        if risk_result.risk_factors:
            report.extend([
                "",
                "## Critical and High Risk Factors",
            ])

            for factor in risk_result.risk_factors:
                if factor.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                    report.extend([
                        f"### {factor.name}",
                        f"**Severity:** {factor.severity.value}",
                        f"**Category:** {factor.category.value}",
                        f"**Description:** {factor.description}",
                        "",
                        f"*Clause:* {factor.clause_text[:200]}...",
                    ])
                    if include_mitigation and factor.mitigation_suggestions:
                        report.append("**Mitigation:**")
                        for m in factor.mitigation_suggestions:
                            report.append(f"  - {m}")
                    report.append("")

        if risk_result.unfavorable_terms:
            report.extend([
                "",
                "## Unfavorable Terms (Priority Order)",
            ])

            sorted_terms = sorted(risk_result.unfavorable_terms, key=lambda x: x.negotiation_priority)
            for term in sorted_terms[:10]:
                report.extend([
                    f"### Priority {term.negotiation_priority}: {term.risk_level.value.upper()}",
                    f"*Issue:* {term.issue_description}",
                    f"*Current:* \"{term.term_text[:150]}...\"",
                ])
                if include_mitigation and term.recommended_alternative:
                    report.append(f"*Suggested:* \"{term.recommended_alternative}\"")
                report.append("")

        if risk_result.recommendations:
            report.extend([
                "",
                "## Recommendations",
            ])
            for i, rec in enumerate(risk_result.recommendations, 1):
                report.append(f"{i}. {rec}")

        return "\n".join(report)

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

    def _parse_category(self, cat_str: str) -> RiskCategory:
        """Parse risk category from string."""
        try:
            return RiskCategory(cat_str.lower().strip())
        except ValueError:
            return RiskCategory.LEGAL

    def _parse_severity(self, sev_str: str) -> RiskLevel:
        """Parse severity from string."""
        try:
            return RiskLevel(sev_str.lower().strip())
        except ValueError:
            return RiskLevel.MEDIUM

    def _parse_risk_factors(self, factors_data: List[Dict]) -> List[RiskFactor]:
        """Parse risk factors from response data."""
        factors = []
        for item in factors_data:
            if not isinstance(item, dict):
                continue
            factors.append(RiskFactor(
                name=item.get("name", "Unspecified"),
                description=item.get("description", ""),
                category=self._parse_category(item.get("category", "legal")),
                severity=self._parse_severity(item.get("severity", "medium")),
                score=float(item.get("score", 0.5)),
                clause_text=item.get("clause_text", item.get("clause", "")),
                clause_location=item.get("clause_location", item.get("location")),
                mitigation_suggestions=item.get("mitigation_suggestions", item.get("mitigations", [])),
                precedent_info=item.get("precedent_info"),
            ))
        return factors

    def _parse_unfavorable_terms(self, terms_data: List[Dict]) -> List[UnfavorableTerm]:
        """Parse unfavorable terms from response data."""
        terms = []
        for item in terms_data:
            if not isinstance(item, dict):
                continue
            terms.append(UnfavorableTerm(
                term_text=item.get("term_text", item.get("text", "")),
                issue_description=item.get("issue_description", item.get("issue", "")),
                risk_level=self._parse_severity(item.get("risk_level", "medium")),
                affected_party=item.get("affected_party", "client"),
                recommended_alternative=item.get("recommended_alternative", item.get("alternative")),
                negotiation_priority=int(item.get("negotiation_priority", item.get("priority", 5))),
                market_standard_deviation=item.get("market_standard_deviation"),
            ))
        return terms

    def _calculate_overall_score(self, factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from factors."""
        if not factors:
            return 0.0

        # Weighted by severity
        weights = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.LOW: 0.2,
            RiskLevel.MINIMAL: 0.1,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for factor in factors:
            weight = weights.get(factor.severity, 0.5)
            weighted_sum += factor.score * weight
            total_weight += weight

        return min(1.0, weighted_sum / max(1.0, total_weight))

    def _calculate_category_scores(self, factors: List[RiskFactor]) -> Dict[RiskCategory, float]:
        """Calculate risk score by category."""
        category_scores: Dict[RiskCategory, List[float]] = defaultdict(list)

        for factor in factors:
            category_scores[factor.category].append(factor.score)

        return {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }

    def _calculate_risk_distribution(self, factors: List[RiskFactor]) -> Dict[str, int]:
        """Calculate distribution of risk severities."""
        distribution: Dict[str, int] = defaultdict(int)
        for factor in factors:
            distribution[factor.severity.value] += 1
        return dict(distribution)

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
