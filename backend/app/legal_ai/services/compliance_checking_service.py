"""
Compliance Checking Service Implementation.

Checks documents against configurable compliance rules and frameworks.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any
from collections import defaultdict

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.compliance_checking import (
    ComplianceCheckingInterface,
    ComplianceFramework,
    ComplianceStatus,
    Severity,
    ComplianceRule,
    ComplianceViolation,
    ComplianceCheckResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

# Built-in compliance rules database
COMPLIANCE_RULES_DB: Dict[ComplianceFramework, List[Dict]] = {
    ComplianceFramework.GDPR: [
        {
            "rule_id": "GDPR-1",
            "name": "Data Subject Rights",
            "description": "Contract must specify data subject rights including access, rectification, erasure",
            "requirement_text": "Contracts involving personal data must clearly outline data subject rights",
            "severity": "critical",
            "article_reference": "Articles 15-22",
        },
        {
            "rule_id": "GDPR-2",
            "name": "Data Processing Purpose",
            "description": "Purpose of data processing must be clearly defined and limited",
            "requirement_text": "Personal data shall be collected for specified, explicit and legitimate purposes",
            "severity": "critical",
            "article_reference": "Article 5(1)(b)",
        },
        {
            "rule_id": "GDPR-3",
            "name": "Data Breach Notification",
            "description": "Contract must include data breach notification procedures",
            "requirement_text": "Processor must notify controller of breaches without undue delay",
            "severity": "major",
            "article_reference": "Article 33",
        },
        {
            "rule_id": "GDPR-4",
            "name": "Sub-processor Authorization",
            "description": "Sub-processor engagement requires prior authorization",
            "requirement_text": "Processor shall not engage another processor without prior authorization",
            "severity": "major",
            "article_reference": "Article 28(2)",
        },
        {
            "rule_id": "GDPR-5",
            "name": "International Transfers",
            "description": "Cross-border data transfers must have appropriate safeguards",
            "requirement_text": "Transfers to third countries require adequate protection measures",
            "severity": "critical",
            "article_reference": "Articles 44-49",
        },
    ],
    ComplianceFramework.HIPAA: [
        {
            "rule_id": "HIPAA-1",
            "name": "Business Associate Agreement",
            "description": "BAA must be in place for PHI sharing",
            "requirement_text": "Covered entities must have BAAs with business associates",
            "severity": "critical",
            "article_reference": "45 CFR 164.502(e)",
        },
        {
            "rule_id": "HIPAA-2",
            "name": "Minimum Necessary Standard",
            "description": "PHI access limited to minimum necessary",
            "requirement_text": "Use and disclosure limited to minimum necessary for purpose",
            "severity": "major",
            "article_reference": "45 CFR 164.502(b)",
        },
        {
            "rule_id": "HIPAA-3",
            "name": "Security Safeguards",
            "description": "Administrative, physical, and technical safeguards required",
            "requirement_text": "Must implement safeguards to protect PHI",
            "severity": "critical",
            "article_reference": "45 CFR 164.306",
        },
    ],
    ComplianceFramework.SOX: [
        {
            "rule_id": "SOX-1",
            "name": "Internal Controls",
            "description": "Contract must support internal control requirements",
            "requirement_text": "Agreements must not circumvent internal financial controls",
            "severity": "critical",
            "article_reference": "Section 404",
        },
        {
            "rule_id": "SOX-2",
            "name": "Audit Trail",
            "description": "Adequate audit trail and record retention",
            "requirement_text": "Must maintain records and audit trail for 7 years",
            "severity": "major",
            "article_reference": "Section 802",
        },
    ],
    ComplianceFramework.CCPA: [
        {
            "rule_id": "CCPA-1",
            "name": "Consumer Rights Notice",
            "description": "Must disclose consumer privacy rights",
            "requirement_text": "Service providers must enable consumer rights exercise",
            "severity": "critical",
            "article_reference": "Cal. Civ. Code 1798.100",
        },
        {
            "rule_id": "CCPA-2",
            "name": "Do Not Sell",
            "description": "Must respect opt-out of data sales",
            "requirement_text": "Must honor consumer's right to opt-out of data sales",
            "severity": "major",
            "article_reference": "Cal. Civ. Code 1798.120",
        },
    ],
}


class ComplianceCheckingService(ComplianceCheckingInterface):
    """
    Service for checking documents against compliance frameworks.

    Supports GDPR, HIPAA, SOX, CCPA, and custom rules.
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
        Initialize compliance checking service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None
        self._rules_db = COMPLIANCE_RULES_DB.copy()

        config = get_service_config("compliance_checking")
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
        logger.info(f"ComplianceCheckingService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("ComplianceCheckingService shutdown")

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

    async def check_compliance(
        self,
        document_text: str,
        frameworks: List[ComplianceFramework],
        document_id: Optional[str] = None,
        custom_rules: Optional[List[ComplianceRule]] = None,
        industry: Optional[str] = None,
    ) -> ComplianceCheckResult:
        """
        Check document compliance against specified frameworks.

        Args:
            document_text: Full document text
            frameworks: Frameworks to check against
            document_id: Optional identifier
            custom_rules: Additional custom rules
            industry: Industry context

        Returns:
            ComplianceCheckResult with findings
        """
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        self._log_request("check_compliance", frameworks=[f.value for f in frameworks])

        try:
            # Gather all rules to check
            all_rules = []
            for framework in frameworks:
                framework_rules = await self.get_framework_rules(framework)
                all_rules.extend(framework_rules)

            if custom_rules:
                all_rules.extend(custom_rules)

            # Check each rule
            violations = []
            passed = 0
            failed = 0

            for rule in all_rules:
                violation = await self.check_rule(document_text, rule)
                if violation.status != ComplianceStatus.COMPLIANT:
                    violations.append(violation)
                    failed += 1
                else:
                    passed += 1

            # Calculate scores
            total_rules = len(all_rules)
            compliance_score = passed / total_rules if total_rules > 0 else 1.0

            # Determine overall status
            critical_violations = [v for v in violations if v.rule.severity == Severity.CRITICAL]
            if critical_violations:
                overall_status = ComplianceStatus.NON_COMPLIANT
            elif violations:
                overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                overall_status = ComplianceStatus.COMPLIANT

            # Calculate framework scores
            framework_scores = self._calculate_framework_scores(violations, frameworks, all_rules)

            # Generate summary
            summary = self._generate_summary(violations, frameworks, compliance_score)

            # Extract critical findings
            critical_findings = [
                f"{v.rule.name}: {v.explanation}"
                for v in violations
                if v.rule.severity == Severity.CRITICAL
            ]

            # Generate recommendations
            recommendations = self._generate_recommendations(violations)

            processing_time = (time.time() - start_time) * 1000

            return ComplianceCheckResult(
                document_id=doc_id,
                frameworks_checked=frameworks,
                overall_status=overall_status,
                compliance_score=compliance_score,
                violations=violations,
                framework_scores=framework_scores,
                summary=summary,
                critical_findings=critical_findings,
                recommendations=recommendations,
                rules_checked=total_rules,
                rules_passed=passed,
                rules_failed=failed,
                processing_time_ms=processing_time,
                model_used=self.model,
            )

        except Exception as e:
            self._log_error("check_compliance", e)
            raise RuntimeError(f"Compliance check failed: {e}")

    async def check_rule(
        self,
        document_text: str,
        rule: ComplianceRule,
    ) -> ComplianceViolation:
        """Check document against a single rule."""
        prompt = f"""Analyze the following contract for compliance with this specific rule:

Rule ID: {rule.rule_id}
Framework: {rule.framework.value.upper()}
Rule Name: {rule.name}
Description: {rule.description}
Requirement: {rule.requirement_text}
Reference: {rule.article_reference or 'N/A'}

Contract text:
{document_text}

Determine:
1. Is the contract compliant with this rule?
2. If not, what specific text or omission violates the rule?
3. What remediation steps are needed?
4. Suggest compliant replacement language if applicable.

Respond with JSON containing: status (compliant, partially_compliant, non_compliant, requires_review), violated_text, explanation, remediation_steps, suggested_language, confidence (0-1)."""

        messages = [
            Message(role="system", content="You are a compliance expert. Analyze carefully and provide accurate assessments. Respond with valid JSON."),
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

            status_str = data.get("status", "requires_review").lower().replace("-", "_")
            try:
                status = ComplianceStatus(status_str)
            except ValueError:
                status = ComplianceStatus.REQUIRES_REVIEW

            return ComplianceViolation(
                rule=rule,
                status=status,
                violated_text=data.get("violated_text", ""),
                location=data.get("location"),
                explanation=data.get("explanation", ""),
                remediation_steps=data.get("remediation_steps", []),
                suggested_language=data.get("suggested_language"),
                confidence=float(data.get("confidence", 0.8)),
            )

        except Exception as e:
            self._log_error("check_rule", e)
            return ComplianceViolation(
                rule=rule,
                status=ComplianceStatus.REQUIRES_REVIEW,
                violated_text="",
                explanation=f"Analysis error: {e}",
                remediation_steps=[],
                confidence=0.0,
            )

    async def get_framework_rules(
        self,
        framework: ComplianceFramework,
        include_guidance: bool = True,
    ) -> List[ComplianceRule]:
        """Get all rules for a compliance framework."""
        rules_data = self._rules_db.get(framework, [])

        rules = []
        for item in rules_data:
            rules.append(ComplianceRule(
                rule_id=item["rule_id"],
                framework=framework,
                name=item["name"],
                description=item["description"],
                requirement_text=item["requirement_text"],
                severity=Severity(item.get("severity", "major")),
                article_reference=item.get("article_reference"),
                guidance=item.get("guidance") if include_guidance else None,
            ))

        return rules

    async def generate_remediation_plan(
        self,
        result: ComplianceCheckResult,
        priority_order: str = "severity",
    ) -> str:
        """Generate a remediation plan for compliance violations."""
        if not result.violations:
            return "No violations found. The document appears to be compliant with all checked frameworks."

        # Sort violations
        if priority_order == "severity":
            severity_order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.INFORMATIONAL: 3}
            sorted_violations = sorted(result.violations, key=lambda v: severity_order.get(v.rule.severity, 4))
        elif priority_order == "framework":
            sorted_violations = sorted(result.violations, key=lambda v: v.rule.framework.value)
        else:
            sorted_violations = result.violations

        plan = [
            "# Compliance Remediation Plan",
            "",
            f"## Overview",
            f"- **Frameworks Checked:** {', '.join([f.value.upper() for f in result.frameworks_checked])}",
            f"- **Compliance Score:** {result.compliance_score:.1%}",
            f"- **Total Violations:** {len(result.violations)}",
            f"- **Critical Issues:** {len([v for v in result.violations if v.rule.severity == Severity.CRITICAL])}",
            "",
            "## Remediation Tasks",
        ]

        for i, violation in enumerate(sorted_violations, 1):
            plan.extend([
                "",
                f"### {i}. [{violation.rule.severity.value.upper()}] {violation.rule.name}",
                f"**Framework:** {violation.rule.framework.value.upper()}",
                f"**Rule:** {violation.rule.rule_id}",
                "",
                f"**Issue:** {violation.explanation}",
                "",
            ])

            if violation.violated_text:
                plan.append(f"**Current Text:** \"{violation.violated_text[:200]}...\"")
                plan.append("")

            if violation.remediation_steps:
                plan.append("**Remediation Steps:**")
                for step in violation.remediation_steps:
                    plan.append(f"  - {step}")
                plan.append("")

            if violation.suggested_language:
                plan.append(f"**Suggested Language:** \"{violation.suggested_language}\"")

        plan.extend([
            "",
            "## Next Steps",
            "1. Address critical violations immediately",
            "2. Review suggested language with legal counsel",
            "3. Update contract templates to prevent future violations",
            "4. Re-run compliance check after changes",
        ])

        return "\n".join(plan)

    async def suggest_compliant_language(
        self,
        violated_text: str,
        rule: ComplianceRule,
        context: Optional[str] = None,
    ) -> str:
        """Suggest compliant replacement language."""
        prompt = f"""Rewrite the following contract language to be compliant with this rule:

Rule: {rule.name}
Framework: {rule.framework.value.upper()}
Requirement: {rule.requirement_text}

Current (non-compliant) text:
{violated_text}

{f"Context: {context}" if context else ""}

Provide compliant replacement language that:
1. Satisfies the regulatory requirement
2. Maintains the original business intent where possible
3. Uses clear, unambiguous language
4. Is legally enforceable

Respond with just the suggested replacement text."""

        messages = [
            Message(role="system", content="You are a compliance and contract drafting expert."),
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
            self._log_error("suggest_compliant_language", e)
            return ""

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

    def _calculate_framework_scores(
        self,
        violations: List[ComplianceViolation],
        frameworks: List[ComplianceFramework],
        all_rules: List[ComplianceRule],
    ) -> Dict[ComplianceFramework, float]:
        """Calculate compliance score per framework."""
        scores = {}

        for framework in frameworks:
            framework_rules = [r for r in all_rules if r.framework == framework]
            framework_violations = [v for v in violations if v.rule.framework == framework]

            if framework_rules:
                scores[framework] = 1.0 - (len(framework_violations) / len(framework_rules))
            else:
                scores[framework] = 1.0

        return scores

    def _generate_summary(
        self,
        violations: List[ComplianceViolation],
        frameworks: List[ComplianceFramework],
        score: float,
    ) -> str:
        """Generate compliance summary."""
        framework_names = ", ".join([f.value.upper() for f in frameworks])

        if not violations:
            return f"The document is fully compliant with all checked frameworks ({framework_names}). No violations were detected."

        critical_count = len([v for v in violations if v.rule.severity == Severity.CRITICAL])
        major_count = len([v for v in violations if v.rule.severity == Severity.MAJOR])

        summary = f"Compliance check against {framework_names} resulted in {len(violations)} violation(s). "

        if critical_count:
            summary += f"There are {critical_count} critical issue(s) requiring immediate attention. "
        if major_count:
            summary += f"Additionally, {major_count} major issue(s) should be addressed. "

        summary += f"Overall compliance score: {score:.1%}."

        return summary

    def _generate_recommendations(
        self,
        violations: List[ComplianceViolation],
    ) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []

        if not violations:
            recommendations.append("Continue regular compliance monitoring")
            return recommendations

        # Group by severity
        critical = [v for v in violations if v.rule.severity == Severity.CRITICAL]
        major = [v for v in violations if v.rule.severity == Severity.MAJOR]

        if critical:
            recommendations.append(f"URGENT: Address {len(critical)} critical compliance violation(s) before contract execution")

        if major:
            recommendations.append(f"Review and remediate {len(major)} major compliance issue(s)")

        # Framework-specific recommendations
        frameworks_violated = set(v.rule.framework for v in violations)
        for framework in frameworks_violated:
            if framework == ComplianceFramework.GDPR:
                recommendations.append("Ensure Data Processing Agreement provisions are complete")
            elif framework == ComplianceFramework.HIPAA:
                recommendations.append("Verify Business Associate Agreement requirements are met")
            elif framework == ComplianceFramework.SOX:
                recommendations.append("Review internal control implications")

        recommendations.append("Consider engaging compliance counsel for detailed review")

        return recommendations
