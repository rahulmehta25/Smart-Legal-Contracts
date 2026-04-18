"""
Interface for compliance checking service.

Defines the contract for checking documents against configurable
compliance rules including GDPR, SOX, and industry-specific regulations.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    FCPA = "fcpa"
    FINRA = "finra"
    CUSTOM = "custom"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


class Severity(str, Enum):
    """Severity of compliance violations."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFORMATIONAL = "informational"


@dataclass
class ComplianceRule:
    """A compliance rule to check against."""
    rule_id: str
    framework: ComplianceFramework
    name: str
    description: str
    requirement_text: str
    severity: Severity
    article_reference: Optional[str] = None
    guidance: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "requirement_text": self.requirement_text,
            "severity": self.severity.value,
            "article_reference": self.article_reference,
            "guidance": self.guidance,
        }


@dataclass
class ComplianceViolation:
    """An identified compliance violation."""
    rule: ComplianceRule
    status: ComplianceStatus
    violated_text: str
    location: Optional[str] = None
    explanation: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    suggested_language: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "rule": self.rule.to_dict(),
            "status": self.status.value,
            "violated_text": self.violated_text,
            "location": self.location,
            "explanation": self.explanation,
            "remediation_steps": self.remediation_steps,
            "suggested_language": self.suggested_language,
            "confidence": self.confidence,
        }


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""
    document_id: str
    frameworks_checked: List[ComplianceFramework]
    overall_status: ComplianceStatus
    compliance_score: float  # 0.0 - 1.0
    violations: List[ComplianceViolation]
    framework_scores: Dict[ComplianceFramework, float]
    summary: str
    critical_findings: List[str]
    recommendations: List[str]
    rules_checked: int
    rules_passed: int
    rules_failed: int
    processing_time_ms: float
    model_used: str

    def get_violations_by_framework(
        self, framework: ComplianceFramework
    ) -> List[ComplianceViolation]:
        """Get violations for a specific framework."""
        return [v for v in self.violations if v.rule.framework == framework]

    def get_critical_violations(self) -> List[ComplianceViolation]:
        """Get all critical severity violations."""
        return [v for v in self.violations if v.rule.severity == Severity.CRITICAL]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "frameworks_checked": [f.value for f in self.frameworks_checked],
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "violations": [v.to_dict() for v in self.violations],
            "framework_scores": {k.value: v for k, v in self.framework_scores.items()},
            "summary": self.summary,
            "critical_findings": self.critical_findings,
            "recommendations": self.recommendations,
            "rules_checked": self.rules_checked,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class ComplianceCheckingInterface(BaseAIService):
    """
    Interface for compliance checking services.

    Implementations check documents against configurable compliance
    rules and frameworks, identifying violations and recommending remediation.
    """

    @abstractmethod
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
            frameworks: Compliance frameworks to check
            document_id: Optional identifier
            custom_rules: Additional custom rules to check
            industry: Industry context for relevant rules

        Returns:
            ComplianceCheckResult with findings
        """
        pass

    @abstractmethod
    async def check_rule(
        self,
        document_text: str,
        rule: ComplianceRule,
    ) -> ComplianceViolation:
        """
        Check document against a single rule.

        Args:
            document_text: Full document text
            rule: Rule to check

        Returns:
            ComplianceViolation result
        """
        pass

    @abstractmethod
    async def get_framework_rules(
        self,
        framework: ComplianceFramework,
        include_guidance: bool = True,
    ) -> List[ComplianceRule]:
        """
        Get all rules for a compliance framework.

        Args:
            framework: Framework to get rules for
            include_guidance: Include implementation guidance

        Returns:
            List of compliance rules
        """
        pass

    @abstractmethod
    async def generate_remediation_plan(
        self,
        result: ComplianceCheckResult,
        priority_order: str = "severity",
    ) -> str:
        """
        Generate a remediation plan for compliance violations.

        Args:
            result: Compliance check result
            priority_order: "severity", "framework", or "effort"

        Returns:
            Formatted remediation plan
        """
        pass

    @abstractmethod
    async def suggest_compliant_language(
        self,
        violated_text: str,
        rule: ComplianceRule,
        context: Optional[str] = None,
    ) -> str:
        """
        Suggest compliant replacement language.

        Args:
            violated_text: Current non-compliant text
            rule: Rule being violated
            context: Surrounding document context

        Returns:
            Suggested compliant language
        """
        pass
