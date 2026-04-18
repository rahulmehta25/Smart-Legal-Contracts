"""
Interface for contract summarization service.

Defines the contract for generating executive summaries of
contracts highlighting key terms, obligations, and deadlines.
"""

from abc import abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class ObligationType(str, Enum):
    """Types of contractual obligations."""
    PAYMENT = "payment"
    DELIVERY = "delivery"
    PERFORMANCE = "performance"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    NOTIFICATION = "notification"
    MAINTENANCE = "maintenance"
    CONFIDENTIALITY = "confidentiality"
    OTHER = "other"


@dataclass
class KeyTerm:
    """A key term extracted from a contract."""
    term_name: str
    description: str
    value: Optional[str] = None
    section_reference: Optional[str] = None
    importance: str = "medium"  # "high", "medium", "low"

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "term_name": self.term_name,
            "description": self.description,
            "value": self.value,
            "section_reference": self.section_reference,
            "importance": self.importance,
        }


@dataclass
class Obligation:
    """A contractual obligation."""
    description: str
    obligated_party: str
    obligation_type: ObligationType
    deadline: Optional[date] = None
    recurring: bool = False
    recurrence_pattern: Optional[str] = None
    penalty_for_breach: Optional[str] = None
    section_reference: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "description": self.description,
            "obligated_party": self.obligated_party,
            "obligation_type": self.obligation_type.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "recurring": self.recurring,
            "recurrence_pattern": self.recurrence_pattern,
            "penalty_for_breach": self.penalty_for_breach,
            "section_reference": self.section_reference,
        }


@dataclass
class ImportantDate:
    """An important date in the contract."""
    date_value: date
    description: str
    date_type: str  # "effective", "expiration", "renewal", "deadline", etc.
    action_required: Optional[str] = None
    days_until: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "date_value": self.date_value.isoformat(),
            "description": self.description,
            "date_type": self.date_type,
            "action_required": self.action_required,
            "days_until": self.days_until,
        }


@dataclass
class ContractSummary:
    """Executive summary of a contract."""
    document_id: str
    title: str
    executive_summary: str
    contract_type: str
    parties: List[str]
    effective_date: Optional[date]
    expiration_date: Optional[date]
    total_value: Optional[str]
    key_terms: List[KeyTerm]
    obligations: List[Obligation]
    important_dates: List[ImportantDate]
    renewal_terms: Optional[str]
    termination_conditions: List[str]
    governing_law: Optional[str]
    dispute_resolution: Optional[str]
    key_risks: List[str]
    action_items: List[str]
    processing_time_ms: float
    model_used: str

    def get_obligations_by_party(self, party: str) -> List[Obligation]:
        """Get all obligations for a specific party."""
        return [o for o in self.obligations if o.obligated_party.lower() == party.lower()]

    def get_upcoming_deadlines(self, days: int = 30) -> List[ImportantDate]:
        """Get deadlines within the specified number of days."""
        return [d for d in self.important_dates if d.days_until is not None and d.days_until <= days]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "contract_type": self.contract_type,
            "parties": self.parties,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "total_value": self.total_value,
            "key_terms": [t.to_dict() for t in self.key_terms],
            "obligations": [o.to_dict() for o in self.obligations],
            "important_dates": [d.to_dict() for d in self.important_dates],
            "renewal_terms": self.renewal_terms,
            "termination_conditions": self.termination_conditions,
            "governing_law": self.governing_law,
            "dispute_resolution": self.dispute_resolution,
            "key_risks": self.key_risks,
            "action_items": self.action_items,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class SummarizationInterface(BaseAIService):
    """
    Interface for contract summarization services.

    Implementations generate executive summaries of contracts
    with structured extraction of key terms, obligations, and dates.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    async def extract_key_terms(
        self,
        document_text: str,
    ) -> List[KeyTerm]:
        """
        Extract key terms from a contract.

        Args:
            document_text: Full contract text

        Returns:
            List of extracted key terms
        """
        pass

    @abstractmethod
    async def extract_obligations(
        self,
        document_text: str,
        party_filter: Optional[str] = None,
    ) -> List[Obligation]:
        """
        Extract contractual obligations.

        Args:
            document_text: Full contract text
            party_filter: Optional filter for specific party

        Returns:
            List of extracted obligations
        """
        pass

    @abstractmethod
    async def extract_dates(
        self,
        document_text: str,
        reference_date: Optional[date] = None,
    ) -> List[ImportantDate]:
        """
        Extract important dates from a contract.

        Args:
            document_text: Full contract text
            reference_date: Reference date for calculating days_until

        Returns:
            List of important dates
        """
        pass

    @abstractmethod
    async def generate_action_items(
        self,
        summary: ContractSummary,
        role: str = "legal_counsel",
    ) -> List[str]:
        """
        Generate action items based on contract summary.

        Args:
            summary: Contract summary to generate actions from
            role: Role perspective for action items

        Returns:
            List of recommended action items
        """
        pass
