"""
Interface for term extraction service.

Defines the contract for extracting key terms including parties,
dates, amounts, obligations, and deadlines into structured data.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PARTY = "party"
    DATE = "date"
    AMOUNT = "amount"
    DURATION = "duration"
    PERCENTAGE = "percentage"
    ADDRESS = "address"
    JURISDICTION = "jurisdiction"
    DEADLINE = "deadline"
    OBLIGATION = "obligation"
    RIGHT = "right"
    CONDITION = "condition"
    DEFINED_TERM = "defined_term"
    REFERENCE = "reference"


@dataclass
class Party:
    """An extracted party from the contract."""
    name: str
    role: str  # "buyer", "seller", "licensor", etc.
    entity_type: str  # "corporation", "individual", "llc", etc.
    jurisdiction: Optional[str] = None
    address: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None
    representative: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "role": self.role,
            "entity_type": self.entity_type,
            "jurisdiction": self.jurisdiction,
            "address": self.address,
            "contact_info": self.contact_info,
            "representative": self.representative,
        }


@dataclass
class MonetaryAmount:
    """An extracted monetary amount."""
    value: float
    currency: str
    description: str
    payment_type: Optional[str] = None  # "one-time", "recurring", "cap"
    frequency: Optional[str] = None  # "monthly", "annually", etc.
    conditions: Optional[str] = None
    section_reference: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "currency": self.currency,
            "description": self.description,
            "payment_type": self.payment_type,
            "frequency": self.frequency,
            "conditions": self.conditions,
            "section_reference": self.section_reference,
        }


@dataclass
class ContractDate:
    """An extracted date from the contract."""
    date_value: Optional[date]
    date_string: str  # Original text representation
    date_type: str  # "effective", "expiration", "deadline", etc.
    description: str
    is_calculated: bool = False  # e.g., "30 days after signing"
    calculation_basis: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "date_value": self.date_value.isoformat() if self.date_value else None,
            "date_string": self.date_string,
            "date_type": self.date_type,
            "description": self.description,
            "is_calculated": self.is_calculated,
            "calculation_basis": self.calculation_basis,
        }


@dataclass
class Deadline:
    """An extracted deadline from the contract."""
    description: str
    due_date: Optional[date]
    due_date_text: str
    responsible_party: str
    action_required: str
    penalty_for_missing: Optional[str] = None
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "description": self.description,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "due_date_text": self.due_date_text,
            "responsible_party": self.responsible_party,
            "action_required": self.action_required,
            "penalty_for_missing": self.penalty_for_missing,
            "is_recurring": self.is_recurring,
            "recurrence_pattern": self.recurrence_pattern,
        }


@dataclass
class ContractObligation:
    """An extracted obligation from the contract."""
    description: str
    obligated_party: str
    obligation_type: str
    trigger_condition: Optional[str] = None
    deadline: Optional[str] = None
    deliverable: Optional[str] = None
    standard_of_performance: Optional[str] = None
    consequences_of_breach: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "description": self.description,
            "obligated_party": self.obligated_party,
            "obligation_type": self.obligation_type,
            "trigger_condition": self.trigger_condition,
            "deadline": self.deadline,
            "deliverable": self.deliverable,
            "standard_of_performance": self.standard_of_performance,
            "consequences_of_breach": self.consequences_of_breach,
        }


@dataclass
class DefinedTerm:
    """A defined term from the contract."""
    term: str
    definition: str
    section_defined: Optional[str] = None
    usage_count: int = 0
    related_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "term": self.term,
            "definition": self.definition,
            "section_defined": self.section_defined,
            "usage_count": self.usage_count,
            "related_terms": self.related_terms,
        }


@dataclass
class TermExtractionResult:
    """Complete term extraction result."""
    document_id: str
    parties: List[Party]
    dates: List[ContractDate]
    amounts: List[MonetaryAmount]
    deadlines: List[Deadline]
    obligations: List[ContractObligation]
    defined_terms: List[DefinedTerm]
    contract_value_summary: Optional[str] = None
    timeline_summary: Optional[str] = None
    raw_entities: Dict[EntityType, List[Dict[str, Any]]] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "parties": [p.to_dict() for p in self.parties],
            "dates": [d.to_dict() for d in self.dates],
            "amounts": [a.to_dict() for a in self.amounts],
            "deadlines": [d.to_dict() for d in self.deadlines],
            "obligations": [o.to_dict() for o in self.obligations],
            "defined_terms": [t.to_dict() for t in self.defined_terms],
            "contract_value_summary": self.contract_value_summary,
            "timeline_summary": self.timeline_summary,
            "raw_entities": {k.value: v for k, v in self.raw_entities.items()},
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
        }


class TermExtractionInterface(BaseAIService):
    """
    Interface for term extraction services.

    Implementations extract structured data from contracts including
    parties, dates, amounts, obligations, and defined terms.
    """

    @abstractmethod
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
            entity_types: Specific entity types to extract (all if None)

        Returns:
            TermExtractionResult with all extracted data
        """
        pass

    @abstractmethod
    async def extract_parties(
        self,
        document_text: str,
    ) -> List[Party]:
        """
        Extract all parties from a contract.

        Args:
            document_text: Full document text

        Returns:
            List of extracted parties
        """
        pass

    @abstractmethod
    async def extract_amounts(
        self,
        document_text: str,
    ) -> List[MonetaryAmount]:
        """
        Extract all monetary amounts.

        Args:
            document_text: Full document text

        Returns:
            List of extracted amounts
        """
        pass

    @abstractmethod
    async def extract_dates(
        self,
        document_text: str,
        reference_date: Optional[date] = None,
    ) -> List[ContractDate]:
        """
        Extract all dates from a contract.

        Args:
            document_text: Full document text
            reference_date: Reference date for relative calculations

        Returns:
            List of extracted dates
        """
        pass

    @abstractmethod
    async def extract_deadlines(
        self,
        document_text: str,
    ) -> List[Deadline]:
        """
        Extract all deadlines from a contract.

        Args:
            document_text: Full document text

        Returns:
            List of extracted deadlines
        """
        pass

    @abstractmethod
    async def extract_defined_terms(
        self,
        document_text: str,
    ) -> List[DefinedTerm]:
        """
        Extract defined terms and their definitions.

        Args:
            document_text: Full document text

        Returns:
            List of defined terms
        """
        pass

    @abstractmethod
    async def generate_term_sheet(
        self,
        extraction_result: TermExtractionResult,
    ) -> str:
        """
        Generate a term sheet from extraction results.

        Args:
            extraction_result: Extraction result to summarize

        Returns:
            Formatted term sheet
        """
        pass
