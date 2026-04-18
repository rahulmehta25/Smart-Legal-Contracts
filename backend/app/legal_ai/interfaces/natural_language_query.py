"""
Interface for natural language query service.

Defines the contract for allowing users to ask questions about
uploaded documents in natural language.
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from app.legal_ai.interfaces.base import BaseAIService


class QueryType(str, Enum):
    """Types of queries that can be answered."""
    FACTUAL = "factual"  # What is the effective date?
    ANALYTICAL = "analytical"  # What are the main risks?
    COMPARATIVE = "comparative"  # How does this differ from standard?
    HYPOTHETICAL = "hypothetical"  # What happens if we breach?
    EXTRACTIVE = "extractive"  # List all payment obligations
    SUMMARIZATION = "summarization"  # Summarize the indemnity clause


class AnswerConfidence(str, Enum):
    """Confidence level of the answer."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class SourceCitation:
    """A citation to source document text."""
    text: str
    section: Optional[str]
    page: Optional[int]
    start_char: Optional[int]
    end_char: Optional[int]
    relevance_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "section": self.section,
            "page": self.page,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "relevance_score": self.relevance_score,
        }


@dataclass
class QueryAnswer:
    """Answer to a natural language query."""
    query: str
    query_type: QueryType
    answer: str
    confidence: AnswerConfidence
    confidence_score: float
    citations: List[SourceCitation]
    reasoning: Optional[str] = None
    follow_up_questions: List[str] = field(default_factory=list)
    related_clauses: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "citations": [c.to_dict() for c in self.citations],
            "reasoning": self.reasoning,
            "follow_up_questions": self.follow_up_questions,
            "related_clauses": self.related_clauses,
            "caveats": self.caveats,
        }


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""
    conversation_id: str
    document_id: str
    document_text: str
    previous_queries: List[str] = field(default_factory=list)
    previous_answers: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    user_role: Optional[str] = None

    def add_turn(self, query: str, answer: str) -> None:
        """Add a conversation turn."""
        self.previous_queries.append(query)
        self.previous_answers.append(answer)


@dataclass
class QuerySuggestion:
    """A suggested query for the user."""
    query: str
    category: str
    relevance: float
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "category": self.category,
            "relevance": self.relevance,
            "description": self.description,
        }


@dataclass
class QueryResult:
    """Complete result of a query operation."""
    answer: QueryAnswer
    processing_time_ms: float
    model_used: str
    tokens_used: int
    conversation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "answer": self.answer.to_dict(),
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "conversation_id": self.conversation_id,
        }


class NaturalLanguageQueryInterface(BaseAIService):
    """
    Interface for natural language query services.

    Implementations allow users to ask questions about documents
    in natural language and receive contextual answers.
    """

    @abstractmethod
    async def query(
        self,
        question: str,
        document_text: str,
        document_id: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        include_citations: bool = True,
    ) -> QueryResult:
        """
        Answer a natural language question about a document.

        Args:
            question: User's question
            document_text: Document to query
            document_id: Optional identifier
            context: Conversation context for follow-ups
            include_citations: Include source citations

        Returns:
            QueryResult with answer
        """
        pass

    @abstractmethod
    async def query_multiple_documents(
        self,
        question: str,
        documents: Dict[str, str],
        aggregate_answers: bool = True,
    ) -> List[QueryResult]:
        """
        Answer a question across multiple documents.

        Args:
            question: User's question
            documents: Dict of document_id -> document_text
            aggregate_answers: Combine into single answer if True

        Returns:
            List of QueryResults
        """
        pass

    @abstractmethod
    async def classify_query(
        self,
        question: str,
    ) -> QueryType:
        """
        Classify the type of query.

        Args:
            question: User's question

        Returns:
            QueryType classification
        """
        pass

    @abstractmethod
    async def suggest_queries(
        self,
        document_text: str,
        num_suggestions: int = 5,
        user_role: Optional[str] = None,
    ) -> List[QuerySuggestion]:
        """
        Suggest relevant queries for a document.

        Args:
            document_text: Document to analyze
            num_suggestions: Number of suggestions
            user_role: User's role for tailored suggestions

        Returns:
            List of suggested queries
        """
        pass

    @abstractmethod
    async def start_conversation(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> ConversationContext:
        """
        Start a new conversation about a document.

        Args:
            document_text: Document to discuss
            document_id: Optional identifier
            user_role: User's role for context

        Returns:
            ConversationContext for the session
        """
        pass

    @abstractmethod
    async def refine_answer(
        self,
        original_answer: QueryAnswer,
        refinement_request: str,
        context: ConversationContext,
    ) -> QueryAnswer:
        """
        Refine a previous answer based on user feedback.

        Args:
            original_answer: Previous answer
            refinement_request: How to refine
            context: Conversation context

        Returns:
            Refined answer
        """
        pass

    @abstractmethod
    async def explain_answer(
        self,
        answer: QueryAnswer,
        explanation_depth: str = "medium",
    ) -> str:
        """
        Provide detailed explanation for an answer.

        Args:
            answer: Answer to explain
            explanation_depth: "brief", "medium", or "detailed"

        Returns:
            Explanation text
        """
        pass
