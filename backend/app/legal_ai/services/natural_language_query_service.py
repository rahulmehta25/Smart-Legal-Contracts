"""
Natural Language Query Service Implementation.

Allows users to ask questions about documents in natural language.
"""

import time
import uuid
import logging
import json
from typing import List, Optional, Dict, Any

from app.legal_ai.interfaces.base import AIProvider, ServiceHealth
from app.legal_ai.interfaces.natural_language_query import (
    NaturalLanguageQueryInterface,
    QueryType,
    AnswerConfidence,
    SourceCitation,
    QueryAnswer,
    ConversationContext,
    QuerySuggestion,
    QueryResult,
)
from app.legal_ai.providers.config import get_service_config
from app.legal_ai.providers.openai_provider import OpenAIProvider
from app.legal_ai.providers.anthropic_provider import AnthropicProvider
from app.legal_ai.providers.vertex_provider import VertexProvider
from app.legal_ai.providers.base_provider import Message

logger = logging.getLogger(__name__)

QUERY_SYSTEM_PROMPT = """You are an expert legal document analyst. Your role is to answer questions about contracts and legal documents accurately and comprehensively.

Guidelines:
1. Base your answers ONLY on the document content provided
2. Cite specific sections or text when answering
3. If information is not in the document, clearly state that
4. Explain legal terms in plain language
5. Highlight any risks or important considerations
6. Suggest relevant follow-up questions

Always provide:
- A clear, direct answer
- Supporting citations from the document
- Confidence level in your answer
- Any caveats or limitations
- Suggested follow-up questions"""

QUERY_PROMPT = """Based on the following contract document, answer this question:

QUESTION: {question}

DOCUMENT:
{document_text}

Provide your response as JSON with:
- answer: Your complete answer
- confidence: "high", "medium", "low", or "uncertain"
- confidence_score: 0.0-1.0
- citations: Array of {{text, section, relevance_score}}
- reasoning: Brief explanation of how you arrived at the answer
- follow_up_questions: Array of relevant follow-up questions
- related_clauses: Array of clause names that are relevant
- caveats: Any limitations or things to note"""


class NaturalLanguageQueryService(NaturalLanguageQueryInterface):
    """
    Service for natural language queries about documents.

    Allows users to ask questions in plain English and receive
    contextual, cited answers from contract documents.
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.ANTHROPIC,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        """
        Initialize natural language query service.

        Args:
            provider: AI provider to use
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout
        """
        super().__init__(provider, model, temperature, max_tokens, timeout)
        self._provider_client = None
        self._conversations: Dict[str, ConversationContext] = {}

        config = get_service_config("natural_language_query")
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
        logger.info(f"NaturalLanguageQueryService initialized with {self.provider.value}/{self.model}")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._provider_client:
            await self._provider_client.shutdown()
        logger.info("NaturalLanguageQueryService shutdown")

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
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not document_text.strip():
            raise ValueError("Document text cannot be empty")

        start_time = time.time()

        self._log_request("query", question_length=len(question))

        try:
            # Classify query type
            query_type = await self.classify_query(question)

            # Build conversation history if context provided
            conversation_history = ""
            if context and context.previous_queries:
                history_parts = []
                for q, a in zip(context.previous_queries[-3:], context.previous_answers[-3:]):
                    history_parts.append(f"Q: {q}\nA: {a}")
                conversation_history = "\n\n".join(history_parts)

            # Build prompt
            prompt = QUERY_PROMPT.format(
                question=question,
                document_text=document_text[:15000],  # Limit for context
            )

            if conversation_history:
                prompt = f"Previous conversation:\n{conversation_history}\n\n{prompt}"

            messages = [
                Message(role="system", content=QUERY_SYSTEM_PROMPT),
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

            # Parse citations
            citations = []
            if include_citations:
                for cit in data.get("citations", []):
                    if isinstance(cit, dict):
                        citations.append(SourceCitation(
                            text=cit.get("text", ""),
                            section=cit.get("section"),
                            page=cit.get("page"),
                            start_char=cit.get("start_char"),
                            end_char=cit.get("end_char"),
                            relevance_score=float(cit.get("relevance_score", 0.8)),
                        ))

            # Parse confidence
            conf_str = data.get("confidence", "medium").lower()
            try:
                confidence = AnswerConfidence(conf_str)
            except ValueError:
                confidence = AnswerConfidence.MEDIUM

            answer = QueryAnswer(
                query=question,
                query_type=query_type,
                answer=data.get("answer", ""),
                confidence=confidence,
                confidence_score=float(data.get("confidence_score", 0.7)),
                citations=citations,
                reasoning=data.get("reasoning"),
                follow_up_questions=data.get("follow_up_questions", []),
                related_clauses=data.get("related_clauses", []),
                caveats=data.get("caveats", []),
            )

            # Update conversation context if provided
            if context:
                context.add_turn(question, answer.answer)

            processing_time = (time.time() - start_time) * 1000

            return QueryResult(
                answer=answer,
                processing_time_ms=processing_time,
                model_used=self.model,
                tokens_used=0,  # Would need provider-specific extraction
                conversation_id=context.conversation_id if context else None,
            )

        except Exception as e:
            self._log_error("query", e)
            raise RuntimeError(f"Query failed: {e}")

    async def query_multiple_documents(
        self,
        question: str,
        documents: Dict[str, str],
        aggregate_answers: bool = True,
    ) -> List[QueryResult]:
        """Answer a question across multiple documents."""
        results = []

        for doc_id, doc_text in documents.items():
            result = await self.query(
                question=question,
                document_text=doc_text,
                document_id=doc_id,
                include_citations=True,
            )
            results.append(result)

        if aggregate_answers and len(results) > 1:
            # Create aggregated answer
            combined_answer = await self._aggregate_answers(question, results)
            results.insert(0, combined_answer)

        return results

    async def classify_query(
        self,
        question: str,
    ) -> QueryType:
        """Classify the type of query."""
        question_lower = question.lower()

        # Simple keyword-based classification
        if any(word in question_lower for word in ["what is", "when is", "who is", "how much", "what date"]):
            return QueryType.FACTUAL
        elif any(word in question_lower for word in ["what are the risks", "analyze", "evaluate", "assess"]):
            return QueryType.ANALYTICAL
        elif any(word in question_lower for word in ["compare", "different", "versus", "vs"]):
            return QueryType.COMPARATIVE
        elif any(word in question_lower for word in ["what if", "what happens if", "suppose", "hypothetically"]):
            return QueryType.HYPOTHETICAL
        elif any(word in question_lower for word in ["list", "all", "enumerate", "extract"]):
            return QueryType.EXTRACTIVE
        elif any(word in question_lower for word in ["summarize", "summary", "overview", "briefly"]):
            return QueryType.SUMMARIZATION
        else:
            return QueryType.FACTUAL

    async def suggest_queries(
        self,
        document_text: str,
        num_suggestions: int = 5,
        user_role: Optional[str] = None,
    ) -> List[QuerySuggestion]:
        """Suggest relevant queries for a document."""
        role_context = f"The user is a {user_role}." if user_role else ""

        prompt = f"""Based on this contract document, suggest {num_suggestions} important questions that someone should ask.
{role_context}

Focus on:
- Key obligations and deadlines
- Risk areas
- Financial terms
- Important conditions
- Rights and remedies

Document (excerpt):
{document_text[:5000]}

Respond with JSON array of suggestions, each with: query, category, relevance (0-1), description."""

        messages = [
            Message(role="system", content="You are a legal document analyst. Suggest insightful questions. Respond with valid JSON."),
            Message(role="user", content=prompt),
        ]

        try:
            if hasattr(self._provider_client, 'complete_json'):
                data = await self._provider_client.complete_json(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1024,
                )
            else:
                response = await self._provider_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1024,
                )
                data = self._parse_json_response(response.content)

            suggestions = []
            items = data if isinstance(data, list) else data.get("suggestions", [])

            for item in items[:num_suggestions]:
                if isinstance(item, dict):
                    suggestions.append(QuerySuggestion(
                        query=item.get("query", ""),
                        category=item.get("category", "general"),
                        relevance=float(item.get("relevance", 0.8)),
                        description=item.get("description"),
                    ))

            return suggestions

        except Exception as e:
            self._log_error("suggest_queries", e)
            return []

    async def start_conversation(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> ConversationContext:
        """Start a new conversation about a document."""
        conversation_id = str(uuid.uuid4())
        doc_id = document_id or str(uuid.uuid4())

        context = ConversationContext(
            conversation_id=conversation_id,
            document_id=doc_id,
            document_text=document_text,
            user_role=user_role,
        )

        self._conversations[conversation_id] = context
        logger.info(f"Started conversation {conversation_id} for document {doc_id}")

        return context

    async def refine_answer(
        self,
        original_answer: QueryAnswer,
        refinement_request: str,
        context: ConversationContext,
    ) -> QueryAnswer:
        """Refine a previous answer based on user feedback."""
        prompt = f"""The user asked a follow-up question about this previous Q&A:

Original Question: {original_answer.query}
Original Answer: {original_answer.answer}

User's refinement request: {refinement_request}

Document context:
{context.document_text[:10000]}

Provide a refined answer that addresses the user's request.

Respond with JSON containing: answer, confidence, citations, reasoning."""

        messages = [
            Message(role="system", content=QUERY_SYSTEM_PROMPT),
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

            # Parse citations
            citations = []
            for cit in data.get("citations", []):
                if isinstance(cit, dict):
                    citations.append(SourceCitation(
                        text=cit.get("text", ""),
                        section=cit.get("section"),
                        page=cit.get("page"),
                        start_char=cit.get("start_char"),
                        end_char=cit.get("end_char"),
                        relevance_score=float(cit.get("relevance_score", 0.8)),
                    ))

            conf_str = data.get("confidence", "medium").lower()
            try:
                confidence = AnswerConfidence(conf_str)
            except ValueError:
                confidence = AnswerConfidence.MEDIUM

            return QueryAnswer(
                query=refinement_request,
                query_type=original_answer.query_type,
                answer=data.get("answer", ""),
                confidence=confidence,
                confidence_score=float(data.get("confidence_score", 0.7)),
                citations=citations,
                reasoning=data.get("reasoning"),
                follow_up_questions=[],
                related_clauses=original_answer.related_clauses,
                caveats=data.get("caveats", []),
            )

        except Exception as e:
            self._log_error("refine_answer", e)
            raise

    async def explain_answer(
        self,
        answer: QueryAnswer,
        explanation_depth: str = "medium",
    ) -> str:
        """Provide detailed explanation for an answer."""
        depth_instructions = {
            "brief": "Provide a brief 2-3 sentence explanation.",
            "medium": "Provide a moderate explanation with key points.",
            "detailed": "Provide a comprehensive explanation covering all aspects.",
        }

        prompt = f"""Explain this answer about a legal document:

Question: {answer.query}
Answer: {answer.answer}
Confidence: {answer.confidence.value}

{depth_instructions.get(explanation_depth, depth_instructions["medium"])}

Include:
- Why this answer is correct
- What document evidence supports it
- Any important nuances
- What the user should consider"""

        messages = [
            Message(role="system", content="You are a legal educator explaining contract analysis."),
            Message(role="user", content=prompt),
        ]

        try:
            response = await self._provider_client.complete(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.content.strip()

        except Exception as e:
            self._log_error("explain_answer", e)
            return f"Unable to generate explanation: {e}"

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

    async def _aggregate_answers(
        self,
        question: str,
        results: List[QueryResult],
    ) -> QueryResult:
        """Aggregate answers from multiple documents."""
        answers_text = "\n\n".join([
            f"Document {i+1}:\n{r.answer.answer}"
            for i, r in enumerate(results)
        ])

        prompt = f"""Aggregate these answers to the question: "{question}"

{answers_text}

Provide a unified answer that:
- Synthesizes findings across all documents
- Notes any contradictions or differences
- Highlights common themes

Respond with JSON containing: answer, confidence, reasoning."""

        messages = [
            Message(role="system", content="You are a legal analyst synthesizing multiple document analyses."),
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

            answer = QueryAnswer(
                query=question,
                query_type=QueryType.ANALYTICAL,
                answer=data.get("answer", ""),
                confidence=AnswerConfidence.MEDIUM,
                confidence_score=0.7,
                citations=[],
                reasoning=data.get("reasoning"),
                follow_up_questions=[],
                related_clauses=[],
                caveats=["This is an aggregated answer from multiple documents"],
            )

            return QueryResult(
                answer=answer,
                processing_time_ms=0,
                model_used=self.model,
                tokens_used=0,
                conversation_id=None,
            )

        except Exception as e:
            self._log_error("_aggregate_answers", e)
            raise
