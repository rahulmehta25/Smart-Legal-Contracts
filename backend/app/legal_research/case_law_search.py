"""
Case Law Search Engine with AI-Powered Similarity Matching

This module provides advanced case law search capabilities using vector embeddings,
semantic similarity, and legal concept understanding.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

from ..db.vector_store import VectorStore
from ..ml.embeddings import EmbeddingService
from ..nlp.language_models import LegalLanguageModel


class CourtLevel(Enum):
    """Court hierarchy levels for precedent weighting."""
    SUPREME_COURT = "supreme_court"
    APPELLATE = "appellate" 
    TRIAL = "trial"
    ADMINISTRATIVE = "administrative"


class Jurisdiction(Enum):
    """Legal jurisdictions."""
    FEDERAL = "federal"
    STATE = "state"
    INTERNATIONAL = "international"
    ADMINISTRATIVE = "administrative"


@dataclass
class CaseCitation:
    """Structured case citation information."""
    case_name: str
    citation: str
    year: int
    court: str
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    parallel_citations: List[str] = None
    
    def __post_init__(self):
        if self.parallel_citations is None:
            self.parallel_citations = []


@dataclass
class CaseMetadata:
    """Comprehensive case metadata."""
    case_id: str
    citation: CaseCitation
    court_level: CourtLevel
    jurisdiction: Jurisdiction
    judges: List[str]
    parties: Dict[str, str]  # plaintiff, defendant, etc.
    legal_areas: List[str]
    procedural_posture: str
    disposition: str
    precedential_value: float
    overruled: bool = False
    overruling_cases: List[str] = None
    
    def __post_init__(self):
        if self.overruling_cases is None:
            self.overruling_cases = []


@dataclass
class CaseContent:
    """Full case content and analysis."""
    metadata: CaseMetadata
    full_text: str
    summary: str
    headnotes: List[str]
    key_holdings: List[str]
    legal_principles: List[str]
    cited_cases: List[str]
    statutes_cited: List[str]
    regulations_cited: List[str]
    opinion_type: str  # majority, dissenting, concurring
    word_count: int
    
    def __post_init__(self):
        if not self.headnotes:
            self.headnotes = []
        if not self.key_holdings:
            self.key_holdings = []
        if not self.legal_principles:
            self.legal_principles = []


@dataclass
class SearchQuery:
    """Structured search query with parameters."""
    query_text: str
    jurisdiction: Optional[Jurisdiction] = None
    court_level: Optional[CourtLevel] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    legal_areas: List[str] = None
    cited_cases: List[str] = None
    cited_statutes: List[str] = None
    include_overruled: bool = False
    precedential_weight: float = 0.5
    similarity_threshold: float = 0.6
    max_results: int = 50
    
    def __post_init__(self):
        if self.legal_areas is None:
            self.legal_areas = []
        if self.cited_cases is None:
            self.cited_cases = []
        if self.cited_statutes is None:
            self.cited_statutes = []


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    case: CaseContent
    relevance_score: float
    similarity_score: float
    precedential_score: float
    matching_segments: List[Dict[str, Any]]
    explanation: str
    citation_network_score: float = 0.0


class LegalConceptExtractor:
    """Extract legal concepts and entities from case text."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.legal_terms_patterns = self._load_legal_patterns()
        
    def _load_legal_patterns(self) -> List[Dict[str, Any]]:
        """Load legal term recognition patterns."""
        return [
            {"pattern": r"\b(?:negligence|negligent|duty of care)\b", "concept": "tort_law"},
            {"pattern": r"\b(?:breach of contract|contractual obligation)\b", "concept": "contract_law"},
            {"pattern": r"\b(?:due process|constitutional right)\b", "concept": "constitutional_law"},
            {"pattern": r"\b(?:summary judgment|motion to dismiss)\b", "concept": "civil_procedure"},
            {"pattern": r"\b(?:hearsay|evidence|admissible)\b", "concept": "evidence_law"},
            {"pattern": r"\b(?:habeas corpus|criminal procedure)\b", "concept": "criminal_law"},
            {"pattern": r"\b(?:intellectual property|patent|trademark)\b", "concept": "ip_law"},
        ]
    
    def extract_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract legal concepts from text."""
        doc = self.nlp(text)
        concepts = defaultdict(list)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                concepts["entities"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Extract legal concepts using patterns
        for pattern_info in self.legal_terms_patterns:
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
            for match in matches:
                concepts[pattern_info["concept"]].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return dict(concepts)


class CaseLawSearchEngine:
    """Advanced case law search engine with AI-powered similarity matching."""
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.legal_model = LegalLanguageModel()
        self.concept_extractor = LegalConceptExtractor()
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.legal_bert = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        
        # Case database (in production, this would be a proper database)
        self.case_database: Dict[str, CaseContent] = {}
        self.citation_network: Dict[str, List[str]] = {}
        self.concept_index: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def index_case(self, case: CaseContent) -> None:
        """Index a case for search with embeddings and metadata."""
        try:
            # Generate embeddings for different parts of the case
            embeddings = {}
            
            # Full text embedding
            embeddings['full_text'] = await self.embedding_service.get_embedding(
                case.full_text, model_type='legal'
            )
            
            # Summary embedding
            embeddings['summary'] = await self.embedding_service.get_embedding(
                case.summary, model_type='legal'
            )
            
            # Key holdings embeddings
            embeddings['holdings'] = []
            for holding in case.key_holdings:
                holding_emb = await self.embedding_service.get_embedding(
                    holding, model_type='legal'
                )
                embeddings['holdings'].append(holding_emb)
            
            # Store in vector database
            await self.vector_store.store_embeddings(
                case.metadata.case_id,
                embeddings,
                metadata=asdict(case.metadata)
            )
            
            # Update in-memory indexes
            self.case_database[case.metadata.case_id] = case
            self._update_citation_network(case)
            self._update_concept_index(case)
            
            self.logger.info(f"Successfully indexed case: {case.metadata.citation.case_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to index case {case.metadata.case_id}: {str(e)}")
            raise
    
    def _update_citation_network(self, case: CaseContent) -> None:
        """Update citation network for precedent analysis."""
        case_id = case.metadata.case_id
        self.citation_network[case_id] = case.cited_cases
        
        # Add reverse citations
        for cited_case in case.cited_cases:
            if cited_case not in self.citation_network:
                self.citation_network[cited_case] = []
    
    def _update_concept_index(self, case: CaseContent) -> None:
        """Update concept-based index."""
        concepts = self.concept_extractor.extract_concepts(case.full_text)
        
        for concept_type, concept_list in concepts.items():
            if concept_type not in self.concept_index:
                self.concept_index[concept_type] = []
            self.concept_index[concept_type].append(case.metadata.case_id)
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform advanced case law search."""
        try:
            self.logger.info(f"Starting case law search: {query.query_text[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(
                query.query_text, model_type='legal'
            )
            
            # Vector similarity search
            similar_cases = await self._vector_similarity_search(
                query_embedding, query
            )
            
            # Concept-based filtering
            concept_filtered = await self._concept_based_filtering(
                similar_cases, query
            )
            
            # Citation network analysis
            citation_enhanced = await self._citation_network_analysis(
                concept_filtered, query
            )
            
            # Final ranking and scoring
            ranked_results = await self._rank_and_score_results(
                citation_enhanced, query
            )
            
            self.logger.info(f"Search completed. Found {len(ranked_results)} results")
            return ranked_results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise
    
    async def _vector_similarity_search(
        self, 
        query_embedding: np.ndarray, 
        query: SearchQuery
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        # Search in vector store
        results = await self.vector_store.similarity_search(
            query_embedding,
            k=query.max_results * 2,  # Get more for filtering
            threshold=query.similarity_threshold
        )
        
        return results
    
    async def _concept_based_filtering(
        self, 
        candidates: List[Tuple[str, float]], 
        query: SearchQuery
    ) -> List[Tuple[str, float]]:
        """Apply concept-based filtering."""
        if not query.legal_areas:
            return candidates
        
        filtered = []
        for case_id, score in candidates:
            case = self.case_database.get(case_id)
            if case and any(area in case.metadata.legal_areas for area in query.legal_areas):
                filtered.append((case_id, score))
        
        return filtered
    
    async def _citation_network_analysis(
        self, 
        candidates: List[Tuple[str, float]], 
        query: SearchQuery
    ) -> List[Tuple[str, float, float]]:
        """Analyze citation networks for precedential value."""
        results = []
        
        for case_id, similarity_score in candidates:
            citation_score = self._calculate_citation_network_score(case_id)
            results.append((case_id, similarity_score, citation_score))
        
        return results
    
    def _calculate_citation_network_score(self, case_id: str) -> float:
        """Calculate citation network importance score."""
        # Count incoming citations (how often this case is cited)
        incoming_citations = sum(
            1 for cited_cases in self.citation_network.values()
            if case_id in cited_cases
        )
        
        # Count outgoing citations (how many cases this case cites)
        outgoing_citations = len(self.citation_network.get(case_id, []))
        
        # Simple PageRank-like score
        return (incoming_citations * 2 + outgoing_citations) / 10.0
    
    async def _rank_and_score_results(
        self, 
        candidates: List[Tuple[str, float, float]], 
        query: SearchQuery
    ) -> List[SearchResult]:
        """Rank and score final results."""
        results = []
        
        for case_id, similarity_score, citation_score in candidates:
            case = self.case_database.get(case_id)
            if not case:
                continue
            
            # Apply jurisdiction and court level filters
            if not self._passes_filters(case, query):
                continue
            
            # Calculate precedential score
            precedential_score = self._calculate_precedential_score(case, query)
            
            # Calculate final relevance score
            relevance_score = (
                similarity_score * 0.4 +
                citation_score * 0.3 +
                precedential_score * 0.3
            )
            
            # Find matching segments
            matching_segments = await self._find_matching_segments(
                case, query.query_text
            )
            
            # Generate explanation
            explanation = self._generate_search_explanation(
                case, similarity_score, citation_score, precedential_score
            )
            
            result = SearchResult(
                case=case,
                relevance_score=relevance_score,
                similarity_score=similarity_score,
                precedential_score=precedential_score,
                matching_segments=matching_segments,
                explanation=explanation,
                citation_network_score=citation_score
            )
            
            results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _passes_filters(self, case: CaseContent, query: SearchQuery) -> bool:
        """Check if case passes query filters."""
        # Jurisdiction filter
        if query.jurisdiction and case.metadata.jurisdiction != query.jurisdiction:
            return False
        
        # Court level filter
        if query.court_level and case.metadata.court_level != query.court_level:
            return False
        
        # Date range filter
        if query.date_range:
            case_year = case.metadata.citation.year
            start_year = query.date_range[0].year
            end_year = query.date_range[1].year
            if not (start_year <= case_year <= end_year):
                return False
        
        # Overruled cases filter
        if not query.include_overruled and case.metadata.overruled:
            return False
        
        return True
    
    def _calculate_precedential_score(self, case: CaseContent, query: SearchQuery) -> float:
        """Calculate precedential value score."""
        score = case.metadata.precedential_value
        
        # Boost for higher court levels
        if case.metadata.court_level == CourtLevel.SUPREME_COURT:
            score += 0.3
        elif case.metadata.court_level == CourtLevel.APPELLATE:
            score += 0.2
        
        # Penalty for overruled cases
        if case.metadata.overruled:
            score -= 0.4
        
        # Recent cases get slight boost
        current_year = datetime.now().year
        case_age = current_year - case.metadata.citation.year
        if case_age < 5:
            score += 0.1
        elif case_age > 20:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _find_matching_segments(
        self, 
        case: CaseContent, 
        query_text: str
    ) -> List[Dict[str, Any]]:
        """Find specific text segments that match the query."""
        segments = []
        
        # Split case text into segments
        text_segments = self._split_into_segments(case.full_text)
        
        # Calculate similarity for each segment
        query_embedding = await self.embedding_service.get_embedding(query_text)
        
        for i, segment in enumerate(text_segments):
            if len(segment.strip()) < 50:  # Skip very short segments
                continue
                
            segment_embedding = await self.embedding_service.get_embedding(segment)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                segment_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.5:  # Threshold for relevant segments
                segments.append({
                    "text": segment[:500] + "..." if len(segment) > 500 else segment,
                    "similarity": float(similarity),
                    "start_position": i * 1000,  # Approximate position
                    "segment_type": self._classify_segment_type(segment)
                })
        
        # Sort by similarity and return top matches
        segments.sort(key=lambda x: x["similarity"], reverse=True)
        return segments[:5]
    
    def _split_into_segments(self, text: str, segment_size: int = 1000) -> List[str]:
        """Split text into overlapping segments."""
        segments = []
        words = text.split()
        
        for i in range(0, len(words), segment_size // 2):
            segment_words = words[i:i + segment_size]
            segment = " ".join(segment_words)
            segments.append(segment)
        
        return segments
    
    def _classify_segment_type(self, segment: str) -> str:
        """Classify type of text segment."""
        segment_lower = segment.lower()
        
        if any(word in segment_lower for word in ["holding", "rule", "principle"]):
            return "holding"
        elif any(word in segment_lower for word in ["fact", "background", "procedural"]):
            return "factual"
        elif any(word in segment_lower for word in ["reasoning", "analysis", "because"]):
            return "reasoning"
        elif any(word in segment_lower for word in ["cited", "see", "accord"]):
            return "citation"
        else:
            return "general"
    
    def _generate_search_explanation(
        self, 
        case: CaseContent, 
        similarity_score: float, 
        citation_score: float, 
        precedential_score: float
    ) -> str:
        """Generate explanation for why this case was returned."""
        explanations = []
        
        if similarity_score > 0.8:
            explanations.append("highly similar legal concepts")
        elif similarity_score > 0.6:
            explanations.append("similar legal concepts")
        
        if citation_score > 0.5:
            explanations.append("frequently cited precedent")
        
        if precedential_score > 0.7:
            explanations.append("strong precedential value")
        
        if case.metadata.court_level == CourtLevel.SUPREME_COURT:
            explanations.append("supreme court authority")
        
        if not explanations:
            explanations.append("relevant to query")
        
        return f"Relevant due to: {', '.join(explanations)}"
    
    async def find_similar_cases(
        self, 
        case_id: str, 
        limit: int = 10
    ) -> List[SearchResult]:
        """Find cases similar to a given case."""
        case = self.case_database.get(case_id)
        if not case:
            raise ValueError(f"Case not found: {case_id}")
        
        # Create search query based on case content
        query = SearchQuery(
            query_text=case.summary,
            jurisdiction=case.metadata.jurisdiction,
            legal_areas=case.metadata.legal_areas,
            max_results=limit + 1  # +1 to exclude the original case
        )
        
        results = await self.search(query)
        
        # Remove the original case from results
        return [r for r in results if r.case.metadata.case_id != case_id]
    
    async def get_citation_analysis(self, case_id: str) -> Dict[str, Any]:
        """Get detailed citation analysis for a case."""
        case = self.case_database.get(case_id)
        if not case:
            raise ValueError(f"Case not found: {case_id}")
        
        # Find cases that cite this case
        citing_cases = []
        for cid, cited_list in self.citation_network.items():
            if case_id in cited_list:
                citing_cases.append(cid)
        
        # Find most cited cases by this case
        cited_cases = case.cited_cases
        
        return {
            "case_id": case_id,
            "case_name": case.metadata.citation.case_name,
            "times_cited": len(citing_cases),
            "cites_count": len(cited_cases),
            "citing_cases": citing_cases[:10],  # Top 10
            "most_cited_cases": cited_cases[:10],  # Top 10
            "citation_network_score": self._calculate_citation_network_score(case_id),
            "precedential_impact": self._assess_precedential_impact(case_id)
        }
    
    def _assess_precedential_impact(self, case_id: str) -> str:
        """Assess the precedential impact of a case."""
        citation_score = self._calculate_citation_network_score(case_id)
        case = self.case_database.get(case_id)
        
        if not case:
            return "unknown"
        
        if case.metadata.court_level == CourtLevel.SUPREME_COURT and citation_score > 1.0:
            return "landmark"
        elif citation_score > 0.8:
            return "highly_influential"
        elif citation_score > 0.5:
            return "influential"
        elif citation_score > 0.2:
            return "moderate"
        else:
            return "limited"
    
    async def search_by_legal_principle(
        self, 
        principle: str, 
        jurisdiction: Optional[Jurisdiction] = None
    ) -> List[SearchResult]:
        """Search cases by legal principle."""
        query = SearchQuery(
            query_text=f"legal principle: {principle}",
            jurisdiction=jurisdiction,
            similarity_threshold=0.7,
            max_results=20
        )
        
        return await self.search(query)
    
    async def search_by_procedural_posture(
        self, 
        posture: str, 
        legal_area: Optional[str] = None
    ) -> List[SearchResult]:
        """Search cases by procedural posture."""
        query_text = f"procedural posture: {posture}"
        if legal_area:
            query_text += f" legal area: {legal_area}"
        
        query = SearchQuery(
            query_text=query_text,
            legal_areas=[legal_area] if legal_area else [],
            max_results=15
        )
        
        return await self.search(query)
    
    async def get_trend_analysis(
        self, 
        legal_concept: str, 
        years: int = 10
    ) -> Dict[str, Any]:
        """Analyze trends in legal concept usage over time."""
        end_year = datetime.now().year
        start_year = end_year - years
        
        # Search for cases mentioning the concept
        query = SearchQuery(
            query_text=legal_concept,
            date_range=(datetime(start_year, 1, 1), datetime(end_year, 12, 31)),
            max_results=1000
        )
        
        results = await self.search(query)
        
        # Analyze by year
        yearly_counts = defaultdict(int)
        court_level_counts = defaultdict(int)
        jurisdiction_counts = defaultdict(int)
        
        for result in results:
            year = result.case.metadata.citation.year
            yearly_counts[year] += 1
            court_level_counts[result.case.metadata.court_level.value] += 1
            jurisdiction_counts[result.case.metadata.jurisdiction.value] += 1
        
        return {
            "concept": legal_concept,
            "time_period": f"{start_year}-{end_year}",
            "total_cases": len(results),
            "yearly_distribution": dict(yearly_counts),
            "court_level_distribution": dict(court_level_counts),
            "jurisdiction_distribution": dict(jurisdiction_counts),
            "trend_direction": self._calculate_trend_direction(yearly_counts)
        }
    
    def _calculate_trend_direction(self, yearly_counts: Dict[int, int]) -> str:
        """Calculate if trend is increasing, decreasing, or stable."""
        if len(yearly_counts) < 2:
            return "insufficient_data"
        
        years = sorted(yearly_counts.keys())
        early_avg = sum(yearly_counts[y] for y in years[:len(years)//2]) / (len(years)//2)
        recent_avg = sum(yearly_counts[y] for y in years[len(years)//2:]) / (len(years) - len(years)//2)
        
        if recent_avg > early_avg * 1.2:
            return "increasing"
        elif recent_avg < early_avg * 0.8:
            return "decreasing"
        else:
            return "stable"