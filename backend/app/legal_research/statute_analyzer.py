"""
Statute Analyzer with NLP Interpretation Capabilities

This module provides comprehensive statute analysis including interpretation,
cross-referencing, amendment tracking, and legal impact assessment.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from collections import defaultdict

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..nlp.language_models import LegalLanguageModel
from ..ml.embeddings import EmbeddingService


class StatuteType(Enum):
    """Types of statutory provisions."""
    FEDERAL_STATUTE = "federal_statute"
    STATE_STATUTE = "state_statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    MUNICIPAL = "municipal"
    ADMINISTRATIVE = "administrative"


class ProvisionType(Enum):
    """Types of legal provisions."""
    DEFINITION = "definition"
    PROHIBITION = "prohibition"
    REQUIREMENT = "requirement"
    PERMISSION = "permission"
    PROCEDURE = "procedure"
    PENALTY = "penalty"
    EXCEPTION = "exception"


class AmendmentType(Enum):
    """Types of statutory amendments."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    REORGANIZATION = "reorganization"
    REPEAL = "repeal"


@dataclass
class StatutoryProvision:
    """Individual statutory provision."""
    provision_id: str
    section_number: str
    subsection: Optional[str]
    title: str
    text: str
    provision_type: ProvisionType
    effective_date: date
    sunset_date: Optional[date] = None
    cross_references: List[str] = None
    
    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []


@dataclass
class StatutorySection:
    """Complete statutory section with provisions."""
    section_id: str
    title: str
    chapter: str
    section_number: str
    provisions: List[StatutoryProvision]
    last_amended: date
    amendment_history: List[Dict[str, Any]] = None
    related_sections: List[str] = None
    
    def __post_init__(self):
        if self.amendment_history is None:
            self.amendment_history = []
        if self.related_sections is None:
            self.related_sections = []


@dataclass
class Statute:
    """Complete statute with metadata."""
    statute_id: str
    title: str
    citation: str
    statute_type: StatuteType
    jurisdiction: str
    enacted_date: date
    effective_date: date
    sections: List[StatutorySection]
    preamble: Optional[str] = None
    purpose_statement: Optional[str] = None
    definitions_section: Optional[str] = None
    
    def __post_init__(self):
        if not self.sections:
            self.sections = []


@dataclass
class StatutoryAmendment:
    """Amendment to a statute."""
    amendment_id: str
    statute_id: str
    section_affected: str
    amendment_type: AmendmentType
    effective_date: date
    amending_statute: str
    old_text: str
    new_text: str
    rationale: Optional[str] = None
    impact_assessment: Optional[str] = None


@dataclass
class InterpretationQuery:
    """Query for statutory interpretation."""
    query_text: str
    statute_citation: Optional[str] = None
    jurisdiction: Optional[str] = None
    legal_context: Optional[str] = None
    fact_pattern: Optional[str] = None
    ambiguity_type: Optional[str] = None


@dataclass
class InterpretationResult:
    """Result of statutory interpretation analysis."""
    relevant_provisions: List[StatutoryProvision]
    interpretation: str
    confidence_score: float
    supporting_authorities: List[str]
    potential_ambiguities: List[Dict[str, Any]]
    practical_application: str
    policy_considerations: List[str]
    related_cases: List[str] = None
    
    def __post_init__(self):
        if self.related_cases is None:
            self.related_cases = []


class LegalLanguageProcessor:
    """Process and analyze legal language in statutes."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.legal_classifier = pipeline(
            "text-classification",
            model="nlpaueb/legal-bert-base-uncased"
        )
        
        # Legal language patterns
        self.modal_verbs = {
            'shall': 'mandatory',
            'must': 'mandatory', 
            'may': 'permissive',
            'should': 'advisory',
            'will': 'future_mandatory',
            'might': 'conditional'
        }
        
        self.legal_connectors = [
            'provided that', 'except that', 'subject to', 'unless',
            'if and only if', 'to the extent that', 'insofar as'
        ]
    
    def analyze_provision_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the grammatical and legal structure of a provision."""
        doc = self.nlp(text)
        
        analysis = {
            'sentence_count': len(list(doc.sents)),
            'modal_analysis': self._analyze_modals(text),
            'conditions': self._extract_conditions(text),
            'exceptions': self._extract_exceptions(text),
            'definitions': self._extract_definitions(text),
            'cross_references': self._extract_cross_references(text),
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'complexity_score': self._calculate_complexity(doc)
        }
        
        return analysis
    
    def _analyze_modals(self, text: str) -> Dict[str, List[str]]:
        """Analyze modal verbs to determine legal obligations."""
        modal_analysis = defaultdict(list)
        
        for modal, obligation_type in self.modal_verbs.items():
            pattern = rf'\b{modal}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                modal_analysis[obligation_type].append({
                    'modal': match.group(),
                    'context': context.strip(),
                    'position': match.start()
                })
        
        return dict(modal_analysis)
    
    def _extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract conditional statements."""
        conditions = []
        
        # Pattern for if-then statements
        if_pattern = r'\bif\b.*?(?:\bthen\b|\b,\b|\b;|\.|$)'
        if_matches = re.finditer(if_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in if_matches:
            conditions.append({
                'type': 'conditional',
                'text': match.group().strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Pattern for unless statements
        unless_pattern = r'\bunless\b.*?(?:\b,\b|\b;|\.|$)'
        unless_matches = re.finditer(unless_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in unless_matches:
            conditions.append({
                'type': 'exception',
                'text': match.group().strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        return conditions
    
    def _extract_exceptions(self, text: str) -> List[Dict[str, Any]]:
        """Extract exception clauses."""
        exceptions = []
        
        exception_patterns = [
            r'\bexcept\b.*?(?:\b,\b|\b;|\.|$)',
            r'\bprovided that\b.*?(?:\b,\b|\b;|\.|$)',
            r'\bhowever\b.*?(?:\b,\b|\b;|\.|$)',
            r'\bnotwithstanding\b.*?(?:\b,\b|\b;|\.|$)'
        ]
        
        for pattern in exception_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                exceptions.append({
                    'text': match.group().strip(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return exceptions
    
    def _extract_definitions(self, text: str) -> List[Dict[str, Any]]:
        """Extract defined terms."""
        definitions = []
        
        # Pattern for "X means Y" or "X shall mean Y"
        definition_pattern = r'"([^"]+)"\s+(?:means?|shall mean)\s+([^.;]+)'
        matches = re.finditer(definition_pattern, text, re.IGNORECASE)
        
        for match in matches:
            definitions.append({
                'term': match.group(1),
                'definition': match.group(2).strip(),
                'start': match.start(),
                'end': match.end()
            })
        
        return definitions
    
    def _extract_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract cross-references to other statutory provisions."""
        references = []
        
        # Pattern for section references
        section_pattern = r'\bsection\s+(\d+(?:\.\d+)*)'
        section_matches = re.finditer(section_pattern, text, re.IGNORECASE)
        
        for match in section_matches:
            references.append({
                'type': 'section',
                'reference': match.group(1),
                'full_text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Pattern for subsection references
        subsection_pattern = r'\bsubsection\s+\(([a-z0-9]+)\)'
        subsection_matches = re.finditer(subsection_pattern, text, re.IGNORECASE)
        
        for match in subsection_matches:
            references.append({
                'type': 'subsection',
                'reference': match.group(1),
                'full_text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return references
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate complexity score based on linguistic features."""
        # Factors contributing to complexity
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        num_clauses = len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
        num_entities = len(doc.ents)
        num_legal_terms = len([token for token in doc if token.text.lower() in 
                              ['shall', 'must', 'may', 'provided', 'unless']])
        
        # Normalize and combine factors
        complexity = (
            min(avg_sentence_length / 20, 1.0) * 0.3 +
            min(num_clauses / 10, 1.0) * 0.3 +
            min(num_entities / 20, 1.0) * 0.2 +
            min(num_legal_terms / 15, 1.0) * 0.2
        )
        
        return complexity


class StatuteAnalyzer:
    """Advanced statute analyzer with NLP interpretation capabilities."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.legal_model = LegalLanguageModel()
        self.language_processor = LegalLanguageProcessor()
        
        # Statute database (in production, this would be a proper database)
        self.statute_database: Dict[str, Statute] = {}
        self.provision_index: Dict[str, StatutoryProvision] = {}
        self.amendment_history: Dict[str, List[StatutoryAmendment]] = {}
        
        # Cross-reference network
        self.reference_graph = nx.DiGraph()
        
        self.logger = logging.getLogger(__name__)
    
    async def index_statute(self, statute: Statute) -> None:
        """Index a statute for analysis and searching."""
        try:
            self.statute_database[statute.statute_id] = statute
            
            # Index individual provisions
            for section in statute.sections:
                for provision in section.provisions:
                    self.provision_index[provision.provision_id] = provision
                    
                    # Generate embeddings for provision
                    embedding = await self.embedding_service.get_embedding(
                        provision.text, model_type='legal'
                    )
                    
                    # Store embedding (would use vector database in production)
                    await self.embedding_service.store_embedding(
                        provision.provision_id, embedding, 
                        {'statute_id': statute.statute_id, 'section': section.section_number}
                    )
            
            # Build cross-reference graph
            self._build_reference_graph(statute)
            
            self.logger.info(f"Successfully indexed statute: {statute.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to index statute {statute.statute_id}: {str(e)}")
            raise
    
    def _build_reference_graph(self, statute: Statute) -> None:
        """Build cross-reference graph for statute navigation."""
        for section in statute.sections:
            for provision in section.provisions:
                # Add node for this provision
                self.reference_graph.add_node(
                    provision.provision_id,
                    statute_id=statute.statute_id,
                    section=section.section_number,
                    title=provision.title
                )
                
                # Add edges for cross-references
                for ref in provision.cross_references:
                    if ref in self.provision_index:
                        self.reference_graph.add_edge(provision.provision_id, ref)
    
    async def interpret_statute(self, query: InterpretationQuery) -> InterpretationResult:
        """Provide comprehensive statutory interpretation."""
        try:
            self.logger.info(f"Starting statutory interpretation for: {query.query_text[:100]}...")
            
            # Find relevant provisions
            relevant_provisions = await self._find_relevant_provisions(query)
            
            # Analyze each relevant provision
            provision_analyses = []
            for provision in relevant_provisions:
                analysis = self.language_processor.analyze_provision_structure(provision.text)
                provision_analyses.append((provision, analysis))
            
            # Generate interpretation
            interpretation = await self._generate_interpretation(
                query, provision_analyses
            )
            
            # Identify potential ambiguities
            ambiguities = await self._identify_ambiguities(
                query, provision_analyses
            )
            
            # Find supporting authorities
            authorities = await self._find_supporting_authorities(relevant_provisions)
            
            # Generate practical application guidance
            practical_application = await self._generate_practical_application(
                query, provision_analyses
            )
            
            # Identify policy considerations
            policy_considerations = await self._identify_policy_considerations(
                relevant_provisions
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                query, provision_analyses
            )
            
            result = InterpretationResult(
                relevant_provisions=relevant_provisions,
                interpretation=interpretation,
                confidence_score=confidence_score,
                supporting_authorities=authorities,
                potential_ambiguities=ambiguities,
                practical_application=practical_application,
                policy_considerations=policy_considerations
            )
            
            self.logger.info("Statutory interpretation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Statutory interpretation failed: {str(e)}")
            raise
    
    async def _find_relevant_provisions(
        self, 
        query: InterpretationQuery
    ) -> List[StatutoryProvision]:
        """Find statutory provisions relevant to the query."""
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(
            query.query_text, model_type='legal'
        )
        
        # Search for similar provisions
        similar_provisions = []
        
        for provision_id, provision in self.provision_index.items():
            # Get provision embedding
            provision_embedding = await self.embedding_service.get_embedding(
                provision.text, model_type='legal'
            )
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                provision_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.6:  # Threshold for relevance
                similar_provisions.append((provision, similarity))
        
        # Sort by similarity and filter
        similar_provisions.sort(key=lambda x: x[1], reverse=True)
        
        # Apply additional filters
        filtered_provisions = []
        for provision, similarity in similar_provisions[:20]:  # Top 20
            if self._passes_query_filters(provision, query):
                filtered_provisions.append(provision)
        
        return filtered_provisions[:10]  # Return top 10
    
    def _passes_query_filters(
        self, 
        provision: StatutoryProvision, 
        query: InterpretationQuery
    ) -> bool:
        """Check if provision passes query-specific filters."""
        # Statute citation filter
        if query.statute_citation:
            statute = self._get_statute_for_provision(provision.provision_id)
            if statute and statute.citation != query.statute_citation:
                return False
        
        # Jurisdiction filter
        if query.jurisdiction:
            statute = self._get_statute_for_provision(provision.provision_id)
            if statute and statute.jurisdiction != query.jurisdiction:
                return False
        
        return True
    
    def _get_statute_for_provision(self, provision_id: str) -> Optional[Statute]:
        """Get the statute that contains a specific provision."""
        for statute in self.statute_database.values():
            for section in statute.sections:
                for provision in section.provisions:
                    if provision.provision_id == provision_id:
                        return statute
        return None
    
    async def _generate_interpretation(
        self, 
        query: InterpretationQuery, 
        provision_analyses: List[Tuple[StatutoryProvision, Dict[str, Any]]]
    ) -> str:
        """Generate statutory interpretation based on analysis."""
        interpretation_parts = []
        
        # Analyze each relevant provision
        for provision, analysis in provision_analyses:
            # Determine the primary legal obligation
            modal_analysis = analysis['modal_analysis']
            primary_obligation = self._determine_primary_obligation(modal_analysis)
            
            interpretation_parts.append(
                f"Section {provision.section_number} {primary_obligation} regarding {provision.title}."
            )
            
            # Add condition analysis
            if analysis['conditions']:
                conditions_text = self._summarize_conditions(analysis['conditions'])
                interpretation_parts.append(f"This applies {conditions_text}.")
            
            # Add exception analysis  
            if analysis['exceptions']:
                exceptions_text = self._summarize_exceptions(analysis['exceptions'])
                interpretation_parts.append(f"However, {exceptions_text}.")
        
        # Combine interpretations
        base_interpretation = " ".join(interpretation_parts)
        
        # Add contextual analysis if fact pattern provided
        if query.fact_pattern:
            contextual_analysis = await self._analyze_fact_pattern_application(
                query.fact_pattern, provision_analyses
            )
            base_interpretation += f"\n\nApplied to your situation: {contextual_analysis}"
        
        return base_interpretation
    
    def _determine_primary_obligation(self, modal_analysis: Dict[str, List[str]]) -> str:
        """Determine the primary legal obligation from modal analysis."""
        if modal_analysis.get('mandatory'):
            return "creates a mandatory obligation"
        elif modal_analysis.get('permissive'):
            return "grants permission"
        elif modal_analysis.get('advisory'):
            return "provides advisory guidance"
        elif modal_analysis.get('conditional'):
            return "establishes conditional requirements"
        else:
            return "establishes provisions"
    
    def _summarize_conditions(self, conditions: List[Dict[str, Any]]) -> str:
        """Summarize conditional statements."""
        if not conditions:
            return ""
        
        condition_types = defaultdict(int)
        for condition in conditions:
            condition_types[condition['type']] += 1
        
        summary_parts = []
        if condition_types['conditional']:
            summary_parts.append(f"when certain conditions are met")
        if condition_types['exception']:
            summary_parts.append(f"unless exceptions apply")
        
        return " and ".join(summary_parts) if summary_parts else "conditionally"
    
    def _summarize_exceptions(self, exceptions: List[Dict[str, Any]]) -> str:
        """Summarize exception clauses."""
        if not exceptions:
            return ""
        
        return f"there are {len(exceptions)} exception(s) that may apply"
    
    async def _analyze_fact_pattern_application(
        self, 
        fact_pattern: str, 
        provision_analyses: List[Tuple[StatutoryProvision, Dict[str, Any]]]
    ) -> str:
        """Analyze how statutes apply to specific fact pattern."""
        # Use legal reasoning model to analyze application
        analysis_prompt = f"""
        Fact Pattern: {fact_pattern}
        
        Relevant Statutory Provisions:
        {chr(10).join([f"- {p[0].title}: {p[0].text[:200]}..." for p in provision_analyses[:3]])}
        
        How do these provisions apply to the fact pattern?
        """
        
        # This would use a legal reasoning model in production
        application_analysis = f"Based on the fact pattern, the statutory provisions appear to apply with specific considerations for the circumstances described."
        
        return application_analysis
    
    async def _identify_ambiguities(
        self, 
        query: InterpretationQuery, 
        provision_analyses: List[Tuple[StatutoryProvision, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify potential ambiguities in statutory interpretation."""
        ambiguities = []
        
        for provision, analysis in provision_analyses:
            # Check for ambiguous terms
            ambiguous_terms = self._find_ambiguous_terms(provision.text)
            if ambiguous_terms:
                ambiguities.append({
                    'type': 'ambiguous_terms',
                    'provision_id': provision.provision_id,
                    'terms': ambiguous_terms,
                    'impact': 'May require clarification for proper application'
                })
            
            # Check for conflicting modals
            modal_analysis = analysis['modal_analysis']
            if len(modal_analysis) > 1:
                ambiguities.append({
                    'type': 'mixed_obligations',
                    'provision_id': provision.provision_id,
                    'modals': list(modal_analysis.keys()),
                    'impact': 'Contains both mandatory and permissive language'
                })
            
            # Check for complex conditions
            if analysis['complexity_score'] > 0.7:
                ambiguities.append({
                    'type': 'high_complexity',
                    'provision_id': provision.provision_id,
                    'score': analysis['complexity_score'],
                    'impact': 'Complex structure may lead to interpretation difficulties'
                })
        
        return ambiguities
    
    def _find_ambiguous_terms(self, text: str) -> List[str]:
        """Find potentially ambiguous terms in statutory text."""
        ambiguous_patterns = [
            r'\breasonable\b', r'\bappropriate\b', r'\bnecessary\b',
            r'\badequate\b', r'\bsubstantial\b', r'\bmaterial\b',
            r'\bsignificant\b', r'\bpromptly\b', r'\btimely\b'
        ]
        
        ambiguous_terms = []
        for pattern in ambiguous_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ambiguous_terms.append(match.group())
        
        return list(set(ambiguous_terms))
    
    async def _find_supporting_authorities(
        self, 
        provisions: List[StatutoryProvision]
    ) -> List[str]:
        """Find supporting authorities (cases, regulations, etc.)."""
        authorities = []
        
        # This would search case law and regulatory databases
        # For now, return placeholder authorities
        for provision in provisions[:3]:
            authorities.extend([
                f"Regulatory guidance on {provision.title}",
                f"Case law interpreting Section {provision.section_number}",
                f"Administrative interpretation of {provision.title}"
            ])
        
        return authorities[:10]
    
    async def _generate_practical_application(
        self, 
        query: InterpretationQuery, 
        provision_analyses: List[Tuple[StatutoryProvision, Dict[str, Any]]]
    ) -> str:
        """Generate practical application guidance."""
        guidance_parts = []
        
        for provision, analysis in provision_analyses[:3]:
            # Generate specific guidance based on provision type
            if provision.provision_type == ProvisionType.REQUIREMENT:
                guidance_parts.append(f"You must comply with {provision.title}")
            elif provision.provision_type == ProvisionType.PROHIBITION:
                guidance_parts.append(f"You must not engage in {provision.title}")
            elif provision.provision_type == ProvisionType.PROCEDURE:
                guidance_parts.append(f"Follow the procedure outlined in {provision.title}")
            elif provision.provision_type == ProvisionType.DEFINITION:
                guidance_parts.append(f"Note the specific definition in {provision.title}")
        
        if guidance_parts:
            return "Practical steps: " + "; ".join(guidance_parts)
        else:
            return "Review the applicable provisions and ensure compliance with all requirements."
    
    async def _identify_policy_considerations(
        self, 
        provisions: List[StatutoryProvision]
    ) -> List[str]:
        """Identify policy considerations underlying the statutes."""
        considerations = []
        
        # Analyze provision types to infer policy goals
        provision_types = [p.provision_type for p in provisions]
        
        if ProvisionType.PROHIBITION in provision_types:
            considerations.append("Public safety and harm prevention")
        
        if ProvisionType.REQUIREMENT in provision_types:
            considerations.append("Ensuring compliance and standards")
        
        if ProvisionType.PROCEDURE in provision_types:
            considerations.append("Due process and fair procedures")
        
        if ProvisionType.DEFINITION in provision_types:
            considerations.append("Legal clarity and certainty")
        
        # Add general considerations
        considerations.extend([
            "Balancing individual rights with public interest",
            "Administrative efficiency and enforceability",
            "Consistency with existing legal framework"
        ])
        
        return considerations[:5]
    
    def _calculate_confidence_score(
        self, 
        query: InterpretationQuery, 
        provision_analyses: List[Tuple[StatutoryProvision, Dict[str, Any]]]
    ) -> float:
        """Calculate confidence score for interpretation."""
        if not provision_analyses:
            return 0.0
        
        # Factors affecting confidence
        num_provisions = len(provision_analyses)
        avg_complexity = np.mean([analysis['complexity_score'] 
                                 for _, analysis in provision_analyses])
        
        # More provisions generally increase confidence
        provision_score = min(num_provisions / 5.0, 1.0) * 0.4
        
        # Lower complexity increases confidence
        complexity_score = (1.0 - avg_complexity) * 0.3
        
        # Specific query factors
        specificity_score = 0.3
        if query.statute_citation:
            specificity_score += 0.1
        if query.fact_pattern:
            specificity_score += 0.1
        if query.legal_context:
            specificity_score += 0.1
        
        confidence = provision_score + complexity_score + specificity_score
        return min(confidence, 1.0)
    
    async def analyze_amendments(self, statute_id: str) -> Dict[str, Any]:
        """Analyze amendment history and impact."""
        statute = self.statute_database.get(statute_id)
        if not statute:
            raise ValueError(f"Statute not found: {statute_id}")
        
        amendments = self.amendment_history.get(statute_id, [])
        
        if not amendments:
            return {
                'statute_id': statute_id,
                'amendment_count': 0,
                'analysis': 'No amendments found'
            }
        
        # Analyze amendment patterns
        amendment_types = defaultdict(int)
        yearly_amendments = defaultdict(int)
        affected_sections = defaultdict(int)
        
        for amendment in amendments:
            amendment_types[amendment.amendment_type.value] += 1
            yearly_amendments[amendment.effective_date.year] += 1
            affected_sections[amendment.section_affected] += 1
        
        # Calculate stability score
        years_since_enactment = (date.today() - statute.enacted_date).days / 365.25
        stability_score = max(0, 1 - (len(amendments) / max(years_since_enactment, 1)))
        
        return {
            'statute_id': statute_id,
            'statute_title': statute.title,
            'amendment_count': len(amendments),
            'amendment_types': dict(amendment_types),
            'yearly_distribution': dict(yearly_amendments),
            'most_amended_sections': dict(sorted(affected_sections.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]),
            'stability_score': stability_score,
            'recent_amendments': [
                {
                    'date': a.effective_date.isoformat(),
                    'type': a.amendment_type.value,
                    'section': a.section_affected,
                    'summary': a.rationale or 'No rationale provided'
                }
                for a in sorted(amendments, key=lambda x: x.effective_date, reverse=True)[:5]
            ]
        }
    
    async def compare_statutes(
        self, 
        statute_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple statutes for similarities and differences."""
        statutes = []
        for statute_id in statute_ids:
            statute = self.statute_database.get(statute_id)
            if statute:
                statutes.append(statute)
        
        if len(statutes) < 2:
            raise ValueError("At least 2 statutes required for comparison")
        
        # Compare statutory structures
        comparison = {
            'statutes_compared': len(statutes),
            'structural_comparison': self._compare_structures(statutes),
            'thematic_analysis': await self._compare_themes(statutes),
            'cross_references': self._analyze_cross_references(statutes),
            'temporal_analysis': self._analyze_temporal_relationships(statutes)
        }
        
        return comparison
    
    def _compare_structures(self, statutes: List[Statute]) -> Dict[str, Any]:
        """Compare structural aspects of statutes."""
        structure_data = []
        
        for statute in statutes:
            structure_data.append({
                'statute_id': statute.statute_id,
                'title': statute.title,
                'section_count': len(statute.sections),
                'total_provisions': sum(len(s.provisions) for s in statute.sections),
                'has_definitions': bool(statute.definitions_section),
                'has_purpose': bool(statute.purpose_statement)
            })
        
        return {
            'individual_structures': structure_data,
            'avg_sections': np.mean([s['section_count'] for s in structure_data]),
            'avg_provisions': np.mean([s['total_provisions'] for s in structure_data])
        }
    
    async def _compare_themes(self, statutes: List[Statute]) -> Dict[str, Any]:
        """Compare thematic content of statutes."""
        # Extract themes from each statute
        statute_themes = {}
        
        for statute in statutes:
            # Combine all text from statute
            full_text = statute.title + " " + (statute.purpose_statement or "")
            for section in statute.sections:
                for provision in section.provisions:
                    full_text += " " + provision.text
            
            # Generate embedding for thematic analysis
            embedding = await self.embedding_service.get_embedding(
                full_text[:5000], model_type='legal'  # Limit text length
            )
            statute_themes[statute.statute_id] = embedding
        
        # Calculate similarity matrix
        similarities = {}
        statute_ids = list(statute_themes.keys())
        
        for i, id1 in enumerate(statute_ids):
            for j, id2 in enumerate(statute_ids[i+1:], i+1):
                similarity = cosine_similarity(
                    statute_themes[id1].reshape(1, -1),
                    statute_themes[id2].reshape(1, -1)
                )[0][0]
                similarities[f"{id1}-{id2}"] = float(similarity)
        
        return {
            'similarity_matrix': similarities,
            'most_similar_pair': max(similarities.items(), key=lambda x: x[1]),
            'least_similar_pair': min(similarities.items(), key=lambda x: x[1])
        }
    
    def _analyze_cross_references(self, statutes: List[Statute]) -> Dict[str, Any]:
        """Analyze cross-references between statutes."""
        cross_refs = defaultdict(int)
        
        for statute in statutes:
            for section in statute.sections:
                for provision in section.provisions:
                    for ref in provision.cross_references:
                        # Check if reference points to another statute in the comparison
                        for other_statute in statutes:
                            if (other_statute.statute_id != statute.statute_id and 
                                ref.startswith(other_statute.citation)):
                                cross_refs[f"{statute.statute_id}->{other_statute.statute_id}"] += 1
        
        return dict(cross_refs)
    
    def _analyze_temporal_relationships(self, statutes: List[Statute]) -> Dict[str, Any]:
        """Analyze temporal relationships between statutes."""
        statutes_by_date = sorted(statutes, key=lambda s: s.enacted_date)
        
        temporal_analysis = {
            'chronological_order': [
                {
                    'statute_id': s.statute_id,
                    'title': s.title,
                    'enacted': s.enacted_date.isoformat(),
                    'effective': s.effective_date.isoformat()
                }
                for s in statutes_by_date
            ],
            'enactment_span_years': (
                statutes_by_date[-1].enacted_date - statutes_by_date[0].enacted_date
            ).days / 365.25,
            'potential_succession': self._identify_potential_succession(statutes_by_date)
        }
        
        return temporal_analysis
    
    def _identify_potential_succession(self, statutes_by_date: List[Statute]) -> List[str]:
        """Identify potential succession or replacement relationships."""
        succession_notes = []
        
        for i in range(len(statutes_by_date) - 1):
            current = statutes_by_date[i]
            next_statute = statutes_by_date[i + 1]
            
            # Check if titles are similar (potential replacement)
            title_similarity = len(set(current.title.lower().split()) & 
                                 set(next_statute.title.lower().split())) / max(
                                     len(current.title.split()), len(next_statute.title.split()))
            
            if title_similarity > 0.5:
                succession_notes.append(
                    f"{next_statute.title} may supersede or amend {current.title}"
                )
        
        return succession_notes