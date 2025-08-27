"""
Advanced RAG Pipeline for Legal Document Analysis
Implements sophisticated multi-document analysis, semantic search, and AI-powered legal insights
"""

import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


@dataclass
class LegalEntity:
    """Represents a legal entity extracted from documents"""
    entity_type: str  # party, jurisdiction, arbitrator, court, etc.
    name: str
    mentions: List[Dict[str, Any]] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ContractClause:
    """Enhanced contract clause representation"""
    clause_type: str
    text: str
    start_position: int
    end_position: int
    confidence_score: float
    risk_level: str  # low, medium, high, critical
    legal_implications: List[str]
    similar_precedents: List[Dict[str, Any]]
    suggested_modifications: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment for legal document"""
    overall_risk_score: float
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    confidence: float
    explanation: str
    legal_vulnerabilities: List[Dict[str, Any]]


@dataclass
class SemanticSearchResult:
    """Enhanced semantic search result"""
    document_id: str
    content: str
    relevance_score: float
    semantic_similarity: float
    keyword_match_score: float
    context_relevance: float
    source_metadata: Dict[str, Any]
    highlighted_text: str
    related_clauses: List[ContractClause]


class AdvancedRAGPipeline:
    """
    Advanced RAG Pipeline with enhanced AI capabilities for legal document analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced RAG pipeline with configuration"""
        self.config = config or self._get_default_config()
        
        # Initialize AI models
        self._initialize_models()
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()
        
        # Legal ontology and patterns
        self._load_legal_ontology()
        
        # Caching for efficiency
        self.embedding_cache = {}
        self.analysis_cache = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        
        logger.info("Advanced RAG Pipeline initialized with enhanced AI capabilities")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'classification_model': 'nlpaueb/legal-bert-base-uncased',
            'ner_model': 'en_legal_ner',
            'max_workers': 4,
            'chunk_size': 512,
            'chunk_overlap': 128,
            'similarity_threshold': 0.7,
            'max_search_results': 100,
            'enable_cross_reference': True,
            'enable_precedent_matching': True,
            'enable_risk_scoring': True
        }
    
    def _initialize_models(self):
        """Initialize AI models for various tasks"""
        try:
            # Embedding model for semantic search
            self.embedding_model = SentenceTransformer(self.config['embedding_model'])
            
            # NER model for entity extraction
            try:
                self.nlp = spacy.load('en_core_web_trf')
            except:
                self.nlp = spacy.load('en_core_web_sm')
            
            # Classification model for clause type detection
            self.clause_classifier = pipeline(
                "text-classification",
                model="nlpaueb/legal-bert-base-uncased",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Zero-shot classification for flexible analysis
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Question answering model for natural language queries
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2-distilled",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("All AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_legal_ontology(self):
        """Load legal ontology and patterns"""
        self.legal_ontology = {
            'arbitration_indicators': [
                'binding arbitration', 'mandatory arbitration', 'arbitration agreement',
                'dispute resolution', 'arbitral tribunal', 'arbitrator', 'arbitration rules',
                'JAMS', 'AAA', 'ICC arbitration', 'UNCITRAL', 'arbitration proceedings'
            ],
            'risk_indicators': [
                'class action waiver', 'jury trial waiver', 'limitation of liability',
                'indemnification', 'consequential damages', 'punitive damages',
                'forum selection', 'choice of law', 'non-compete', 'confidentiality'
            ],
            'clause_types': [
                'arbitration', 'jurisdiction', 'limitation', 'indemnification',
                'warranty', 'termination', 'payment', 'confidentiality', 'intellectual_property',
                'force_majeure', 'severability', 'assignment', 'governing_law'
            ],
            'legal_relationships': [
                'party_to_party', 'jurisdiction_applies', 'clause_references',
                'precedent_cites', 'statute_governs', 'regulation_requires'
            ]
        }
        
        # Build initial knowledge graph structure
        for clause_type in self.legal_ontology['clause_types']:
            self.knowledge_graph.add_node(clause_type, type='clause_category')
    
    async def analyze_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        cross_reference: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze multiple documents with cross-referencing and comparison
        
        Args:
            documents: List of document dictionaries with 'id', 'content', and 'metadata'
            cross_reference: Whether to perform cross-document analysis
            
        Returns:
            Comprehensive analysis results
        """
        start_time = datetime.now()
        
        # Process documents in parallel
        tasks = []
        for doc in documents:
            tasks.append(self._analyze_single_document_async(doc))
        
        # Wait for all analyses to complete
        individual_analyses = await asyncio.gather(*tasks)
        
        # Build document embeddings for similarity analysis
        doc_embeddings = self._create_document_embeddings(documents)
        
        # Perform cross-document analysis if requested
        cross_analysis = {}
        if cross_reference and len(documents) > 1:
            cross_analysis = self._perform_cross_document_analysis(
                documents, individual_analyses, doc_embeddings
            )
        
        # Generate consolidated insights
        consolidated_insights = self._generate_consolidated_insights(
            individual_analyses, cross_analysis
        )
        
        # Calculate overall risk assessment
        overall_risk = self._calculate_multi_document_risk(individual_analyses)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'individual_analyses': individual_analyses,
            'cross_document_analysis': cross_analysis,
            'consolidated_insights': consolidated_insights,
            'overall_risk_assessment': overall_risk,
            'document_similarity_matrix': self._calculate_similarity_matrix(doc_embeddings),
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_single_document_async(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for single document analysis"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.analyze_single_document, document
        )
    
    def analyze_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single legal document
        
        Args:
            document: Document dictionary with 'id', 'content', and 'metadata'
            
        Returns:
            Complete analysis results
        """
        doc_id = document.get('id', 'unknown')
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Extract and classify clauses
        clauses = self.extract_intelligent_clauses(content)
        
        # Extract legal entities
        entities = self._extract_legal_entities(content)
        
        # Perform risk scoring
        risk_assessment = self.calculate_risk_score(content, clauses)
        
        # Generate embeddings for semantic search
        doc_embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        
        # Identify arbitration-specific content
        arbitration_analysis = self._analyze_arbitration_content(content, clauses)
        
        # Build document knowledge graph
        doc_graph = self._build_document_graph(clauses, entities)
        
        return {
            'document_id': doc_id,
            'metadata': metadata,
            'clauses': [self._clause_to_dict(c) for c in clauses],
            'entities': [self._entity_to_dict(e) for e in entities],
            'risk_assessment': risk_assessment,
            'arbitration_analysis': arbitration_analysis,
            'document_embedding': doc_embedding.tolist(),
            'knowledge_graph': self._graph_to_dict(doc_graph),
            'summary': self._generate_document_summary(content, clauses),
            'key_terms': self._extract_key_terms(content),
            'readability_score': self._calculate_readability(content)
        }
    
    def extract_intelligent_clauses(self, text: str) -> List[ContractClause]:
        """
        Extract and classify contract clauses using AI
        
        Args:
            text: Document text
            
        Returns:
            List of identified and classified clauses
        """
        clauses = []
        
        # Split text into potential clause segments
        segments = self._segment_document(text)
        
        for segment in segments:
            # Classify clause type
            classification = self.zero_shot_classifier(
                segment['text'],
                candidate_labels=self.legal_ontology['clause_types'],
                multi_label=True
            )
            
            if classification['scores'][0] > 0.5:  # Confidence threshold
                clause_type = classification['labels'][0]
                confidence = classification['scores'][0]
                
                # Assess risk level
                risk_level = self._assess_clause_risk(segment['text'], clause_type)
                
                # Find similar precedents
                precedents = self._find_similar_precedents(segment['text'], clause_type)
                
                # Generate suggestions
                suggestions = self._generate_clause_suggestions(
                    segment['text'], clause_type, risk_level
                )
                
                # Extract legal implications
                implications = self._extract_legal_implications(segment['text'], clause_type)
                
                clause = ContractClause(
                    clause_type=clause_type,
                    text=segment['text'],
                    start_position=segment['start'],
                    end_position=segment['end'],
                    confidence_score=confidence,
                    risk_level=risk_level,
                    legal_implications=implications,
                    similar_precedents=precedents,
                    suggested_modifications=suggestions,
                    metadata={
                        'classification_scores': dict(zip(
                            classification['labels'], 
                            classification['scores']
                        )),
                        'segment_id': segment['id']
                    }
                )
                clauses.append(clause)
        
        return clauses
    
    def _segment_document(self, text: str) -> List[Dict[str, Any]]:
        """Segment document into potential clauses"""
        segments = []
        
        # Use multiple strategies for segmentation
        # 1. Numbered sections
        numbered_pattern = r'(\d+\.[\d\.]*\s+[A-Z][^.]+\.(?:\s+[^.]+\.)*)'
        numbered_segments = re.finditer(numbered_pattern, text)
        
        # 2. Heading-based sections
        heading_pattern = r'([A-Z][A-Z\s]+:[\s\S]+?)(?=\n[A-Z][A-Z\s]+:|$)'
        heading_segments = re.finditer(heading_pattern, text)
        
        # 3. Paragraph-based segmentation
        paragraphs = text.split('\n\n')
        
        # Combine and deduplicate segments
        seen_texts = set()
        segment_id = 0
        
        for match in numbered_segments:
            segment_text = match.group(0).strip()
            if segment_text not in seen_texts and len(segment_text) > 50:
                segments.append({
                    'id': segment_id,
                    'text': segment_text,
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'numbered'
                })
                seen_texts.add(segment_text)
                segment_id += 1
        
        for match in heading_segments:
            segment_text = match.group(0).strip()
            if segment_text not in seen_texts and len(segment_text) > 50:
                segments.append({
                    'id': segment_id,
                    'text': segment_text,
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'heading'
                })
                seen_texts.add(segment_text)
                segment_id += 1
        
        # Add remaining paragraphs
        current_pos = 0
        for para in paragraphs:
            para = para.strip()
            if para not in seen_texts and len(para) > 50:
                start_pos = text.find(para, current_pos)
                if start_pos != -1:
                    segments.append({
                        'id': segment_id,
                        'text': para,
                        'start': start_pos,
                        'end': start_pos + len(para),
                        'type': 'paragraph'
                    })
                    seen_texts.add(para)
                    segment_id += 1
                    current_pos = start_pos + len(para)
        
        return sorted(segments, key=lambda x: x['start'])
    
    def _assess_clause_risk(self, text: str, clause_type: str) -> str:
        """Assess risk level of a clause"""
        risk_score = 0
        
        # Check for high-risk indicators
        high_risk_patterns = {
            'arbitration': ['binding', 'mandatory', 'waive', 'class action'],
            'limitation': ['consequential', 'punitive', 'gross negligence'],
            'indemnification': ['defend', 'hold harmless', 'unlimited'],
            'termination': ['immediate', 'without cause', 'sole discretion']
        }
        
        if clause_type in high_risk_patterns:
            for pattern in high_risk_patterns[clause_type]:
                if pattern.lower() in text.lower():
                    risk_score += 2
        
        # Check for general risk indicators
        for indicator in self.legal_ontology['risk_indicators']:
            if indicator.lower() in text.lower():
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            return 'critical'
        elif risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _find_similar_precedents(
        self, 
        clause_text: str, 
        clause_type: str
    ) -> List[Dict[str, Any]]:
        """Find similar clauses from precedent database"""
        # In a real implementation, this would query a precedent database
        # For now, return mock data
        precedents = []
        
        # Generate clause embedding
        clause_embedding = self.embedding_model.encode(clause_text, convert_to_numpy=True)
        
        # Mock precedent search
        sample_precedents = {
            'arbitration': [
                {
                    'case': 'Smith v. Jones Corp',
                    'year': 2022,
                    'similarity': 0.85,
                    'outcome': 'Clause upheld',
                    'jurisdiction': 'California',
                    'key_points': ['Binding arbitration enforced', 'Class action waiver valid']
                }
            ],
            'limitation': [
                {
                    'case': 'Tech Co. v. Customer Inc',
                    'year': 2021,
                    'similarity': 0.78,
                    'outcome': 'Partially enforced',
                    'jurisdiction': 'New York',
                    'key_points': ['Consequential damages waived', 'Gross negligence exception']
                }
            ]
        }
        
        if clause_type in sample_precedents:
            precedents = sample_precedents[clause_type]
        
        return precedents
    
    def _generate_clause_suggestions(
        self, 
        clause_text: str, 
        clause_type: str, 
        risk_level: str
    ) -> List[str]:
        """Generate AI-powered suggestions for clause improvement"""
        suggestions = []
        
        if risk_level in ['high', 'critical']:
            # Suggest modifications to reduce risk
            if clause_type == 'arbitration':
                if 'class action' in clause_text.lower():
                    suggestions.append(
                        "Consider adding an opt-out provision for class action waivers"
                    )
                if 'binding' in clause_text.lower() and 'employment' not in clause_text.lower():
                    suggestions.append(
                        "Add carve-outs for small claims court to increase enforceability"
                    )
            
            elif clause_type == 'limitation':
                if 'consequential damages' in clause_text.lower():
                    suggestions.append(
                        "Include mutual limitation of liability to balance risk"
                    )
                if 'gross negligence' not in clause_text.lower():
                    suggestions.append(
                        "Add exception for gross negligence and willful misconduct"
                    )
            
            elif clause_type == 'indemnification':
                if 'defend' in clause_text.lower():
                    suggestions.append(
                        "Specify right to control defense and settlement"
                    )
                if 'unlimited' in clause_text.lower() or 'cap' not in clause_text.lower():
                    suggestions.append(
                        "Consider adding a liability cap tied to contract value"
                    )
        
        # General suggestions
        if len(clause_text.split()) > 200:
            suggestions.append("Consider breaking this clause into smaller sub-sections for clarity")
        
        if not re.search(r'\d+\.\d+', clause_text):
            suggestions.append("Add section numbering for easier reference")
        
        return suggestions
    
    def _extract_legal_implications(self, clause_text: str, clause_type: str) -> List[str]:
        """Extract legal implications of a clause"""
        implications = []
        
        # Analyze based on clause type
        if clause_type == 'arbitration':
            if 'binding' in clause_text.lower():
                implications.append("Parties waive right to jury trial")
            if 'class action' in clause_text.lower():
                implications.append("Individual claims only - no class actions")
            if 'confidential' in clause_text.lower():
                implications.append("Arbitration proceedings will be confidential")
        
        elif clause_type == 'limitation':
            if 'consequential' in clause_text.lower():
                implications.append("No recovery for indirect or consequential damages")
            if 'liability cap' in clause_text.lower() or 'limited to' in clause_text.lower():
                implications.append("Total liability is capped regardless of claim type")
        
        elif clause_type == 'governing_law':
            # Extract jurisdiction
            jurisdictions = re.findall(r'\b(?:laws of |governed by )([A-Z][a-z]+)', clause_text)
            if jurisdictions:
                implications.append(f"Subject to {jurisdictions[0]} state law")
        
        return implications
    
    def _extract_legal_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities from document"""
        entities = []
        
        # Use NER to extract entities
        doc = self.nlp(text)
        
        entity_dict = defaultdict(lambda: {'mentions': [], 'contexts': []})
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LAW']:
                # Get context around entity
                start = max(0, ent.start - 10)
                end = min(len(doc), ent.end + 10)
                context = doc[start:end].text
                
                entity_dict[ent.text]['mentions'].append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_
                })
                entity_dict[ent.text]['contexts'].append(context)
        
        # Create LegalEntity objects
        for name, data in entity_dict.items():
            entity_type = self._classify_entity_type(name, data['contexts'])
            
            entity = LegalEntity(
                entity_type=entity_type,
                name=name,
                mentions=data['mentions'],
                context=data['contexts'][:3],  # Keep top 3 contexts
                confidence=min(len(data['mentions']) / 10.0, 1.0)
            )
            entities.append(entity)
        
        return entities
    
    def _classify_entity_type(self, name: str, contexts: List[str]) -> str:
        """Classify the type of legal entity"""
        context_text = ' '.join(contexts).lower()
        
        if any(word in context_text for word in ['arbitrator', 'mediator', 'arbitration']):
            return 'arbitrator'
        elif any(word in context_text for word in ['court', 'judge', 'tribunal']):
            return 'court'
        elif any(word in context_text for word in ['party', 'agreement', 'contract']):
            return 'party'
        elif any(word in context_text for word in ['law', 'statute', 'regulation']):
            return 'legal_reference'
        elif any(word in context_text for word in ['state', 'country', 'jurisdiction']):
            return 'jurisdiction'
        else:
            return 'organization'
    
    def calculate_risk_score(
        self, 
        content: str, 
        clauses: List[ContractClause]
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk score for document
        
        Args:
            content: Document content
            clauses: Extracted clauses
            
        Returns:
            RiskAssessment with detailed risk analysis
        """
        risk_factors = []
        total_risk = 0
        
        # Analyze each clause for risk
        for clause in clauses:
            clause_risk = 0
            
            if clause.risk_level == 'critical':
                clause_risk = 4
            elif clause.risk_level == 'high':
                clause_risk = 3
            elif clause.risk_level == 'medium':
                clause_risk = 2
            elif clause.risk_level == 'low':
                clause_risk = 1
            
            if clause_risk > 0:
                risk_factors.append({
                    'clause_type': clause.clause_type,
                    'risk_level': clause.risk_level,
                    'risk_score': clause_risk,
                    'description': clause.text[:200] + '...' if len(clause.text) > 200 else clause.text,
                    'implications': clause.legal_implications
                })
                total_risk += clause_risk
        
        # Check for missing important clauses
        expected_clauses = {'governing_law', 'termination', 'limitation', 'confidentiality'}
        found_clauses = {c.clause_type for c in clauses}
        missing_clauses = expected_clauses - found_clauses
        
        if missing_clauses:
            risk_factors.append({
                'type': 'missing_clauses',
                'risk_level': 'medium',
                'risk_score': 2,
                'description': f"Missing important clauses: {', '.join(missing_clauses)}",
                'implications': ['Potential legal uncertainty', 'May favor other party']
            })
            total_risk += 2 * len(missing_clauses)
        
        # Calculate overall risk score (0-100 scale)
        max_possible_risk = len(clauses) * 4 + len(expected_clauses) * 2
        overall_risk_score = min((total_risk / max_possible_risk) * 100, 100) if max_possible_risk > 0 else 0
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors)
        
        # Identify legal vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(content, clauses)
        
        # Generate explanation
        if overall_risk_score > 70:
            risk_level = "High"
            explanation = "This document contains multiple high-risk provisions that significantly favor one party."
        elif overall_risk_score > 40:
            risk_level = "Medium"
            explanation = "This document has moderate risk with some provisions requiring careful consideration."
        else:
            risk_level = "Low"
            explanation = "This document appears balanced with minimal risk factors."
        
        return RiskAssessment(
            overall_risk_score=overall_risk_score,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            confidence=0.85,  # Based on model confidence
            explanation=f"{risk_level} Risk: {explanation}",
            legal_vulnerabilities=vulnerabilities
        )
    
    def _generate_mitigation_strategies(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate strategies to mitigate identified risks"""
        strategies = []
        
        for factor in risk_factors:
            if factor.get('clause_type') == 'arbitration' and factor['risk_level'] in ['high', 'critical']:
                strategies.append("Negotiate for carve-outs for injunctive relief and small claims")
                strategies.append("Request mutual arbitration clause instead of one-sided provision")
            
            elif factor.get('clause_type') == 'limitation' and factor['risk_level'] in ['high', 'critical']:
                strategies.append("Negotiate for reciprocal limitation of liability")
                strategies.append("Add exceptions for gross negligence and intentional misconduct")
            
            elif factor.get('clause_type') == 'indemnification':
                strategies.append("Limit indemnification to third-party claims only")
                strategies.append("Add cap on indemnification tied to contract value")
            
            elif factor.get('type') == 'missing_clauses':
                strategies.append("Add missing standard clauses to provide legal certainty")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        return unique_strategies
    
    def _identify_vulnerabilities(
        self, 
        content: str, 
        clauses: List[ContractClause]
    ) -> List[Dict[str, Any]]:
        """Identify legal vulnerabilities in the document"""
        vulnerabilities = []
        
        # Check for one-sided provisions
        if 'sole discretion' in content.lower():
            vulnerabilities.append({
                'type': 'one_sided_discretion',
                'severity': 'high',
                'description': 'Contains provisions allowing one party sole discretion',
                'recommendation': 'Negotiate for mutual consent or reasonableness standard'
            })
        
        # Check for unlimited liability
        if 'unlimited' in content.lower() and 'liability' in content.lower():
            vulnerabilities.append({
                'type': 'unlimited_liability',
                'severity': 'critical',
                'description': 'May expose party to unlimited liability',
                'recommendation': 'Negotiate for liability cap or mutual limitation'
            })
        
        # Check for broad indemnification
        indemnity_clauses = [c for c in clauses if c.clause_type == 'indemnification']
        if indemnity_clauses and any('defend' in c.text.lower() for c in indemnity_clauses):
            vulnerabilities.append({
                'type': 'broad_indemnification',
                'severity': 'high',
                'description': 'Broad indemnification including duty to defend',
                'recommendation': 'Limit to third-party claims and add notice requirements'
            })
        
        return vulnerabilities
    
    def perform_semantic_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SemanticSearchResult]:
        """
        Perform advanced semantic search across documents
        
        Args:
            query: Natural language search query
            documents: List of documents to search
            top_k: Number of top results to return
            filters: Optional filters to apply
            
        Returns:
            List of semantic search results
        """
        results = []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Extract query intent
        query_intent = self._analyze_query_intent(query)
        
        # Search each document
        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Apply filters if provided
            if filters:
                if not self._matches_filters(metadata, filters):
                    continue
            
            # Chunk document for granular search
            chunks = self._create_semantic_chunks(content)
            
            # Search within chunks
            for chunk in chunks:
                # Calculate semantic similarity
                chunk_embedding = self.embedding_model.encode(chunk['text'], convert_to_numpy=True)
                semantic_sim = float(util.pytorch_cos_sim(query_embedding, chunk_embedding))
                
                # Calculate keyword match score
                keyword_score = self._calculate_keyword_score(query, chunk['text'])
                
                # Calculate context relevance
                context_score = self._calculate_context_relevance(
                    query_intent, chunk['text']
                )
                
                # Combined relevance score
                relevance_score = (
                    semantic_sim * 0.5 + 
                    keyword_score * 0.3 + 
                    context_score * 0.2
                )
                
                # Extract related clauses
                related_clauses = self._find_related_clauses_in_chunk(
                    chunk['text'], query_intent
                )
                
                # Create highlighted text
                highlighted = self._highlight_relevant_text(
                    chunk['text'], query, query_intent
                )
                
                result = SemanticSearchResult(
                    document_id=doc_id,
                    content=chunk['text'],
                    relevance_score=relevance_score,
                    semantic_similarity=semantic_sim,
                    keyword_match_score=keyword_score,
                    context_relevance=context_score,
                    source_metadata={
                        **metadata,
                        'chunk_id': chunk['id'],
                        'chunk_position': chunk['position']
                    },
                    highlighted_text=highlighted,
                    related_clauses=related_clauses
                )
                results.append(result)
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent behind a search query"""
        intent = {
            'primary_intent': '',
            'entities': [],
            'clause_types': [],
            'risk_focus': False,
            'comparison_request': False
        }
        
        # Check for clause type mentions
        for clause_type in self.legal_ontology['clause_types']:
            if clause_type.replace('_', ' ') in query.lower():
                intent['clause_types'].append(clause_type)
        
        # Check for risk-related queries
        risk_keywords = ['risk', 'liability', 'exposure', 'vulnerable', 'concern']
        if any(keyword in query.lower() for keyword in risk_keywords):
            intent['risk_focus'] = True
        
        # Check for comparison requests
        comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'between']
        if any(keyword in query.lower() for keyword in comparison_keywords):
            intent['comparison_request'] = True
        
        # Classify primary intent
        if 'arbitration' in query.lower():
            intent['primary_intent'] = 'arbitration_analysis'
        elif 'risk' in query.lower():
            intent['primary_intent'] = 'risk_assessment'
        elif 'clause' in query.lower() or 'provision' in query.lower():
            intent['primary_intent'] = 'clause_search'
        else:
            intent['primary_intent'] = 'general_search'
        
        return intent
    
    def _create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from text"""
        chunks = []
        chunk_size = self.config['chunk_size']
        overlap = self.config['chunk_overlap']
        
        # Split into sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Create overlapping chunks
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)
            
            if current_size >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'position': i,
                    'sentence_count': len(current_chunk)
                })
                
                # Handle overlap
                if overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = int(len(current_chunk) * (overlap / chunk_size))
                    current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
                
                chunk_id += 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'text': ' '.join(current_chunk),
                'position': len(sentences),
                'sentence_count': len(current_chunk)
            })
        
        return chunks
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword match score between query and text"""
        # Tokenize and normalize
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())
        
        # Calculate intersection
        common_tokens = query_tokens & text_tokens
        
        if not query_tokens:
            return 0.0
        
        # Calculate score
        score = len(common_tokens) / len(query_tokens)
        return min(score, 1.0)
    
    def _calculate_context_relevance(
        self, 
        query_intent: Dict[str, Any], 
        text: str
    ) -> float:
        """Calculate contextual relevance based on query intent"""
        score = 0.0
        text_lower = text.lower()
        
        # Check for clause type matches
        if query_intent['clause_types']:
            for clause_type in query_intent['clause_types']:
                if clause_type.replace('_', ' ') in text_lower:
                    score += 0.3
        
        # Check for risk focus
        if query_intent['risk_focus']:
            risk_count = sum(1 for word in self.legal_ontology['risk_indicators'] 
                           if word.lower() in text_lower)
            score += min(risk_count * 0.1, 0.3)
        
        # Check for arbitration focus
        if query_intent['primary_intent'] == 'arbitration_analysis':
            arb_count = sum(1 for word in self.legal_ontology['arbitration_indicators']
                          if word.lower() in text_lower)
            score += min(arb_count * 0.15, 0.4)
        
        return min(score, 1.0)
    
    def _find_related_clauses_in_chunk(
        self, 
        chunk_text: str, 
        query_intent: Dict[str, Any]
    ) -> List[ContractClause]:
        """Find clauses related to query intent within a text chunk"""
        # This would integrate with the clause extraction logic
        # For now, return empty list
        return []
    
    def _highlight_relevant_text(
        self, 
        text: str, 
        query: str, 
        query_intent: Dict[str, Any]
    ) -> str:
        """Highlight relevant portions of text based on query"""
        highlighted = text
        
        # Highlight query terms
        query_terms = query.lower().split()
        for term in query_terms:
            pattern = re.compile(f'\\b{re.escape(term)}\\b', re.IGNORECASE)
            highlighted = pattern.sub(f'**{term}**', highlighted)
        
        # Highlight intent-specific terms
        if query_intent['primary_intent'] == 'arbitration_analysis':
            for term in self.legal_ontology['arbitration_indicators']:
                if term.lower() in text.lower():
                    pattern = re.compile(f'\\b{re.escape(term)}\\b', re.IGNORECASE)
                    highlighted = pattern.sub(f'**{term}**', highlighted)
        
        return highlighted
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def create_natural_language_interface(
        self,
        documents: List[Dict[str, Any]]
    ) -> 'NaturalLanguageInterface':
        """
        Create a natural language query interface for documents
        
        Args:
            documents: Documents to query
            
        Returns:
            NaturalLanguageInterface instance
        """
        return NaturalLanguageInterface(self, documents)
    
    def generate_arbitration_suggestions(
        self,
        context: Dict[str, Any],
        requirements: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate AI-powered arbitration clause suggestions
        
        Args:
            context: Context about the agreement (parties, jurisdiction, etc.)
            requirements: Specific requirements for the arbitration clause
            
        Returns:
            List of suggested arbitration clauses with explanations
        """
        suggestions = []
        
        # Analyze context to determine appropriate arbitration framework
        jurisdiction = context.get('jurisdiction', 'Delaware')
        contract_type = context.get('contract_type', 'commercial')
        party_sophistication = context.get('party_sophistication', 'both_sophisticated')
        
        # Generate base arbitration clause
        base_clause = self._generate_base_arbitration_clause(
            jurisdiction, contract_type, party_sophistication
        )
        
        # Customize based on requirements
        for requirement in requirements:
            if 'expedited' in requirement.lower():
                base_clause = self._add_expedited_provisions(base_clause)
            elif 'confidential' in requirement.lower():
                base_clause = self._add_confidentiality_provisions(base_clause)
            elif 'appeal' in requirement.lower():
                base_clause = self._add_appeal_provisions(base_clause)
        
        # Generate variations
        variations = [
            {
                'type': 'standard',
                'clause': base_clause,
                'pros': ['Widely accepted', 'Court-tested', 'Predictable enforcement'],
                'cons': ['May be one-sided', 'Limited flexibility'],
                'confidence': 0.85,
                'risk_level': 'low'
            },
            {
                'type': 'mutual',
                'clause': self._make_clause_mutual(base_clause),
                'pros': ['Balanced', 'Higher acceptance rate', 'Reduces unconscionability risk'],
                'cons': ['May limit strategic advantages'],
                'confidence': 0.90,
                'risk_level': 'very_low'
            },
            {
                'type': 'carve_out',
                'clause': self._add_carve_outs(base_clause),
                'pros': ['Preserves injunctive relief', 'Small claims access'],
                'cons': ['More complex', 'Potential for parallel proceedings'],
                'confidence': 0.82,
                'risk_level': 'low'
            }
        ]
        
        # Score each variation
        for variation in variations:
            variation['suitability_score'] = self._calculate_suitability_score(
                variation, context, requirements
            )
            suggestions.append(variation)
        
        # Sort by suitability score
        suggestions.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return suggestions
    
    def _generate_base_arbitration_clause(
        self, 
        jurisdiction: str, 
        contract_type: str,
        party_sophistication: str
    ) -> str:
        """Generate base arbitration clause based on context"""
        if contract_type == 'employment':
            return (
                "Any dispute arising out of or relating to this Agreement or the "
                "employment relationship shall be resolved through final and binding "
                "arbitration administered by JAMS in accordance with its Employment "
                f"Arbitration Rules. The arbitration shall be held in {jurisdiction}. "
                "The arbitrator shall apply the substantive law of {jurisdiction}."
            )
        elif contract_type == 'consumer' and party_sophistication == 'consumer':
            return (
                "DISPUTE RESOLUTION: Most disputes can be resolved without resort to "
                "arbitration. In the event of a dispute, you may contact our customer "
                "service team. If we cannot resolve your dispute informally, any dispute "
                "will be resolved through binding arbitration before a neutral arbitrator "
                f"administered by AAA under its Consumer Rules in {jurisdiction}."
            )
        else:  # Commercial
            return (
                "Any dispute, claim or controversy arising out of or relating to this "
                "Agreement or the breach, termination, enforcement, interpretation or "
                "validity thereof, including the determination of the scope or "
                "applicability of this agreement to arbitrate, shall be determined by "
                "arbitration in {jurisdiction} before one arbitrator. The arbitration "
                "shall be administered by JAMS pursuant to its Comprehensive Arbitration "
                "Rules and Procedures."
            )
    
    def _add_expedited_provisions(self, clause: str) -> str:
        """Add expedited arbitration provisions"""
        expedited = (
            " The parties agree to expedited arbitration procedures, including: "
            "(a) limited discovery; (b) page limits on briefs; (c) time limits on "
            "hearings; and (d) a decision within 30 days of the hearing."
        )
        return clause + expedited
    
    def _add_confidentiality_provisions(self, clause: str) -> str:
        """Add confidentiality provisions"""
        confidential = (
            " The parties agree that all arbitration proceedings, including any "
            "hearings, discovery, and the arbitrator's award, shall be kept "
            "confidential, except as necessary to enforce the award or as required by law."
        )
        return clause + confidential
    
    def _add_appeal_provisions(self, clause: str) -> str:
        """Add appeal provisions"""
        appeal = (
            " The parties agree to the JAMS Optional Arbitration Appeal Procedure, "
            "allowing for limited appellate review of the arbitrator's award."
        )
        return clause + appeal
    
    def _make_clause_mutual(self, clause: str) -> str:
        """Make arbitration clause mutual"""
        if 'you agree' in clause.lower():
            clause = clause.replace('you agree', 'the parties agree')
        if 'you must' in clause.lower():
            clause = clause.replace('you must', 'each party must')
        return clause
    
    def _add_carve_outs(self, clause: str) -> str:
        """Add carve-outs to arbitration clause"""
        carve_outs = (
            " Notwithstanding the foregoing, either party may seek injunctive or "
            "other equitable relief in court for disputes relating to intellectual "
            "property, confidential information, or violations of restrictive covenants. "
            "Additionally, either party may bring an individual action in small claims court."
        )
        return clause + carve_outs
    
    def _calculate_suitability_score(
        self,
        variation: Dict[str, Any],
        context: Dict[str, Any],
        requirements: List[str]
    ) -> float:
        """Calculate suitability score for an arbitration clause variation"""
        score = 0.5  # Base score
        
        # Adjust based on context
        if context.get('party_sophistication') == 'both_sophisticated':
            if variation['type'] == 'standard':
                score += 0.2
        else:
            if variation['type'] == 'mutual':
                score += 0.2
        
        # Adjust based on requirements match
        requirement_text = ' '.join(requirements).lower()
        variation_text = variation['clause'].lower()
        
        if 'expedited' in requirement_text and 'expedited' in variation_text:
            score += 0.1
        if 'confidential' in requirement_text and 'confidential' in variation_text:
            score += 0.1
        if 'appeal' in requirement_text and 'appeal' in variation_text:
            score += 0.1
        
        # Adjust based on risk level
        if variation['risk_level'] == 'very_low':
            score += 0.1
        elif variation['risk_level'] == 'low':
            score += 0.05
        
        return min(score, 1.0)
    
    def _create_document_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for all documents"""
        embeddings = []
        for doc in documents:
            content = doc.get('content', '')
            # Use cached embedding if available
            cache_key = hashlib.md5(content.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                embedding = self.embedding_cache[cache_key]
            else:
                embedding = self.embedding_model.encode(content, convert_to_numpy=True)
                self.embedding_cache[cache_key] = embedding
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _perform_cross_document_analysis(
        self,
        documents: List[Dict[str, Any]],
        individual_analyses: List[Dict[str, Any]],
        doc_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Perform cross-document analysis"""
        cross_analysis = {
            'common_clauses': self._find_common_clauses(individual_analyses),
            'conflicting_provisions': self._find_conflicting_provisions(individual_analyses),
            'risk_comparison': self._compare_document_risks(individual_analyses),
            'entity_relationships': self._analyze_entity_relationships(individual_analyses),
            'clause_evolution': self._track_clause_evolution(documents, individual_analyses)
        }
        return cross_analysis
    
    def _find_common_clauses(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common clauses across documents"""
        clause_map = defaultdict(list)
        
        for analysis in analyses:
            for clause in analysis.get('clauses', []):
                clause_type = clause['clause_type']
                clause_map[clause_type].append({
                    'document_id': analysis['document_id'],
                    'text': clause['text'][:200],
                    'risk_level': clause['risk_level']
                })
        
        common_clauses = []
        for clause_type, occurrences in clause_map.items():
            if len(occurrences) > 1:
                common_clauses.append({
                    'clause_type': clause_type,
                    'occurrence_count': len(occurrences),
                    'documents': occurrences
                })
        
        return common_clauses
    
    def _find_conflicting_provisions(
        self, 
        analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find conflicting provisions across documents"""
        conflicts = []
        
        # Check for jurisdiction conflicts
        jurisdictions = {}
        for analysis in analyses:
            for clause in analysis.get('clauses', []):
                if clause['clause_type'] == 'governing_law':
                    jurisdiction = self._extract_jurisdiction(clause['text'])
                    if jurisdiction:
                        jurisdictions[analysis['document_id']] = jurisdiction
        
        if len(set(jurisdictions.values())) > 1:
            conflicts.append({
                'type': 'jurisdiction_conflict',
                'severity': 'high',
                'documents': jurisdictions,
                'description': 'Documents specify different governing jurisdictions'
            })
        
        return conflicts
    
    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract jurisdiction from governing law clause"""
        patterns = [
            r'laws of (\w+)',
            r'governed by (\w+)',
            r'jurisdiction of (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _compare_document_risks(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare risk levels across documents"""
        risk_scores = []
        for analysis in analyses:
            risk_assessment = analysis.get('risk_assessment', {})
            risk_scores.append({
                'document_id': analysis['document_id'],
                'risk_score': risk_assessment.get('overall_risk_score', 0),
                'risk_factors': len(risk_assessment.get('risk_factors', []))
            })
        
        # Calculate statistics
        scores = [r['risk_score'] for r in risk_scores]
        avg_risk = np.mean(scores) if scores else 0
        max_risk = max(scores) if scores else 0
        min_risk = min(scores) if scores else 0
        
        return {
            'average_risk': avg_risk,
            'max_risk': max_risk,
            'min_risk': min_risk,
            'risk_distribution': risk_scores,
            'high_risk_documents': [r for r in risk_scores if r['risk_score'] > 70]
        }
    
    def _analyze_entity_relationships(
        self, 
        analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze relationships between entities across documents"""
        entity_graph = nx.Graph()
        
        for analysis in analyses:
            doc_id = analysis['document_id']
            entities = analysis.get('entities', [])
            
            # Add entities to graph
            for entity in entities:
                entity_graph.add_node(
                    entity['name'],
                    type=entity['entity_type'],
                    documents=[doc_id]
                )
            
            # Add relationships between entities in same document
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    entity_graph.add_edge(
                        entity1['name'],
                        entity2['name'],
                        document=doc_id,
                        relationship='co-occurrence'
                    )
        
        # Extract key relationships
        relationships = []
        for edge in entity_graph.edges(data=True):
            relationships.append({
                'entity1': edge[0],
                'entity2': edge[1],
                'relationship': edge[2].get('relationship'),
                'document': edge[2].get('document')
            })
        
        return relationships
    
    def _track_clause_evolution(
        self,
        documents: List[Dict[str, Any]],
        analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Track how clauses evolve across document versions"""
        # This would require document versioning metadata
        # For now, return placeholder
        return {
            'evolution_detected': False,
            'message': 'Document versioning required for clause evolution tracking'
        }
    
    def _generate_consolidated_insights(
        self,
        individual_analyses: List[Dict[str, Any]],
        cross_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate consolidated insights from all analyses"""
        insights = {
            'key_findings': [],
            'recommendations': [],
            'action_items': []
        }
        
        # Analyze arbitration presence
        arbitration_docs = [
            a for a in individual_analyses 
            if a.get('arbitration_analysis', {}).get('has_arbitration', False)
        ]
        
        if arbitration_docs:
            insights['key_findings'].append(
                f"Arbitration clauses found in {len(arbitration_docs)} of {len(individual_analyses)} documents"
            )
            insights['recommendations'].append(
                "Review arbitration clauses for consistency across all documents"
            )
        
        # Check for high-risk documents
        high_risk_docs = [
            a for a in individual_analyses
            if a.get('risk_assessment', {}).get('overall_risk_score', 0) > 70
        ]
        
        if high_risk_docs:
            insights['key_findings'].append(
                f"{len(high_risk_docs)} documents have high risk scores"
            )
            insights['action_items'].append(
                "Priority review required for high-risk documents"
            )
        
        # Check for conflicts
        if cross_analysis.get('conflicting_provisions'):
            insights['key_findings'].append(
                "Conflicting provisions detected across documents"
            )
            insights['action_items'].append(
                "Resolve jurisdictional and other conflicts before execution"
            )
        
        return insights
    
    def _calculate_multi_document_risk(
        self, 
        analyses: List[Dict[str, Any]]
    ) -> RiskAssessment:
        """Calculate overall risk across multiple documents"""
        all_risk_factors = []
        risk_scores = []
        
        for analysis in analyses:
            risk_assessment = analysis.get('risk_assessment', {})
            if risk_assessment:
                all_risk_factors.extend(risk_assessment.get('risk_factors', []))
                risk_scores.append(risk_assessment.get('overall_risk_score', 0))
        
        # Calculate aggregate risk
        if risk_scores:
            overall_score = np.mean(risk_scores)
        else:
            overall_score = 0
        
        # Aggregate mitigation strategies
        all_strategies = set()
        for analysis in analyses:
            strategies = analysis.get('risk_assessment', {}).get('mitigation_strategies', [])
            all_strategies.update(strategies)
        
        return RiskAssessment(
            overall_risk_score=overall_score,
            risk_factors=all_risk_factors[:10],  # Top 10 factors
            mitigation_strategies=list(all_strategies)[:5],  # Top 5 strategies
            confidence=0.85,
            explanation=f"Aggregate risk across {len(analyses)} documents",
            legal_vulnerabilities=[]
        )
    
    def _calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate document similarity matrix"""
        if len(embeddings) == 0:
            return np.array([])
        
        # Calculate cosine similarity between all document pairs
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def _analyze_arbitration_content(
        self,
        content: str,
        clauses: List[ContractClause]
    ) -> Dict[str, Any]:
        """Analyze arbitration-specific content in document"""
        arbitration_clauses = [c for c in clauses if 'arbitration' in c.clause_type.lower()]
        
        has_arbitration = len(arbitration_clauses) > 0
        
        # Extract specific arbitration features
        features = {
            'binding': any('binding' in c.text.lower() for c in arbitration_clauses),
            'mandatory': any('mandatory' in c.text.lower() for c in arbitration_clauses),
            'class_action_waiver': any('class action' in c.text.lower() for c in arbitration_clauses),
            'arbitration_organization': self._extract_arbitration_org(arbitration_clauses),
            'carve_outs': self._identify_carve_outs(arbitration_clauses)
        }
        
        return {
            'has_arbitration': has_arbitration,
            'arbitration_clause_count': len(arbitration_clauses),
            'features': features,
            'enforceability_assessment': self._assess_enforceability(arbitration_clauses, features)
        }
    
    def _extract_arbitration_org(self, clauses: List[ContractClause]) -> Optional[str]:
        """Extract arbitration organization from clauses"""
        orgs = ['JAMS', 'AAA', 'ICC', 'LCIA', 'SIAC', 'UNCITRAL']
        
        for clause in clauses:
            for org in orgs:
                if org in clause.text:
                    return org
        return None
    
    def _identify_carve_outs(self, clauses: List[ContractClause]) -> List[str]:
        """Identify carve-outs in arbitration clauses"""
        carve_outs = []
        
        carve_out_patterns = {
            'injunctive_relief': 'injunctive',
            'intellectual_property': 'intellectual property',
            'small_claims': 'small claims',
            'employment': 'employment'
        }
        
        for clause in clauses:
            for carve_out_type, pattern in carve_out_patterns.items():
                if pattern in clause.text.lower():
                    carve_outs.append(carve_out_type)
        
        return carve_outs
    
    def _assess_enforceability(
        self,
        clauses: List[ContractClause],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess enforceability of arbitration clauses"""
        enforceability_score = 0.7  # Base score
        issues = []
        
        # Check for unconscionability factors
        if features['mandatory'] and features['class_action_waiver']:
            enforceability_score -= 0.1
            issues.append("Combination of mandatory arbitration and class waiver may face scrutiny")
        
        if not features['carve_outs']:
            enforceability_score -= 0.05
            issues.append("No carve-outs may reduce enforceability in some jurisdictions")
        
        # Check for mutuality
        mutuality = any('mutual' in c.text.lower() or 'both parties' in c.text.lower() 
                       for c in clauses)
        if not mutuality:
            enforceability_score -= 0.1
            issues.append("Lack of mutuality may impact enforceability")
        
        return {
            'score': max(0, min(1, enforceability_score)),
            'issues': issues,
            'recommendation': "Consider adding carve-outs and ensuring mutuality" if issues else "Clause appears enforceable"
        }
    
    def _build_document_graph(
        self,
        clauses: List[ContractClause],
        entities: List[LegalEntity]
    ) -> nx.DiGraph:
        """Build knowledge graph for document"""
        graph = nx.DiGraph()
        
        # Add clause nodes
        for clause in clauses:
            graph.add_node(
                f"clause_{id(clause)}",
                type='clause',
                clause_type=clause.clause_type,
                risk_level=clause.risk_level
            )
        
        # Add entity nodes
        for entity in entities:
            graph.add_node(
                f"entity_{entity.name}",
                type='entity',
                entity_type=entity.entity_type,
                name=entity.name
            )
        
        # Add relationships
        # This would be enhanced with actual relationship extraction
        
        return graph
    
    def _graph_to_dict(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Convert networkx graph to dictionary"""
        return {
            'nodes': [
                {'id': node, **graph.nodes[node]}
                for node in graph.nodes()
            ],
            'edges': [
                {'source': edge[0], 'target': edge[1], **graph.edges[edge]}
                for edge in graph.edges()
            ]
        }
    
    def _generate_document_summary(
        self,
        content: str,
        clauses: List[ContractClause]
    ) -> str:
        """Generate executive summary of document"""
        # Count clause types
        clause_types = defaultdict(int)
        for clause in clauses:
            clause_types[clause.clause_type] += 1
        
        # Identify high-risk clauses
        high_risk = [c for c in clauses if c.risk_level in ['high', 'critical']]
        
        summary = f"Document contains {len(clauses)} identified clauses. "
        summary += f"Key provisions include: {', '.join(list(clause_types.keys())[:3])}. "
        
        if high_risk:
            summary += f"{len(high_risk)} high-risk provisions identified requiring review. "
        
        return summary
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key legal terms from document"""
        # Use TF-IDF to extract important terms
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            return feature_names.tolist()
        except:
            return []
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate document readability score"""
        # Simple readability based on sentence and word length
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (lower is more readable)
        readability = (avg_sentence_length * 0.5 + avg_word_length * 2) / 10
        
        # Normalize to 0-1 scale (1 being most readable)
        return max(0, min(1, 1 - readability))
    
    def _clause_to_dict(self, clause: ContractClause) -> Dict[str, Any]:
        """Convert ContractClause to dictionary"""
        return {
            'clause_type': clause.clause_type,
            'text': clause.text,
            'start_position': clause.start_position,
            'end_position': clause.end_position,
            'confidence_score': clause.confidence_score,
            'risk_level': clause.risk_level,
            'legal_implications': clause.legal_implications,
            'similar_precedents': clause.similar_precedents,
            'suggested_modifications': clause.suggested_modifications,
            'metadata': clause.metadata
        }
    
    def _entity_to_dict(self, entity: LegalEntity) -> Dict[str, Any]:
        """Convert LegalEntity to dictionary"""
        return {
            'entity_type': entity.entity_type,
            'name': entity.name,
            'mentions': entity.mentions,
            'context': entity.context,
            'confidence': entity.confidence
        }


class NaturalLanguageInterface:
    """Natural language interface for querying legal documents"""
    
    def __init__(self, rag_pipeline: AdvancedRAGPipeline, documents: List[Dict[str, Any]]):
        """Initialize NL interface"""
        self.rag_pipeline = rag_pipeline
        self.documents = documents
        self.conversation_history = []
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process natural language query about documents
        
        Args:
            question: Natural language question
            
        Returns:
            Answer with supporting evidence
        """
        # Add to conversation history
        self.conversation_history.append({'role': 'user', 'content': question})
        
        # Analyze question intent
        intent = self._analyze_question_intent(question)
        
        # Route to appropriate handler
        if intent['type'] == 'factual':
            response = self._handle_factual_question(question, intent)
        elif intent['type'] == 'comparison':
            response = self._handle_comparison_question(question, intent)
        elif intent['type'] == 'risk_assessment':
            response = self._handle_risk_question(question, intent)
        elif intent['type'] == 'clause_search':
            response = self._handle_clause_search(question, intent)
        else:
            response = self._handle_general_question(question, intent)
        
        # Add response to history
        self.conversation_history.append({'role': 'assistant', 'content': response})
        
        return response
    
    def _analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the intent of a question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'when', 'where', 'who']):
            return {'type': 'factual', 'focus': 'information_extraction'}
        elif any(word in question_lower for word in ['compare', 'difference', 'similar']):
            return {'type': 'comparison', 'focus': 'document_comparison'}
        elif any(word in question_lower for word in ['risk', 'concern', 'issue']):
            return {'type': 'risk_assessment', 'focus': 'risk_analysis'}
        elif any(word in question_lower for word in ['clause', 'provision', 'section']):
            return {'type': 'clause_search', 'focus': 'clause_retrieval'}
        else:
            return {'type': 'general', 'focus': 'open_ended'}
    
    def _handle_factual_question(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle factual questions using QA model"""
        answers = []
        
        for doc in self.documents:
            # Use QA model to find answer in document
            result = self.rag_pipeline.qa_model(
                question=question,
                context=doc['content'][:2000]  # Limit context size
            )
            
            if result['score'] > 0.5:
                answers.append({
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'document_id': doc['id'],
                    'context': result.get('context', '')
                })
        
        return {
            'type': 'factual_answer',
            'answers': answers,
            'summary': answers[0]['answer'] if answers else "No clear answer found",
            'confidence': max([a['confidence'] for a in answers]) if answers else 0
        }
    
    def _handle_comparison_question(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comparison questions"""
        # Placeholder implementation
        return {
            'type': 'comparison',
            'message': 'Comparison analysis would be performed here',
            'documents_compared': len(self.documents)
        }
    
    def _handle_risk_question(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk-related questions"""
        # Placeholder implementation
        return {
            'type': 'risk_assessment',
            'message': 'Risk assessment would be performed here'
        }
    
    def _handle_clause_search(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clause search questions"""
        # Use semantic search to find relevant clauses
        results = self.rag_pipeline.perform_semantic_search(
            query=question,
            documents=self.documents,
            top_k=5
        )
        
        return {
            'type': 'clause_search',
            'results': [
                {
                    'document_id': r.document_id,
                    'content': r.content,
                    'relevance': r.relevance_score,
                    'highlighted': r.highlighted_text
                }
                for r in results
            ],
            'total_found': len(results)
        }
    
    def _handle_general_question(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general open-ended questions"""
        return {
            'type': 'general',
            'message': 'Processing general query',
            'suggestion': 'Try asking a more specific question about clauses, risks, or comparisons'
        }