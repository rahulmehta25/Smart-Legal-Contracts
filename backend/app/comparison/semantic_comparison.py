"""
Semantic document comparison using NLP and transformer models.
Analyzes meaning-based differences beyond surface-level text changes.
"""

import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
from datetime import datetime

from .diff_engine import DiffResult, DiffType, DiffLevel, ComparisonResult


class SemanticChangeType(Enum):
    """Types of semantic changes detected."""
    MEANING_PRESERVED = "meaning_preserved"  # Different words, same meaning
    MEANING_STRENGTHENED = "meaning_strengthened"  # Stronger assertion
    MEANING_WEAKENED = "meaning_weakened"  # Weaker assertion
    MEANING_NEGATED = "meaning_negated"  # Opposite meaning
    MEANING_SHIFTED = "meaning_shifted"  # Different but related meaning
    CONCEPT_ADDED = "concept_added"  # New concept introduced
    CONCEPT_REMOVED = "concept_removed"  # Concept eliminated
    RELATIONSHIP_CHANGED = "relationship_changed"  # Changed relationships between concepts


@dataclass
class SemanticDiff:
    """Represents a semantic difference between document segments."""
    change_type: SemanticChangeType
    old_segment: str
    new_segment: str
    old_concepts: List[str]
    new_concepts: List[str]
    semantic_similarity: float
    intent_change_score: float
    confidence: float
    explanation: str
    position: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptEntity:
    """Represents a semantic concept or entity."""
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float
    embedding: np.ndarray = None
    relations: List[str] = field(default_factory=list)


class SemanticAnalyzer:
    """Handles semantic analysis of document content."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize semantic analyzer.
        
        Args:
            model_name: spaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logging.warning(f"Model {model_name} not found, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize sentence transformer for semantic embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Legal domain-specific patterns
        self.legal_patterns = self._initialize_legal_patterns()
        
    def _initialize_legal_patterns(self) -> Dict[str, List[str]]:
        """Initialize legal domain-specific patterns and keywords."""
        return {
            'obligations': [
                'shall', 'must', 'required to', 'obligated to', 'bound to',
                'responsible for', 'duty to', 'covenant to'
            ],
            'permissions': [
                'may', 'can', 'allowed to', 'permitted to', 'authorized to',
                'entitled to', 'right to'
            ],
            'prohibitions': [
                'shall not', 'must not', 'cannot', 'prohibited from',
                'forbidden to', 'restricted from', 'banned from'
            ],
            'conditions': [
                'if', 'when', 'unless', 'provided that', 'subject to',
                'in the event that', 'on condition that'
            ],
            'temporal': [
                'immediately', 'within', 'no later than', 'prior to',
                'after', 'during', 'throughout', 'until'
            ],
            'liability': [
                'liable', 'responsible', 'damages', 'compensation',
                'indemnify', 'hold harmless', 'loss', 'penalty'
            ]
        }
    
    def extract_concepts(self, text: str) -> List[ConceptEntity]:
        """
        Extract semantic concepts and entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted concepts and entities
        """
        doc = self.nlp(text)
        concepts = []
        
        # Extract named entities
        for ent in doc.ents:
            concepts.append(ConceptEntity(
                text=ent.text,
                label=ent.label_,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=1.0  # spaCy doesn't provide confidence scores
            ))
        
        # Extract key phrases (noun phrases)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                concepts.append(ConceptEntity(
                    text=chunk.text,
                    label="NOUN_PHRASE",
                    start_pos=chunk.start_char,
                    end_pos=chunk.end_char,
                    confidence=0.8
                ))
        
        # Extract legal-specific patterns
        concepts.extend(self._extract_legal_concepts(text))
        
        return concepts
    
    def _extract_legal_concepts(self, text: str) -> List[ConceptEntity]:
        """Extract legal domain-specific concepts."""
        concepts = []
        text_lower = text.lower()
        
        for category, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(re.escape(pattern.lower()), text_lower)
                for match in matches:
                    concepts.append(ConceptEntity(
                        text=text[match.start():match.end()],
                        label=f"LEGAL_{category.upper()}",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9
                    ))
        
        return concepts
    
    def compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two text segments.
        
        Args:
            text_a: First text segment
            text_b: Second text segment
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.sentence_model:
            # Fallback to spaCy similarity
            doc_a = self.nlp(text_a)
            doc_b = self.nlp(text_b)
            return doc_a.similarity(doc_b)
        
        # Use sentence transformers for better semantic understanding
        embeddings = self.sentence_model.encode([text_a, text_b])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def analyze_intent_change(self, old_text: str, new_text: str) -> Tuple[float, str]:
        """
        Analyze change in intent or meaning between text segments.
        
        Args:
            old_text: Original text
            new_text: Modified text
            
        Returns:
            Tuple of (intent_change_score, explanation)
        """
        old_concepts = self.extract_concepts(old_text)
        new_concepts = self.extract_concepts(new_text)
        
        # Analyze changes in legal modalities
        old_modalities = self._extract_modalities(old_text)
        new_modalities = self._extract_modalities(new_text)
        
        intent_score = 0.0
        explanations = []
        
        # Check for changes in obligation strength
        if old_modalities['obligations'] != new_modalities['obligations']:
            if len(new_modalities['obligations']) > len(old_modalities['obligations']):
                intent_score += 0.3
                explanations.append("Obligations strengthened")
            else:
                intent_score += 0.2
                explanations.append("Obligations weakened")
        
        # Check for permission changes
        if old_modalities['permissions'] != new_modalities['permissions']:
            intent_score += 0.2
            explanations.append("Permission scope changed")
        
        # Check for prohibition changes
        if old_modalities['prohibitions'] != new_modalities['prohibitions']:
            intent_score += 0.4
            explanations.append("Prohibition rules modified")
        
        # Check for temporal condition changes
        if old_modalities['temporal'] != new_modalities['temporal']:
            intent_score += 0.3
            explanations.append("Temporal conditions modified")
        
        # Check for concept additions/removals
        old_concept_texts = set(c.text.lower() for c in old_concepts)
        new_concept_texts = set(c.text.lower() for c in new_concepts)
        
        added_concepts = new_concept_texts - old_concept_texts
        removed_concepts = old_concept_texts - new_concept_texts
        
        if added_concepts:
            intent_score += min(0.2, len(added_concepts) * 0.05)
            explanations.append(f"New concepts added: {', '.join(list(added_concepts)[:3])}")
        
        if removed_concepts:
            intent_score += min(0.2, len(removed_concepts) * 0.05)
            explanations.append(f"Concepts removed: {', '.join(list(removed_concepts)[:3])}")
        
        return min(1.0, intent_score), "; ".join(explanations)
    
    def _extract_modalities(self, text: str) -> Dict[str, Set[str]]:
        """Extract modal expressions from text."""
        text_lower = text.lower()
        modalities = defaultdict(set)
        
        for category, patterns in self.legal_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    modalities[category].add(pattern)
        
        return dict(modalities)


class SemanticComparisonEngine:
    """
    Main engine for semantic document comparison.
    Combines surface-level and meaning-based analysis.
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        self.analyzer = SemanticAnalyzer(model_name)
        
    def compare_documents_semantically(self, doc_a: str, doc_b: str,
                                     chunk_size: int = 500) -> List[SemanticDiff]:
        """
        Perform semantic comparison of two documents.
        
        Args:
            doc_a: First document content
            doc_b: Second document content
            chunk_size: Size of text chunks for comparison
            
        Returns:
            List of semantic differences found
        """
        # Split documents into semantic chunks
        chunks_a = self._split_into_semantic_chunks(doc_a, chunk_size)
        chunks_b = self._split_into_semantic_chunks(doc_b, chunk_size)
        
        semantic_diffs = []
        
        # Compare chunks using alignment
        aligned_chunks = self._align_chunks(chunks_a, chunks_b)
        
        for chunk_a, chunk_b, alignment_score in aligned_chunks:
            if chunk_a and chunk_b:
                # Both chunks exist - compare them
                diff = self._compare_chunk_pair(chunk_a, chunk_b)
                if diff:
                    semantic_diffs.append(diff)
            elif chunk_a:
                # Chunk removed
                semantic_diffs.append(SemanticDiff(
                    change_type=SemanticChangeType.CONCEPT_REMOVED,
                    old_segment=chunk_a['text'],
                    new_segment="",
                    old_concepts=[c.text for c in self.analyzer.extract_concepts(chunk_a['text'])],
                    new_concepts=[],
                    semantic_similarity=0.0,
                    intent_change_score=0.5,
                    confidence=0.9,
                    explanation="Entire section removed",
                    position=(chunk_a['start'], chunk_a['end'])
                ))
            elif chunk_b:
                # Chunk added
                semantic_diffs.append(SemanticDiff(
                    change_type=SemanticChangeType.CONCEPT_ADDED,
                    old_segment="",
                    new_segment=chunk_b['text'],
                    old_concepts=[],
                    new_concepts=[c.text for c in self.analyzer.extract_concepts(chunk_b['text'])],
                    semantic_similarity=0.0,
                    intent_change_score=0.5,
                    confidence=0.9,
                    explanation="New section added",
                    position=(chunk_b['start'], chunk_b['end'])
                ))
        
        return semantic_diffs
    
    def _split_into_semantic_chunks(self, text: str, chunk_size: int) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks based on sentence boundaries.
        
        Args:
            text: Input text
            chunk_size: Target size for chunks
            
        Returns:
            List of chunk dictionaries with text, start, and end positions
        """
        doc = self.analyzer.nlp(text)
        chunks = []
        current_chunk = ""
        chunk_start = 0
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sentence_text) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start': chunk_start,
                    'end': chunk_start + len(current_chunk)
                })
                current_chunk = sentence_text
                chunk_start = sent.start_char
            else:
                if current_chunk:
                    current_chunk += " " + sentence_text
                else:
                    current_chunk = sentence_text
                    chunk_start = sent.start_char
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start': chunk_start,
                'end': chunk_start + len(current_chunk)
            })
        
        return chunks
    
    def _align_chunks(self, chunks_a: List[Dict], chunks_b: List[Dict]]) -> List[Tuple]:
        """
        Align chunks between documents based on semantic similarity.
        
        Args:
            chunks_a: Chunks from first document
            chunks_b: Chunks from second document
            
        Returns:
            List of aligned chunk tuples
        """
        if not chunks_a and not chunks_b:
            return []
        
        if not chunks_a:
            return [(None, chunk, 0.0) for chunk in chunks_b]
        
        if not chunks_b:
            return [(chunk, None, 0.0) for chunk in chunks_a]
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(chunks_a), len(chunks_b)))
        
        for i, chunk_a in enumerate(chunks_a):
            for j, chunk_b in enumerate(chunks_b):
                similarity = self.analyzer.compute_semantic_similarity(
                    chunk_a['text'], chunk_b['text']
                )
                similarity_matrix[i][j] = similarity
        
        # Simple greedy alignment - can be improved with Hungarian algorithm
        aligned_chunks = []
        used_a = set()
        used_b = set()
        
        # Find best matches first
        while True:
            max_sim = 0.0
            best_i, best_j = -1, -1
            
            for i in range(len(chunks_a)):
                if i in used_a:
                    continue
                for j in range(len(chunks_b)):
                    if j in used_b:
                        continue
                    if similarity_matrix[i][j] > max_sim:
                        max_sim = similarity_matrix[i][j]
                        best_i, best_j = i, j
            
            if max_sim < 0.3:  # Threshold for meaningful alignment
                break
            
            aligned_chunks.append((chunks_a[best_i], chunks_b[best_j], max_sim))
            used_a.add(best_i)
            used_b.add(best_j)
        
        # Add unmatched chunks
        for i in range(len(chunks_a)):
            if i not in used_a:
                aligned_chunks.append((chunks_a[i], None, 0.0))
        
        for j in range(len(chunks_b)):
            if j not in used_b:
                aligned_chunks.append((None, chunks_b[j], 0.0))
        
        return aligned_chunks
    
    def _compare_chunk_pair(self, chunk_a: Dict, chunk_b: Dict) -> Optional[SemanticDiff]:
        """
        Compare a pair of aligned chunks.
        
        Args:
            chunk_a: Chunk from first document
            chunk_b: Chunk from second document
            
        Returns:
            SemanticDiff if significant difference found, None otherwise
        """
        text_a = chunk_a['text']
        text_b = chunk_b['text']
        
        # Skip if texts are identical
        if text_a.strip() == text_b.strip():
            return None
        
        # Compute semantic similarity
        semantic_similarity = self.analyzer.compute_semantic_similarity(text_a, text_b)
        
        # Analyze intent change
        intent_change_score, explanation = self.analyzer.analyze_intent_change(text_a, text_b)
        
        # Extract concepts
        old_concepts = self.analyzer.extract_concepts(text_a)
        new_concepts = self.analyzer.extract_concepts(text_b)
        
        # Determine change type
        change_type = self._determine_change_type(
            text_a, text_b, semantic_similarity, intent_change_score,
            old_concepts, new_concepts
        )
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            semantic_similarity, intent_change_score, len(text_a), len(text_b)
        )
        
        return SemanticDiff(
            change_type=change_type,
            old_segment=text_a,
            new_segment=text_b,
            old_concepts=[c.text for c in old_concepts],
            new_concepts=[c.text for c in new_concepts],
            semantic_similarity=semantic_similarity,
            intent_change_score=intent_change_score,
            confidence=confidence,
            explanation=explanation,
            position=(chunk_a['start'], chunk_a['end']),
            metadata={
                'chunk_a_size': len(text_a),
                'chunk_b_size': len(text_b),
                'concept_overlap': len(set(c.text for c in old_concepts) & 
                                     set(c.text for c in new_concepts))
            }
        )
    
    def _determine_change_type(self, text_a: str, text_b: str, 
                              semantic_similarity: float, intent_change_score: float,
                              old_concepts: List[ConceptEntity], 
                              new_concepts: List[ConceptEntity]) -> SemanticChangeType:
        """Determine the type of semantic change."""
        
        # High semantic similarity but different text = meaning preserved
        if semantic_similarity > 0.8 and intent_change_score < 0.2:
            return SemanticChangeType.MEANING_PRESERVED
        
        # High intent change = significant meaning change
        if intent_change_score > 0.6:
            # Check for negation patterns
            if self._has_negation_change(text_a, text_b):
                return SemanticChangeType.MEANING_NEGATED
            elif intent_change_score > 0.8:
                return SemanticChangeType.MEANING_SHIFTED
            else:
                return SemanticChangeType.RELATIONSHIP_CHANGED
        
        # Check for strengthening/weakening patterns
        old_modalities = self.analyzer._extract_modalities(text_a)
        new_modalities = self.analyzer._extract_modalities(text_b)
        
        if self._is_strengthened(old_modalities, new_modalities):
            return SemanticChangeType.MEANING_STRENGTHENED
        elif self._is_weakened(old_modalities, new_modalities):
            return SemanticChangeType.MEANING_WEAKENED
        
        # Check for concept changes
        old_concept_texts = set(c.text.lower() for c in old_concepts)
        new_concept_texts = set(c.text.lower() for c in new_concepts)
        
        if len(new_concept_texts - old_concept_texts) > len(old_concept_texts - new_concept_texts):
            return SemanticChangeType.CONCEPT_ADDED
        elif len(old_concept_texts - new_concept_texts) > len(new_concept_texts - old_concept_texts):
            return SemanticChangeType.CONCEPT_REMOVED
        
        # Default to meaning shifted
        return SemanticChangeType.MEANING_SHIFTED
    
    def _has_negation_change(self, text_a: str, text_b: str) -> bool:
        """Check if there's a negation change between texts."""
        negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor'}
        
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        neg_a = len(words_a & negation_words)
        neg_b = len(words_b & negation_words)
        
        return abs(neg_a - neg_b) > 0
    
    def _is_strengthened(self, old_modalities: Dict, new_modalities: Dict) -> bool:
        """Check if the new text has stronger modalities."""
        strong_obligations = {'must', 'shall', 'required to'}
        
        old_strong = len(old_modalities.get('obligations', set()) & strong_obligations)
        new_strong = len(new_modalities.get('obligations', set()) & strong_obligations)
        
        return new_strong > old_strong
    
    def _is_weakened(self, old_modalities: Dict, new_modalities: Dict) -> bool:
        """Check if the new text has weaker modalities."""
        weak_permissions = {'may', 'can', 'might'}
        
        old_weak = len(old_modalities.get('permissions', set()) & weak_permissions)
        new_weak = len(new_modalities.get('permissions', set()) & weak_permissions)
        
        return new_weak > old_weak
    
    def _calculate_confidence(self, semantic_similarity: float, intent_change_score: float,
                            text_a_length: int, text_b_length: int) -> float:
        """Calculate confidence score for the semantic difference."""
        base_confidence = 0.5
        
        # Higher confidence for larger text differences
        length_factor = min(abs(text_a_length - text_b_length) / max(text_a_length, text_b_length), 0.3)
        
        # Higher confidence for clear semantic differences
        semantic_factor = (1.0 - semantic_similarity) * 0.3
        
        # Higher confidence for clear intent changes
        intent_factor = intent_change_score * 0.2
        
        confidence = base_confidence + length_factor + semantic_factor + intent_factor
        
        return min(1.0, max(0.1, confidence))


# Utility functions

def analyze_document_semantics(document: str) -> Dict[str, Any]:
    """
    Analyze semantic properties of a single document.
    
    Args:
        document: Document content to analyze
        
    Returns:
        Dictionary with semantic analysis results
    """
    analyzer = SemanticAnalyzer()
    
    concepts = analyzer.extract_concepts(document)
    concept_categories = defaultdict(list)
    
    for concept in concepts:
        concept_categories[concept.label].append(concept.text)
    
    return {
        'total_concepts': len(concepts),
        'concept_categories': dict(concept_categories),
        'legal_modalities': analyzer._extract_modalities(document),
        'document_length': len(document),
        'semantic_complexity': len(concepts) / max(1, len(document.split()))
    }


def compare_documents_with_semantics(doc_a: str, doc_b: str) -> Dict[str, Any]:
    """
    Perform comprehensive semantic comparison of two documents.
    
    Args:
        doc_a: First document content
        doc_b: Second document content
        
    Returns:
        Comprehensive comparison results
    """
    engine = SemanticComparisonEngine()
    
    semantic_diffs = engine.compare_documents_semantically(doc_a, doc_b)
    
    # Overall semantic similarity
    overall_similarity = engine.analyzer.compute_semantic_similarity(doc_a, doc_b)
    
    # Categorize differences
    diff_categories = defaultdict(list)
    for diff in semantic_diffs:
        diff_categories[diff.change_type.value].append(diff)
    
    return {
        'overall_semantic_similarity': overall_similarity,
        'total_semantic_differences': len(semantic_diffs),
        'differences_by_category': {k: len(v) for k, v in diff_categories.items()},
        'semantic_diffs': semantic_diffs,
        'analysis_timestamp': datetime.now().isoformat()
    }