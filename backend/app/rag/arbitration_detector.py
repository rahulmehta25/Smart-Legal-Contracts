"""
Main arbitration clause detection module.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np

from .patterns import ArbitrationPatterns, ArbitrationPattern
from .embeddings import EmbeddingGenerator, EmbeddingSimilarityScorer, SemanticChunker
from .retriever import DocumentRetriever, HybridRetriever, RetrievalConfig

logger = logging.getLogger(__name__)


class ArbitrationType(Enum):
    """Types of arbitration clauses."""
    BINDING = "binding"
    NON_BINDING = "non-binding"
    VOLUNTARY = "voluntary"
    MANDATORY = "mandatory"
    UNKNOWN = "unknown"


class ClauseType(Enum):
    """Types of legal clauses related to arbitration."""
    ARBITRATION = "arbitration"
    CLASS_ACTION_WAIVER = "class_action_waiver"
    JURY_TRIAL_WAIVER = "jury_trial_waiver"
    DISPUTE_RESOLUTION = "dispute_resolution"
    OPT_OUT = "opt_out"


@dataclass
class ArbitrationClause:
    """Represents a detected arbitration clause."""
    text: str
    confidence_score: float
    arbitration_type: ArbitrationType
    clause_types: List[ClauseType]
    location: Dict[str, int]  # start_char, end_char
    details: Dict[str, Any]
    pattern_matches: List[str]
    semantic_score: float
    keyword_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'confidence_score': self.confidence_score,
            'arbitration_type': self.arbitration_type.value,
            'clause_types': [ct.value for ct in self.clause_types],
            'location': self.location,
            'details': self.details,
            'pattern_matches': self.pattern_matches,
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score
        }


@dataclass
class DetectionResult:
    """Overall detection result for a document."""
    document_id: str
    has_arbitration: bool
    confidence: float
    clauses: List[ArbitrationClause]
    summary: Dict[str, Any]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'document_id': self.document_id,
            'has_arbitration': self.has_arbitration,
            'confidence': self.confidence,
            'clauses': [clause.to_dict() for clause in self.clauses],
            'summary': self.summary,
            'processing_time': self.processing_time
        }


class ArbitrationDetector:
    """Main class for detecting arbitration clauses in documents."""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize the arbitration detector.
        
        Args:
            config: Configuration for retrieval and detection
        """
        self.config = config or RetrievalConfig()
        self.patterns = ArbitrationPatterns()
        self.embedding_generator = EmbeddingGenerator()
        self.retriever = DocumentRetriever(config)
        self.similarity_scorer = EmbeddingSimilarityScorer(self.embedding_generator)
        self.chunker = SemanticChunker(
            self.embedding_generator,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        
        # Detection thresholds
        self.arbitration_threshold = 0.6
        self.high_confidence_threshold = 0.8
        self.clause_merge_threshold = 100  # characters
    
    def detect(self, document_text: str, document_id: Optional[str] = None) -> DetectionResult:
        """
        Detect arbitration clauses in a document.
        
        Args:
            document_text: Full text of the document
            document_id: Optional document identifier
            
        Returns:
            Detection result with found clauses
        """
        import time
        start_time = time.time()
        
        document_id = document_id or "document"
        
        # Index document for retrieval
        self.retriever.retriever.index_document(document_id, document_text)
        
        # Search for arbitration clauses
        search_results = self.retriever.search_arbitration_clauses(document_id)
        
        # Detect clauses from search results
        detected_clauses = self._detect_clauses_from_chunks(search_results, document_text)
        
        # Merge nearby clauses
        merged_clauses = self._merge_nearby_clauses(detected_clauses)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(merged_clauses)
        
        # Generate summary
        summary = self._generate_summary(merged_clauses)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            document_id=document_id,
            has_arbitration=len(merged_clauses) > 0 and overall_confidence > self.arbitration_threshold,
            confidence=overall_confidence,
            clauses=merged_clauses,
            summary=summary,
            processing_time=processing_time
        )
    
    def _detect_clauses_from_chunks(self, 
                                   chunks: List[Dict[str, Any]], 
                                   full_text: str) -> List[ArbitrationClause]:
        """
        Detect arbitration clauses from retrieved chunks.
        
        Args:
            chunks: Retrieved text chunks
            full_text: Full document text
            
        Returns:
            List of detected arbitration clauses
        """
        clauses = []
        
        for chunk in chunks:
            # Skip low-scoring chunks
            if chunk.get('rerank_score', chunk.get('score', 0)) < self.arbitration_threshold:
                continue
            
            # Analyze chunk for arbitration content
            clause = self._analyze_chunk(chunk, full_text)
            
            if clause and clause.confidence_score >= self.arbitration_threshold:
                clauses.append(clause)
        
        return clauses
    
    def _analyze_chunk(self, chunk: Dict[str, Any], full_text: str) -> Optional[ArbitrationClause]:
        """
        Analyze a chunk for arbitration clause content.
        
        Args:
            chunk: Text chunk to analyze
            full_text: Full document text
            
        Returns:
            Detected arbitration clause or None
        """
        chunk_text = chunk['text']
        
        # Pattern matching
        pattern_matches, pattern_score, arbitration_types = self._match_patterns(chunk_text)
        
        # Semantic similarity scoring
        semantic_score = self.similarity_scorer.compute_arbitration_similarity(chunk_text)
        
        # Keyword scoring
        keyword_score = self._compute_keyword_score(chunk_text)
        
        # Combine scores
        confidence_score = self._combine_scores(pattern_score, semantic_score, keyword_score)
        
        # Determine arbitration type
        arbitration_type = self._determine_arbitration_type(
            chunk_text, 
            arbitration_types,
            pattern_matches
        )
        
        # Identify clause types
        clause_types = self._identify_clause_types(chunk_text)
        
        # Extract details
        details = ArbitrationPatterns.extract_arbitration_details(chunk_text)
        
        # Find location in full text
        location = self._find_location(chunk_text, full_text, chunk.get('start_char'))
        
        if confidence_score >= self.arbitration_threshold:
            return ArbitrationClause(
                text=chunk_text,
                confidence_score=confidence_score,
                arbitration_type=arbitration_type,
                clause_types=clause_types,
                location=location,
                details=details,
                pattern_matches=pattern_matches,
                semantic_score=semantic_score,
                keyword_score=keyword_score
            )
        
        return None
    
    def _match_patterns(self, text: str) -> Tuple[List[str], float, List[str]]:
        """
        Match arbitration patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (matched patterns, total score, arbitration types)
        """
        text_lower = text.lower()
        matched_patterns = []
        total_score = 0
        arbitration_types = []
        
        for pattern in ArbitrationPatterns.get_all_patterns():
            matched = False
            
            if pattern.pattern_type == 'regex':
                if re.search(pattern.pattern, text_lower):
                    matched = True
            else:
                if pattern.pattern in text_lower:
                    matched = True
            
            if matched:
                matched_patterns.append(pattern.pattern)
                total_score += pattern.weight
                if pattern.arbitration_type not in arbitration_types:
                    arbitration_types.append(pattern.arbitration_type)
        
        # Check negative indicators
        for negative_pattern, weight in ArbitrationPatterns.get_negative_indicators():
            if negative_pattern in text_lower:
                total_score += weight  # weight is negative
        
        # Normalize score
        total_score = max(0, min(1, total_score))
        
        return matched_patterns, total_score, arbitration_types
    
    def _compute_keyword_score(self, text: str) -> float:
        """
        Compute keyword-based score for text.
        
        Args:
            text: Text to score
            
        Returns:
            Keyword score (0-1)
        """
        text_lower = text.lower()
        score = 0
        word_count = len(text_lower.split())
        
        # Count keyword occurrences
        keyword_counts = {}
        
        for pattern, weight, _ in ArbitrationPatterns.HIGH_CONFIDENCE_KEYWORDS:
            count = text_lower.count(pattern)
            if count > 0:
                keyword_counts[pattern] = (count, weight)
        
        for pattern, weight, _ in ArbitrationPatterns.MEDIUM_CONFIDENCE_KEYWORDS:
            count = text_lower.count(pattern)
            if count > 0:
                keyword_counts[pattern] = (count, weight * 0.7)
        
        # Calculate weighted score
        for pattern, (count, weight) in keyword_counts.items():
            # Diminishing returns for multiple occurrences
            occurrence_score = min(1.0, count / 3)
            score += weight * occurrence_score
        
        # Normalize by text length
        if word_count > 0:
            density_factor = min(1.0, 50 / word_count)  # Prefer shorter, focused text
            score *= (1 + density_factor) / 2
        
        return min(1.0, score)
    
    def _combine_scores(self, pattern_score: float, semantic_score: float, 
                       keyword_score: float) -> float:
        """
        Combine different scoring methods.
        
        Args:
            pattern_score: Pattern matching score
            semantic_score: Semantic similarity score
            keyword_score: Keyword density score
            
        Returns:
            Combined confidence score (0-1)
        """
        # Weighted combination with boost for high individual scores
        base_score = (
            0.4 * pattern_score +
            0.35 * semantic_score +
            0.25 * keyword_score
        )
        
        # Boost if any individual score is very high
        max_score = max(pattern_score, semantic_score, keyword_score)
        if max_score > 0.9:
            base_score = min(1.0, base_score * 1.2)
        elif max_score > 0.8:
            base_score = min(1.0, base_score * 1.1)
        
        return base_score
    
    def _determine_arbitration_type(self, text: str, pattern_types: List[str],
                                   pattern_matches: List[str]) -> ArbitrationType:
        """
        Determine the type of arbitration clause.
        
        Args:
            text: Clause text
            pattern_types: Types from pattern matching
            pattern_matches: Matched patterns
            
        Returns:
            Arbitration type
        """
        text_lower = text.lower()
        
        # Check for explicit binding/mandatory indicators
        if 'binding' in pattern_types or any('binding' in p for p in pattern_matches):
            return ArbitrationType.BINDING
        
        if any(word in text_lower for word in ['mandatory', 'required', 'must', 'shall']):
            return ArbitrationType.MANDATORY
        
        # Check for non-binding indicators
        if 'non-binding' in pattern_types or 'non-binding' in text_lower:
            return ArbitrationType.NON_BINDING
        
        # Check for voluntary indicators
        if any(word in text_lower for word in ['voluntary', 'may elect', 'optional']):
            return ArbitrationType.VOLUNTARY
        
        # Default to binding if strong arbitration language present
        if any(word in text_lower for word in ['agree to arbitrate', 'submit to arbitration']):
            return ArbitrationType.BINDING
        
        return ArbitrationType.UNKNOWN
    
    def _identify_clause_types(self, text: str) -> List[ClauseType]:
        """
        Identify specific types of clauses present.
        
        Args:
            text: Clause text
            
        Returns:
            List of clause types
        """
        text_lower = text.lower()
        clause_types = []
        
        # Check for arbitration
        if any(word in text_lower for word in ['arbitrat', 'dispute resolution']):
            clause_types.append(ClauseType.ARBITRATION)
        
        # Check for class action waiver
        if any(phrase in text_lower for phrase in ['class action', 'class proceed', 
                                                    'representative action', 'collective']):
            clause_types.append(ClauseType.CLASS_ACTION_WAIVER)
        
        # Check for jury trial waiver
        if any(phrase in text_lower for phrase in ['jury trial', 'jury waiver', 
                                                    'waive.*jury', 'trial by jury']):
            clause_types.append(ClauseType.JURY_TRIAL_WAIVER)
        
        # Check for general dispute resolution
        if 'dispute' in text_lower and ClauseType.ARBITRATION not in clause_types:
            clause_types.append(ClauseType.DISPUTE_RESOLUTION)
        
        # Check for opt-out provisions
        if any(phrase in text_lower for phrase in ['opt out', 'opt-out', 'reject', 
                                                    'decline arbitration']):
            clause_types.append(ClauseType.OPT_OUT)
        
        return clause_types if clause_types else [ClauseType.ARBITRATION]
    
    def _find_location(self, chunk_text: str, full_text: str, 
                      hint_start: Optional[int] = None) -> Dict[str, int]:
        """
        Find the location of chunk in full text.
        
        Args:
            chunk_text: Text chunk to locate
            full_text: Full document text
            hint_start: Optional hint for start position
            
        Returns:
            Dictionary with start_char and end_char
        """
        # Try exact match first
        start_pos = full_text.find(chunk_text)
        
        if start_pos == -1:
            # Try normalized match
            normalized_chunk = ' '.join(chunk_text.split())
            normalized_full = ' '.join(full_text.split())
            start_pos = normalized_full.find(normalized_chunk)
            
            if start_pos != -1:
                # Adjust for original text
                # This is approximate
                ratio = len(full_text) / len(normalized_full)
                start_pos = int(start_pos * ratio)
        
        if start_pos == -1 and hint_start is not None:
            # Use hint if available
            start_pos = hint_start
        
        if start_pos == -1:
            start_pos = 0  # Fallback
        
        end_pos = start_pos + len(chunk_text)
        
        return {
            'start_char': start_pos,
            'end_char': end_pos
        }
    
    def _merge_nearby_clauses(self, clauses: List[ArbitrationClause]) -> List[ArbitrationClause]:
        """
        Merge clauses that are close together.
        
        Args:
            clauses: List of detected clauses
            
        Returns:
            List of merged clauses
        """
        if not clauses:
            return []
        
        # Sort by location
        sorted_clauses = sorted(clauses, key=lambda c: c.location['start_char'])
        
        merged = []
        current_group = [sorted_clauses[0]]
        
        for clause in sorted_clauses[1:]:
            last_in_group = current_group[-1]
            
            # Check if close enough to merge
            distance = clause.location['start_char'] - last_in_group.location['end_char']
            
            if distance <= self.clause_merge_threshold:
                current_group.append(clause)
            else:
                # Merge current group and start new one
                merged.append(self._merge_clause_group(current_group))
                current_group = [clause]
        
        # Merge last group
        if current_group:
            merged.append(self._merge_clause_group(current_group))
        
        return merged
    
    def _merge_clause_group(self, clauses: List[ArbitrationClause]) -> ArbitrationClause:
        """
        Merge a group of clauses into one.
        
        Args:
            clauses: Group of clauses to merge
            
        Returns:
            Merged clause
        """
        if len(clauses) == 1:
            return clauses[0]
        
        # Combine text
        combined_text = ' '.join(c.text for c in clauses)
        
        # Average scores
        avg_confidence = np.mean([c.confidence_score for c in clauses])
        avg_semantic = np.mean([c.semantic_score for c in clauses])
        avg_keyword = np.mean([c.keyword_score for c in clauses])
        
        # Combine clause types
        all_clause_types = []
        for c in clauses:
            all_clause_types.extend(c.clause_types)
        clause_types = list(set(all_clause_types))
        
        # Determine overall arbitration type (prefer most restrictive)
        arbitration_type = ArbitrationType.UNKNOWN
        type_priority = [
            ArbitrationType.BINDING,
            ArbitrationType.MANDATORY,
            ArbitrationType.NON_BINDING,
            ArbitrationType.VOLUNTARY,
            ArbitrationType.UNKNOWN
        ]
        
        for priority_type in type_priority:
            if any(c.arbitration_type == priority_type for c in clauses):
                arbitration_type = priority_type
                break
        
        # Combine details
        combined_details = {}
        for c in clauses:
            combined_details.update(c.details)
        
        # Combine pattern matches
        all_patterns = []
        for c in clauses:
            all_patterns.extend(c.pattern_matches)
        pattern_matches = list(set(all_patterns))
        
        # Calculate location span
        location = {
            'start_char': min(c.location['start_char'] for c in clauses),
            'end_char': max(c.location['end_char'] for c in clauses)
        }
        
        return ArbitrationClause(
            text=combined_text,
            confidence_score=avg_confidence,
            arbitration_type=arbitration_type,
            clause_types=clause_types,
            location=location,
            details=combined_details,
            pattern_matches=pattern_matches,
            semantic_score=avg_semantic,
            keyword_score=avg_keyword
        )
    
    def _calculate_overall_confidence(self, clauses: List[ArbitrationClause]) -> float:
        """
        Calculate overall confidence that document contains arbitration.
        
        Args:
            clauses: Detected clauses
            
        Returns:
            Overall confidence score (0-1)
        """
        if not clauses:
            return 0.0
        
        # Get highest individual confidence
        max_confidence = max(c.confidence_score for c in clauses)
        
        # Average confidence of high-scoring clauses
        high_scoring = [c for c in clauses if c.confidence_score >= self.high_confidence_threshold]
        if high_scoring:
            avg_high = np.mean([c.confidence_score for c in high_scoring])
        else:
            avg_high = 0
        
        # Boost for multiple clauses
        multi_clause_boost = min(0.2, len(clauses) * 0.05)
        
        # Combine
        overall = max(max_confidence, avg_high) + multi_clause_boost
        
        return min(1.0, overall)
    
    def _generate_summary(self, clauses: List[ArbitrationClause]) -> Dict[str, Any]:
        """
        Generate summary of detection results.
        
        Args:
            clauses: Detected clauses
            
        Returns:
            Summary dictionary
        """
        if not clauses:
            return {
                'has_binding_arbitration': False,
                'has_class_action_waiver': False,
                'has_jury_trial_waiver': False,
                'has_opt_out': False,
                'arbitration_provider': None,
                'num_clauses': 0
            }
        
        # Aggregate information
        has_binding = any(
            c.arbitration_type in [ArbitrationType.BINDING, ArbitrationType.MANDATORY]
            for c in clauses
        )
        
        has_class_waiver = any(
            ClauseType.CLASS_ACTION_WAIVER in c.clause_types
            for c in clauses
        )
        
        has_jury_waiver = any(
            ClauseType.JURY_TRIAL_WAIVER in c.clause_types
            for c in clauses
        )
        
        has_opt_out = any(
            ClauseType.OPT_OUT in c.clause_types or c.details.get('opt_out_available')
            for c in clauses
        )
        
        # Find provider
        provider = None
        for c in clauses:
            if c.details.get('provider'):
                provider = c.details['provider']
                break
        
        return {
            'has_binding_arbitration': has_binding,
            'has_class_action_waiver': has_class_waiver,
            'has_jury_trial_waiver': has_jury_waiver,
            'has_opt_out': has_opt_out,
            'arbitration_provider': provider,
            'num_clauses': len(clauses),
            'highest_confidence': max(c.confidence_score for c in clauses),
            'arbitration_types': list(set(c.arbitration_type.value for c in clauses))
        }
    
    def detect_from_file(self, file_path: str) -> DetectionResult:
        """
        Detect arbitration clauses from a file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Detection result
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        document_id = Path(file_path).stem
        return self.detect(text, document_id)