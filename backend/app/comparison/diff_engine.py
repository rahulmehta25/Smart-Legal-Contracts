"""
Advanced document comparison engine with multiple diff algorithms.
Implements character-level, word-level, and paragraph-level comparison.
"""

import difflib
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict
import numpy as np
from datetime import datetime


class DiffType(Enum):
    """Types of differences that can be detected."""
    INSERTION = "insertion"
    DELETION = "deletion"
    MODIFICATION = "modification"
    MOVE = "move"
    NO_CHANGE = "no_change"


class DiffLevel(Enum):
    """Levels of granularity for diff comparison."""
    CHARACTER = "character"
    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"


@dataclass
class DiffResult:
    """Represents a single difference between documents."""
    diff_type: DiffType
    level: DiffLevel
    old_content: str
    new_content: str
    old_position: Tuple[int, int]  # (start, end)
    new_position: Tuple[int, int]  # (start, end)
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Complete comparison result between two documents."""
    document_a_id: str
    document_b_id: str
    timestamp: datetime
    similarity_score: float
    differences: List[DiffResult]
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MyersAlgorithm:
    """Implementation of Myers' diff algorithm for efficient longest common subsequence."""
    
    @staticmethod
    def compute_lcs(seq_a: List[str], seq_b: List[str]) -> List[Tuple[int, int]]:
        """
        Compute the longest common subsequence using Myers' algorithm.
        
        Args:
            seq_a: First sequence
            seq_b: Second sequence
            
        Returns:
            List of (index_a, index_b) tuples representing matching elements
        """
        m, n = len(seq_a), len(seq_b)
        if m == 0 or n == 0:
            return []
        
        # Create the edit graph
        max_d = m + n
        v = {1: 0}
        trace = []
        
        for d in range(max_d + 1):
            trace.append(v.copy())
            for k in range(-d, d + 1, 2):
                if k == -d or (k != d and v[k - 1] < v[k + 1]):
                    x = v[k + 1]
                else:
                    x = v[k - 1] + 1
                
                y = x - k
                
                # Extend diagonal as far as possible
                while x < m and y < n and seq_a[x] == seq_b[y]:
                    x += 1
                    y += 1
                
                v[k] = x
                
                if x >= m and y >= n:
                    return MyersAlgorithm._backtrack(trace, seq_a, seq_b, d, k)
        
        return []
    
    @staticmethod
    def _backtrack(trace: List[Dict], seq_a: List[str], seq_b: List[str], 
                   d: int, k: int) -> List[Tuple[int, int]]:
        """Backtrack through the edit graph to find the LCS."""
        x, y = len(seq_a), len(seq_b)
        lcs = []
        
        for i in range(d, -1, -1):
            v = trace[i]
            k_prev = k
            
            if k == -i or (k != i and v.get(k - 1, -1) < v.get(k + 1, -1)):
                k_prev = k + 1
            else:
                k_prev = k - 1
            
            x_prev = v.get(k_prev, 0)
            y_prev = x_prev - k_prev
            
            # Add diagonal moves (matches) to LCS
            while x > x_prev and y > y_prev:
                x -= 1
                y -= 1
                if seq_a[x] == seq_b[y]:
                    lcs.append((x, y))
            
            if i > 0:
                k = k_prev
                x, y = x_prev, y_prev
        
        return list(reversed(lcs))


class FuzzyMatcher:
    """Handles fuzzy matching for moved sections and similar content."""
    
    @staticmethod
    def similarity_ratio(text_a: str, text_b: str) -> float:
        """Calculate similarity ratio between two text strings."""
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()
    
    @staticmethod
    def find_moved_sections(old_paragraphs: List[str], new_paragraphs: List[str], 
                          threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find paragraphs that have been moved between documents.
        
        Returns:
            List of (old_index, new_index, similarity) tuples
        """
        moved_sections = []
        
        for i, old_para in enumerate(old_paragraphs):
            if not old_para.strip():
                continue
                
            best_match_idx = -1
            best_similarity = 0.0
            
            for j, new_para in enumerate(new_paragraphs):
                if not new_para.strip():
                    continue
                    
                similarity = FuzzyMatcher.similarity_ratio(old_para, new_para)
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = j
            
            if best_match_idx != -1 and best_match_idx != i:
                moved_sections.append((i, best_match_idx, best_similarity))
        
        return moved_sections


class TextPreprocessor:
    """Handles text preprocessing and normalization for comparison."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text.strip())
    
    @staticmethod
    def remove_formatting(text: str) -> str:
        """Remove common formatting characters."""
        # Remove common markdown/HTML formatting
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold markdown
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic markdown
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code markdown
        return text
    
    @staticmethod
    def split_into_words(text: str) -> List[str]:
        """Split text into words, preserving punctuation context."""
        return re.findall(r'\w+|[^\w\s]', text)
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]


class AdvancedDiffEngine:
    """
    Advanced document comparison engine with multiple diff algorithms.
    Supports character, word, sentence, and paragraph level comparisons.
    """
    
    def __init__(self, ignore_whitespace: bool = True, ignore_formatting: bool = False):
        self.ignore_whitespace = ignore_whitespace
        self.ignore_formatting = ignore_formatting
        self.preprocessor = TextPreprocessor()
        self.fuzzy_matcher = FuzzyMatcher()
        
    def compare_documents(self, doc_a: str, doc_b: str, 
                         levels: List[DiffLevel] = None) -> ComparisonResult:
        """
        Compare two documents at multiple levels of granularity.
        
        Args:
            doc_a: First document content
            doc_b: Second document content
            levels: List of comparison levels to perform
            
        Returns:
            ComparisonResult with all differences found
        """
        if levels is None:
            levels = [DiffLevel.WORD, DiffLevel.PARAGRAPH]
        
        # Preprocess documents
        processed_a = self._preprocess_document(doc_a)
        processed_b = self._preprocess_document(doc_b)
        
        differences = []
        statistics = {
            'total_chars_a': len(doc_a),
            'total_chars_b': len(doc_b),
            'total_words_a': len(self.preprocessor.split_into_words(doc_a)),
            'total_words_b': len(self.preprocessor.split_into_words(doc_b)),
        }
        
        # Perform comparisons at each requested level
        for level in levels:
            level_diffs = self._compare_at_level(processed_a, processed_b, level)
            differences.extend(level_diffs)
        
        # Calculate overall similarity score
        similarity_score = self._calculate_similarity_score(processed_a, processed_b)
        
        # Update statistics
        statistics.update({
            'total_differences': len(differences),
            'insertions': len([d for d in differences if d.diff_type == DiffType.INSERTION]),
            'deletions': len([d for d in differences if d.diff_type == DiffType.DELETION]),
            'modifications': len([d for d in differences if d.diff_type == DiffType.MODIFICATION]),
            'moves': len([d for d in differences if d.diff_type == DiffType.MOVE]),
        })
        
        return ComparisonResult(
            document_a_id=hashlib.md5(doc_a.encode()).hexdigest()[:8],
            document_b_id=hashlib.md5(doc_b.encode()).hexdigest()[:8],
            timestamp=datetime.now(),
            similarity_score=similarity_score,
            differences=differences,
            statistics=statistics
        )
    
    def _preprocess_document(self, document: str) -> str:
        """Preprocess document according to engine settings."""
        processed = document
        
        if self.ignore_formatting:
            processed = self.preprocessor.remove_formatting(processed)
            
        if self.ignore_whitespace:
            processed = self.preprocessor.normalize_whitespace(processed)
            
        return processed
    
    def _compare_at_level(self, doc_a: str, doc_b: str, level: DiffLevel) -> List[DiffResult]:
        """Compare documents at a specific granularity level."""
        if level == DiffLevel.CHARACTER:
            return self._character_level_diff(doc_a, doc_b)
        elif level == DiffLevel.WORD:
            return self._word_level_diff(doc_a, doc_b)
        elif level == DiffLevel.SENTENCE:
            return self._sentence_level_diff(doc_a, doc_b)
        elif level == DiffLevel.PARAGRAPH:
            return self._paragraph_level_diff(doc_a, doc_b)
        else:
            return []
    
    def _character_level_diff(self, doc_a: str, doc_b: str) -> List[DiffResult]:
        """Perform character-level diff using difflib."""
        differences = []
        
        matcher = difflib.SequenceMatcher(None, doc_a, doc_b)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
                
            old_content = doc_a[i1:i2]
            new_content = doc_b[j1:j2]
            
            if tag == 'delete':
                diff_type = DiffType.DELETION
                new_content = ""
                new_position = (j1, j1)
            elif tag == 'insert':
                diff_type = DiffType.INSERTION
                old_content = ""
                old_position = (i1, i1)
            else:  # replace
                diff_type = DiffType.MODIFICATION
            
            differences.append(DiffResult(
                diff_type=diff_type,
                level=DiffLevel.CHARACTER,
                old_content=old_content,
                new_content=new_content,
                old_position=(i1, i2),
                new_position=(j1, j2),
                confidence=1.0,
                context=self._get_context(doc_a, i1, i2, 50)
            ))
        
        return differences
    
    def _word_level_diff(self, doc_a: str, doc_b: str) -> List[DiffResult]:
        """Perform word-level diff using Myers algorithm."""
        words_a = self.preprocessor.split_into_words(doc_a)
        words_b = self.preprocessor.split_into_words(doc_b)
        
        # Use Myers algorithm for efficient LCS
        lcs_matches = MyersAlgorithm.compute_lcs(words_a, words_b)
        
        differences = []
        a_idx = b_idx = 0
        match_idx = 0
        
        while a_idx < len(words_a) or b_idx < len(words_b):
            # Check if we have a match at current position
            if (match_idx < len(lcs_matches) and 
                lcs_matches[match_idx][0] == a_idx and 
                lcs_matches[match_idx][1] == b_idx):
                # Skip matched words
                a_idx += 1
                b_idx += 1
                match_idx += 1
                continue
            
            # Determine the type of difference
            if a_idx < len(words_a) and b_idx < len(words_b):
                # Modification
                differences.append(DiffResult(
                    diff_type=DiffType.MODIFICATION,
                    level=DiffLevel.WORD,
                    old_content=words_a[a_idx],
                    new_content=words_b[b_idx],
                    old_position=(a_idx, a_idx + 1),
                    new_position=(b_idx, b_idx + 1),
                    confidence=0.9
                ))
                a_idx += 1
                b_idx += 1
            elif a_idx < len(words_a):
                # Deletion
                differences.append(DiffResult(
                    diff_type=DiffType.DELETION,
                    level=DiffLevel.WORD,
                    old_content=words_a[a_idx],
                    new_content="",
                    old_position=(a_idx, a_idx + 1),
                    new_position=(b_idx, b_idx),
                    confidence=1.0
                ))
                a_idx += 1
            else:
                # Insertion
                differences.append(DiffResult(
                    diff_type=DiffType.INSERTION,
                    level=DiffLevel.WORD,
                    old_content="",
                    new_content=words_b[b_idx],
                    old_position=(a_idx, a_idx),
                    new_position=(b_idx, b_idx + 1),
                    confidence=1.0
                ))
                b_idx += 1
        
        return differences
    
    def _sentence_level_diff(self, doc_a: str, doc_b: str) -> List[DiffResult]:
        """Perform sentence-level comparison."""
        sentences_a = self.preprocessor.split_into_sentences(doc_a)
        sentences_b = self.preprocessor.split_into_sentences(doc_b)
        
        differences = []
        matcher = difflib.SequenceMatcher(None, sentences_a, sentences_b)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            old_content = ' '.join(sentences_a[i1:i2])
            new_content = ' '.join(sentences_b[j1:j2])
            
            if tag == 'delete':
                diff_type = DiffType.DELETION
                new_content = ""
            elif tag == 'insert':
                diff_type = DiffType.INSERTION
                old_content = ""
            else:
                diff_type = DiffType.MODIFICATION
            
            differences.append(DiffResult(
                diff_type=diff_type,
                level=DiffLevel.SENTENCE,
                old_content=old_content,
                new_content=new_content,
                old_position=(i1, i2),
                new_position=(j1, j2),
                confidence=0.95
            ))
        
        return differences
    
    def _paragraph_level_diff(self, doc_a: str, doc_b: str) -> List[DiffResult]:
        """Perform paragraph-level comparison with move detection."""
        paragraphs_a = self.preprocessor.split_into_paragraphs(doc_a)
        paragraphs_b = self.preprocessor.split_into_paragraphs(doc_b)
        
        differences = []
        
        # First, detect moved paragraphs
        moved_sections = self.fuzzy_matcher.find_moved_sections(paragraphs_a, paragraphs_b)
        moved_a_indices = set(m[0] for m in moved_sections)
        moved_b_indices = set(m[1] for m in moved_sections)
        
        # Add move differences
        for old_idx, new_idx, similarity in moved_sections:
            differences.append(DiffResult(
                diff_type=DiffType.MOVE,
                level=DiffLevel.PARAGRAPH,
                old_content=paragraphs_a[old_idx],
                new_content=paragraphs_b[new_idx],
                old_position=(old_idx, old_idx + 1),
                new_position=(new_idx, new_idx + 1),
                confidence=similarity,
                metadata={'similarity': similarity}
            ))
        
        # Then perform regular diff on non-moved paragraphs
        filtered_a = [p for i, p in enumerate(paragraphs_a) if i not in moved_a_indices]
        filtered_b = [p for i, p in enumerate(paragraphs_b) if i not in moved_b_indices]
        
        matcher = difflib.SequenceMatcher(None, filtered_a, filtered_b)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            old_content = '\n\n'.join(filtered_a[i1:i2])
            new_content = '\n\n'.join(filtered_b[j1:j2])
            
            if tag == 'delete':
                diff_type = DiffType.DELETION
                new_content = ""
            elif tag == 'insert':
                diff_type = DiffType.INSERTION
                old_content = ""
            else:
                diff_type = DiffType.MODIFICATION
            
            differences.append(DiffResult(
                diff_type=diff_type,
                level=DiffLevel.PARAGRAPH,
                old_content=old_content,
                new_content=new_content,
                old_position=(i1, i2),
                new_position=(j1, j2),
                confidence=0.9
            ))
        
        return differences
    
    def _calculate_similarity_score(self, doc_a: str, doc_b: str) -> float:
        """Calculate overall similarity score between documents."""
        return difflib.SequenceMatcher(None, doc_a, doc_b).ratio()
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Get context around a difference."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]


# Utility functions for external use

def compare_documents_simple(doc_a: str, doc_b: str, 
                           ignore_whitespace: bool = True) -> ComparisonResult:
    """
    Simple document comparison function.
    
    Args:
        doc_a: First document content
        doc_b: Second document content
        ignore_whitespace: Whether to ignore whitespace differences
        
    Returns:
        ComparisonResult with differences
    """
    engine = AdvancedDiffEngine(ignore_whitespace=ignore_whitespace)
    return engine.compare_documents(doc_a, doc_b)


def get_similarity_score(doc_a: str, doc_b: str) -> float:
    """Get similarity score between two documents."""
    return difflib.SequenceMatcher(None, doc_a, doc_b).ratio()


def find_moved_sections(doc_a: str, doc_b: str, threshold: float = 0.8) -> List[Dict]:
    """Find sections that have been moved between documents."""
    preprocessor = TextPreprocessor()
    matcher = FuzzyMatcher()
    
    paragraphs_a = preprocessor.split_into_paragraphs(doc_a)
    paragraphs_b = preprocessor.split_into_paragraphs(doc_b)
    
    moved_sections = matcher.find_moved_sections(paragraphs_a, paragraphs_b, threshold)
    
    return [
        {
            'old_index': old_idx,
            'new_index': new_idx,
            'similarity': similarity,
            'content': paragraphs_a[old_idx][:100] + '...' if len(paragraphs_a[old_idx]) > 100 else paragraphs_a[old_idx]
        }
        for old_idx, new_idx, similarity in moved_sections
    ]