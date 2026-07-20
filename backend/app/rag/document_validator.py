"""
Document type validation for arbitration clause detection.
Prevents false positives by validating that documents are actually legal documents
before attempting arbitration clause detection.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents that can be processed."""
    LEGAL_DOCUMENT = "legal_document"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    NON_LEGAL = "non_legal"
    ACADEMIC = "academic"
    RECIPE = "recipe"
    STORY = "story"
    QUIZ = "quiz"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of document type validation."""
    is_legal_document: bool
    document_type: DocumentType
    confidence: float
    reasons: List[str]
    warning_flags: List[str]


class DocumentValidator:
    """Validates document type before arbitration detection."""
    
    def __init__(self):
        """Initialize the document validator."""
        self.legal_document_indicators = self._build_legal_indicators()
        self.non_legal_indicators = self._build_non_legal_indicators()
        self.structure_patterns = self._build_structure_patterns()
        
        # Minimum requirements for legal document validation - ENHANCED
        self.min_legal_indicators = 4  # Increased from 3
        self.min_confidence_threshold = 0.75  # Increased from 0.7
        self.max_non_legal_score = 0.25  # Decreased from 0.3 for stricter filtering
    
    def validate_document(self, text: str) -> ValidationResult:
        """
        Validate if a document is a legal document suitable for arbitration detection.
        
        Args:
            text: Document text to validate
            
        Returns:
            ValidationResult with assessment
        """
        if not text or len(text.strip()) < 50:
            return ValidationResult(
                is_legal_document=False,
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                reasons=["Document too short to analyze"],
                warning_flags=["insufficient_content"]
            )
        
        text_lower = text.lower()
        text_words = text_lower.split()
        
        # Score legal indicators
        legal_score, legal_reasons = self._score_legal_indicators(text_lower, text_words)
        
        # Score non-legal indicators
        non_legal_score, non_legal_flags = self._score_non_legal_indicators(text_lower, text_words)
        
        # Check document structure
        structure_score, structure_reasons = self._analyze_document_structure(text)
        
        # Determine document type
        document_type = self._classify_document_type(text_lower, legal_score, non_legal_score)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            legal_score, non_legal_score, structure_score, len(text_words)
        )
        
        # Determine if suitable for legal analysis
        is_legal = self._is_legal_document(
            legal_score, non_legal_score, overall_confidence, document_type
        )
        
        # Compile reasons
        all_reasons = legal_reasons + structure_reasons
        if non_legal_flags:
            all_reasons.append(f"Non-legal indicators detected: {', '.join(non_legal_flags[:3])}")
        
        return ValidationResult(
            is_legal_document=is_legal,
            document_type=document_type,
            confidence=overall_confidence,
            reasons=all_reasons,
            warning_flags=non_legal_flags
        )
    
    def _build_legal_indicators(self) -> Dict[str, float]:
        """Build legal document indicators with weights."""
        return {
            # Legal document types
            "terms of service": 0.9,
            "terms of use": 0.9,
            "service agreement": 0.9,
            "license agreement": 0.9,
            "privacy policy": 0.8,
            "end user license": 0.9,
            "software license": 0.8,
            "user agreement": 0.8,
            "terms and conditions": 0.9,
            "service terms": 0.8,
            
            # Legal language
            "whereas": 0.7,
            "wherefore": 0.7,
            "hereby": 0.6,
            "heretofore": 0.7,
            "hereafter": 0.6,
            "party agrees": 0.7,
            "parties agree": 0.7,
            "shall be": 0.5,
            "shall not": 0.5,
            "may not": 0.4,
            "you agree": 0.6,
            "user agrees": 0.7,
            "customer agrees": 0.7,
            
            # Legal concepts
            "jurisdiction": 0.6,
            "governing law": 0.8,
            "applicable law": 0.7,
            "legal proceedings": 0.7,
            "court of law": 0.7,
            "binding agreement": 0.8,
            "contract": 0.5,
            "agreement": 0.4,
            "liability": 0.6,
            "damages": 0.6,
            "indemnify": 0.8,
            "indemnification": 0.8,
            "breach": 0.6,
            "violation": 0.5,
            "enforce": 0.4,
            "enforcement": 0.5,
            "termination": 0.5,
            "terminate": 0.4,
            
            # Legal structure words
            "section": 0.3,
            "subsection": 0.4,
            "clause": 0.4,
            "provision": 0.5,
            "article": 0.3,
        }
    
    def _build_non_legal_indicators(self) -> Dict[str, float]:
        """Build non-legal document indicators with weights."""
        return {
            # Academic/Quiz indicators - ENHANCED
            "question": 1.0,
            "answer": 0.9,
            "multiple choice": 1.0,
            "choose the correct": 1.0,
            "select the best": 1.0,
            "which of the following": 1.0,
            "test question": 1.0,
            "exam": 1.0,
            "quiz": 1.0,
            "homework": 1.0,
            "assignment": 0.9,
            "study guide": 1.0,
            "grade": 0.8,
            "score": 0.6,
            "points": 0.5,
            "solve for": 0.9,
            "calculate": 0.8,
            "find the value": 0.9,
            "what is x": 0.9,
            "if x =": 0.9,
            "equation": 0.9,
            
            # Recipe indicators
            "ingredients": 0.9,
            "recipe": 0.9,
            "cooking": 0.8,
            "bake": 0.7,
            "tablespoon": 0.8,
            "teaspoon": 0.8,
            "cup": 0.6,
            "oven": 0.7,
            "preheat": 0.8,
            "mix": 0.5,
            "stir": 0.6,
            "serve": 0.5,
            "minutes": 0.3,
            "degrees": 0.4,
            
            # Story/Narrative indicators
            "once upon a time": 0.9,
            "the end": 0.7,
            "chapter": 0.6,
            "character": 0.6,
            "plot": 0.7,
            "story": 0.5,
            "tale": 0.6,
            "narrator": 0.7,
            "protagonist": 0.7,
            
            # Math/Science indicators - ENHANCED
            "formula": 0.9,
            "x equals": 1.0,
            "y equals": 1.0,
            "theorem": 1.0,
            "proof": 0.8,
            "hypothesis": 0.9,
            "experiment": 0.8,
            "coefficient": 0.9,
            "probability": 0.8,
            "algebra": 0.9,
            "geometry": 0.9,
            "calculus": 0.9,
            
            # Technical manual indicators
            "step 1": 0.6,
            "step 2": 0.6,
            "instruction": 0.5,
            "manual": 0.5,
            "tutorial": 0.6,
            "how to": 0.5,
            "troubleshooting": 0.6,
        }
    
    def _build_structure_patterns(self) -> Dict[str, float]:
        """Build document structure patterns."""
        return {
            # Legal document structure
            "numbered_sections": 0.6,  # 1., 2., 3. etc.
            "lettered_sections": 0.5,  # a), b), c) etc.
            "legal_headers": 0.7,      # "TERMS OF SERVICE", etc.
            "signature_blocks": 0.8,   # signature, date fields
            "effective_date": 0.7,     # "Effective Date:", etc.
            
            # Non-legal structure
            "question_numbering": -0.8,  # Q1, Q2, etc.
            "recipe_format": -0.8,       # ingredient lists
            "dialogue_format": -0.6,     # "He said:", etc.
        }
    
    def _score_legal_indicators(self, text_lower: str, text_words: List[str]) -> Tuple[float, List[str]]:
        """Score legal document indicators."""
        score = 0.0
        reasons = []
        indicators_found = []
        
        for indicator, weight in self.legal_document_indicators.items():
            if indicator in text_lower:
                score += weight
                indicators_found.append(indicator)
                
                # Bonus for multiple occurrences (diminishing returns)
                count = text_lower.count(indicator)
                if count > 1:
                    score += weight * 0.2 * min(count - 1, 3)
        
        # Normalize score
        normalized_score = min(1.0, score / 5.0)  # Normalize to 0-1 range
        
        if indicators_found:
            reasons.append(f"Found {len(indicators_found)} legal indicators")
            if len(indicators_found) >= 5:
                reasons.append("Strong legal language patterns detected")
        
        return normalized_score, reasons
    
    def _score_non_legal_indicators(self, text_lower: str, text_words: List[str]) -> Tuple[float, List[str]]:
        """Score non-legal document indicators."""
        score = 0.0
        flags = []
        
        for indicator, weight in self.non_legal_indicators.items():
            if indicator in text_lower:
                score += weight
                flags.append(indicator)
        
        # Check for question patterns - ENHANCED
        question_patterns = [
            r'\b\d+\.\s*[A-Z].*\?',     # "1. What is...?"
            r'\bq\d+[\.:;]\s',           # "Q1:", "Q1.", "Q1;"
            r'\bquestion\s*\d+[\.:;]',   # "Question 1:", "Question 2."
            r'\b[a-d]\)\s',             # "a) ", "b) ", "c) ", "d) "
            r'choose.*correct',          # "choose the correct"
            r'select.*best',             # "select the best"
            r'which.*following',         # "which of the following"
            r'solve\s+for\s+[xy]',       # "solve for x", "solve for y"
            r'if\s+[xy]\s*=',            # "if x =", "if y ="
            r'calculate.*value',         # "calculate the value"
            r'find\s+[xy]\s+when',       # "find x when", "find y when"
            r'mathematics\s+test',       # "Mathematics Test"
            r'chapter\s*\d+.*test',      # "Chapter 7 Test"
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                score += 0.8
                flags.append("question_format")
                break
        
        # Check for recipe patterns - ENHANCED
        recipe_patterns = [
            r'\d+\s*(?:cups?|tbsp|tsp|oz|lbs?|tablespoons?|teaspoons?)\s',  # measurements
            r'preheat.*(?:oven|degrees)',                                    # cooking instructions
            r'bake.*(?:minutes|hours)',                                      # timing
            r'ingredients?\s*:',                                            # ingredient lists
            r'instructions?\s*:',                                           # instruction lists
            r'\d+\s*degrees?\s*[fc]',                                       # temperature
            r'mix.*until.*combined',                                         # mixing instructions
            r'stir.*occasionally',                                           # stirring instructions
            r'serve.*hot',                                                   # serving instructions
        ]
        
        for pattern in recipe_patterns:
            if re.search(pattern, text_lower):
                score += 0.7
                flags.append("recipe_format")
                break
        
        # Check for academic/math content patterns - NEW
        academic_patterns = [
            r'\b(?:algebra|geometry|calculus|trigonometry)\b',  # math subjects
            r'\b(?:physics|chemistry|biology)\b',               # science subjects
            r'chapter\s*\d+',                                   # textbook chapters
            r'lesson\s*\d+',                                    # lesson numbers
            r'exercise\s*\d+',                                  # exercise numbers
            r'test\s*\d+',                                      # test numbers
            r'page\s*\d+',                                      # page references
        ]
        
        for pattern in academic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.8
                flags.append("academic_content")
                break
        
        # Check for story/narrative patterns - NEW
        narrative_patterns = [
            r'once upon a time',
            r'the end',
            r'chapter \d+',
            r'\b(?:he|she)\s+(?:said|asked|replied)',
            r'character named',
            r'protagonist',
            r'plot',
            r'story continues',
        ]
        
        for pattern in narrative_patterns:
            if re.search(pattern, text_lower):
                score += 0.7
                flags.append("narrative_content")
                break
        
        # Normalize score with higher threshold
        normalized_score = min(1.0, score / 2.5)  # More sensitive threshold
        
        return normalized_score, flags[:5]  # Limit flags
    
    def _analyze_document_structure(self, text: str) -> Tuple[float, List[str]]:
        """Analyze document structure patterns."""
        score = 0.0
        reasons = []
        
        # Check for legal document headers
        legal_headers = [
            r'^[A-Z\s]{10,}$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z][A-Za-z\s]+',  # Numbered sections
            r'^ARTICLE\s+[IVX]+',  # Article headers
            r'^SECTION\s+\d+',     # Section headers
        ]
        
        lines = text.split('\n')
        header_count = 0
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if len(line) > 10:
                for pattern in legal_headers:
                    if re.match(pattern, line):
                        header_count += 1
                        break
        
        if header_count >= 2:
            score += 0.6
            reasons.append("Legal document structure detected")
        
        # Check for signature blocks or effective dates
        signature_patterns = [
            r'signature.*date',
            r'effective\s+date',
            r'last\s+updated',
            r'version\s+\d+',
        ]
        
        text_lower = text.lower()
        for pattern in signature_patterns:
            if re.search(pattern, text_lower):
                score += 0.3
                reasons.append("Document metadata found")
                break
        
        return min(1.0, score), reasons
    
    def _classify_document_type(self, text_lower: str, legal_score: float, non_legal_score: float) -> DocumentType:
        """Classify the document type based on content analysis."""
        # Check for non-legal types FIRST - prioritize these
        if "quiz" in text_lower or ("question" in text_lower and non_legal_score > 0.5):
            return DocumentType.QUIZ
        elif "recipe" in text_lower or "ingredients" in text_lower or "preheat" in text_lower:
            return DocumentType.RECIPE
        elif ("chapter" in text_lower and ("story" in text_lower or "character" in text_lower or "protagonist" in text_lower)) or "once upon a time" in text_lower:
            return DocumentType.STORY
        elif any(word in text_lower for word in ["theorem", "equation", "proof", "algebra", "calculus", "mathematics"]):
            return DocumentType.ACADEMIC
        
        # Only check legal types if no strong non-legal indicators
        if non_legal_score > 0.4:  # High non-legal score = non-legal document
            return DocumentType.NON_LEGAL
        
        # Check for specific legal document types
        if ("terms of service" in text_lower or "terms of use" in text_lower) and non_legal_score < 0.3:
            return DocumentType.TERMS_OF_SERVICE
        elif "privacy policy" in text_lower and non_legal_score < 0.3:
            return DocumentType.PRIVACY_POLICY
        elif any(word in text_lower for word in ["agreement", "contract"]) and legal_score > 0.6 and non_legal_score < 0.2:
            return DocumentType.CONTRACT
        
        # Determine based on scores with stricter thresholds
        if legal_score > 0.7 and non_legal_score < 0.2:
            return DocumentType.LEGAL_DOCUMENT
        elif non_legal_score > 0.3:
            return DocumentType.NON_LEGAL
        else:
            return DocumentType.UNKNOWN
    
    def _calculate_confidence(self, legal_score: float, non_legal_score: float, 
                            structure_score: float, word_count: int) -> float:
        """Calculate overall confidence in document classification."""
        # Base confidence from scores
        if legal_score > non_legal_score:
            base_confidence = legal_score
        else:
            base_confidence = 1.0 - non_legal_score
        
        # Adjust for structure
        confidence = (base_confidence + structure_score) / 2
        
        # Adjust for document length
        if word_count < 100:
            confidence *= 0.7  # Lower confidence for short docs
        elif word_count > 1000:
            confidence = min(1.0, confidence * 1.1)  # Boost for longer docs
        
        return min(1.0, max(0.0, confidence))
    
    def _is_legal_document(self, legal_score: float, non_legal_score: float, 
                          confidence: float, document_type: DocumentType) -> bool:
        """Determine if document is suitable for legal analysis."""
        # Explicit non-legal types - ALWAYS reject these
        if document_type in [DocumentType.NON_LEGAL, DocumentType.QUIZ, 
                            DocumentType.RECIPE, DocumentType.STORY, DocumentType.ACADEMIC]:
            return False
        
        # Hard rejection if non-legal score is too high - NEW
        if non_legal_score > 0.4:  # If non-legal indicators are strong, reject
            return False
        
        # Strong legal indicators - but still check non-legal score
        if document_type in [DocumentType.TERMS_OF_SERVICE, DocumentType.PRIVACY_POLICY, 
                            DocumentType.LEGAL_DOCUMENT]:
            # Even for strong legal types, reject if non-legal score is high
            return non_legal_score <= self.max_non_legal_score
        
        # For contracts and agreements, be more careful
        if document_type in [DocumentType.CONTRACT, DocumentType.AGREEMENT]:
            # These could be recipe or story contracts, so be stricter
            return (
                legal_score >= 0.7 and  # Much higher threshold
                non_legal_score <= 0.15 and  # Very low non-legal tolerance
                confidence >= 0.8 and
                legal_score > (non_legal_score * 3)  # 3x legal vs non-legal
            )
        
        # Score-based determination - ENHANCED with stricter requirements
        return (
            legal_score >= 0.6 and  # Increased from 0.5
            non_legal_score <= 0.15 and  # Much stricter non-legal threshold
            confidence >= self.min_confidence_threshold and
            legal_score > (non_legal_score * 3)  # Legal score must be 3x non-legal score
        )
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get a human-readable summary of validation result."""
        summary = f"Document Type: {result.document_type.value.replace('_', ' ').title()}\n"
        summary += f"Legal Document: {'Yes' if result.is_legal_document else 'No'}\n"
        summary += f"Confidence: {result.confidence:.2f}\n"
        
        if result.reasons:
            summary += f"Reasons: {'; '.join(result.reasons[:3])}\n"
        
        if result.warning_flags:
            summary += f"Warning Flags: {', '.join(result.warning_flags[:3])}\n"
        
        return summary