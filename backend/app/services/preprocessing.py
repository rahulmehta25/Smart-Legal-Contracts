"""
Text Preprocessing Service

Comprehensive text preprocessing for extracted documents:
- Text cleaning and normalization
- Section detection and structure analysis
- Paragraph and sentence segmentation
- Language detection and handling
- Encoding normalization
- Legal document structure recognition
- Content quality assessment
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import string

# Language detection
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import polyglot
from polyglot.detect import Detector, LangDetectException as PolyglotLangDetectException

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, LineTokenizer
from nltk.corpus import stopwords
import ftfy  # Fix text encoding issues
import chardet
import unidecode

# Statistical analysis
import numpy as np
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ContentType(Enum):
    """Document content types"""
    LEGAL_CONTRACT = "legal_contract"
    LEGAL_TERMS = "legal_terms"
    TECHNICAL_DOCUMENT = "technical_document"
    BUSINESS_DOCUMENT = "business_document"
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    GENERAL_TEXT = "general_text"
    UNKNOWN = "unknown"


class TextQuality(Enum):
    """Text quality levels"""
    EXCELLENT = "excellent"  # >95% readable text
    GOOD = "good"           # 80-95% readable text
    FAIR = "fair"           # 60-80% readable text
    POOR = "poor"           # <60% readable text


@dataclass
class TextSection:
    """A logical section of text"""
    title: Optional[str]
    content: str
    level: int  # Hierarchy level (1=top level, 2=subsection, etc.)
    section_type: str  # header, paragraph, list, table, etc.
    position: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextStatistics:
    """Text statistical analysis"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    character_count: int
    avg_words_per_sentence: float
    avg_sentences_per_paragraph: float
    readability_score: float
    lexical_diversity: float
    most_common_words: List[Tuple[str, int]]
    language_confidence: float


@dataclass
class PreprocessingResult:
    """Complete preprocessing result"""
    cleaned_text: str
    sections: List[TextSection]
    statistics: TextStatistics
    detected_language: str
    content_type: ContentType
    text_quality: TextQuality
    encoding_issues_fixed: List[str]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextPreprocessor:
    """Advanced text preprocessing and analysis"""
    
    def __init__(self, 
                 target_language: Optional[str] = None,
                 clean_ocr_artifacts: bool = True,
                 normalize_whitespace: bool = True,
                 remove_boilerplate: bool = True):
        """
        Initialize text preprocessor
        
        Args:
            target_language: Expected language (for better processing)
            clean_ocr_artifacts: Clean common OCR errors
            normalize_whitespace: Normalize spacing and line breaks
            remove_boilerplate: Remove common boilerplate text
        """
        self.target_language = target_language
        self.clean_ocr_artifacts = clean_ocr_artifacts
        self.normalize_whitespace = normalize_whitespace
        self.remove_boilerplate = remove_boilerplate
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Legal document patterns
        self.legal_patterns = [
            r'arbitration',
            r'terms of service',
            r'privacy policy',
            r'user agreement',
            r'license agreement',
            r'terms and conditions',
            r'dispute resolution',
            r'governing law',
            r'jurisdiction',
            r'waiver',
            r'liability',
            r'damages',
            r'indemnification'
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning"""
        
        # OCR artifacts
        self.ocr_patterns = [
            (re.compile(r'\s+'), ' '),  # Multiple spaces
            (re.compile(r'([a-z])([A-Z])'), r'\1 \2'),  # Missing spaces between words
            (re.compile(r'(\w)([.!?])(\w)'), r'\1\2 \3'),  # Missing spaces after punctuation
            (re.compile(r'[|]'), 'l'),  # Common OCR error: | instead of l
            (re.compile(r'[0O](?=[a-z])'), 'o'),  # 0 or O instead of o
            (re.compile(r'(?<=[a-z])[1I](?=[a-z])'), 'l'),  # 1 or I instead of l
            (re.compile(r'\s*-\s*\n\s*'), ''),  # Remove hyphenation across lines
        ]
        
        # Section headers
        self.header_patterns = [
            re.compile(r'^(?:ARTICLE|SECTION|CHAPTER)\s+[IVX\d]+\.?\s*(.+)', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^(\d+\.(?:\d+\.)*)\s*(.+)', re.MULTILINE),
            re.compile(r'^([A-Z][A-Z\s]{3,}):?\s*$', re.MULTILINE),  # ALL CAPS headers
        ]
        
        # Boilerplate patterns
        self.boilerplate_patterns = [
            re.compile(r'Page \d+ of \d+', re.IGNORECASE),
            re.compile(r'Confidential and Proprietary', re.IGNORECASE),
            re.compile(r'© \d{4}.*?All rights reserved\.?', re.IGNORECASE),
            re.compile(r'\[END OF DOCUMENT\]', re.IGNORECASE),
            re.compile(r'Generated on \d{1,2}\/\d{1,2}\/\d{4}', re.IGNORECASE),
        ]
    
    def preprocess(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> PreprocessingResult:
        """
        Preprocess text with comprehensive cleaning and analysis
        
        Args:
            text: Input text to preprocess
            metadata: Optional metadata about the document
            
        Returns:
            PreprocessingResult with cleaned text and analysis
        """
        
        if not text or not text.strip():
            return self._empty_result()
        
        original_text = text
        encoding_issues_fixed = []
        warnings = []
        
        # Fix encoding issues
        if self._has_encoding_issues(text):
            text = ftfy.fix_text(text)
            encoding_issues_fixed.append("Fixed text encoding issues")
        
        # Detect language
        detected_language, lang_confidence = self._detect_language(text)
        if not detected_language:
            warnings.append("Could not detect document language")
            detected_language = self.target_language or 'en'
            lang_confidence = 0.0
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract sections
        sections = self._extract_sections(cleaned_text)
        
        # Analyze content type
        content_type = self._detect_content_type(cleaned_text, metadata)
        
        # Calculate statistics
        statistics = self._calculate_statistics(cleaned_text, detected_language, lang_confidence)
        
        # Assess text quality
        text_quality = self._assess_text_quality(cleaned_text, statistics)
        
        return PreprocessingResult(
            cleaned_text=cleaned_text,
            sections=sections,
            statistics=statistics,
            detected_language=detected_language,
            content_type=content_type,
            text_quality=text_quality,
            encoding_issues_fixed=encoding_issues_fixed,
            warnings=warnings,
            metadata=metadata or {}
        )
    
    def _has_encoding_issues(self, text: str) -> bool:
        """Check if text has encoding issues"""
        
        # Check for common encoding issues
        encoding_indicators = [
            'â€™',  # Smart quote issues
            'â€œ',  # Smart quote issues
            'â€',   # General encoding issues
            'Ã',    # Latin-1 to UTF-8 issues
            '\\x',  # Escaped characters
            '\\u',  # Unicode escapes
        ]
        
        return any(indicator in text for indicator in encoding_indicators)
    
    def _detect_language(self, text: str) -> Tuple[Optional[str], float]:
        """Detect text language with confidence score"""
        
        # Use langdetect first (faster)
        try:
            detected = langdetect.detect_langs(text)
            if detected:
                return detected[0].lang, detected[0].prob
        except LangDetectException:
            pass
        
        # Fallback to polyglot
        try:
            detector = Detector(text)
            return detector.language.code, detector.language.confidence
        except PolyglotLangDetectException:
            pass
        
        return None, 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        cleaned = text
        
        # Fix OCR artifacts
        if self.clean_ocr_artifacts:
            for pattern, replacement in self.ocr_patterns:
                cleaned = pattern.sub(replacement, cleaned)
        
        # Remove boilerplate
        if self.remove_boilerplate:
            for pattern in self.boilerplate_patterns:
                cleaned = pattern.sub('', cleaned)
        
        # Normalize unicode
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            # Fix common line break issues
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple empty lines
            cleaned = re.sub(r'(?<=[.!?])\n(?=[A-Z])', ' ', cleaned)  # Line breaks in sentences
            cleaned = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', cleaned)  # Line breaks in words
            
            # Clean up spacing
            cleaned = re.sub(r'\t+', ' ', cleaned)  # Tabs to spaces
            cleaned = re.sub(r' +', ' ', cleaned)   # Multiple spaces
            cleaned = re.sub(r'^ +| +$', '', cleaned, flags=re.MULTILINE)  # Leading/trailing spaces
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        return cleaned.strip()
    
    def _extract_sections(self, text: str) -> List[TextSection]:
        """Extract logical sections from text"""
        
        sections = []
        position = 0
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a header
            header_match = self._find_header(para)
            if header_match:
                sections.append(TextSection(
                    title=header_match['title'],
                    content=para,
                    level=header_match['level'],
                    section_type='header',
                    position=position,
                    word_count=len(para.split()),
                    metadata={'pattern': header_match['pattern']}
                ))
            else:
                # Check if it's a list
                if self._is_list_item(para):
                    sections.append(TextSection(
                        title=None,
                        content=para,
                        level=0,
                        section_type='list',
                        position=position,
                        word_count=len(para.split())
                    ))
                else:
                    # Regular paragraph
                    sections.append(TextSection(
                        title=None,
                        content=para,
                        level=0,
                        section_type='paragraph',
                        position=position,
                        word_count=len(para.split())
                    ))
            
            position += 1
        
        return sections
    
    def _find_header(self, text: str) -> Optional[Dict[str, Any]]:
        """Find if text is a header"""
        
        text_line = text.split('\n')[0].strip()  # Use first line
        
        # Check against header patterns
        for i, pattern in enumerate(self.header_patterns):
            match = pattern.match(text_line)
            if match:
                if len(match.groups()) >= 2:
                    level = len(match.group(1).split('.')) if '.' in match.group(1) else 1
                    title = match.group(2).strip()
                else:
                    level = 1
                    title = match.group(1).strip() if match.groups() else text_line
                
                return {
                    'title': title,
                    'level': level,
                    'pattern': f'pattern_{i}'
                }
        
        # Check if line is very short and looks like a header
        if (len(text_line) < 100 and 
            len(text_line.split()) < 10 and
            not text_line.endswith('.') and
            not text_line.endswith(',') and
            (text_line.isupper() or text_line.istitle())):
            
            return {
                'title': text_line,
                'level': 1,
                'pattern': 'short_title'
            }
        
        return None
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text appears to be a list item"""
        
        list_patterns = [
            r'^\s*[-•▪▫◦‣⁃]\s+',  # Bullet points
            r'^\s*\d+\.\s+',       # Numbered lists
            r'^\s*[a-z]\.\s+',     # Lettered lists
            r'^\s*[ivx]+\.\s+',    # Roman numerals
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_content_type(self, text: str, metadata: Optional[Dict[str, Any]]) -> ContentType:
        """Detect the type of document content"""
        
        text_lower = text.lower()
        
        # Check for legal document patterns
        legal_score = sum(1 for pattern in self.legal_patterns 
                         if re.search(pattern, text_lower))
        
        if legal_score >= 3:
            if any(term in text_lower for term in ['terms of service', 'user agreement', 'terms and conditions']):
                return ContentType.LEGAL_TERMS
            else:
                return ContentType.LEGAL_CONTRACT
        
        # Check for technical content
        technical_terms = [
            'api', 'algorithm', 'software', 'system', 'database', 'server',
            'implementation', 'configuration', 'protocol', 'framework'
        ]
        technical_score = sum(1 for term in technical_terms if term in text_lower)
        
        if technical_score >= 3:
            return ContentType.TECHNICAL_DOCUMENT
        
        # Check for academic content
        academic_terms = [
            'abstract', 'methodology', 'hypothesis', 'research', 'study',
            'analysis', 'conclusion', 'bibliography', 'references'
        ]
        academic_score = sum(1 for term in academic_terms if term in text_lower)
        
        if academic_score >= 3:
            return ContentType.ACADEMIC_PAPER
        
        # Check for business content
        business_terms = [
            'company', 'business', 'market', 'revenue', 'profit', 'customer',
            'service', 'product', 'strategy', 'management'
        ]
        business_score = sum(1 for term in business_terms if term in text_lower)
        
        if business_score >= 3:
            return ContentType.BUSINESS_DOCUMENT
        
        return ContentType.GENERAL_TEXT
    
    def _calculate_statistics(self, text: str, language: str, lang_confidence: float) -> TextStatistics:
        """Calculate comprehensive text statistics"""
        
        # Basic counts
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        character_count = len(text)
        
        # Calculate averages
        avg_words_per_sentence = word_count / max(1, sentence_count)
        avg_sentences_per_paragraph = sentence_count / max(1, paragraph_count)
        
        # Calculate lexical diversity (Type-Token Ratio)
        unique_words = set(word.lower() for word in words if word.isalpha())
        lexical_diversity = len(unique_words) / max(1, len([w for w in words if w.isalpha()]))
        
        # Calculate readability score (simplified)
        readability_score = self._calculate_readability(text, words, sentences)
        
        # Most common words (excluding stopwords)
        try:
            stop_words = set(stopwords.words(language if language in stopwords.fileids() else 'english'))
        except:
            stop_words = set()
        
        content_words = [word.lower() for word in words 
                        if word.isalpha() and word.lower() not in stop_words]
        most_common = Counter(content_words).most_common(10)
        
        return TextStatistics(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            character_count=character_count,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_sentences_per_paragraph=avg_sentences_per_paragraph,
            readability_score=readability_score,
            lexical_diversity=lexical_diversity,
            most_common_words=most_common,
            language_confidence=lang_confidence
        )
    
    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        
        if not words or not sentences:
            return 0.0
        
        # Count syllables (approximation)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count += 1
            return count
        
        total_syllables = sum(count_syllables(word) for word in words if word.isalpha())
        
        # Flesch Reading Ease formula
        asl = len(words) / len(sentences)  # Average sentence length
        asw = total_syllables / len(words)  # Average syllables per word
        
        score = 206.835 - (1.015 * asl) - (84.6 * asw)
        return max(0, min(100, score))  # Clamp to 0-100
    
    def _assess_text_quality(self, text: str, statistics: TextStatistics) -> TextQuality:
        """Assess overall text quality"""
        
        # Quality indicators
        quality_score = 0
        total_checks = 0
        
        # Check readability
        if statistics.readability_score > 60:
            quality_score += 1
        total_checks += 1
        
        # Check lexical diversity
        if statistics.lexical_diversity > 0.4:
            quality_score += 1
        total_checks += 1
        
        # Check for reasonable sentence length
        if 10 <= statistics.avg_words_per_sentence <= 25:
            quality_score += 1
        total_checks += 1
        
        # Check for reasonable paragraph structure
        if 2 <= statistics.avg_sentences_per_paragraph <= 8:
            quality_score += 1
        total_checks += 1
        
        # Check for presence of structure
        if statistics.paragraph_count > 1:
            quality_score += 1
        total_checks += 1
        
        # Check for OCR artifacts or gibberish
        gibberish_ratio = self._calculate_gibberish_ratio(text)
        if gibberish_ratio < 0.1:
            quality_score += 1
        total_checks += 1
        
        quality_percentage = quality_score / total_checks
        
        if quality_percentage >= 0.9:
            return TextQuality.EXCELLENT
        elif quality_percentage >= 0.7:
            return TextQuality.GOOD
        elif quality_percentage >= 0.5:
            return TextQuality.FAIR
        else:
            return TextQuality.POOR
    
    def _calculate_gibberish_ratio(self, text: str) -> float:
        """Calculate ratio of gibberish/noise in text"""
        
        words = word_tokenize(text)
        if not words:
            return 1.0
        
        gibberish_indicators = 0
        
        for word in words:
            if not word.isalpha():
                continue
            
            # Check for common OCR/gibberish patterns
            if (len(word) == 1 or 
                re.search(r'[0-9]', word) or
                len(set(word)) == 1 or  # Repeated character
                not re.search(r'[aeiou]', word.lower()) and len(word) > 3):  # No vowels
                gibberish_indicators += 1
        
        alpha_words = [w for w in words if w.isalpha()]
        return gibberish_indicators / max(1, len(alpha_words))
    
    def _empty_result(self) -> PreprocessingResult:
        """Return empty preprocessing result"""
        
        return PreprocessingResult(
            cleaned_text="",
            sections=[],
            statistics=TextStatistics(
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                character_count=0,
                avg_words_per_sentence=0.0,
                avg_sentences_per_paragraph=0.0,
                readability_score=0.0,
                lexical_diversity=0.0,
                most_common_words=[],
                language_confidence=0.0
            ),
            detected_language='unknown',
            content_type=ContentType.UNKNOWN,
            text_quality=TextQuality.POOR,
            encoding_issues_fixed=[],
            warnings=["Empty or invalid text input"]
        )


# Convenience function
def preprocess_text(text: str, 
                   target_language: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> PreprocessingResult:
    """
    Preprocess text with default settings
    
    Args:
        text: Text to preprocess
        target_language: Expected language
        metadata: Optional document metadata
        
    Returns:
        PreprocessingResult with cleaned and analyzed text
    """
    preprocessor = TextPreprocessor(target_language=target_language)
    return preprocessor.preprocess(text, metadata)