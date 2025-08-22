"""
Multi-language support for arbitration clause detection.

This module provides language detection, translation pipelines, and 
language-specific legal terminology mappings for global arbitration
clause detection across different legal systems.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from functools import lru_cache
import time

# Language detection
from langdetect import detect, LangDetectError
from langdetect.detector import PROFILES_DIRECTORY
import polyglot
from polyglot.detect import Detector

# Translation
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    MarianMTModel, MarianTokenizer,
    pipeline
)
import torch

# Google Translate fallback
try:
    from google.cloud import translate_v2 as translate
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False

# Configuration
import yaml
from pathlib import Path


logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for arbitration clause detection."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    JAPANESE = "ja"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    KOREAN = "ko"


@dataclass
class DetectionResult:
    """Result of language detection."""
    language: str
    confidence: float
    detected_by: str
    is_supported: bool
    legal_system: Optional[str] = None


@dataclass
class TranslationResult:
    """Result of text translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    translation_method: str
    preserved_terms: List[str]
    processing_time: float


class LanguageDetector:
    """Advanced language detection with multiple fallback methods."""
    
    def __init__(self):
        self.cache = {}
        self.cache_max_size = 1000
        
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> DetectionResult:
        """
        Detect language using multiple methods with fallbacks.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with language, confidence, and metadata
        """
        if len(text.strip()) < 10:
            return DetectionResult(
                language="unknown",
                confidence=0.0,
                detected_by="insufficient_text",
                is_supported=False
            )
        
        # Primary detection with langdetect
        try:
            detected_lang = detect(text)
            confidence = self._calculate_langdetect_confidence(text, detected_lang)
            
            return DetectionResult(
                language=detected_lang,
                confidence=confidence,
                detected_by="langdetect",
                is_supported=self._is_supported_language(detected_lang),
                legal_system=self._get_legal_system(detected_lang)
            )
        except LangDetectError:
            logger.warning("langdetect failed, trying polyglot")
        
        # Fallback to polyglot
        try:
            detector = Detector(text)
            detected_lang = detector.language.code
            confidence = detector.language.confidence
            
            return DetectionResult(
                language=detected_lang,
                confidence=confidence,
                detected_by="polyglot",
                is_supported=self._is_supported_language(detected_lang),
                legal_system=self._get_legal_system(detected_lang)
            )
        except Exception as e:
            logger.warning(f"Polyglot detection failed: {e}")
        
        # Character-based detection fallback
        return self._character_based_detection(text)
    
    def _calculate_langdetect_confidence(self, text: str, language: str) -> float:
        """Calculate confidence score for langdetect result."""
        try:
            from langdetect import detect_langs
            probabilities = detect_langs(text)
            for prob in probabilities:
                if prob.lang == language:
                    return prob.prob
        except:
            pass
        return 0.7  # Default confidence
    
    def _character_based_detection(self, text: str) -> DetectionResult:
        """Basic character-based language detection."""
        char_counts = {
            'chinese': 0,
            'japanese': 0,
            'arabic': 0,
            'cyrillic': 0,
            'latin': 0
        }
        
        for char in text:
            code = ord(char)
            if 0x4e00 <= code <= 0x9fff:  # CJK Unified Ideographs
                char_counts['chinese'] += 1
            elif 0x3040 <= code <= 0x309f or 0x30a0 <= code <= 0x30ff:  # Hiragana/Katakana
                char_counts['japanese'] += 1
            elif 0x0600 <= code <= 0x06ff:  # Arabic
                char_counts['arabic'] += 1
            elif 0x0400 <= code <= 0x04ff:  # Cyrillic
                char_counts['cyrillic'] += 1
            elif 0x0020 <= code <= 0x007f:  # Basic Latin
                char_counts['latin'] += 1
        
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return DetectionResult("unknown", 0.0, "character_based", False)
        
        max_script = max(char_counts, key=char_counts.get)
        confidence = char_counts[max_script] / total_chars
        
        script_to_lang = {
            'chinese': 'zh',
            'japanese': 'ja',
            'arabic': 'ar',
            'cyrillic': 'ru',
            'latin': 'en'  # Default to English for Latin script
        }
        
        detected_lang = script_to_lang.get(max_script, 'unknown')
        
        return DetectionResult(
            language=detected_lang,
            confidence=confidence,
            detected_by="character_based",
            is_supported=self._is_supported_language(detected_lang),
            legal_system=self._get_legal_system(detected_lang)
        )
    
    def _is_supported_language(self, language: str) -> bool:
        """Check if language is supported."""
        supported_codes = [lang.value for lang in SupportedLanguage]
        return language in supported_codes or language.split('-')[0] in supported_codes
    
    def _get_legal_system(self, language: str) -> Optional[str]:
        """Get legal system associated with language."""
        legal_systems = {
            'en': 'common_law',
            'es': 'civil_law',
            'fr': 'civil_law',
            'de': 'civil_law',
            'zh': 'civil_law',
            'ja': 'civil_law',
            'pt': 'civil_law',
            'it': 'civil_law',
            'ru': 'civil_law',
            'ar': 'islamic_law',
            'ko': 'civil_law'
        }
        base_lang = language.split('-')[0]
        return legal_systems.get(base_lang)


class TranslationPipeline:
    """High-performance translation pipeline with caching and batch processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.models = {}
        self.tokenizers = {}
        self.translation_cache = {}
        self.cache_max_size = 10000
        self.config = self._load_config(config_path)
        self.google_client = self._init_google_translate()
        
        # Initialize commonly used models
        self._preload_models()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load translation configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'models': {
                'marian': {
                    'en-es': 'Helsinki-NLP/opus-mt-en-es',
                    'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
                    'en-de': 'Helsinki-NLP/opus-mt-en-de',
                    'es-en': 'Helsinki-NLP/opus-mt-es-en',
                    'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
                    'de-en': 'Helsinki-NLP/opus-mt-de-en'
                },
                'multilingual': 'facebook/mbart-large-50-many-to-many-mmt'
            },
            'fallback_to_google': True,
            'max_length': 512,
            'batch_size': 8
        }
    
    def _init_google_translate(self):
        """Initialize Google Translate client if available."""
        if GOOGLE_TRANSLATE_AVAILABLE and self.config.get('google_translate_api_key'):
            try:
                return translate.Client(api_key=self.config['google_translate_api_key'])
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate: {e}")
        return None
    
    def _preload_models(self):
        """Preload commonly used translation models."""
        common_pairs = ['en-es', 'en-fr', 'en-de', 'es-en', 'fr-en', 'de-en']
        
        for pair in common_pairs:
            try:
                self._load_marian_model(pair)
            except Exception as e:
                logger.warning(f"Failed to preload model for {pair}: {e}")
    
    def _load_marian_model(self, language_pair: str) -> Tuple[Any, Any]:
        """Load Marian translation model and tokenizer."""
        if language_pair in self.models:
            return self.models[language_pair], self.tokenizers[language_pair]
        
        model_name = self.config['models']['marian'].get(language_pair)
        if not model_name:
            raise ValueError(f"No model configured for language pair: {language_pair}")
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.models[language_pair] = model
            self.tokenizers[language_pair] = tokenizer
            
            logger.info(f"Loaded Marian model for {language_pair}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load Marian model for {language_pair}: {e}")
            raise
    
    def _cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{text}|{source_lang}|{target_lang}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def translate_text(
        self, 
        text: str, 
        source_language: str, 
        target_language: str = "en",
        preserve_legal_terms: bool = True
    ) -> TranslationResult:
        """
        Translate text with legal term preservation.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            preserve_legal_terms: Whether to preserve legal terminology
            
        Returns:
            TranslationResult with translation and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._cache_key(text, source_language, target_language)
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        # Clean up cache if too large
        if len(self.translation_cache) > self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.translation_cache.keys())[:100]
            for key in keys_to_remove:
                del self.translation_cache[key]
        
        preserved_terms = []
        processed_text = text
        
        if preserve_legal_terms:
            processed_text, preserved_terms = self._preserve_legal_terms(text, source_language)
        
        # Try translation methods in order of preference
        translation_methods = [
            self._translate_with_marian,
            self._translate_with_multilingual_model,
            self._translate_with_google
        ]
        
        translation_result = None
        for method in translation_methods:
            try:
                translation_result = await method(
                    processed_text, source_language, target_language
                )
                break
            except Exception as e:
                logger.warning(f"Translation method {method.__name__} failed: {e}")
                continue
        
        if not translation_result:
            # Fallback to identity translation
            translation_result = TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.1,
                translation_method="identity",
                preserved_terms=preserved_terms,
                processing_time=time.time() - start_time
            )
        else:
            translation_result.processing_time = time.time() - start_time
            translation_result.preserved_terms = preserved_terms
        
        # Restore preserved terms
        if preserve_legal_terms and preserved_terms:
            translation_result.translated_text = self._restore_legal_terms(
                translation_result.translated_text, preserved_terms
            )
        
        # Cache the result
        self.translation_cache[cache_key] = translation_result
        
        return translation_result
    
    async def _translate_with_marian(
        self, text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """Translate using Marian models."""
        language_pair = f"{source_lang}-{target_lang}"
        
        try:
            model, tokenizer = self._load_marian_model(language_pair)
        except:
            raise ValueError(f"Marian model not available for {language_pair}")
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.8,  # Default confidence for Marian
            translation_method="marian",
            preserved_terms=[],
            processing_time=0.0
        )
    
    async def _translate_with_multilingual_model(
        self, text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """Translate using multilingual model (mBart)."""
        model_name = self.config['models']['multilingual']
        
        try:
            translator = pipeline(
                "translation",
                model=model_name,
                tokenizer=model_name,
                src_lang=source_lang,
                tgt_lang=target_lang,
                device=0 if torch.cuda.is_available() else -1
            )
            
            result = translator(text, max_length=512)
            translated_text = result[0]['translation_text']
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.75,
                translation_method="multilingual",
                preserved_terms=[],
                processing_time=0.0
            )
        except Exception as e:
            raise ValueError(f"Multilingual translation failed: {e}")
    
    async def _translate_with_google(
        self, text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """Translate using Google Translate API."""
        if not self.google_client:
            raise ValueError("Google Translate not available")
        
        try:
            result = self.google_client.translate(
                text,
                source_language=source_lang,
                target_language=target_lang
            )
            
            confidence = getattr(result, 'confidence', 0.7)
            
            return TranslationResult(
                original_text=text,
                translated_text=result['translatedText'],
                source_language=source_lang,
                target_language=target_lang,
                confidence=confidence,
                translation_method="google",
                preserved_terms=[],
                processing_time=0.0
            )
        except Exception as e:
            raise ValueError(f"Google Translate failed: {e}")
    
    def _preserve_legal_terms(self, text: str, language: str) -> Tuple[str, List[str]]:
        """Preserve legal terms during translation."""
        # This would be expanded with comprehensive legal term dictionaries
        legal_terms = {
            'en': ['arbitration', 'jurisdiction', 'binding', 'dispute resolution', 'mediation'],
            'es': ['arbitraje', 'jurisdicción', 'vinculante', 'resolución de disputas'],
            'fr': ['arbitrage', 'juridiction', 'contraignant', 'résolution des différends'],
            'de': ['Schiedsverfahren', 'Gerichtsbarkeit', 'bindend', 'Streitbeilegung'],
            'zh': ['仲裁', '管辖权', '有约束力', '争议解决'],
            'ja': ['仲裁', '管轄権', '拘束力', '紛争解決']
        }
        
        terms_to_preserve = legal_terms.get(language, [])
        preserved_terms = []
        processed_text = text
        
        for i, term in enumerate(terms_to_preserve):
            if term.lower() in text.lower():
                placeholder = f"__LEGAL_TERM_{i}__"
                processed_text = processed_text.replace(term, placeholder)
                preserved_terms.append((placeholder, term))
        
        return processed_text, preserved_terms
    
    def _restore_legal_terms(self, text: str, preserved_terms: List[Tuple[str, str]]) -> str:
        """Restore preserved legal terms after translation."""
        for placeholder, original_term in preserved_terms:
            text = text.replace(placeholder, original_term)
        return text
    
    async def translate_batch(
        self, 
        texts: List[str], 
        source_language: str, 
        target_language: str = "en"
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch for better performance.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of TranslationResult objects
        """
        batch_size = self.config.get('batch_size', 8)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [
                self.translate_text(text, source_language, target_language)
                for text in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results


class MultilingualProcessor:
    """Main processor for multi-language arbitration clause detection."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.language_detector = LanguageDetector()
        self.translation_pipeline = TranslationPipeline(config_path)
        self.legal_terminology = self._load_legal_terminology()
        
    def _load_legal_terminology(self) -> Dict[str, Dict[str, List[str]]]:
        """Load language-specific legal terminology mappings."""
        return {
            'arbitration_keywords': {
                'en': [
                    'arbitration', 'arbitrator', 'arbitral', 'binding arbitration',
                    'dispute resolution', 'AAA', 'JAMS', 'ICC arbitration',
                    'final and binding', 'arbitration clause', 'arbitration agreement'
                ],
                'es': [
                    'arbitraje', 'árbitro', 'arbitral', 'arbitraje vinculante',
                    'resolución de disputas', 'resolución de conflictos',
                    'tribunal de arbitraje', 'laudo arbitral'
                ],
                'fr': [
                    'arbitrage', 'arbitre', 'arbitral', 'arbitrage contraignant',
                    'résolution des différends', 'tribunal arbitral',
                    'sentence arbitrale', 'clause d\'arbitrage'
                ],
                'de': [
                    'Schiedsverfahren', 'Schiedsrichter', 'schiedsgerichtlich',
                    'bindendes Schiedsverfahren', 'Streitbeilegung',
                    'Schiedsgericht', 'Schiedsspruch'
                ],
                'zh': [
                    '仲裁', '仲裁员', '仲裁庭', '约束性仲裁',
                    '争议解决', '仲裁条款', '仲裁协议', '仲裁裁决'
                ],
                'ja': [
                    '仲裁', '仲裁人', '仲裁廷', '拘束力のある仲裁',
                    '紛争解決', '仲裁条項', '仲裁合意', '仲裁判断'
                ]
            },
            'jurisdiction_keywords': {
                'en': [
                    'jurisdiction', 'governing law', 'applicable law',
                    'court', 'legal proceedings', 'exclusive jurisdiction'
                ],
                'es': [
                    'jurisdicción', 'ley aplicable', 'ley rectora',
                    'tribunal', 'procedimientos legales'
                ],
                'fr': [
                    'juridiction', 'loi applicable', 'droit applicable',
                    'tribunal', 'procédures judiciaires'
                ],
                'de': [
                    'Gerichtsbarkeit', 'anwendbares Recht', 'maßgebliches Recht',
                    'Gericht', 'Gerichtsverfahren'
                ],
                'zh': [
                    '管辖权', '适用法律', '管辖法律',
                    '法院', '法律程序'
                ],
                'ja': [
                    '管轄権', '準拠法', '適用法',
                    '裁判所', '法的手続き'
                ]
            }
        }
    
    async def process_document(
        self, 
        text: str, 
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process a document for arbitration clause detection across languages.
        
        Args:
            text: Document text to process
            target_language: Target language for analysis
            
        Returns:
            Dict containing detection results, translation info, and analysis
        """
        # Detect source language
        detection_result = self.language_detector.detect_language(text)
        
        # If already in target language, process directly
        if detection_result.language == target_language:
            return {
                'source_language': detection_result.language,
                'target_language': target_language,
                'translation_needed': False,
                'detection_result': detection_result,
                'original_text': text,
                'processed_text': text,
                'legal_keywords': self._extract_keywords(text, detection_result.language),
                'confidence_score': detection_result.confidence
            }
        
        # Translate to target language
        translation_result = await self.translation_pipeline.translate_text(
            text, 
            detection_result.language, 
            target_language,
            preserve_legal_terms=True
        )
        
        return {
            'source_language': detection_result.language,
            'target_language': target_language,
            'translation_needed': True,
            'detection_result': detection_result,
            'translation_result': translation_result,
            'original_text': text,
            'processed_text': translation_result.translated_text,
            'legal_keywords': self._extract_keywords(
                translation_result.translated_text, 
                target_language
            ),
            'confidence_score': min(
                detection_result.confidence, 
                translation_result.confidence
            )
        }
    
    def _extract_keywords(self, text: str, language: str) -> Dict[str, List[str]]:
        """Extract legal keywords from text in specified language."""
        text_lower = text.lower()
        found_keywords = {}
        
        for category, lang_keywords in self.legal_terminology.items():
            keywords = lang_keywords.get(language, [])
            found = [keyword for keyword in keywords if keyword.lower() in text_lower]
            if found:
                found_keywords[category] = found
        
        return found_keywords
    
    async def process_batch(
        self, 
        documents: List[str], 
        target_language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document texts
            target_language: Target language for analysis
            
        Returns:
            List of processing results
        """
        tasks = [
            self.process_document(doc, target_language) 
            for doc in documents
        ]
        return await asyncio.gather(*tasks)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        return [
            {
                'code': lang.value,
                'name': lang.name.replace('_', ' ').title(),
                'legal_system': self.language_detector._get_legal_system(lang.value)
            }
            for lang in SupportedLanguage
        ]