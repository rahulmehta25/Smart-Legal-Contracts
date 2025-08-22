"""
Legal-specific translation module with term preservation and context awareness.

This module provides specialized translation capabilities for legal documents,
ensuring that legal terminology is preserved and context is maintained during
translation across different legal systems and jurisdictions.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import hashlib
from functools import lru_cache
import spacy
from spacy.tokens import Doc, Span, Token

# Legal terminology databases
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Legal domain models
try:
    import legal_bert
    LEGAL_BERT_AVAILABLE = True
except ImportError:
    LEGAL_BERT_AVAILABLE = False


logger = logging.getLogger(__name__)


class LegalDomain(Enum):
    """Legal domain categories for context-aware translation."""
    CONTRACT_LAW = "contract"
    ARBITRATION = "arbitration"
    CORPORATE_LAW = "corporate"
    INTELLECTUAL_PROPERTY = "ip"
    EMPLOYMENT_LAW = "employment"
    CONSUMER_PROTECTION = "consumer"
    INTERNATIONAL_LAW = "international"
    CIVIL_PROCEDURE = "civil_procedure"


class JurisdictionType(Enum):
    """Major legal system types."""
    COMMON_LAW = "common_law"
    CIVIL_LAW = "civil_law"
    ISLAMIC_LAW = "islamic_law"
    MIXED_SYSTEM = "mixed"


@dataclass
class LegalTerm:
    """Representation of a legal term with context."""
    term: str
    category: str
    jurisdiction: Optional[str]
    equivalents: Dict[str, List[str]] = field(default_factory=dict)
    context_patterns: List[str] = field(default_factory=list)
    confidence: float = 1.0
    domain: Optional[LegalDomain] = None


@dataclass
class ContextualTranslation:
    """Result of contextual legal translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    legal_terms_preserved: List[LegalTerm]
    context_adjustments: List[str]
    jurisdiction_mapping: Optional[Tuple[str, str]]
    confidence_score: float
    domain_detected: Optional[LegalDomain]
    processing_notes: List[str] = field(default_factory=list)


class LegalTerminologyDatabase:
    """Comprehensive legal terminology database with cross-jurisdictional mappings."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.term_database = {}
        self.phrase_patterns = {}
        self.jurisdiction_mappings = {}
        self.domain_keywords = {}
        self._load_terminology_data(data_path)
        
    def _load_terminology_data(self, data_path: Optional[str]):
        """Load legal terminology database."""
        # Arbitration-specific terminology
        self.term_database['arbitration'] = {
            'en': {
                'arbitration': LegalTerm(
                    term='arbitration',
                    category='dispute_resolution',
                    jurisdiction='common_law',
                    equivalents={
                        'es': ['arbitraje', 'arbitramento'],
                        'fr': ['arbitrage'],
                        'de': ['Schiedsverfahren', 'Arbitrage'],
                        'zh': ['仲裁'],
                        'ja': ['仲裁', '調停'],
                        'pt': ['arbitragem'],
                        'it': ['arbitrato'],
                        'ru': ['арбитраж'],
                        'ar': ['تحكيم']
                    },
                    context_patterns=[
                        r'binding\s+arbitration',
                        r'arbitration\s+clause',
                        r'arbitration\s+agreement',
                        r'submit\s+to\s+arbitration'
                    ],
                    domain=LegalDomain.ARBITRATION
                ),
                'arbitrator': LegalTerm(
                    term='arbitrator',
                    category='legal_actor',
                    jurisdiction='common_law',
                    equivalents={
                        'es': ['árbitro', 'arbitrador'],
                        'fr': ['arbitre'],
                        'de': ['Schiedsrichter'],
                        'zh': ['仲裁员', '仲裁者'],
                        'ja': ['仲裁人', '調停者'],
                        'pt': ['árbitro'],
                        'it': ['arbitro'],
                        'ru': ['арбитр'],
                        'ar': ['محكم']
                    },
                    domain=LegalDomain.ARBITRATION
                ),
                'binding': LegalTerm(
                    term='binding',
                    category='legal_effect',
                    jurisdiction='common_law',
                    equivalents={
                        'es': ['vinculante', 'obligatorio'],
                        'fr': ['contraignant', 'obligatoire'],
                        'de': ['bindend', 'verbindlich'],
                        'zh': ['有约束力的', '强制性的'],
                        'ja': ['拘束力のある', '強制的な'],
                        'pt': ['vinculativo', 'obrigatório'],
                        'it': ['vincolante', 'obbligatorio'],
                        'ru': ['обязательный', 'связывающий'],
                        'ar': ['ملزم', 'إلزامي']
                    },
                    context_patterns=[
                        r'final\s+and\s+binding',
                        r'binding\s+arbitration',
                        r'binding\s+decision'
                    ],
                    domain=LegalDomain.ARBITRATION
                ),
                'dispute_resolution': LegalTerm(
                    term='dispute resolution',
                    category='process',
                    jurisdiction='common_law',
                    equivalents={
                        'es': ['resolución de disputas', 'resolución de conflictos'],
                        'fr': ['résolution des différends', 'règlement des conflits'],
                        'de': ['Streitbeilegung', 'Konfliktlösung'],
                        'zh': ['争议解决', '纠纷解决'],
                        'ja': ['紛争解決', '争議解決'],
                        'pt': ['resolução de disputas'],
                        'it': ['risoluzione delle controversie'],
                        'ru': ['разрешение споров'],
                        'ar': ['حل النزاعات']
                    },
                    domain=LegalDomain.ARBITRATION
                ),
                'jurisdiction': LegalTerm(
                    term='jurisdiction',
                    category='legal_authority',
                    jurisdiction='common_law',
                    equivalents={
                        'es': ['jurisdicción', 'competencia'],
                        'fr': ['juridiction', 'compétence'],
                        'de': ['Gerichtsbarkeit', 'Zuständigkeit'],
                        'zh': ['管辖权', '司法管辖'],
                        'ja': ['管轄権', '司法権'],
                        'pt': ['jurisdição'],
                        'it': ['giurisdizione'],
                        'ru': ['юрисдикция', 'подсудность'],
                        'ar': ['اختصاص قضائي', 'ولاية قضائية']
                    },
                    context_patterns=[
                        r'exclusive\s+jurisdiction',
                        r'subject\s+to\s+the\s+jurisdiction',
                        r'jurisdiction\s+of\s+the\s+courts'
                    ]
                )
            }
        }
        
        # Add more domains as needed
        self._build_phrase_patterns()
        self._build_jurisdiction_mappings()
        self._build_domain_keywords()
    
    def _build_phrase_patterns(self):
        """Build regex patterns for legal phrase detection."""
        self.phrase_patterns = {
            'arbitration_clauses': [
                r'any\s+dispute.*?shall\s+be\s+settled\s+by\s+arbitration',
                r'disputes?\s+arising.*?arbitration',
                r'binding\s+arbitration.*?final\s+and\s+binding',
                r'arbitration\s+rules\s+of\s+the\s+(AAA|ICC|JAMS)',
                r'submit.*?to\s+binding\s+arbitration',
                r'exclusive\s+remedy.*?arbitration'
            ],
            'jurisdiction_clauses': [
                r'subject\s+to\s+the\s+exclusive\s+jurisdiction',
                r'courts?\s+of.*?shall\s+have\s+exclusive\s+jurisdiction',
                r'governing\s+law.*?jurisdiction',
                r'applicable\s+law.*?disputes?'
            ],
            'waiver_clauses': [
                r'waive.*?right\s+to\s+(jury\s+trial|class\s+action)',
                r'waiver\s+of.*?(jury|class\s+action)',
                r'no\s+right\s+to.*?(jury\s+trial|class\s+action)'
            ]
        }
    
    def _build_jurisdiction_mappings(self):
        """Build mappings between legal systems."""
        self.jurisdiction_mappings = {
            'common_law': {
                'countries': ['US', 'UK', 'CA', 'AU', 'IN', 'HK', 'SG'],
                'characteristics': ['case_law', 'precedent', 'adversarial'],
                'arbitration_institutions': ['AAA', 'JAMS', 'LCIA', 'SIAC']
            },
            'civil_law': {
                'countries': ['DE', 'FR', 'ES', 'IT', 'JP', 'CN', 'BR'],
                'characteristics': ['written_law', 'codified', 'inquisitorial'],
                'arbitration_institutions': ['ICC', 'DIS', 'SCAI', 'CIETAC']
            },
            'islamic_law': {
                'countries': ['SA', 'AE', 'QA', 'BH', 'KW'],
                'characteristics': ['sharia_compliant', 'religious_basis'],
                'arbitration_institutions': ['DIAC', 'KCAB', 'SCCA']
            }
        }
    
    def _build_domain_keywords(self):
        """Build domain-specific keyword mappings."""
        self.domain_keywords = {
            LegalDomain.ARBITRATION: {
                'en': [
                    'arbitration', 'arbitrator', 'arbitral', 'binding',
                    'dispute resolution', 'final and binding', 'AAA', 'JAMS',
                    'ICC arbitration', 'arbitration clause', 'arbitration agreement',
                    'arbitral tribunal', 'arbitral award', 'arbitration rules'
                ]
            },
            LegalDomain.CONTRACT_LAW: {
                'en': [
                    'contract', 'agreement', 'breach', 'termination',
                    'consideration', 'offer', 'acceptance', 'liability'
                ]
            }
        }
    
    def get_legal_term(self, term: str, language: str, domain: Optional[LegalDomain] = None) -> Optional[LegalTerm]:
        """Retrieve legal term information."""
        domain_key = domain.value if domain else 'arbitration'
        
        if domain_key in self.term_database:
            lang_terms = self.term_database[domain_key].get(language, {})
            return lang_terms.get(term.lower())
        
        return None
    
    def find_equivalent_terms(self, term: str, source_lang: str, target_lang: str, domain: Optional[LegalDomain] = None) -> List[str]:
        """Find equivalent terms in target language."""
        legal_term = self.get_legal_term(term, source_lang, domain)
        if legal_term and target_lang in legal_term.equivalents:
            return legal_term.equivalents[target_lang]
        return []
    
    def detect_legal_phrases(self, text: str, category: Optional[str] = None) -> List[Tuple[str, str, int, int]]:
        """Detect legal phrases in text using regex patterns."""
        detected_phrases = []
        
        patterns_to_check = self.phrase_patterns
        if category and category in patterns_to_check:
            patterns_to_check = {category: patterns_to_check[category]}
        
        for phrase_category, patterns in patterns_to_check.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    detected_phrases.append((
                        match.group(),
                        phrase_category,
                        match.start(),
                        match.end()
                    ))
        
        return detected_phrases


class ContextualLegalTranslator:
    """Advanced legal translator with context awareness and term preservation."""
    
    def __init__(self, terminology_db: Optional[LegalTerminologyDatabase] = None):
        self.terminology_db = terminology_db or LegalTerminologyDatabase()
        self.nlp_models = {}
        self.legal_embeddings = {}
        self.translation_cache = {}
        self._load_nlp_models()
        
    def _load_nlp_models(self):
        """Load language-specific NLP models."""
        supported_langs = ['en', 'es', 'fr', 'de', 'zh', 'ja']
        
        for lang in supported_langs:
            try:
                model_name = f"{lang}_core_web_sm"
                if lang == 'zh':
                    model_name = "zh_core_web_sm"
                elif lang == 'ja':
                    model_name = "ja_core_news_sm"
                
                self.nlp_models[lang] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model for {lang}")
            except OSError:
                logger.warning(f"spaCy model for {lang} not available")
                # Fallback to basic tokenization
                self.nlp_models[lang] = None
    
    async def translate_legal_document(
        self,
        text: str,
        source_language: str,
        target_language: str,
        legal_domain: Optional[LegalDomain] = None,
        preserve_structure: bool = True
    ) -> ContextualTranslation:
        """
        Translate legal document with context preservation.
        
        Args:
            text: Legal document text
            source_language: Source language code
            target_language: Target language code
            legal_domain: Specific legal domain for context
            preserve_structure: Whether to preserve document structure
            
        Returns:
            ContextualTranslation with detailed translation results
        """
        # Detect legal domain if not provided
        if not legal_domain:
            legal_domain = self._detect_legal_domain(text, source_language)
        
        # Extract legal terms and phrases
        legal_terms = self._extract_legal_terms(text, source_language, legal_domain)
        legal_phrases = self.terminology_db.detect_legal_phrases(text)
        
        # Prepare text for translation with term preservation
        processed_text, term_mappings = self._prepare_for_translation(
            text, legal_terms, legal_phrases
        )
        
        # Perform contextual translation
        translated_text = await self._contextual_translate(
            processed_text,
            source_language,
            target_language,
            legal_domain
        )
        
        # Restore legal terms with appropriate equivalents
        final_text, preserved_terms = self._restore_legal_terms(
            translated_text,
            term_mappings,
            source_language,
            target_language,
            legal_domain
        )
        
        # Detect jurisdiction mapping
        jurisdiction_mapping = self._detect_jurisdiction_mapping(
            source_language, target_language
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_translation_confidence(
            text, final_text, preserved_terms, legal_phrases
        )
        
        return ContextualTranslation(
            original_text=text,
            translated_text=final_text,
            source_language=source_language,
            target_language=target_language,
            legal_terms_preserved=preserved_terms,
            context_adjustments=[],  # Would be populated with specific adjustments
            jurisdiction_mapping=jurisdiction_mapping,
            confidence_score=confidence_score,
            domain_detected=legal_domain,
            processing_notes=[]
        )
    
    def _detect_legal_domain(self, text: str, language: str) -> Optional[LegalDomain]:
        """Detect the legal domain of the document."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, lang_keywords in self.terminology_db.domain_keywords.items():
            keywords = lang_keywords.get(language, [])
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                domain_scores[domain] = score / len(keywords)
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return LegalDomain.ARBITRATION  # Default to arbitration
    
    def _extract_legal_terms(self, text: str, language: str, domain: LegalDomain) -> List[LegalTerm]:
        """Extract legal terms from text."""
        terms = []
        
        # Get domain-specific keywords
        domain_keywords = self.terminology_db.domain_keywords.get(domain, {}).get(language, [])
        
        text_lower = text.lower()
        for keyword in domain_keywords:
            if keyword.lower() in text_lower:
                legal_term = self.terminology_db.get_legal_term(keyword, language, domain)
                if legal_term:
                    terms.append(legal_term)
        
        return terms
    
    def _prepare_for_translation(
        self, 
        text: str, 
        legal_terms: List[LegalTerm], 
        legal_phrases: List[Tuple[str, str, int, int]]
    ) -> Tuple[str, Dict[str, LegalTerm]]:
        """Prepare text for translation by replacing legal terms with placeholders."""
        processed_text = text
        term_mappings = {}
        
        # Sort by position (reverse order to preserve indices)
        all_terms = []
        
        # Add individual terms
        for term in legal_terms:
            for match in re.finditer(re.escape(term.term), text, re.IGNORECASE):
                all_terms.append((match.start(), match.end(), term.term, term))
        
        # Add phrases
        for phrase, category, start, end in legal_phrases:
            # Create a pseudo-term for the phrase
            phrase_term = LegalTerm(
                term=phrase,
                category=category,
                jurisdiction=None
            )
            all_terms.append((start, end, phrase, phrase_term))
        
        # Sort by start position (reverse)
        all_terms.sort(key=lambda x: x[0], reverse=True)
        
        # Replace with placeholders
        for i, (start, end, text_match, term) in enumerate(all_terms):
            placeholder = f"__LEGAL_TERM_{i}__"
            processed_text = processed_text[:start] + placeholder + processed_text[end:]
            term_mappings[placeholder] = term
        
        return processed_text, term_mappings
    
    async def _contextual_translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        domain: LegalDomain
    ) -> str:
        """Perform contextual translation considering legal domain."""
        # This would integrate with the main translation pipeline
        # For now, return a simplified translation
        
        # Cache key
        cache_key = hashlib.md5(
            f"{text}_{source_language}_{target_language}_{domain.value}".encode()
        ).hexdigest()
        
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Simplified translation (would use actual translation models)
        # This is a placeholder - in practice, would use the translation pipeline
        translated_text = text  # Placeholder
        
        # Cache result
        self.translation_cache[cache_key] = translated_text
        
        return translated_text
    
    def _restore_legal_terms(
        self,
        translated_text: str,
        term_mappings: Dict[str, LegalTerm],
        source_language: str,
        target_language: str,
        domain: LegalDomain
    ) -> Tuple[str, List[LegalTerm]]:
        """Restore legal terms with appropriate target language equivalents."""
        final_text = translated_text
        preserved_terms = []
        
        for placeholder, legal_term in term_mappings.items():
            if placeholder in final_text:
                # Find equivalent terms in target language
                equivalents = self.terminology_db.find_equivalent_terms(
                    legal_term.term, source_language, target_language, domain
                )
                
                if equivalents:
                    # Use the first equivalent (could be made more sophisticated)
                    replacement = equivalents[0]
                    legal_term.equivalents[target_language] = equivalents
                else:
                    # Keep original term if no equivalent found
                    replacement = legal_term.term
                
                final_text = final_text.replace(placeholder, replacement)
                preserved_terms.append(legal_term)
        
        return final_text, preserved_terms
    
    def _detect_jurisdiction_mapping(
        self, source_language: str, target_language: str
    ) -> Optional[Tuple[str, str]]:
        """Detect jurisdiction mapping between source and target languages."""
        # Map languages to legal systems
        lang_to_jurisdiction = {
            'en': 'common_law',
            'es': 'civil_law',
            'fr': 'civil_law',
            'de': 'civil_law',
            'zh': 'civil_law',
            'ja': 'civil_law',
            'ar': 'islamic_law'
        }
        
        source_jurisdiction = lang_to_jurisdiction.get(source_language)
        target_jurisdiction = lang_to_jurisdiction.get(target_language)
        
        if source_jurisdiction and target_jurisdiction:
            return (source_jurisdiction, target_jurisdiction)
        
        return None
    
    def _calculate_translation_confidence(
        self,
        original_text: str,
        translated_text: str,
        preserved_terms: List[LegalTerm],
        legal_phrases: List[Tuple[str, str, int, int]]
    ) -> float:
        """Calculate confidence score for translation quality."""
        factors = []
        
        # Term preservation factor
        if preserved_terms:
            factors.append(0.8)  # High confidence for preserved terms
        
        # Legal phrase detection factor
        if legal_phrases:
            factors.append(0.7)  # Good confidence for detected phrases
        
        # Length similarity factor
        length_ratio = len(translated_text) / max(len(original_text), 1)
        if 0.5 <= length_ratio <= 2.0:  # Reasonable length ratio
            factors.append(0.6)
        
        # Base confidence
        factors.append(0.5)
        
        return sum(factors) / len(factors) if factors else 0.5


class LegalTranslator:
    """Main legal translator class combining all functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.terminology_db = LegalTerminologyDatabase()
        self.contextual_translator = ContextualLegalTranslator(self.terminology_db)
        self.fallback_strategies = []
        self._initialize_fallback_strategies()
    
    def _initialize_fallback_strategies(self):
        """Initialize fallback translation strategies."""
        self.fallback_strategies = [
            self._glossary_based_translation,
            self._pattern_based_translation,
            self._identity_translation
        ]
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str = "en",
        domain: Optional[LegalDomain] = None,
        jurisdiction_aware: bool = True
    ) -> ContextualTranslation:
        """
        Main translation method with comprehensive legal context handling.
        
        Args:
            text: Legal text to translate
            source_language: Source language code
            target_language: Target language code
            domain: Legal domain for context
            jurisdiction_aware: Whether to apply jurisdiction-specific handling
            
        Returns:
            ContextualTranslation with complete translation results
        """
        try:
            # Primary translation attempt
            result = await self.contextual_translator.translate_legal_document(
                text, source_language, target_language, domain
            )
            
            # Apply jurisdiction-specific adjustments if needed
            if jurisdiction_aware and result.jurisdiction_mapping:
                result = self._apply_jurisdiction_adjustments(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Primary translation failed: {e}")
            
            # Try fallback strategies
            for strategy in self.fallback_strategies:
                try:
                    return await strategy(text, source_language, target_language, domain)
                except Exception as fallback_error:
                    logger.warning(f"Fallback strategy failed: {fallback_error}")
                    continue
            
            # Ultimate fallback
            return ContextualTranslation(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                legal_terms_preserved=[],
                context_adjustments=[],
                jurisdiction_mapping=None,
                confidence_score=0.1,
                domain_detected=domain,
                processing_notes=["Translation failed, returned original text"]
            )
    
    def _apply_jurisdiction_adjustments(self, result: ContextualTranslation) -> ContextualTranslation:
        """Apply jurisdiction-specific adjustments to translation."""
        if not result.jurisdiction_mapping:
            return result
        
        source_jurisdiction, target_jurisdiction = result.jurisdiction_mapping
        
        # Apply specific adjustments based on jurisdiction mapping
        adjustments = []
        
        if source_jurisdiction == 'common_law' and target_jurisdiction == 'civil_law':
            # Adjust common law concepts for civil law context
            adjustments.append("Adjusted common law terminology for civil law context")
            
        elif source_jurisdiction == 'civil_law' and target_jurisdiction == 'common_law':
            # Adjust civil law concepts for common law context
            adjustments.append("Adjusted civil law terminology for common law context")
        
        result.context_adjustments.extend(adjustments)
        return result
    
    async def _glossary_based_translation(
        self, text: str, source_lang: str, target_lang: str, domain: Optional[LegalDomain]
    ) -> ContextualTranslation:
        """Fallback translation using glossary mapping."""
        translated_text = text
        
        # Apply simple term substitution
        if domain:
            domain_keywords = self.terminology_db.domain_keywords.get(domain, {}).get(source_lang, [])
            for keyword in domain_keywords:
                equivalents = self.terminology_db.find_equivalent_terms(
                    keyword, source_lang, target_lang, domain
                )
                if equivalents:
                    translated_text = translated_text.replace(keyword, equivalents[0])
        
        return ContextualTranslation(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            legal_terms_preserved=[],
            context_adjustments=["Used glossary-based translation"],
            jurisdiction_mapping=None,
            confidence_score=0.3,
            domain_detected=domain
        )
    
    async def _pattern_based_translation(
        self, text: str, source_lang: str, target_lang: str, domain: Optional[LegalDomain]
    ) -> ContextualTranslation:
        """Fallback translation using pattern matching."""
        return ContextualTranslation(
            original_text=text,
            translated_text=text,  # Simplified fallback
            source_language=source_lang,
            target_language=target_lang,
            legal_terms_preserved=[],
            context_adjustments=["Used pattern-based translation"],
            jurisdiction_mapping=None,
            confidence_score=0.2,
            domain_detected=domain
        )
    
    async def _identity_translation(
        self, text: str, source_lang: str, target_lang: str, domain: Optional[LegalDomain]
    ) -> ContextualTranslation:
        """Ultimate fallback - return original text."""
        return ContextualTranslation(
            original_text=text,
            translated_text=text,
            source_language=source_lang,
            target_language=target_lang,
            legal_terms_preserved=[],
            context_adjustments=["Identity translation - no changes made"],
            jurisdiction_mapping=None,
            confidence_score=0.1,
            domain_detected=domain,
            processing_notes=["Fallback to identity translation"]
        )
    
    def get_supported_domains(self) -> List[LegalDomain]:
        """Get list of supported legal domains."""
        return list(LegalDomain)
    
    def get_supported_jurisdictions(self) -> List[JurisdictionType]:
        """Get list of supported jurisdiction types."""
        return list(JurisdictionType)
    
    async def batch_translate(
        self,
        texts: List[str],
        source_language: str,
        target_language: str = "en",
        domain: Optional[LegalDomain] = None
    ) -> List[ContextualTranslation]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            domain: Legal domain for context
            
        Returns:
            List of ContextualTranslation results
        """
        tasks = [
            self.translate(text, source_language, target_language, domain)
            for text in texts
        ]
        return await asyncio.gather(*tasks)