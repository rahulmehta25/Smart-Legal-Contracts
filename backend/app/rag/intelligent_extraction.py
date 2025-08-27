"""
Intelligent Contract Clause Extraction and Analysis
Uses advanced NLP and pattern recognition for legal document understanding
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClause:
    """Enhanced extracted clause with deep analysis"""
    id: str
    type: str
    subtype: Optional[str]
    text: str
    normalized_text: str
    start_position: int
    end_position: int
    confidence: float
    entities: List[Dict[str, Any]]
    obligations: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    temporal_elements: List[Dict[str, Any]]
    cross_references: List[str]
    legal_concepts: List[str]
    enforceability_score: float
    clarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractStructure:
    """Represents the hierarchical structure of a contract"""
    sections: List[Dict[str, Any]]
    subsections: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    definitions: Dict[str, str]
    cross_reference_map: Dict[str, List[str]]
    hierarchy_tree: Dict[str, Any]


class IntelligentClauseExtractor:
    """Advanced clause extraction with deep legal understanding"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize intelligent extractor"""
        self.config = config or self._get_default_config()
        self._initialize_models()
        self._load_legal_patterns()
        self._initialize_extractors()
        
        logger.info("Intelligent Clause Extractor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_deep_parsing': True,
            'enable_entity_recognition': True,
            'enable_obligation_extraction': True,
            'enable_temporal_extraction': True,
            'confidence_threshold': 0.6,
            'use_legal_bert': True,
            'max_clause_length': 2000,
            'min_clause_length': 20
        }
    
    def _initialize_models(self):
        """Initialize NLP models for extraction"""
        try:
            # Load spaCy model for linguistic analysis
            try:
                self.nlp = spacy.load('en_legal_ner')
            except:
                self.nlp = spacy.load('en_core_web_trf')
            
            # Load transformer models for classification
            if self.config['use_legal_bert']:
                self.clause_classifier = pipeline(
                    "text-classification",
                    model="nlpaueb/legal-bert-base-uncased",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Token classification for entity extraction
            self.token_classifier = pipeline(
                "token-classification",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to basic spaCy
            self.nlp = spacy.load('en_core_web_sm')
    
    def _load_legal_patterns(self):
        """Load comprehensive legal patterns and templates"""
        self.legal_patterns = {
            'obligation_patterns': [
                r'\b(shall|must|will|agrees? to|undertakes? to|commits? to|is required to)\b',
                r'\b(responsible for|obligated to|bound to|duty to)\b'
            ],
            'right_patterns': [
                r'\b(may|can|is entitled to|has the right to|is authorized to)\b',
                r'\b(at its sole discretion|at its option|reserves the right)\b'
            ],
            'condition_patterns': [
                r'\b(if|when|where|provided that|subject to|in the event|upon)\b',
                r'\b(unless|except|notwithstanding|conditioned upon)\b'
            ],
            'temporal_patterns': [
                r'\b(within|before|after|during|following|prior to|no later than)\b\s+\d+\s*(days?|months?|years?|hours?)',
                r'\b(effective date|termination date|expiration|commencement)\b',
                r'\b(immediately|promptly|forthwith|as soon as practicable)\b'
            ],
            'definition_patterns': [
                r'"([^"]+)"\s+(?:means|refers to|shall mean|is defined as)',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:means|refers to|shall mean)',
                r'(?:hereinafter|hereafter)\s+"([^"]+)"'
            ],
            'cross_reference_patterns': [
                r'Section\s+\d+(?:\.\d+)*',
                r'Clause\s+\d+(?:\.\d+)*',
                r'Article\s+[IVXLCDM]+',
                r'Exhibit\s+[A-Z]',
                r'Schedule\s+\d+',
                r'Appendix\s+[A-Z]'
            ]
        }
        
        # Clause type patterns
        self.clause_type_patterns = {
            'payment': [
                r'payment|invoice|fee|cost|price|charge|billing|reimburse'
            ],
            'termination': [
                r'terminat|expir|end|cancel|dissolve|breach|default'
            ],
            'confidentiality': [
                r'confidential|proprietary|non-disclosure|secret|private'
            ],
            'liability': [
                r'liabilit|damages|indemnif|loss|claim|responsible'
            ],
            'warranty': [
                r'warrant|guarantee|represent|covenant|assure'
            ],
            'arbitration': [
                r'arbitrat|mediat|dispute resolution|binding|tribunal'
            ],
            'intellectual_property': [
                r'intellectual property|copyright|patent|trademark|trade secret'
            ],
            'assignment': [
                r'assign|transfer|delegate|successor|merger'
            ],
            'force_majeure': [
                r'force majeure|act of god|unforeseeable|beyond control'
            ],
            'governing_law': [
                r'governing law|jurisdiction|venue|applicable law|laws of'
            ]
        }
    
    def _initialize_extractors(self):
        """Initialize specialized extractors"""
        self.entity_extractor = EntityExtractor(self.nlp)
        self.obligation_extractor = ObligationExtractor(self.legal_patterns)
        self.temporal_extractor = TemporalExtractor(self.legal_patterns)
        self.structure_analyzer = ContractStructureAnalyzer()
    
    def extract_all_clauses(self, document_text: str) -> List[ExtractedClause]:
        """
        Extract all clauses from a document with deep analysis
        
        Args:
            document_text: Full document text
            
        Returns:
            List of extracted and analyzed clauses
        """
        # Analyze document structure
        structure = self.structure_analyzer.analyze(document_text)
        
        # Extract raw clauses
        raw_clauses = self._extract_raw_clauses(document_text, structure)
        
        # Process each clause
        extracted_clauses = []
        for idx, raw_clause in enumerate(raw_clauses):
            clause = self._process_clause(
                raw_clause,
                clause_id=f"clause_{idx}",
                structure=structure
            )
            if clause:
                extracted_clauses.append(clause)
        
        # Analyze cross-references
        self._analyze_cross_references(extracted_clauses, structure)
        
        # Score clauses
        self._score_clauses(extracted_clauses)
        
        return extracted_clauses
    
    def _extract_raw_clauses(
        self,
        text: str,
        structure: ContractStructure
    ) -> List[Dict[str, Any]]:
        """Extract raw clause segments from text"""
        raw_clauses = []
        
        # Method 1: Structure-based extraction
        for section in structure.sections:
            # Extract section content
            section_text = text[section['start']:section['end']]
            
            # Split into subsections or paragraphs
            if section.get('subsections'):
                for subsection in section['subsections']:
                    raw_clauses.append({
                        'text': subsection['text'],
                        'start': subsection['start'],
                        'end': subsection['end'],
                        'type': 'subsection',
                        'parent_section': section.get('title', '')
                    })
            else:
                # Split by paragraphs
                paragraphs = self._split_into_paragraphs(section_text)
                for para in paragraphs:
                    if len(para['text']) >= self.config['min_clause_length']:
                        raw_clauses.append({
                            'text': para['text'],
                            'start': section['start'] + para['start'],
                            'end': section['start'] + para['end'],
                            'type': 'paragraph',
                            'parent_section': section.get('title', '')
                        })
        
        # Method 2: Pattern-based extraction for missed clauses
        pattern_clauses = self._extract_by_patterns(text)
        
        # Merge and deduplicate
        all_clauses = self._merge_clauses(raw_clauses, pattern_clauses)
        
        return all_clauses
    
    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split text into paragraphs"""
        paragraphs = []
        
        # Split by double newlines or numbered items
        para_pattern = r'(?:\n\n|\n(?=\d+\.)|\n(?=[A-Z]))'
        splits = re.split(para_pattern, text)
        
        current_pos = 0
        for split_text in splits:
            if split_text.strip():
                paragraphs.append({
                    'text': split_text.strip(),
                    'start': current_pos,
                    'end': current_pos + len(split_text)
                })
            current_pos += len(split_text) + 2  # Account for delimiter
        
        return paragraphs
    
    def _extract_by_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract clauses using legal patterns"""
        pattern_clauses = []
        
        # Look for numbered sections
        section_pattern = r'(\d+(?:\.\d+)*)\s+([A-Z][^.]+\.(?:\s+[^.]+\.)*)'
        
        for match in re.finditer(section_pattern, text, re.MULTILINE):
            clause_text = match.group(0)
            if len(clause_text) >= self.config['min_clause_length']:
                pattern_clauses.append({
                    'text': clause_text,
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'numbered_section',
                    'number': match.group(1)
                })
        
        return pattern_clauses
    
    def _merge_clauses(
        self,
        structure_clauses: List[Dict[str, Any]],
        pattern_clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate clauses from different extraction methods"""
        all_clauses = structure_clauses.copy()
        
        # Add pattern clauses that don't overlap with structure clauses
        for pattern_clause in pattern_clauses:
            overlaps = False
            for struct_clause in structure_clauses:
                if (pattern_clause['start'] >= struct_clause['start'] and
                    pattern_clause['end'] <= struct_clause['end']):
                    overlaps = True
                    break
            
            if not overlaps:
                all_clauses.append(pattern_clause)
        
        # Sort by position
        all_clauses.sort(key=lambda x: x['start'])
        
        return all_clauses
    
    def _process_clause(
        self,
        raw_clause: Dict[str, Any],
        clause_id: str,
        structure: ContractStructure
    ) -> Optional[ExtractedClause]:
        """Process a raw clause into an ExtractedClause object"""
        text = raw_clause['text']
        
        # Skip if too short or too long
        if (len(text) < self.config['min_clause_length'] or
            len(text) > self.config['max_clause_length']):
            return None
        
        # Classify clause type
        clause_type, subtype, confidence = self._classify_clause(text)
        
        # Skip if confidence too low
        if confidence < self.config['confidence_threshold']:
            return None
        
        # Extract entities
        entities = self.entity_extractor.extract(text) if self.config['enable_entity_recognition'] else []
        
        # Extract obligations
        obligations = self.obligation_extractor.extract(text) if self.config['enable_obligation_extraction'] else []
        
        # Extract conditions
        conditions = self._extract_conditions(text)
        
        # Extract temporal elements
        temporal_elements = self.temporal_extractor.extract(text) if self.config['enable_temporal_extraction'] else []
        
        # Find cross-references
        cross_references = self._find_cross_references(text)
        
        # Extract legal concepts
        legal_concepts = self._extract_legal_concepts(text)
        
        # Normalize text for comparison
        normalized_text = self._normalize_text(text)
        
        # Create ExtractedClause object
        clause = ExtractedClause(
            id=clause_id,
            type=clause_type,
            subtype=subtype,
            text=text,
            normalized_text=normalized_text,
            start_position=raw_clause['start'],
            end_position=raw_clause['end'],
            confidence=confidence,
            entities=entities,
            obligations=obligations,
            conditions=conditions,
            temporal_elements=temporal_elements,
            cross_references=cross_references,
            legal_concepts=legal_concepts,
            enforceability_score=0.0,  # Will be calculated later
            clarity_score=0.0,  # Will be calculated later
            metadata={
                'extraction_type': raw_clause['type'],
                'parent_section': raw_clause.get('parent_section', ''),
                'clause_number': raw_clause.get('number', '')
            }
        )
        
        return clause
    
    def _classify_clause(self, text: str) -> Tuple[str, Optional[str], float]:
        """Classify clause type and subtype"""
        # First, try pattern-based classification
        for clause_type, patterns in self.clause_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Use model for confidence if available
                    if hasattr(self, 'clause_classifier'):
                        try:
                            result = self.clause_classifier(text[:512])  # Limit length
                            confidence = result[0]['score'] if result else 0.8
                        except:
                            confidence = 0.8
                    else:
                        confidence = 0.8
                    
                    # Determine subtype
                    subtype = self._determine_subtype(text, clause_type)
                    
                    return clause_type, subtype, confidence
        
        # Fallback to model-based classification
        if hasattr(self, 'clause_classifier'):
            try:
                result = self.clause_classifier(text[:512])
                if result:
                    label = result[0]['label'].lower()
                    confidence = result[0]['score']
                    return label, None, confidence
            except:
                pass
        
        # Default classification
        return 'general', None, 0.5
    
    def _determine_subtype(self, text: str, clause_type: str) -> Optional[str]:
        """Determine clause subtype based on content"""
        subtypes = {
            'termination': {
                'for_cause': ['breach', 'default', 'violation'],
                'for_convenience': ['convenience', 'without cause', 'any reason'],
                'automatic': ['automatically', 'expiration', 'end of term']
            },
            'payment': {
                'fixed_fee': ['fixed', 'flat fee', 'lump sum'],
                'recurring': ['monthly', 'annually', 'subscription'],
                'milestone': ['milestone', 'completion', 'deliverable']
            },
            'liability': {
                'limitation': ['limit', 'cap', 'maximum'],
                'exclusion': ['exclude', 'not liable', 'no liability'],
                'indemnification': ['indemnify', 'defend', 'hold harmless']
            }
        }
        
        if clause_type in subtypes:
            text_lower = text.lower()
            for subtype, keywords in subtypes[clause_type].items():
                if any(keyword in text_lower for keyword in keywords):
                    return subtype
        
        return None
    
    def _extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract conditional statements from clause"""
        conditions = []
        
        for pattern in self.legal_patterns['condition_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract condition context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                conditions.append({
                    'trigger': match.group(0),
                    'context': context,
                    'position': match.start(),
                    'type': self._classify_condition(match.group(0))
                })
        
        return conditions
    
    def _classify_condition(self, trigger: str) -> str:
        """Classify the type of condition"""
        trigger_lower = trigger.lower()
        
        if trigger_lower in ['if', 'when', 'where']:
            return 'prerequisite'
        elif trigger_lower in ['unless', 'except']:
            return 'exception'
        elif trigger_lower in ['provided that', 'subject to']:
            return 'qualification'
        elif trigger_lower in ['upon', 'following']:
            return 'temporal'
        else:
            return 'general'
    
    def _find_cross_references(self, text: str) -> List[str]:
        """Find cross-references to other sections"""
        references = []
        
        for pattern in self.legal_patterns['cross_reference_patterns']:
            matches = re.finditer(pattern, text)
            for match in matches:
                references.append(match.group(0))
        
        return list(set(references))  # Remove duplicates
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts mentioned in clause"""
        concepts = []
        
        legal_concepts_list = [
            'good faith', 'reasonable efforts', 'material breach',
            'force majeure', 'act of god', 'gross negligence',
            'willful misconduct', 'liquidated damages', 'consequential damages',
            'fiduciary duty', 'due diligence', 'arm\'s length',
            'time is of the essence', 'severability', 'entire agreement',
            'survival', 'waiver', 'estoppel', 'consideration'
        ]
        
        text_lower = text.lower()
        for concept in legal_concepts_list:
            if concept in text_lower:
                concepts.append(concept)
        
        return concepts
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra whitespace
        normalized = ' '.join(text.split())
        
        # Remove section numbers
        normalized = re.sub(r'^\d+(?:\.\d+)*\s+', '', normalized)
        
        # Standardize common legal phrases
        replacements = {
            r'\bshall\b': 'must',
            r'\bprovided that\b': 'if',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'under',
            r'\bprior to\b': 'before'
        }
        
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _analyze_cross_references(
        self,
        clauses: List[ExtractedClause],
        structure: ContractStructure
    ):
        """Analyze cross-references between clauses"""
        # Build reference map
        reference_map = defaultdict(list)
        
        for clause in clauses:
            for ref in clause.cross_references:
                # Find target clause
                target_clause = self._find_referenced_clause(ref, clauses, structure)
                if target_clause:
                    reference_map[clause.id].append(target_clause.id)
                    # Add bidirectional reference
                    clause.metadata['references_to'] = clause.metadata.get('references_to', []) + [target_clause.id]
                    target_clause.metadata['referenced_by'] = target_clause.metadata.get('referenced_by', []) + [clause.id]
    
    def _find_referenced_clause(
        self,
        reference: str,
        clauses: List[ExtractedClause],
        structure: ContractStructure
    ) -> Optional[ExtractedClause]:
        """Find the clause being referenced"""
        # Extract section number from reference
        match = re.search(r'\d+(?:\.\d+)*', reference)
        if match:
            ref_number = match.group(0)
            
            # Find clause with matching number
            for clause in clauses:
                if clause.metadata.get('clause_number') == ref_number:
                    return clause
        
        return None
    
    def _score_clauses(self, clauses: List[ExtractedClause]):
        """Score clauses for enforceability and clarity"""
        for clause in clauses:
            # Calculate enforceability score
            clause.enforceability_score = self._calculate_enforceability_score(clause)
            
            # Calculate clarity score
            clause.clarity_score = self._calculate_clarity_score(clause)
    
    def _calculate_enforceability_score(self, clause: ExtractedClause) -> float:
        """Calculate enforceability score for a clause"""
        score = 0.7  # Base score
        
        # Check for clear obligations
        if clause.obligations:
            score += 0.1
        
        # Check for defined terms
        if any(term in clause.text for term in ['"', 'defined', 'means']):
            score += 0.05
        
        # Check for ambiguous language
        ambiguous_terms = ['reasonable', 'appropriate', 'timely', 'prompt', 'material']
        ambiguous_count = sum(1 for term in ambiguous_terms if term in clause.text.lower())
        score -= ambiguous_count * 0.05
        
        # Check for conditions
        if clause.conditions:
            score += 0.05  # Conditions add clarity
        
        # Check for temporal elements
        if clause.temporal_elements:
            score += 0.05  # Specific timeframes improve enforceability
        
        return max(0.0, min(1.0, score))
    
    def _calculate_clarity_score(self, clause: ExtractedClause) -> float:
        """Calculate clarity score for a clause"""
        # Simple readability metrics
        words = clause.text.split()
        sentences = clause.text.split('.')
        
        if not words or not sentences:
            return 0.5
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Complexity based on sentence length
        if avg_sentence_length < 15:
            clarity = 0.9
        elif avg_sentence_length < 25:
            clarity = 0.7
        elif avg_sentence_length < 35:
            clarity = 0.5
        else:
            clarity = 0.3
        
        # Adjust for legal jargon
        jargon_terms = ['whereas', 'heretofore', 'witnesseth', 'aforementioned', 'hereinafter']
        jargon_count = sum(1 for term in jargon_terms if term in clause.text.lower())
        clarity -= jargon_count * 0.1
        
        # Adjust for structure
        if clause.metadata.get('extraction_type') == 'numbered_section':
            clarity += 0.1  # Numbered sections are clearer
        
        return max(0.0, min(1.0, clarity))


class EntityExtractor:
    """Extract legal entities from text"""
    
    def __init__(self, nlp):
        self.nlp = nlp
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'legal_role': self._determine_legal_role(ent.text, text)
            }
            entities.append(entity)
        
        return entities
    
    def _determine_legal_role(self, entity_text: str, context: str) -> str:
        """Determine the legal role of an entity"""
        context_lower = context.lower()
        entity_lower = entity_text.lower()
        
        # Check surrounding context
        if 'party' in context_lower or 'agreement between' in context_lower:
            return 'party'
        elif 'arbitrator' in context_lower or 'mediator' in context_lower:
            return 'arbitrator'
        elif 'court' in context_lower or 'tribunal' in context_lower:
            return 'judicial_body'
        elif 'jurisdiction' in context_lower or 'governed by' in context_lower:
            return 'jurisdiction'
        else:
            return 'organization'


class ObligationExtractor:
    """Extract obligations and rights from clauses"""
    
    def __init__(self, legal_patterns: Dict[str, List[str]]):
        self.patterns = legal_patterns
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract obligations from text"""
        obligations = []
        
        # Extract obligation statements
        for pattern in self.patterns['obligation_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                obligation = self._extract_obligation_details(text, match)
                if obligation:
                    obligations.append(obligation)
        
        # Extract rights (negative obligations)
        for pattern in self.patterns['right_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                right = self._extract_right_details(text, match)
                if right:
                    obligations.append(right)
        
        return obligations
    
    def _extract_obligation_details(
        self,
        text: str,
        match: re.Match
    ) -> Dict[str, Any]:
        """Extract details of an obligation"""
        # Get context around the match
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 150)
        context = text[start:end]
        
        # Try to identify the obligated party
        party = self._identify_party(context, match.start() - start)
        
        # Extract the obligation action
        action = self._extract_action(context, match.end() - start)
        
        return {
            'type': 'obligation',
            'trigger': match.group(0),
            'party': party,
            'action': action,
            'full_text': context.strip(),
            'position': match.start(),
            'strength': self._determine_obligation_strength(match.group(0))
        }
    
    def _extract_right_details(
        self,
        text: str,
        match: re.Match
    ) -> Dict[str, Any]:
        """Extract details of a right"""
        # Get context around the match
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 150)
        context = text[start:end]
        
        # Try to identify the party with the right
        party = self._identify_party(context, match.start() - start)
        
        # Extract the right action
        action = self._extract_action(context, match.end() - start)
        
        return {
            'type': 'right',
            'trigger': match.group(0),
            'party': party,
            'action': action,
            'full_text': context.strip(),
            'position': match.start(),
            'discretionary': 'discretion' in match.group(0).lower()
        }
    
    def _identify_party(self, context: str, trigger_pos: int) -> str:
        """Identify the party in an obligation/right"""
        # Look for party indicators before the trigger
        before_trigger = context[:trigger_pos] if trigger_pos > 0 else ''
        
        # Common party references
        party_patterns = [
            r'([Tt]he )?(Party|Buyer|Seller|Company|Customer|Client|Vendor|Contractor|Employee|Employer)',
            r'([Ee]ach|[Bb]oth|[Ee]ither) [Pp]art(y|ies)',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s+shall|\s+must|\s+will|\s+may)'
        ]
        
        for pattern in party_patterns:
            match = re.search(pattern, before_trigger[-100:] if len(before_trigger) > 100 else before_trigger)
            if match:
                return match.group(0)
        
        return 'unspecified party'
    
    def _extract_action(self, context: str, start_pos: int) -> str:
        """Extract the action from an obligation/right"""
        # Get text after the trigger
        after_trigger = context[start_pos:] if start_pos < len(context) else ''
        
        # Extract up to the next period or comma
        match = re.search(r'^[^.,;]+', after_trigger)
        if match:
            return match.group(0).strip()
        
        return after_trigger[:100].strip() if after_trigger else 'unspecified action'
    
    def _determine_obligation_strength(self, trigger: str) -> str:
        """Determine the strength of an obligation"""
        trigger_lower = trigger.lower()
        
        if any(word in trigger_lower for word in ['shall', 'must', 'will', 'required']):
            return 'mandatory'
        elif any(word in trigger_lower for word in ['should', 'ought']):
            return 'recommended'
        elif any(word in trigger_lower for word in ['may', 'can']):
            return 'optional'
        else:
            return 'uncertain'


class TemporalExtractor:
    """Extract temporal elements from clauses"""
    
    def __init__(self, legal_patterns: Dict[str, List[str]]):
        self.patterns = legal_patterns
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal elements from text"""
        temporal_elements = []
        
        for pattern in self.patterns['temporal_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal = {
                    'text': match.group(0),
                    'type': self._classify_temporal(match.group(0)),
                    'position': match.start(),
                    'duration': self._extract_duration(match.group(0)),
                    'urgency': self._determine_urgency(match.group(0))
                }
                temporal_elements.append(temporal)
        
        return temporal_elements
    
    def _classify_temporal(self, temporal_text: str) -> str:
        """Classify the type of temporal element"""
        text_lower = temporal_text.lower()
        
        if any(word in text_lower for word in ['within', 'before', 'no later than']):
            return 'deadline'
        elif any(word in text_lower for word in ['after', 'following']):
            return 'trigger'
        elif any(word in text_lower for word in ['during', 'throughout']):
            return 'duration'
        elif any(word in text_lower for word in ['effective', 'commencement']):
            return 'start_date'
        elif any(word in text_lower for word in ['expiration', 'termination']):
            return 'end_date'
        else:
            return 'general'
    
    def _extract_duration(self, temporal_text: str) -> Optional[Dict[str, Any]]:
        """Extract duration information"""
        # Look for number + time unit
        match = re.search(r'(\d+)\s*(days?|months?|years?|hours?|weeks?)', temporal_text, re.IGNORECASE)
        if match:
            return {
                'value': int(match.group(1)),
                'unit': match.group(2).rstrip('s').lower()
            }
        return None
    
    def _determine_urgency(self, temporal_text: str) -> str:
        """Determine urgency level"""
        text_lower = temporal_text.lower()
        
        if any(word in text_lower for word in ['immediately', 'forthwith', 'promptly']):
            return 'immediate'
        elif any(word in text_lower for word in ['as soon as', 'expeditious']):
            return 'urgent'
        elif re.search(r'\d+\s*days?', text_lower):
            days_match = re.search(r'(\d+)\s*days?', text_lower)
            days = int(days_match.group(1))
            if days <= 7:
                return 'urgent'
            elif days <= 30:
                return 'normal'
            else:
                return 'long_term'
        else:
            return 'normal'


class ContractStructureAnalyzer:
    """Analyze the hierarchical structure of contracts"""
    
    def analyze(self, text: str) -> ContractStructure:
        """Analyze document structure"""
        sections = self._extract_sections(text)
        subsections = self._extract_subsections(text, sections)
        paragraphs = self._extract_paragraphs(text)
        definitions = self._extract_definitions(text)
        cross_reference_map = self._build_cross_reference_map(text)
        hierarchy_tree = self._build_hierarchy_tree(sections, subsections, paragraphs)
        
        return ContractStructure(
            sections=sections,
            subsections=subsections,
            paragraphs=paragraphs,
            definitions=definitions,
            cross_reference_map=cross_reference_map,
            hierarchy_tree=hierarchy_tree
        )
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract main sections from document"""
        sections = []
        
        # Pattern for main sections (e.g., "1. DEFINITIONS", "ARTICLE I")
        patterns = [
            r'^(\d+)\.\s+([A-Z][A-Z\s]+)$',
            r'^ARTICLE\s+([IVXLCDM]+)\.?\s*-?\s*(.+)$',
            r'^SECTION\s+(\d+)\.?\s*-?\s*(.+)$'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                sections.append({
                    'number': match.group(1),
                    'title': match.group(2).strip() if match.lastindex > 1 else '',
                    'start': match.start(),
                    'end': match.end(),
                    'level': 1
                })
        
        # Sort by position
        sections.sort(key=lambda x: x['start'])
        
        # Update end positions
        for i in range(len(sections) - 1):
            sections[i]['end'] = sections[i + 1]['start']
        if sections:
            sections[-1]['end'] = len(text)
        
        return sections
    
    def _extract_subsections(
        self,
        text: str,
        sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract subsections within sections"""
        subsections = []
        
        for section in sections:
            section_text = text[section['start']:section['end']]
            
            # Pattern for subsections (e.g., "1.1", "(a)", "(i)")
            patterns = [
                r'^(\d+\.\d+)\s+(.+)$',
                r'^\(([a-z])\)\s+(.+)$',
                r'^\(([ivx]+)\)\s+(.+)$'
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, section_text, re.MULTILINE):
                    subsections.append({
                        'number': match.group(1),
                        'title': match.group(2).strip() if match.lastindex > 1 else '',
                        'start': section['start'] + match.start(),
                        'end': section['start'] + match.end(),
                        'parent_section': section['number'],
                        'level': 2
                    })
        
        return subsections
    
    def _extract_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Extract paragraphs from document"""
        paragraphs = []
        
        # Split by double newlines
        para_splits = text.split('\n\n')
        
        current_pos = 0
        for i, para in enumerate(para_splits):
            if para.strip():
                paragraphs.append({
                    'id': f'para_{i}',
                    'text': para.strip(),
                    'start': current_pos,
                    'end': current_pos + len(para),
                    'word_count': len(para.split())
                })
            current_pos += len(para) + 2  # Account for double newline
        
        return paragraphs
    
    def _extract_definitions(self, text: str) -> Dict[str, str]:
        """Extract defined terms from document"""
        definitions = {}
        
        # Look for definition patterns
        patterns = [
            r'"([^"]+)"\s+(?:means|shall mean|refers to)\s+([^.]+)\.',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+means\s+([^.]+)\.',
            r'defined as\s+"([^"]+)"\s+([^.]+)\.'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1)
                definition = match.group(2).strip()
                definitions[term] = definition
        
        return definitions
    
    def _build_cross_reference_map(self, text: str) -> Dict[str, List[str]]:
        """Build map of cross-references"""
        cross_refs = defaultdict(list)
        
        # Find all section references
        ref_pattern = r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)'
        
        # Split text into sections for context
        sections = text.split('\n\n')
        
        for i, section in enumerate(sections):
            for match in re.finditer(ref_pattern, section):
                referenced_section = match.group(1)
                cross_refs[f'section_{i}'].append(referenced_section)
        
        return dict(cross_refs)
    
    def _build_hierarchy_tree(
        self,
        sections: List[Dict[str, Any]],
        subsections: List[Dict[str, Any]],
        paragraphs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build hierarchical tree structure"""
        tree = {'root': [], 'sections': {}}
        
        for section in sections:
            section_node = {
                'number': section['number'],
                'title': section['title'],
                'subsections': [],
                'paragraphs': []
            }
            
            # Add subsections
            for subsection in subsections:
                if subsection.get('parent_section') == section['number']:
                    section_node['subsections'].append({
                        'number': subsection['number'],
                        'title': subsection['title']
                    })
            
            # Add paragraphs within section
            for para in paragraphs:
                if (para['start'] >= section['start'] and 
                    para['end'] <= section['end']):
                    section_node['paragraphs'].append(para['id'])
            
            tree['sections'][section['number']] = section_node
            tree['root'].append(section['number'])
        
        return tree