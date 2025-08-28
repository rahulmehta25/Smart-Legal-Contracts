"""Core arbitration detection pipeline integrating all components."""
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import hashlib
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ArbitrationClause:
    """Complete arbitration clause with metadata."""
    full_text: str
    summary: str
    location: Dict[str, Union[int, str]]
    confidence: float
    clause_type: str  # mandatory, optional, etc.
    key_provisions: List[str]
    detection_method: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class ArbitrationDetectionPipeline:
    """Complete detection pipeline for arbitration clauses."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize the complete detection pipeline."""
        # Import here to avoid circular dependencies
        from ..models.legal_bert_detector import LegalBERTDetector
        from ..document.section_detector import DocumentStructureAnalyzer
        
        self.bert_detector = LegalBERTDetector()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            try:
                import redis
                self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                # Test connection
                self.cache.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis not available, disabling cache: {e}")
                self.cache_enabled = False
                self.cache = None
        else:
            self.cache = None
    
    def detect_arbitration_clause(self, filepath: str) -> Optional[ArbitrationClause]:
        """
        Main entry point for arbitration detection.
        
        Args:
            filepath: Path to document
            
        Returns:
            ArbitrationClause object if found, None otherwise
        """
        # Validate file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        # Check cache
        if self.cache_enabled:
            cached_result = self._check_cache(filepath)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
        
        try:
            # Step 1: Structural analysis to find candidate sections
            logger.info(f"Analyzing document structure: {filepath}")
            candidate_sections = self.structure_analyzer.find_arbitration_sections(
                filepath, threshold=0.3
            )
            
            if not candidate_sections:
                logger.info("No candidate sections found")
                # Try analyzing full document
                full_text = self.structure_analyzer.extract_full_text(filepath)
                if full_text:
                    detection_result = self.bert_detector.detect(full_text)
                    if detection_result.is_arbitration:
                        return self._create_clause_from_detection(
                            detection_result, full_text, filepath
                        )
                return None
            
            # Step 2: Deep analysis on candidate sections
            best_result = None
            best_confidence = 0.0
            best_section = None
            
            for section in candidate_sections[:5]:  # Check top 5 candidates
                # Run Legal-BERT detection
                detection_result = self.bert_detector.detect(section.content)
                
                if detection_result.is_arbitration and detection_result.confidence > best_confidence:
                    best_result = detection_result
                    best_confidence = detection_result.confidence
                    best_section = section
            
            if best_result and best_section:
                # Step 3: Extract complete clause
                full_clause = self._extract_full_clause(best_section, best_result)
                
                # Step 4: Analyze clause provisions
                provisions = self._analyze_provisions(full_clause)
                
                arbitration_clause = ArbitrationClause(
                    full_text=full_clause,
                    summary=self._generate_summary(full_clause),
                    location={
                        'start_page': best_section.start_page,
                        'end_page': best_section.end_page,
                        'section_title': best_section.title,
                        'section_number': best_section.section_number or 'N/A'
                    },
                    confidence=best_confidence,
                    clause_type=provisions['type'],
                    key_provisions=provisions['key_points'],
                    detection_method='Legal-BERT + Pattern Matching + Structure Analysis'
                )
                
                # Cache result
                if self.cache_enabled:
                    self._cache_result(filepath, arbitration_clause)
                
                logger.info(f"Arbitration clause detected with confidence: {best_confidence:.2%}")
                return arbitration_clause
            
            logger.info("No arbitration clause detected in candidate sections")
            return None
            
        except Exception as e:
            logger.error(f"Error in detection pipeline: {e}")
            return None
    
    def _create_clause_from_detection(self, detection_result, full_text: str, filepath: str) -> ArbitrationClause:
        """Create arbitration clause from detection result."""
        provisions = self._analyze_provisions(full_text)
        
        return ArbitrationClause(
            full_text=full_text[:5000],  # Limit to 5000 chars
            summary=self._generate_summary(full_text),
            location={
                'start_page': 1,
                'end_page': 1,
                'section_title': 'Full Document',
                'section_number': 'N/A'
            },
            confidence=detection_result.confidence,
            clause_type=provisions['type'],
            key_provisions=provisions['key_points'],
            detection_method='Legal-BERT + Pattern Matching'
        )
    
    def _extract_full_clause(self, section, detection_result) -> str:
        """Extract the complete arbitration clause text."""
        # Use pattern matcher to find exact boundaries
        from ..models.pattern_matcher import ArbitrationPatternMatcher
        matcher = ArbitrationPatternMatcher()
        
        # Find the most relevant portion
        if detection_result.pattern_matches:
            # Use pattern matches to find boundaries
            for pattern in detection_result.pattern_matches:
                if 'arbitration' in pattern.lower():
                    # Found relevant pattern, extract surrounding context
                    return section.content[:3000]  # Return first 3000 chars
        
        # Default: return section content
        return section.content[:3000]
    
    def _analyze_provisions(self, clause_text: str) -> Dict:
        """Analyze key provisions in the arbitration clause."""
        provisions = {
            'type': 'unknown',
            'key_points': []
        }
        
        clause_lower = clause_text.lower()
        
        # Determine type
        if any(word in clause_lower for word in ['mandatory', 'shall', 'must', 'required']):
            provisions['type'] = 'mandatory'
        elif any(word in clause_lower for word in ['may', 'option', 'elect', 'choose']):
            provisions['type'] = 'optional'
        elif 'opt-out' in clause_lower or 'opt out' in clause_lower:
            provisions['type'] = 'mandatory_with_opt_out'
        else:
            provisions['type'] = 'standard'
        
        # Extract key points
        key_point_checks = [
            ('class action waiver', 'Class action waiver'),
            ('opt-out', 'Opt-out provision'),
            ('opt out', 'Opt-out provision'),
            ('jams', 'JAMS arbitration'),
            ('aaa', 'AAA arbitration'),
            ('american arbitration association', 'AAA arbitration'),
            ('confidential', 'Confidentiality requirement'),
            ('jury trial', 'Jury trial waiver'),
            ('binding', 'Binding arbitration'),
            ('final', 'Final and binding'),
            ('individual', 'Individual arbitration only'),
            ('30 days', '30-day opt-out period'),
            ('60 days', '60-day opt-out period'),
            ('fees', 'Fee provisions'),
            ('costs', 'Cost allocation'),
            ('venue', 'Specified venue'),
            ('governing law', 'Governing law specified'),
        ]
        
        for check_text, provision_name in key_point_checks:
            if check_text in clause_lower and provision_name not in provisions['key_points']:
                provisions['key_points'].append(provision_name)
        
        # Limit to top 10 provisions
        provisions['key_points'] = provisions['key_points'][:10]
        
        return provisions
    
    def _generate_summary(self, clause_text: str) -> str:
        """Generate a brief summary of the clause."""
        # Simple extractive summary - take first 200 characters
        summary = clause_text[:200].strip()
        
        # Clean up summary
        summary = ' '.join(summary.split())  # Normalize whitespace
        
        if len(clause_text) > 200:
            # Find end of sentence
            for punct in ['.', '!', '?', ';']:
                idx = summary.rfind(punct)
                if idx > 100:  # At least 100 chars
                    summary = summary[:idx+1]
                    break
            else:
                summary += "..."
        
        return summary
    
    def _check_cache(self, filepath: str) -> Optional[ArbitrationClause]:
        """Check cache for previous analysis."""
        if not self.cache:
            return None
            
        try:
            # Generate cache key from file hash
            with open(filepath, 'rb') as f:
                # Read in chunks for large files
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                cache_key = f"arbitration:{file_hash.hexdigest()}"
            
            cached_json = self.cache.get(cache_key)
            if cached_json:
                cached_dict = json.loads(cached_json)
                return ArbitrationClause(**cached_dict)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, filepath: str, result: ArbitrationClause):
        """Cache analysis result."""
        if not self.cache:
            return
            
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                cache_key = f"arbitration:{file_hash.hexdigest()}"
            
            self.cache.setex(
                cache_key,
                86400,  # 24 hour TTL
                result.to_json()
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def batch_detect(self, filepaths: List[str]) -> List[Optional[ArbitrationClause]]:
        """
        Detect arbitration clauses in multiple documents.
        
        Args:
            filepaths: List of document paths
            
        Returns:
            List of ArbitrationClause objects or None for each document
        """
        results = []
        for filepath in filepaths:
            logger.info(f"Processing: {filepath}")
            result = self.detect_arbitration_clause(filepath)
            results.append(result)
        
        return results
    
    def detect_from_text(self, text: str) -> Optional[ArbitrationClause]:
        """
        Detect arbitration clause directly from text.
        
        Args:
            text: Document text
            
        Returns:
            ArbitrationClause object if found, None otherwise
        """
        try:
            # Run detection
            detection_result = self.bert_detector.detect(text)
            
            if detection_result.is_arbitration:
                provisions = self._analyze_provisions(text)
                
                return ArbitrationClause(
                    full_text=text[:5000],
                    summary=self._generate_summary(text),
                    location={
                        'start_page': 1,
                        'end_page': 1,
                        'section_title': 'Text Input',
                        'section_number': 'N/A'
                    },
                    confidence=detection_result.confidence,
                    clause_type=provisions['type'],
                    key_provisions=provisions['key_points'],
                    detection_method='Legal-BERT + Pattern Matching'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting from text: {e}")
            return None