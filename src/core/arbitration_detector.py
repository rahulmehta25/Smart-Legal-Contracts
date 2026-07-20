from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib
import json

from models.legal_bert_detector import LegalBERTDetector, DetectionResult
from document.section_detector import DocumentStructureAnalyzer, DocumentSection

@dataclass
class ArbitrationClause:
    """Complete arbitration clause with metadata"""
    full_text: str
    summary: str
    location: Dict[str, int]  # page numbers, section info
    confidence: float
    clause_type: str  # mandatory, optional, etc.
    key_provisions: List[str]
    detection_method: str

class ArbitrationDetectionPipeline:
    def __init__(self, cache_enabled: bool = True):
        """Initialize the complete detection pipeline"""
        self.bert_detector = LegalBERTDetector()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            try:
                import redis
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                # Test connection
                self.cache.ping()
            except:
                print("Redis not available. Caching disabled.")
                self.cache_enabled = False
                self.cache = None
    
    def detect_arbitration_clause(self, filepath: str) -> Optional[ArbitrationClause]:
        """
        Main entry point for arbitration detection
        
        Args:
            filepath: Path to document
            
        Returns:
            ArbitrationClause object if found, None otherwise
        """
        # Check cache
        if self.cache_enabled and self.cache:
            cached_result = self._check_cache(filepath)
            if cached_result:
                return cached_result
        
        # Step 1: Structural analysis to find candidate sections
        candidate_sections = self.structure_analyzer.find_arbitration_sections(
            filepath, threshold=0.3
        )
        
        if not candidate_sections:
            print("No candidate sections found")
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
                    'section_number': best_section.section_number
                },
                confidence=best_confidence,
                clause_type=provisions['type'],
                key_provisions=provisions['key_points'],
                detection_method='Legal-BERT + Pattern Matching'
            )
            
            # Cache result
            if self.cache_enabled and self.cache:
                self._cache_result(filepath, arbitration_clause)
            
            return arbitration_clause
        
        return None
    
    def _extract_full_clause(self, section: DocumentSection, 
                           detection: DetectionResult) -> str:
        """Extract the complete arbitration clause text"""
        # For now, return the section content
        # In production, use more sophisticated extraction
        return section.content
    
    def _analyze_provisions(self, clause_text: str) -> Dict:
        """Analyze key provisions in the arbitration clause"""
        provisions = {
            'type': 'unknown',
            'key_points': []
        }
        
        clause_lower = clause_text.lower()
        
        # Determine type
        if 'mandatory' in clause_lower or 'shall' in clause_lower:
            provisions['type'] = 'mandatory'
        elif 'may' in clause_lower or 'option' in clause_lower:
            provisions['type'] = 'optional'
        
        # Extract key points
        if 'class action waiver' in clause_lower:
            provisions['key_points'].append('Class action waiver')
        if 'opt-out' in clause_lower or 'opt out' in clause_lower:
            provisions['key_points'].append('Opt-out provision')
        if 'jams' in clause_lower or 'aaa' in clause_lower:
            provisions['key_points'].append('Specified arbitration organization')
        if 'confidential' in clause_lower:
            provisions['key_points'].append('Confidentiality requirement')
        
        return provisions
    
    def _generate_summary(self, clause_text: str) -> str:
        """Generate a brief summary of the clause"""
        # Simple extractive summary - take first 200 characters
        # In production, use more sophisticated summarization
        summary = clause_text[:200].strip()
        if len(clause_text) > 200:
            summary += "..."
        return summary
    
    def _check_cache(self, filepath: str) -> Optional[ArbitrationClause]:
        """Check cache for previous analysis"""
        # Generate cache key from file hash
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        try:
            cached = self.cache.get(f"arbitration:{file_hash}")
            if cached:
                return ArbitrationClause(**json.loads(cached))
        except:
            pass
        
        return None
    
    def _cache_result(self, filepath: str, result: ArbitrationClause):
        """Cache analysis result"""
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        try:
            self.cache.setex(
                f"arbitration:{file_hash}",
                86400,  # 24 hour TTL
                json.dumps(result.__dict__, default=str)
            )
        except:
            pass
