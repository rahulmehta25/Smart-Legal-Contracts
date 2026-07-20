import re
from typing import Dict, List
import spacy

class ArbitrationPatternMatcher:
    def __init__(self):
        """Initialize pattern matching system"""
        # Load spaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define arbitration-specific patterns
        self.patterns = self._load_patterns()
        self.keywords = self._load_keywords()
        
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for arbitration detection"""
        return {
            'mandatory_arbitration': [
                r'shall\s+be\s+(?:finally\s+)?(?:settled|resolved)\s+by\s+arbitration',
                r'must\s+be\s+(?:submitted|referred)\s+to\s+arbitration',
                r'agrees?\s+to\s+(?:binding\s+)?arbitration',
                r'subject\s+to\s+(?:final\s+and\s+)?binding\s+arbitration',
            ],
            'arbitration_rules': [
                r'(?:AAA|JAMS|ICC|LCIA|UNCITRAL)\s+(?:rules?|procedures?)',
                r'American\s+Arbitration\s+Association',
                r'International\s+Chamber\s+of\s+Commerce',
            ],
            'class_action_waiver': [
                r'waive[sd]?\s+(?:any\s+)?right\s+to\s+(?:a\s+)?class\s+action',
                r'no\s+class\s+(?:or\s+collective\s+)?action',
                r'prohibited\s+from\s+bringing\s+(?:a\s+)?class\s+action',
            ],
            'opt_out': [
                r'opt[\s-]?out\s+of\s+(?:this\s+)?arbitration',
                r'reject\s+(?:this\s+)?arbitration\s+(?:agreement|provision)',
                r'(?:30|thirty|60|sixty)\s+days?\s+to\s+opt[\s-]?out',
            ],
            'venue': [
                r'arbitration\s+shall\s+(?:take\s+place|be\s+conducted)\s+in',
                r'venue\s+for\s+arbitration',
                r'seat\s+of\s+(?:the\s+)?arbitration',
            ]
        }
    
    def _load_keywords(self) -> Dict[str, float]:
        """Load weighted keywords for arbitration detection"""
        return {
            # High confidence keywords
            'arbitration': 0.9,
            'arbitrator': 0.9,
            'arbitral': 0.85,
            'JAMS': 0.85,
            'AAA': 0.85,
            
            # Medium confidence keywords
            'dispute resolution': 0.6,
            'binding': 0.5,
            'waive': 0.5,
            'class action': 0.6,
            
            # Context keywords (lower weight)
            'dispute': 0.3,
            'resolve': 0.3,
            'settlement': 0.3,
            'mediation': 0.4,
        }
    
    def match(self, text: str) -> Dict:
        """
        Perform pattern matching on text
        
        Returns:
            Dictionary with matches and confidence score
        """
        text_lower = text.lower()
        matches = []
        pattern_scores = []
        
        # Check regex patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(f"{category}: {pattern[:50]}...")
                    # Different weights for different categories
                    weight = 0.9 if category == 'mandatory_arbitration' else 0.7
                    pattern_scores.append(weight)
        
        # Check keywords
        keyword_score = 0.0
        for keyword, weight in self.keywords.items():
            if keyword.lower() in text_lower:
                matches.append(f"keyword: {keyword}")
                keyword_score += weight
        
        # Normalize keyword score
        keyword_score = min(1.0, keyword_score / 3.0)
        
        # Calculate overall confidence
        if pattern_scores:
            pattern_confidence = max(pattern_scores)
        else:
            pattern_confidence = 0.0
            
        overall_confidence = max(pattern_confidence, keyword_score)
        
        return {
            'matches': matches[:10],  # Top 10 matches
            'confidence': overall_confidence,
            'pattern_confidence': pattern_confidence,
            'keyword_confidence': keyword_score
        }
