"""Pattern matching system for arbitration detection."""
import re
from typing import Dict, List
import spacy
import logging

logger = logging.getLogger(__name__)

class ArbitrationPatternMatcher:
    """Advanced pattern matching for arbitration clause detection."""
    
    def __init__(self):
        """Initialize pattern matching system."""
        # Load spaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define arbitration-specific patterns
        self.patterns = self._load_patterns()
        self.keywords = self._load_keywords()
        
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for arbitration detection."""
        return {
            'mandatory_arbitration': [
                r'shall\s+be\s+(?:finally\s+)?(?:settled|resolved)\s+by\s+arbitration',
                r'must\s+be\s+(?:submitted|referred)\s+to\s+arbitration',
                r'agrees?\s+to\s+(?:binding\s+)?arbitration',
                r'subject\s+to\s+(?:final\s+and\s+)?binding\s+arbitration',
                r'mandatory\s+arbitration',
                r'compulsory\s+arbitration',
                r'required\s+to\s+arbitrate',
                r'shall\s+arbitrate',
            ],
            'arbitration_rules': [
                r'(?:AAA|JAMS|ICC|LCIA|UNCITRAL)\s+(?:rules?|procedures?)',
                r'American\s+Arbitration\s+Association',
                r'International\s+Chamber\s+of\s+Commerce',
                r'London\s+Court\s+of\s+International\s+Arbitration',
                r'Judicial\s+Arbitration\s+and\s+Mediation\s+Services',
                r'Federal\s+Arbitration\s+Act',
                r'arbitration\s+rules',
            ],
            'class_action_waiver': [
                r'waive[sd]?\s+(?:any\s+)?right\s+to\s+(?:a\s+)?class\s+action',
                r'no\s+class\s+(?:or\s+collective\s+)?action',
                r'prohibited\s+from\s+bringing\s+(?:a\s+)?class\s+action',
                r'class\s+action\s+waiver',
                r'individual\s+basis\s+only',
                r'no\s+representative\s+actions',
                r'waive\s+class\s+proceedings',
            ],
            'opt_out': [
                r'opt[\s-]?out\s+of\s+(?:this\s+)?arbitration',
                r'reject\s+(?:this\s+)?arbitration\s+(?:agreement|provision)',
                r'(?:30|thirty|60|sixty)\s+days?\s+to\s+opt[\s-]?out',
                r'decline\s+arbitration',
                r'right\s+to\s+reject',
                r'election\s+to\s+arbitrate',
            ],
            'venue': [
                r'arbitration\s+shall\s+(?:take\s+place|be\s+conducted)\s+in',
                r'venue\s+for\s+arbitration',
                r'seat\s+of\s+(?:the\s+)?arbitration',
                r'location\s+of\s+arbitration',
                r'arbitration\s+proceedings?\s+(?:shall|will)\s+be\s+held',
            ],
            'jury_trial_waiver': [
                r'waive[sd]?\s+(?:any\s+)?right\s+to\s+(?:a\s+)?jury\s+trial',
                r'no\s+jury\s+trial',
                r'waive\s+trial\s+by\s+jury',
                r'forfeit\s+jury\s+trial',
            ],
            'dispute_resolution': [
                r'dispute\s+resolution\s+(?:procedure|process|mechanism)',
                r'alternative\s+dispute\s+resolution',
                r'ADR\s+(?:procedure|process)',
                r'mediation\s+(?:and|or)\s+arbitration',
                r'escalation\s+to\s+arbitration',
            ]
        }
    
    def _load_keywords(self) -> Dict[str, float]:
        """Load weighted keywords for arbitration detection."""
        return {
            # High confidence keywords
            'arbitration': 0.9,
            'arbitrator': 0.9,
            'arbitral': 0.85,
            'JAMS': 0.85,
            'AAA': 0.85,
            'ICC': 0.85,
            'LCIA': 0.85,
            'UNCITRAL': 0.8,
            
            # Medium confidence keywords
            'dispute resolution': 0.6,
            'binding': 0.5,
            'waive': 0.5,
            'class action': 0.6,
            'jury trial': 0.5,
            'individual basis': 0.5,
            
            # Context keywords (lower weight)
            'dispute': 0.3,
            'resolve': 0.3,
            'settlement': 0.3,
            'mediation': 0.4,
            'claims': 0.3,
            'proceedings': 0.3,
            'agreement': 0.2,
            'terms': 0.2,
        }
    
    def match(self, text: str) -> Dict:
        """
        Perform pattern matching on text.
        
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
                    if category == 'mandatory_arbitration':
                        weight = 0.95
                    elif category in ['class_action_waiver', 'jury_trial_waiver']:
                        weight = 0.85
                    elif category == 'arbitration_rules':
                        weight = 0.8
                    elif category == 'opt_out':
                        weight = 0.6  # Opt-out still means arbitration exists
                    else:
                        weight = 0.7
                    pattern_scores.append(weight)
        
        # Check keywords
        keyword_score = 0.0
        keyword_matches = []
        for keyword, weight in self.keywords.items():
            if keyword.lower() in text_lower:
                keyword_matches.append(f"keyword: {keyword}")
                keyword_score += weight
        
        # Add unique keyword matches to matches list
        matches.extend(keyword_matches[:5])  # Limit to top 5 keyword matches
        
        # Normalize keyword score
        keyword_score = min(1.0, keyword_score / 3.0)
        
        # Calculate overall confidence
        if pattern_scores:
            pattern_confidence = max(pattern_scores)
        else:
            pattern_confidence = 0.0
            
        # Combine pattern and keyword confidence
        overall_confidence = max(pattern_confidence, keyword_score)
        
        # Boost confidence if multiple strong indicators
        if len([s for s in pattern_scores if s > 0.8]) > 2:
            overall_confidence = min(1.0, overall_confidence * 1.1)
        
        return {
            'matches': matches[:10],  # Top 10 matches
            'confidence': overall_confidence,
            'pattern_confidence': pattern_confidence,
            'keyword_confidence': keyword_score,
            'num_patterns': len(pattern_scores),
            'num_keywords': len(keyword_matches)
        }
    
    def extract_clause_boundaries(self, text: str, match_position: int) -> Tuple[int, int]:
        """
        Extract the boundaries of the arbitration clause.
        
        Args:
            text: Full text
            match_position: Position of arbitration match
            
        Returns:
            Tuple of (start_idx, end_idx) for the clause
        """
        # Use spaCy to find sentence boundaries
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Find the sentence containing the match
        for i, sent in enumerate(sentences):
            if sent.start_char <= match_position < sent.end_char:
                # Include surrounding sentences for context
                start_idx = sentences[max(0, i-1)].start_char
                end_idx = sentences[min(len(sentences)-1, i+2)].end_char
                return start_idx, end_idx
        
        # Fallback to simple approach
        start_idx = max(0, match_position - 500)
        end_idx = min(len(text), match_position + 500)
        return start_idx, end_idx
    
    def analyze_clause_type(self, text: str) -> str:
        """
        Analyze the type of arbitration clause.
        
        Args:
            text: Clause text
            
        Returns:
            Type of arbitration clause
        """
        text_lower = text.lower()
        
        if re.search(r'mandatory|shall|must|required', text_lower):
            return 'mandatory'
        elif re.search(r'may|option|elect|choose', text_lower):
            return 'optional'
        elif re.search(r'opt[\s-]?out', text_lower):
            return 'mandatory_with_opt_out'
        else:
            return 'standard'