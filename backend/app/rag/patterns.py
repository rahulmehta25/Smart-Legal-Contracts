"""
Legal patterns and indicators for arbitration clause detection.
"""

from typing import List, Dict, Set
import re
from dataclasses import dataclass


@dataclass
class ArbitrationPattern:
    """Represents an arbitration pattern with its weight and type."""
    pattern: str
    weight: float
    pattern_type: str  # 'keyword', 'phrase', 'regex'
    arbitration_type: str  # 'binding', 'non-binding', 'general', 'class-action-waiver'


class ArbitrationPatterns:
    """Collection of patterns for detecting arbitration clauses."""
    
    # High-confidence keywords that strongly indicate arbitration
    HIGH_CONFIDENCE_KEYWORDS = [
        ("binding arbitration", 0.95, "binding"),
        ("mandatory arbitration", 0.95, "binding"),
        ("compulsory arbitration", 0.95, "binding"),
        ("agree to arbitrate", 0.9, "binding"),
        ("submit to arbitration", 0.9, "binding"),
        ("arbitration agreement", 0.85, "general"),
        ("arbitration provision", 0.85, "general"),
        ("arbitration clause", 0.85, "general"),
        ("final and binding arbitration", 0.95, "binding"),
        ("exclusive remedy", 0.8, "binding"),
    ]
    
    # Medium-confidence keywords
    MEDIUM_CONFIDENCE_KEYWORDS = [
        ("arbitration", 0.7, "general"),
        ("arbitrate", 0.65, "general"),
        ("arbitrator", 0.6, "general"),
        ("dispute resolution", 0.5, "general"),
        ("alternative dispute resolution", 0.55, "general"),
        ("adr", 0.4, "general"),
        ("mediation and arbitration", 0.75, "general"),
        ("non-binding arbitration", 0.7, "non-binding"),
        ("voluntary arbitration", 0.6, "non-binding"),
    ]
    
    # Class action waiver indicators
    CLASS_ACTION_WAIVER_PATTERNS = [
        ("class action waiver", 0.95, "class-action-waiver"),
        ("no class actions", 0.9, "class-action-waiver"),
        ("waive any right to a class action", 0.95, "class-action-waiver"),
        ("individual basis only", 0.85, "class-action-waiver"),
        ("not participate in a class action", 0.9, "class-action-waiver"),
        ("prohibit class arbitration", 0.9, "class-action-waiver"),
        ("no representative actions", 0.85, "class-action-waiver"),
    ]
    
    # Jury trial waiver patterns
    JURY_TRIAL_WAIVER_PATTERNS = [
        ("waive right to jury trial", 0.95, "binding"),
        ("waive jury trial", 0.95, "binding"),
        ("no right to jury trial", 0.9, "binding"),
        ("forfeit jury trial", 0.9, "binding"),
        ("jury trial waiver", 0.95, "binding"),
        ("waive trial by jury", 0.95, "binding"),
        ("no jury", 0.7, "binding"),
    ]
    
    # Legal phrases that indicate arbitration context
    LEGAL_PHRASES = [
        ("disputes arising out of or relating to", 0.6, "general"),
        ("any dispute or claim", 0.5, "general"),
        ("resolution of disputes", 0.5, "general"),
        ("settle by arbitration", 0.85, "binding"),
        ("governed by the federal arbitration act", 0.9, "binding"),
        ("federal arbitration act", 0.85, "binding"),
        ("faa", 0.6, "general"),
        ("rules of arbitration", 0.7, "general"),
        ("arbitration rules", 0.7, "general"),
        ("shall be settled", 0.4, "general"),
        ("final and binding", 0.75, "binding"),
        ("sole and exclusive remedy", 0.8, "binding"),
    ]
    
    # Arbitration provider indicators
    ARBITRATION_PROVIDERS = [
        ("american arbitration association", 0.8, "general"),
        ("aaa", 0.6, "general"),
        ("jams", 0.7, "general"),
        ("judicial arbitration and mediation services", 0.8, "general"),
        ("international chamber of commerce", 0.7, "general"),
        ("icc arbitration", 0.8, "general"),
        ("uncitral", 0.7, "general"),
        ("lcia", 0.7, "general"),
        ("london court of international arbitration", 0.8, "general"),
        ("national arbitration forum", 0.7, "general"),
        ("finra", 0.6, "general"),
    ]
    
    # Negative indicators (reduce confidence)
    NEGATIVE_INDICATORS = [
        ("may elect arbitration", -0.3),
        ("optional arbitration", -0.3),
        ("court of competent jurisdiction", -0.2),
        ("right to sue", -0.25),
        ("litigation", -0.15),
        ("small claims court", -0.2),
        ("excluded from arbitration", -0.4),
        ("not subject to arbitration", -0.5),
        ("arbitration is voluntary", -0.4),
    ]
    
    # Regex patterns for complex matching
    REGEX_PATTERNS = [
        (r"(you|user|customer|party|parties)\s+(agree|agrees|consent|consents)\s+to\s+arbitrat", 0.85, "binding"),
        (r"arbitration\s+shall\s+be\s+(final|binding|exclusive)", 0.9, "binding"),
        (r"must\s+arbitrate\s+(any|all)\s+dispute", 0.9, "binding"),
        (r"required\s+to\s+arbitrate", 0.85, "binding"),
        (r"subject\s+to\s+(binding|mandatory)\s+arbitration", 0.9, "binding"),
        (r"resolve[sd]?\s+through\s+arbitration", 0.75, "general"),
        (r"by\s+arbitration\s+in\s+accordance", 0.8, "general"),
        (r"waive[sd]?\s+(your|any|the)\s+right\s+to", 0.7, "binding"),
        (r"arbitration\s+is\s+the\s+(exclusive|sole)\s+remedy", 0.9, "binding"),
    ]
    
    @classmethod
    def get_all_patterns(cls) -> List[ArbitrationPattern]:
        """Get all patterns as ArbitrationPattern objects."""
        patterns = []
        
        # Add keyword patterns
        for keyword, weight, arb_type in cls.HIGH_CONFIDENCE_KEYWORDS:
            patterns.append(ArbitrationPattern(keyword, weight, 'keyword', arb_type))
        
        for keyword, weight, arb_type in cls.MEDIUM_CONFIDENCE_KEYWORDS:
            patterns.append(ArbitrationPattern(keyword, weight, 'keyword', arb_type))
        
        for keyword, weight, arb_type in cls.CLASS_ACTION_WAIVER_PATTERNS:
            patterns.append(ArbitrationPattern(keyword, weight, 'keyword', arb_type))
        
        for keyword, weight, arb_type in cls.JURY_TRIAL_WAIVER_PATTERNS:
            patterns.append(ArbitrationPattern(keyword, weight, 'keyword', arb_type))
        
        for phrase, weight, arb_type in cls.LEGAL_PHRASES:
            patterns.append(ArbitrationPattern(phrase, weight, 'phrase', arb_type))
        
        for provider, weight, arb_type in cls.ARBITRATION_PROVIDERS:
            patterns.append(ArbitrationPattern(provider, weight, 'keyword', arb_type))
        
        # Add regex patterns
        for regex, weight, arb_type in cls.REGEX_PATTERNS:
            patterns.append(ArbitrationPattern(regex, weight, 'regex', arb_type))
        
        return patterns
    
    @classmethod
    def get_negative_indicators(cls) -> List[tuple]:
        """Get negative indicators that reduce confidence."""
        return cls.NEGATIVE_INDICATORS
    
    @classmethod
    def extract_arbitration_details(cls, text: str) -> Dict[str, any]:
        """Extract specific details about arbitration from text."""
        details = {
            'provider': None,
            'location': None,
            'rules': None,
            'class_action_waiver': False,
            'jury_trial_waiver': False,
            'opt_out_available': False,
            'time_limit': None,
            'fees': None,
        }
        
        text_lower = text.lower()
        
        # Check for providers
        for provider, _, _ in cls.ARBITRATION_PROVIDERS:
            if provider in text_lower:
                details['provider'] = provider.upper()
                break
        
        # Check for class action waiver
        for pattern, _, _ in cls.CLASS_ACTION_WAIVER_PATTERNS:
            if pattern in text_lower:
                details['class_action_waiver'] = True
                break
        
        # Check for jury trial waiver
        for pattern, _, _ in cls.JURY_TRIAL_WAIVER_PATTERNS:
            if pattern in text_lower:
                details['jury_trial_waiver'] = True
                break
        
        # Check for opt-out
        opt_out_patterns = [
            r"opt[\s-]?out\s+of\s+arbitration",
            r"reject\s+this\s+arbitration",
            r"decline\s+arbitration",
            r"within\s+\d+\s+days?\s+of",
        ]
        for pattern in opt_out_patterns:
            if re.search(pattern, text_lower):
                details['opt_out_available'] = True
                # Try to extract time limit
                time_match = re.search(r"within\s+(\d+)\s+days?", text_lower)
                if time_match:
                    details['time_limit'] = f"{time_match.group(1)} days"
                break
        
        # Check for location
        location_patterns = [
            r"arbitration\s+(?:shall|will|must)\s+(?:take\s+place|occur|be\s+held)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"venue\s+for\s+arbitration\s+(?:is|shall\s+be|will\s+be)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                details['location'] = match.group(1)
                break
        
        # Check for fee arrangements
        fee_patterns = [
            r"(each\s+party\s+(?:pays|bears)\s+(?:its|their)\s+own)",
            r"(filing\s+fees?\s+(?:paid|borne)\s+by)",
            r"(we\s+will\s+pay\s+(?:all|the)\s+(?:arbitration|filing)\s+fees)",
        ]
        for pattern in fee_patterns:
            match = re.search(pattern, text_lower)
            if match:
                details['fees'] = match.group(1)
                break
        
        return details
    
    @classmethod
    def get_context_window_size(cls) -> int:
        """Get recommended context window size for pattern matching."""
        return 500  # characters around the matched pattern
    
    @classmethod
    def preprocess_text(cls, text: str) -> str:
        """Preprocess text for better pattern matching."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[•·■□▪]', '', text)
        # Normalize quotes
        text = re.sub(r'[""''`´]', "'", text)
        return text.strip()