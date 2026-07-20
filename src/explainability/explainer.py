import numpy as np
from typing import List, Dict, Tuple
import shap
from lime.lime_text import LimeTextExplainer

class ArbitrationExplainer:
    def __init__(self, detector):
        """Initialize explainability module"""
        self.detector = detector
        try:
            self.lime_explainer = LimeTextExplainer(class_names=['Not Arbitration', 'Arbitration'])
        except ImportError:
            print("LIME not available. Some explainability features will be limited.")
            self.lime_explainer = None
        
    def explain_detection(self, text: str, detection_result) -> Dict:
        """
        Provide detailed explanation for detection
        
        Args:
            text: Input text
            detection_result: Result from detector
            
        Returns:
            Detailed explanation dictionary
        """
        explanation = {
            'confidence_breakdown': self._explain_confidence(detection_result),
            'key_indicators': self._extract_key_indicators(text, detection_result),
            'pattern_analysis': self._explain_patterns(detection_result),
            'interpretability': self._generate_lime_explanation(text) if self.lime_explainer else {},
            'decision_path': self._trace_decision_path(text, detection_result)
        }
        
        return explanation
    
    def _explain_confidence(self, detection_result) -> Dict:
        """Break down confidence scores"""
        return {
            'overall_confidence': detection_result.confidence,
            'semantic_confidence': detection_result.semantic_score,
            'pattern_confidence': len(detection_result.pattern_matches) / 10,  # Normalize
            'explanation': self._generate_confidence_explanation(detection_result)
        }
    
    def _generate_confidence_explanation(self, detection_result) -> str:
        """Generate human-readable confidence explanation"""
        if detection_result.confidence > 0.9:
            return "Very high confidence - multiple strong indicators present"
        elif detection_result.confidence > 0.7:
            return "High confidence - clear arbitration language detected"
        elif detection_result.confidence > 0.5:
            return "Moderate confidence - some arbitration indicators present"
        else:
            return "Low confidence - few arbitration indicators found"
    
    def _extract_key_indicators(self, text: str, detection_result) -> List[Dict]:
        """Extract and highlight key indicators"""
        indicators = []
        
        # Pattern-based indicators
        for pattern_match in detection_result.pattern_matches[:5]:
            indicators.append({
                'type': 'pattern',
                'description': pattern_match,
                'importance': 'high'
            })
        
        # Keyword indicators
        key_phrases = [
            'binding arbitration', 'class action waiver', 
            'JAMS', 'AAA', 'dispute resolution'
        ]
        
        text_lower = text.lower()
        for phrase in key_phrases:
            if phrase.lower() in text_lower:
                # Find position
                start_idx = text_lower.index(phrase.lower())
                end_idx = start_idx + len(phrase)
                
                indicators.append({
                    'type': 'keyword',
                    'description': f"Found '{phrase}'",
                    'text_snippet': text[max(0, start_idx-20):min(len(text), end_idx+20)],
                    'importance': 'medium'
                })
        
        return indicators
    
    def _explain_patterns(self, detection_result) -> Dict:
        """Explain pattern matching results"""
        pattern_categories = {}
        
        for match in detection_result.pattern_matches:
            category = match.split(':')[0] if ':' in match else 'general'
            if category not in pattern_categories:
                pattern_categories[category] = []
            pattern_categories[category].append(match)
        
        return {
            'categories_found': list(pattern_categories.keys()),
            'pattern_details': pattern_categories,
            'interpretation': self._interpret_patterns(pattern_categories)
        }
    
    def _interpret_patterns(self, pattern_categories: Dict) -> str:
        """Interpret pattern matching results"""
        if 'mandatory_arbitration' in pattern_categories:
            return "Mandatory arbitration clause detected - binding on parties"
        elif 'class_action_waiver' in pattern_categories:
            return "Class action waiver present - individual arbitration only"
        elif pattern_categories:
            return "Arbitration-related language detected"
        else:
            return "No specific arbitration patterns found"
    
    def _generate_lime_explanation(self, text: str) -> Dict:
        """Generate LIME explanation for interpretability"""
        if not self.lime_explainer:
            return {'error': 'LIME not available'}
        
        def predict_proba(texts):
            """Prediction function for LIME"""
            results = []
            for t in texts:
                detection = self.detector.detect(t)
                prob_not_arb = 1 - detection.confidence
                prob_arb = detection.confidence
                results.append([prob_not_arb, prob_arb])
            return np.array(results)
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            text[:1000],  # Limit text length for performance
            predict_proba,
            num_features=10
        )
        
        # Extract important words
        important_words = []
        for word, importance in exp.as_list():
            important_words.append({
                'word': word,
                'importance': importance,
                'impact': 'positive' if importance > 0 else 'negative'
            })
        
        return {
            'important_words': important_words,
            'visualization': exp.as_html()  # Can be rendered in UI
        }
    
    def _trace_decision_path(self, text: str, detection_result) -> List[str]:
        """Trace the decision path for transparency"""
        path = []
        
        # Step 1: Document intake
        path.append("1. Document received and preprocessed")
        
        # Step 2: Section detection
        if hasattr(detection_result, 'section_detected'):
            path.append("2. Relevant section identified through structure analysis")
        else:
            path.append("2. Full document analyzed (no clear sections)")
        
        # Step 3: Pattern matching
        if detection_result.pattern_matches:
            path.append(f"3. Pattern matching found {len(detection_result.pattern_matches)} matches")
        else:
            path.append("3. No pattern matches found")
        
        # Step 4: Semantic analysis
        path.append(f"4. Semantic analysis score: {detection_result.semantic_score:.2f}")
        
        # Step 5: Final decision
        if detection_result.is_arbitration:
            path.append(f"5. DETECTED: Arbitration clause with {detection_result.confidence:.1%} confidence")
        else:
            path.append(f"5. NOT DETECTED: Confidence below threshold ({detection_result.confidence:.1%})")
        
        return path

class VisualExplainer:
    """Generate visual explanations for UI"""
    
    def generate_confidence_chart(self, explanation: Dict) -> Dict:
        """Generate data for confidence visualization"""
        breakdown = explanation['confidence_breakdown']
        
        return {
            'type': 'bar_chart',
            'data': [
                {'category': 'Semantic Analysis', 'score': breakdown['semantic_confidence']},
                {'category': 'Pattern Matching', 'score': breakdown['pattern_confidence']},
                {'category': 'Overall', 'score': breakdown['overall_confidence']}
            ],
            'title': 'Confidence Score Breakdown',
            'y_axis': 'Confidence Score (0-1)',
            'x_axis': 'Detection Method'
        }
    
    def generate_indicator_highlight(self, text: str, indicators: List[Dict]) -> str:
        """Generate HTML with highlighted indicators"""
        html = text
        
        # Sort indicators by position (if available)
        for indicator in indicators:
            if 'text_snippet' in indicator:
                snippet = indicator['text_snippet']
                # Wrap in span for highlighting
                highlighted = f'<span class="highlight-{indicator["importance"]}">{snippet}</span>'
                html = html.replace(snippet, highlighted)
        
        return html
