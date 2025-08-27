"""
Model Serving and Prediction Module for Arbitration Clause Detection

This module provides a production-ready interface for loading trained models
and making predictions with confidence scoring and fallback mechanisms.
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.app.ml.features import LegalFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArbitrationPredictor:
    """
    Production-ready arbitration clause detection predictor
    """
    
    def __init__(self, models_dir: str = None, model_name: str = "ensemble"):
        """
        Initialize predictor with trained models
        
        Args:
            models_dir: Directory containing trained models
            model_name: Name of the primary model to use ('ensemble', 'random_forest', etc.)
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.primary_model_name = model_name
        self.models = {}
        self.feature_extractor = None
        self.model_performance = {}
        self.config = {}
        self.fallback_rules = None
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        # Initialize the predictor
        self._load_models()
        self._load_fallback_rules()
    
    def _load_models(self):
        """Load trained models and feature extractor"""
        
        try:
            # Load feature extractor
            feature_path = self.models_dir / "feature_extractor.pkl"
            if feature_path.exists():
                self.feature_extractor = LegalFeatureExtractor.load(str(feature_path))
                logger.info("Feature extractor loaded successfully")
            else:
                raise FileNotFoundError(f"Feature extractor not found at {feature_path}")
            
            # Load configuration
            config_path = self.models_dir / "training_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Training configuration loaded")
            
            # Load model performance metrics
            performance_path = self.models_dir / "model_performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
                logger.info("Model performance metrics loaded")
            
            # Load available models
            model_files = list(self.models_dir.glob("*_model.pkl"))
            
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
            
            if not self.models:
                raise ValueError("No models could be loaded")
            
            # Set primary model
            if self.primary_model_name not in self.models:
                logger.warning(f"Primary model '{self.primary_model_name}' not found. Using best available model.")
                best_model = self._get_best_model()
                self.primary_model_name = best_model
            
            logger.info(f"Primary model set to: {self.primary_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _get_best_model(self) -> str:
        """Get the best performing model based on F1 score"""
        
        if not self.model_performance:
            # Fallback to first available model
            return list(self.models.keys())[0]
        
        best_model = max(
            self.model_performance.items(),
            key=lambda x: x[1].get('f1_score', 0)
        )[0]
        
        return best_model
    
    def _load_fallback_rules(self):
        """Load rule-based fallback patterns"""
        
        self.fallback_rules = {
            'strong_positive_patterns': [
                r'\bbinding\s+arbitration\b',
                r'\bmandatory\s+arbitration\b',
                r'\bwaive.*jury\s+trial\b',
                r'\bclass\s+action\s+waiver\b',
                r'\bAAA\s+rules\b',
                r'\bJAMS\s+arbitration\b',
                r'\barbitration\s+agreement\b'
            ],
            'moderate_positive_patterns': [
                r'\barbitration.*dispute\b',
                r'\bdispute.*arbitration\b',
                r'\barbitral\s+proceedings\b',
                r'\bfinal\s+and\s+binding\b',
                r'\bindividual\s+arbitration\b'
            ],
            'negative_patterns': [
                r'\bcourt\s+proceedings\b',
                r'\bjudicial\s+proceedings\b',
                r'\blitigation\b',
                r'\bfederal\s+court\b',
                r'\bstate\s+court\b',
                r'\bexclusive\s+jurisdiction\b'
            ]
        }
    
    def predict_single(self, text: str, return_confidence: bool = True, 
                      use_fallback: bool = True) -> Dict[str, Any]:
        """
        Predict arbitration clause presence for a single text
        
        Args:
            text: Text to analyze
            return_confidence: Whether to return confidence scores
            use_fallback: Whether to use rule-based fallback
        
        Returns:
            Dictionary with prediction results
        """
        
        try:
            # Extract features
            features = self.feature_extractor.transform([text])
            
            # Get prediction from primary model
            primary_model = self.models[self.primary_model_name]
            
            # Handle different model types
            if hasattr(primary_model, 'predict_proba'):
                prob = primary_model.predict_proba(features)[0]
                prediction = int(prob[1] > 0.5)
                confidence = float(prob[1])
            else:
                prediction = int(primary_model.predict(features)[0])
                confidence = 0.5  # Default confidence for models without probability
            
            # Determine confidence level
            if confidence >= self.confidence_thresholds['high_confidence']:
                confidence_level = 'high'
            elif confidence >= self.confidence_thresholds['medium_confidence']:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            result = {
                'prediction': prediction,
                'has_arbitration': bool(prediction),
                'confidence_score': confidence,
                'confidence_level': confidence_level,
                'model_used': self.primary_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add ensemble predictions if available
            if return_confidence and len(self.models) > 1:
                ensemble_predictions = self._get_ensemble_predictions(features)
                result['ensemble_predictions'] = ensemble_predictions
                result['prediction_consensus'] = self._calculate_consensus(ensemble_predictions)
            
            # Apply rule-based fallback if confidence is low or model disagrees
            if use_fallback and (confidence_level == 'low' or self._should_use_fallback(text, prediction)):
                fallback_result = self._apply_fallback_rules(text)
                result['fallback_prediction'] = fallback_result
                
                # Override prediction if fallback is strongly confident
                if fallback_result['confidence'] > 0.7 and fallback_result['prediction'] != prediction:
                    result['prediction'] = fallback_result['prediction']
                    result['has_arbitration'] = bool(fallback_result['prediction'])
                    result['model_used'] = 'rule_based_fallback'
                    result['confidence_override'] = True
            
            # Add explanatory features
            if return_confidence:
                result['explanation'] = self._generate_explanation(text, features, prediction)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            # Emergency fallback to rules only
            if use_fallback:
                fallback_result = self._apply_fallback_rules(text)
                return {
                    'prediction': fallback_result['prediction'],
                    'has_arbitration': bool(fallback_result['prediction']),
                    'confidence_score': fallback_result['confidence'],
                    'confidence_level': 'fallback',
                    'model_used': 'emergency_fallback',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise
    
    def predict_batch(self, texts: List[str], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Predict arbitration clauses for multiple texts
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
        
        Returns:
            List of prediction results
        """
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Extract features for batch
                features = self.feature_extractor.transform(batch_texts)
                
                # Get predictions
                primary_model = self.models[self.primary_model_name]
                
                if hasattr(primary_model, 'predict_proba'):
                    probabilities = primary_model.predict_proba(features)[:, 1]
                    predictions = (probabilities > 0.5).astype(int)
                else:
                    predictions = primary_model.predict(features)
                    probabilities = np.full(len(predictions), 0.5)
                
                # Process each prediction
                for j, (text, pred, prob) in enumerate(zip(batch_texts, predictions, probabilities)):
                    result = {
                        'prediction': int(pred),
                        'has_arbitration': bool(pred),
                        'confidence_score': float(prob),
                        'confidence_level': self._get_confidence_level(float(prob)),
                        'model_used': self.primary_model_name,
                        'batch_index': i + j,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Batch prediction failed for batch starting at index {i}: {e}")
                
                # Process individually with fallback
                for j, text in enumerate(batch_texts):
                    try:
                        result = self.predict_single(text, return_confidence=False, use_fallback=True)
                        result['batch_index'] = i + j
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"Individual fallback failed for text {i+j}: {e2}")
                        results.append({
                            'prediction': 0,
                            'has_arbitration': False,
                            'confidence_score': 0.0,
                            'confidence_level': 'error',
                            'model_used': 'error_fallback',
                            'batch_index': i + j,
                            'error': str(e2),
                            'timestamp': datetime.now().isoformat()
                        })
        
        return results
    
    def _get_ensemble_predictions(self, features) -> Dict[str, Any]:
        """Get predictions from all available models"""
        
        ensemble_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features)[0, 1]
                    pred = int(prob > 0.5)
                else:
                    pred = int(model.predict(features)[0])
                    prob = 0.5
                
                ensemble_predictions[model_name] = {
                    'prediction': pred,
                    'confidence': float(prob)
                }
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_name}: {e}")
        
        return ensemble_predictions
    
    def _calculate_consensus(self, ensemble_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus across multiple models"""
        
        predictions = [p['prediction'] for p in ensemble_predictions.values()]
        confidences = [p['confidence'] for p in ensemble_predictions.values()]
        
        if not predictions:
            return {'consensus': 0, 'agreement_ratio': 0.0, 'avg_confidence': 0.0}
        
        consensus = int(np.mean(predictions) > 0.5)
        agreement_ratio = sum(1 for p in predictions if p == consensus) / len(predictions)
        avg_confidence = np.mean(confidences)
        
        return {
            'consensus': consensus,
            'agreement_ratio': agreement_ratio,
            'avg_confidence': avg_confidence,
            'total_models': len(predictions)
        }
    
    def _should_use_fallback(self, text: str, prediction: int) -> bool:
        """Determine if fallback rules should be applied"""
        
        # Use fallback if text is very short or very long
        word_count = len(text.split())
        if word_count < 10 or word_count > 1000:
            return True
        
        # Use fallback if no clear arbitration keywords found
        text_lower = text.lower()
        arbitration_keywords = ['arbitration', 'arbitrator', 'arbitral', 'dispute resolution']
        
        if prediction == 1 and not any(keyword in text_lower for keyword in arbitration_keywords):
            return True
        
        return False
    
    def _apply_fallback_rules(self, text: str) -> Dict[str, Any]:
        """Apply rule-based classification as fallback"""
        
        text_lower = text.lower()
        score = 0
        matched_patterns = []
        
        # Check strong positive patterns
        for pattern in self.fallback_rules['strong_positive_patterns']:
            import re
            if re.search(pattern, text_lower):
                score += 2
                matched_patterns.append(f"strong: {pattern}")
        
        # Check moderate positive patterns  
        for pattern in self.fallback_rules['moderate_positive_patterns']:
            import re
            if re.search(pattern, text_lower):
                score += 1
                matched_patterns.append(f"moderate: {pattern}")
        
        # Check negative patterns
        for pattern in self.fallback_rules['negative_patterns']:
            import re
            if re.search(pattern, text_lower):
                score -= 1
                matched_patterns.append(f"negative: {pattern}")
        
        # Determine prediction and confidence
        if score >= 2:
            prediction = 1
            confidence = 0.8
        elif score >= 1:
            prediction = 1
            confidence = 0.6
        elif score <= -1:
            prediction = 0
            confidence = 0.7
        else:
            prediction = 0
            confidence = 0.3
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'score': score,
            'matched_patterns': matched_patterns
        }
    
    def _generate_explanation(self, text: str, features, prediction: int) -> Dict[str, Any]:
        """Generate explanation for the prediction"""
        
        explanation = {
            'prediction_reasoning': [],
            'key_indicators': [],
            'text_statistics': {}
        }
        
        text_lower = text.lower()
        
        # Key arbitration indicators
        arbitration_indicators = {
            'binding arbitration': 'strong indicator',
            'mandatory arbitration': 'strong indicator', 
            'jury trial waiver': 'strong indicator',
            'class action waiver': 'strong indicator',
            'arbitration': 'moderate indicator',
            'dispute resolution': 'moderate indicator',
            'AAA': 'moderate indicator',
            'JAMS': 'moderate indicator'
        }
        
        for indicator, strength in arbitration_indicators.items():
            if indicator in text_lower:
                explanation['key_indicators'].append({
                    'term': indicator,
                    'strength': strength,
                    'context': self._extract_context(text, indicator)
                })
        
        # Text statistics
        words = text.split()
        explanation['text_statistics'] = {
            'word_count': len(words),
            'character_count': len(text),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
        
        # Prediction reasoning
        if prediction == 1:
            explanation['prediction_reasoning'].append("Text contains arbitration-related language")
            if len(explanation['key_indicators']) > 0:
                explanation['prediction_reasoning'].append(f"Found {len(explanation['key_indicators'])} key indicators")
        else:
            explanation['prediction_reasoning'].append("No strong arbitration indicators found")
            if 'court' in text_lower or 'litigation' in text_lower:
                explanation['prediction_reasoning'].append("Text mentions court/litigation proceedings")
        
        return explanation
    
    def _extract_context(self, text: str, term: str, context_window: int = 50) -> str:
        """Extract context around a term"""
        
        text_lower = text.lower()
        term_lower = term.lower()
        
        index = text_lower.find(term_lower)
        if index == -1:
            return ""
        
        start = max(0, index - context_window)
        end = min(len(text), index + len(term) + context_window)
        
        return text[start:end].strip()
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string"""
        
        if confidence >= self.confidence_thresholds['high_confidence']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            return 'medium'
        else:
            return 'low'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            'models_available': list(self.models.keys()),
            'primary_model': self.primary_model_name,
            'feature_extractor_loaded': self.feature_extractor is not None,
            'model_performance': self.model_performance,
            'config': self.config,
            'confidence_thresholds': self.confidence_thresholds
        }
    
    def validate_prediction(self, text: str, expected_label: int) -> Dict[str, Any]:
        """Validate a prediction against expected result"""
        
        result = self.predict_single(text, return_confidence=True, use_fallback=True)
        
        validation = {
            'correct': result['prediction'] == expected_label,
            'prediction': result['prediction'],
            'expected': expected_label,
            'confidence': result['confidence_score'],
            'agreement': result.get('prediction_consensus', {}).get('agreement_ratio', 0.0)
        }
        
        return validation


def main():
    """Test the predictor"""
    
    # Test texts
    test_texts = [
        "Any dispute arising out of these terms shall be resolved through binding arbitration in Delaware.",
        "You waive your right to jury trial and class action participation for all disputes.",
        "Legal action may be brought in federal or state court in California.",
        "This privacy policy explains how we collect and use your personal information.",
        "BINDING ARBITRATION: All disputes MUST be resolved through individual arbitration with AAA.",
        "The parties may choose arbitration or litigation to resolve disputes.",
        "All claims must be submitted to binding arbitration administered by JAMS."
    ]
    
    expected_labels = [1, 1, 0, 0, 1, 0, 1]  # Expected arbitration presence
    
    try:
        # Initialize predictor
        predictor = ArbitrationPredictor()
        
        print("Model Information:")
        info = predictor.get_model_info()
        print(f"Available models: {info['models_available']}")
        print(f"Primary model: {info['primary_model']}")
        
        print("\nTesting individual predictions:")
        print("-" * 80)
        
        for i, text in enumerate(test_texts):
            result = predictor.predict_single(text, return_confidence=True)
            
            print(f"\nText {i+1}: {text[:60]}...")
            print(f"Prediction: {'ARBITRATION' if result['has_arbitration'] else 'NO ARBITRATION'}")
            print(f"Confidence: {result['confidence_score']:.3f} ({result['confidence_level']})")
            print(f"Model used: {result['model_used']}")
            
            if 'explanation' in result:
                key_indicators = result['explanation']['key_indicators']
                if key_indicators:
                    print(f"Key indicators: {[ind['term'] for ind in key_indicators]}")
        
        print("\nTesting batch prediction:")
        print("-" * 40)
        
        batch_results = predictor.predict_batch(test_texts)
        
        correct_predictions = 0
        for i, (result, expected) in enumerate(zip(batch_results, expected_labels)):
            is_correct = result['prediction'] == expected
            correct_predictions += is_correct
            
            print(f"Text {i+1}: {'✓' if is_correct else '✗'} "
                  f"(Pred: {result['prediction']}, Expected: {expected}, "
                  f"Conf: {result['confidence_score']:.3f})")
        
        accuracy = correct_predictions / len(test_texts)
        print(f"\nBatch accuracy: {accuracy:.3f} ({correct_predictions}/{len(test_texts)})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure models are trained and saved first by running train.py")


if __name__ == "__main__":
    main()