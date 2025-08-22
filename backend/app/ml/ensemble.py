"""
Ensemble approach for arbitration clause detection
Combines rule-based, semantic similarity, and statistical classifiers
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import re
from collections import defaultdict
import pickle
import os
from datetime import datetime

from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import mlflow

from .feature_extraction import LegalFeatureExtractor
from .classifier import ArbitrationClassifier
from .ner import LegalEntityRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleBasedDetector:
    """
    Rule-based arbitration clause detector using legal patterns and keywords
    """
    
    def __init__(self, precision_focus: bool = True):
        self.precision_focus = precision_focus
        self.rules = self._create_detection_rules()
        self.weights = self._create_rule_weights()
        
    def _create_detection_rules(self) -> List[Dict]:
        """
        Create comprehensive rule set for arbitration detection
        """
        rules = [
            # High confidence rules (strong indicators)
            {
                "name": "binding_arbitration_explicit",
                "pattern": r"\bbinding\s+arbitration\b",
                "confidence": 0.95,
                "required_context": None
            },
            {
                "name": "final_arbitration_explicit", 
                "pattern": r"\bfinal\s+(?:and\s+)?(?:binding\s+)?arbitration\b",
                "confidence": 0.90,
                "required_context": None
            },
            {
                "name": "arbitration_administered_by",
                "pattern": r"\barbitration\s+administered\s+by\b",
                "confidence": 0.90,
                "required_context": None
            },
            {
                "name": "submit_to_arbitration",
                "pattern": r"\bsubmit\s+(?:any\s+)?(?:dispute|controversy|claim)s?\s+to\s+arbitration\b",
                "confidence": 0.85,
                "required_context": None
            },
            
            # Medium confidence rules
            {
                "name": "dispute_arbitration_combo",
                "pattern": r"\b(?:any|all)\s+(?:dispute|controversy|claim)s?\s+.{0,50}\s+arbitration\b",
                "confidence": 0.75,
                "required_context": None
            },
            {
                "name": "arbitral_tribunal",
                "pattern": r"\barbitral\s+tribunal\b",
                "confidence": 0.70,
                "required_context": None
            },
            {
                "name": "arbitration_rules_reference",
                "pattern": r"\b(?:under|pursuant\s+to|in\s+accordance\s+with)\s+(?:the\s+)?(?:rules|procedures)\s+of\s+(?:the\s+)?(?:aaa|jams|icc|lcia|siac|uncitral)\b",
                "confidence": 0.80,
                "required_context": None
            },
            {
                "name": "exclusive_arbitration",
                "pattern": r"\bexclusively\s+through\s+(?:binding\s+)?arbitration\b",
                "confidence": 0.85,
                "required_context": None
            },
            
            # Waiver patterns (strong indicators)
            {
                "name": "jury_trial_waiver",
                "pattern": r"\bwaiv(?:e|er|ing)\s+(?:their|the)\s+right\s+to\s+(?:a\s+)?jury\s+trial\b",
                "confidence": 0.80,
                "required_context": ["arbitration", "dispute"]
            },
            {
                "name": "class_action_waiver", 
                "pattern": r"\bclass\s+action\s+waiver\b",
                "confidence": 0.75,
                "required_context": ["arbitration", "dispute"]
            },
            
            # Institutional arbitration
            {
                "name": "aaa_arbitration",
                "pattern": r"\b(?:american\s+arbitration\s+association|aaa)\b",
                "confidence": 0.70,
                "required_context": ["arbitration", "dispute", "rules"]
            },
            {
                "name": "jams_arbitration",
                "pattern": r"\bjams\b",
                "confidence": 0.70,
                "required_context": ["arbitration", "dispute", "rules"]
            },
            {
                "name": "icc_arbitration",
                "pattern": r"\b(?:international\s+chamber\s+of\s+commerce|icc)\b",
                "confidence": 0.70,
                "required_context": ["arbitration", "dispute", "rules"]
            }
        ]
        
        return rules
    
    def _create_rule_weights(self) -> Dict[str, float]:
        """
        Create weights for different rule types
        """
        return {
            "explicit_arbitration": 3.0,
            "institutional_reference": 2.5,
            "procedural_terms": 2.0,
            "waiver_terms": 1.5,
            "contextual_indicators": 1.0
        }
    
    def _check_required_context(self, text: str, required_context: List[str]) -> bool:
        """
        Check if required context terms are present in text
        """
        if not required_context:
            return True
        
        text_lower = text.lower()
        return any(context_term in text_lower for context_term in required_context)
    
    def _apply_negation_rules(self, text: str, base_score: float) -> float:
        """
        Apply negation rules to reduce false positives
        """
        negation_patterns = [
            r"\bno\s+arbitration\b",
            r"\bnot\s+subject\s+to\s+arbitration\b", 
            r"\bexcept\s+for\s+arbitration\b",
            r"\bresolved\s+in\s+court\b",
            r"\bcourt\s+of\s+competent\s+jurisdiction\b",
            r"\bjudicial\s+proceedings\b"
        ]
        
        negation_count = 0
        for pattern in negation_patterns:
            negation_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Reduce score based on negation strength
        negation_penalty = min(0.8, negation_count * 0.3)
        return base_score * (1 - negation_penalty)
    
    def detect_arbitration(self, text: str) -> Dict[str, Any]:
        """
        Detect arbitration clauses using rule-based approach
        """
        text_lower = text.lower()
        total_score = 0.0
        triggered_rules = []
        
        # Apply each rule
        for rule in self.rules:
            pattern = rule["pattern"]
            confidence = rule["confidence"]
            required_context = rule["required_context"]
            
            # Check for pattern match
            matches = re.findall(pattern, text_lower)
            
            if matches and self._check_required_context(text_lower, required_context):
                rule_score = len(matches) * confidence
                total_score += rule_score
                
                triggered_rules.append({
                    "name": rule["name"],
                    "matches": len(matches),
                    "confidence": confidence,
                    "score": rule_score
                })
        
        # Apply negation rules
        total_score = self._apply_negation_rules(text_lower, total_score)
        
        # Normalize score (adjust based on empirical analysis)
        max_possible_score = 10.0  # Empirically determined
        normalized_score = min(1.0, total_score / max_possible_score)
        
        # Determine prediction based on threshold
        if self.precision_focus:
            threshold = 0.7  # High threshold for precision
        else:
            threshold = 0.5  # Balanced threshold
        
        prediction = 1 if normalized_score >= threshold else 0
        
        return {
            "prediction": prediction,
            "confidence": normalized_score,
            "raw_score": total_score,
            "triggered_rules": triggered_rules,
            "num_rules_triggered": len(triggered_rules)
        }
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict for multiple texts
        """
        predictions = []
        for text in texts:
            result = self.detect_arbitration(text)
            predictions.append(result["prediction"])
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for multiple texts
        """
        probabilities = []
        for text in texts:
            result = self.detect_arbitration(text)
            confidence = result["confidence"]
            probabilities.append([1 - confidence, confidence])
        
        return np.array(probabilities)


class SemanticSimilarityDetector:
    """
    Semantic similarity-based arbitration detection using embeddings
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model)
        self.arbitration_templates = self._create_arbitration_templates()
        self.template_embeddings = None
        self.threshold = 0.75
    
    def _create_arbitration_templates(self) -> List[str]:
        """
        Create template arbitration clauses for similarity comparison
        """
        templates = [
            "Any dispute arising under this agreement shall be resolved through binding arbitration.",
            "All controversies shall be settled by final and binding arbitration.",
            "The parties agree to submit any dispute to arbitration.",
            "Disputes will be resolved exclusively through binding arbitration.",
            "Any claim shall be subject to final arbitration.",
            "The parties waive their right to jury trial and agree to arbitration.",
            "All disputes shall be finally settled by arbitration administered by AAA.",
            "Any controversy arising out of this contract shall be settled by arbitration.",
            "Disputes must be resolved through individual arbitration proceedings.",
            "The arbitral tribunal shall have exclusive jurisdiction over disputes."
        ]
        return templates
    
    def fit(self, arbitration_texts: List[str] = None):
        """
        Fit the semantic detector by creating template embeddings
        """
        if arbitration_texts:
            # Use provided arbitration texts as templates
            self.arbitration_templates.extend(arbitration_texts)
        
        # Generate embeddings for templates
        self.template_embeddings = self.embedding_model.encode(self.arbitration_templates)
        logger.info(f"Semantic detector fitted with {len(self.arbitration_templates)} templates")
    
    def _calculate_similarity_score(self, text: str) -> float:
        """
        Calculate semantic similarity score with arbitration templates
        """
        if self.template_embeddings is None:
            self.fit()
        
        # Get embedding for input text
        text_embedding = self.embedding_model.encode([text])[0]
        
        # Calculate similarities with all templates
        similarities = []
        for template_embedding in self.template_embeddings:
            similarity = np.dot(text_embedding, template_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(template_embedding)
            )
            similarities.append(similarity)
        
        # Return maximum similarity
        return max(similarities)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict arbitration clauses based on semantic similarity
        """
        predictions = []
        for text in texts:
            similarity_score = self._calculate_similarity_score(text)
            prediction = 1 if similarity_score >= self.threshold else 0
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities based on semantic similarity
        """
        probabilities = []
        for text in texts:
            similarity_score = self._calculate_similarity_score(text)
            # Convert similarity to probability
            prob_positive = similarity_score
            prob_negative = 1 - similarity_score
            probabilities.append([prob_negative, prob_positive])
        
        return np.array(probabilities)


class ArbitrationEnsemble:
    """
    Ensemble classifier combining rule-based, semantic, and statistical approaches
    """
    
    def __init__(self,
                 model_save_path: str = "backend/models",
                 experiment_name: str = "arbitration_ensemble",
                 precision_focus: bool = True):
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        self.precision_focus = precision_focus
        
        # Initialize component models
        self.rule_based_detector = RuleBasedDetector(precision_focus=precision_focus)
        self.semantic_detector = SemanticSimilarityDetector()
        self.statistical_classifier = ArbitrationClassifier(model_save_path=model_save_path)
        self.feature_extractor = LegalFeatureExtractor()
        
        # Ensemble configuration
        self.ensemble_weights = {"rule_based": 0.4, "semantic": 0.3, "statistical": 0.3}
        self.voting_classifier = None
        self.calibrated_ensemble = None
        self.optimal_threshold = 0.5
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        os.makedirs(model_save_path, exist_ok=True)
    
    def train_ensemble(self,
                      train_texts: List[str],
                      train_labels: List[int],
                      val_texts: List[str] = None,
                      val_labels: List[int] = None) -> Dict[str, Any]:
        """
        Train the ensemble model with all components
        """
        logger.info("Training ensemble model...")
        
        with mlflow.start_run(run_name="ensemble_training"):
            results = {}
            
            # 1. Train semantic detector
            logger.info("Training semantic detector...")
            arbitration_texts = [text for text, label in zip(train_texts, train_labels) if label == 1]
            self.semantic_detector.fit(arbitration_texts)
            
            # 2. Train statistical classifier
            logger.info("Training statistical classifier...")
            # Extract features for statistical model
            train_features_df, train_embeddings = self.feature_extractor.extract_features_batch(train_texts)
            
            if val_texts and val_labels:
                val_features_df, val_embeddings = self.feature_extractor.extract_features_batch(val_texts)
                
                # Combine features
                train_features_combined = np.hstack([train_features_df.values, train_embeddings])
                val_features_combined = np.hstack([val_features_df.values, val_embeddings])
                
                # Train statistical models
                statistical_results = self.statistical_classifier.train_statistical_classifier(
                    train_features_combined, train_labels, val_features_combined, val_labels
                )
                results['statistical_training'] = statistical_results
            else:
                train_features_combined = np.hstack([train_features_df.values, train_embeddings])
                statistical_results = self.statistical_classifier.train_statistical_classifier(
                    train_features_combined, train_labels
                )
                results['statistical_training'] = statistical_results
            
            # 3. Create voting ensemble
            logger.info("Creating voting ensemble...")
            
            # Note: For sklearn VotingClassifier, we need sklearn-compatible estimators
            # Here we'll implement a custom ensemble approach
            
            # 4. Optimize ensemble weights
            if val_texts and val_labels:
                logger.info("Optimizing ensemble weights...")
                optimal_weights = self._optimize_ensemble_weights(val_texts, val_labels)
                self.ensemble_weights = optimal_weights
                results['optimal_weights'] = optimal_weights
                
                # Find optimal threshold
                val_predictions = self._get_ensemble_predictions(val_texts)
                self.optimal_threshold = self._find_optimal_threshold(val_labels, val_predictions)
                results['optimal_threshold'] = self.optimal_threshold
            
            # Log ensemble configuration
            mlflow.log_params({
                "ensemble_weights": self.ensemble_weights,
                "optimal_threshold": self.optimal_threshold,
                "precision_focus": self.precision_focus
            })
            
            logger.info("Ensemble training completed")
            return results
    
    def _get_individual_predictions(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Get predictions from all individual models
        """
        predictions = {}
        
        # Rule-based predictions
        predictions['rule_based'] = self.rule_based_detector.predict_proba(texts)[:, 1]
        
        # Semantic predictions
        predictions['semantic'] = self.semantic_detector.predict_proba(texts)[:, 1]
        
        # Statistical predictions
        if 'statistical' in self.statistical_classifier.models:
            # Extract features for statistical model
            features_df, embeddings = self.feature_extractor.extract_features_batch(texts)
            features_combined = np.hstack([features_df.values, embeddings])
            predictions['statistical'] = self.statistical_classifier.models['statistical'].predict_proba(features_combined)[:, 1]
        else:
            # Fallback to rule-based if statistical model not trained
            predictions['statistical'] = predictions['rule_based']
        
        return predictions
    
    def _get_ensemble_predictions(self, texts: List[str]) -> np.ndarray:
        """
        Get weighted ensemble predictions
        """
        individual_preds = self._get_individual_predictions(texts)
        
        # Weighted combination
        ensemble_probs = (
            self.ensemble_weights['rule_based'] * individual_preds['rule_based'] +
            self.ensemble_weights['semantic'] * individual_preds['semantic'] +
            self.ensemble_weights['statistical'] * individual_preds['statistical']
        )
        
        return ensemble_probs
    
    def _optimize_ensemble_weights(self, val_texts: List[str], val_labels: List[int]) -> Dict[str, float]:
        """
        Optimize ensemble weights for best performance
        """
        logger.info("Optimizing ensemble weights...")
        
        individual_preds = self._get_individual_predictions(val_texts)
        
        best_weights = None
        best_score = 0.0
        
        # Grid search over weight combinations
        weight_combinations = [
            {"rule_based": 0.5, "semantic": 0.3, "statistical": 0.2},
            {"rule_based": 0.4, "semantic": 0.3, "statistical": 0.3},
            {"rule_based": 0.3, "semantic": 0.4, "statistical": 0.3},
            {"rule_based": 0.3, "semantic": 0.3, "statistical": 0.4},
            {"rule_based": 0.6, "semantic": 0.2, "statistical": 0.2},
            {"rule_based": 0.2, "semantic": 0.6, "statistical": 0.2},
            {"rule_based": 0.2, "semantic": 0.2, "statistical": 0.6},
            {"rule_based": 0.33, "semantic": 0.33, "statistical": 0.34}
        ]
        
        for weights in weight_combinations:
            # Calculate ensemble predictions
            ensemble_probs = (
                weights['rule_based'] * individual_preds['rule_based'] +
                weights['semantic'] * individual_preds['semantic'] +
                weights['statistical'] * individual_preds['statistical']
            )
            
            # Find optimal threshold for this weight combination
            threshold = self._find_optimal_threshold(val_labels, ensemble_probs)
            ensemble_preds = (ensemble_probs >= threshold).astype(int)
            
            # Calculate precision (our primary metric)
            precision = precision_score(val_labels, ensemble_preds, zero_division=0)
            
            if precision > best_score:
                best_score = precision
                best_weights = weights
        
        logger.info(f"Best ensemble weights: {best_weights} (precision: {best_score:.4f})")
        return best_weights
    
    def _find_optimal_threshold(self, y_true: List[int], y_proba: np.ndarray) -> float:
        """
        Find optimal threshold for high precision
        """
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        if self.precision_focus:
            # Find threshold that achieves target precision (95%)
            target_precision = 0.95
            high_precision_indices = precision >= target_precision
            
            if np.any(high_precision_indices):
                # Choose threshold with highest recall among high precision options
                valid_indices = np.where(high_precision_indices)[0]
                best_index = valid_indices[np.argmax(recall[valid_indices])]
                optimal_threshold = thresholds[best_index]
            else:
                # Fallback to highest precision threshold
                optimal_threshold = thresholds[np.argmax(precision)]
        else:
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        return optimal_threshold
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make ensemble predictions
        """
        ensemble_probs = self._get_ensemble_predictions(texts)
        predictions = (ensemble_probs >= self.optimal_threshold).astype(int)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get ensemble prediction probabilities
        """
        ensemble_probs = self._get_ensemble_predictions(texts)
        return np.column_stack([1 - ensemble_probs, ensemble_probs])
    
    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """
        Make prediction with detailed explanation from each component
        """
        individual_preds = self._get_individual_predictions([text])
        ensemble_prob = self._get_ensemble_predictions([text])[0]
        final_prediction = int(ensemble_prob >= self.optimal_threshold)
        
        # Get detailed explanations
        rule_explanation = self.rule_based_detector.detect_arbitration(text)
        
        return {
            "final_prediction": final_prediction,
            "ensemble_probability": ensemble_prob,
            "individual_predictions": {
                "rule_based": individual_preds['rule_based'][0],
                "semantic": individual_preds['semantic'][0],
                "statistical": individual_preds['statistical'][0]
            },
            "ensemble_weights": self.ensemble_weights,
            "optimal_threshold": self.optimal_threshold,
            "rule_based_explanation": rule_explanation
        }
    
    def evaluate_ensemble(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of ensemble performance
        """
        logger.info("Evaluating ensemble performance...")
        
        with mlflow.start_run(run_name="ensemble_evaluation", nested=True):
            # Get predictions
            ensemble_probs = self._get_ensemble_predictions(test_texts)
            ensemble_preds = (ensemble_probs >= self.optimal_threshold).astype(int)
            
            # Get individual model predictions for comparison
            individual_preds = self._get_individual_predictions(test_texts)
            
            # Calculate metrics for ensemble
            ensemble_metrics = {
                "ensemble_precision": precision_score(test_labels, ensemble_preds),
                "ensemble_recall": recall_score(test_labels, ensemble_preds),
                "ensemble_f1": f1_score(test_labels, ensemble_preds),
                "ensemble_auc": roc_auc_score(test_labels, ensemble_probs)
            }
            
            # Calculate metrics for individual models
            for model_name, probs in individual_preds.items():
                preds = (probs >= 0.5).astype(int)
                ensemble_metrics.update({
                    f"{model_name}_precision": precision_score(test_labels, preds),
                    f"{model_name}_recall": recall_score(test_labels, preds),
                    f"{model_name}_f1": f1_score(test_labels, preds),
                    f"{model_name}_auc": roc_auc_score(test_labels, probs)
                })
            
            # Log metrics
            mlflow.log_metrics(ensemble_metrics)
            
            logger.info(f"Ensemble evaluation complete: {ensemble_metrics}")
            return ensemble_metrics
    
    def save_ensemble(self, version: str = None) -> str:
        """
        Save the complete ensemble model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ensemble_data = {
            'rule_based_detector': self.rule_based_detector,
            'semantic_detector': self.semantic_detector,
            'statistical_classifier': self.statistical_classifier,
            'feature_extractor': self.feature_extractor,
            'ensemble_weights': self.ensemble_weights,
            'optimal_threshold': self.optimal_threshold,
            'precision_focus': self.precision_focus,
            'version': version,
            'created_at': datetime.now().isoformat()
        }
        
        ensemble_path = os.path.join(self.model_save_path, f"arbitration_ensemble_v{version}.pkl")
        
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info(f"Ensemble saved to {ensemble_path}")
        return ensemble_path
    
    def load_ensemble(self, ensemble_path: str):
        """
        Load a saved ensemble model
        """
        with open(ensemble_path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.rule_based_detector = ensemble_data['rule_based_detector']
        self.semantic_detector = ensemble_data['semantic_detector']
        self.statistical_classifier = ensemble_data['statistical_classifier']
        self.feature_extractor = ensemble_data['feature_extractor']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.optimal_threshold = ensemble_data['optimal_threshold']
        self.precision_focus = ensemble_data['precision_focus']
        
        logger.info(f"Ensemble loaded from {ensemble_path}")


def demo_ensemble():
    """
    Demonstrate ensemble functionality
    """
    # Sample data
    arbitration_texts = [
        "Any dispute arising under this agreement shall be resolved through binding arbitration administered by the American Arbitration Association.",
        "All controversies shall be settled by final and binding arbitration under AAA rules.",
        "The parties agree to submit any dispute to arbitration under JAMS procedures.",
    ]
    
    non_arbitration_texts = [
        "Any dispute shall be resolved in the courts of New York.",
        "The parties retain the right to seek judicial remedies.",
        "This agreement shall be governed by the laws of California.",
    ]
    
    # Create ensemble
    ensemble = ArbitrationEnsemble()
    
    # Train ensemble
    train_texts = arbitration_texts + non_arbitration_texts
    train_labels = [1] * len(arbitration_texts) + [0] * len(non_arbitration_texts)
    
    results = ensemble.train_ensemble(train_texts, train_labels)
    
    # Test prediction with explanation
    test_text = "Disputes will be resolved exclusively through binding arbitration."
    explanation = ensemble.predict_with_explanation(test_text)
    
    print("Ensemble Demo Results:")
    print(f"Text: {test_text}")
    print(f"Prediction: {explanation['final_prediction']}")
    print(f"Ensemble probability: {explanation['ensemble_probability']:.3f}")
    print(f"Individual predictions: {explanation['individual_predictions']}")


if __name__ == "__main__":
    demo_ensemble()