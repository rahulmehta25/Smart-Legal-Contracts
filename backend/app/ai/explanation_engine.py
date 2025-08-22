"""
Explanation engine for model interpretability using LIME, SHAP, and custom methods.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import shap
import lime
import lime.lime_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import logging
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Comprehensive explanation for a model prediction."""
    prediction: Any
    confidence: float
    feature_importance: Dict[str, float]
    decision_path: List[str]
    counterfactual: str
    plain_language: str
    visualizations: Dict[str, str]  # Base64 encoded images
    evidence: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]


@dataclass
class FeatureContribution:
    """Contribution of a feature to the prediction."""
    feature_name: str
    value: Any
    contribution: float
    direction: str  # positive, negative, neutral
    explanation: str


@dataclass
class DecisionNode:
    """Node in a decision tree explanation."""
    feature: str
    threshold: float
    decision: str
    confidence: float
    samples: int
    explanation: str


class ExplainableModel:
    """Wrapper for making any model explainable."""
    
    def __init__(self, model: Any, model_type: str = "classifier"):
        self.model = model
        self.model_type = model_type
        self.feature_names = None
        
    def predict(self, X):
        """Make predictions."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif hasattr(self.model, 'forward'):
            # Handle PyTorch models
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X = torch.tensor(X, dtype=torch.float32)
                output = self.model(X)
                if self.model_type == "classifier":
                    return torch.argmax(output, dim=1).numpy()
                else:
                    return output.numpy()
        else:
            raise ValueError("Model must have predict or forward method")
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'forward'):
            # Handle PyTorch models
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X = torch.tensor(X, dtype=torch.float32)
                output = self.model(X)
                if self.model_type == "classifier":
                    return torch.softmax(output, dim=1).numpy()
                else:
                    return output.numpy()
        else:
            # Fallback for models without probability
            predictions = self.predict(X)
            # Convert to one-hot for classifiers
            if self.model_type == "classifier":
                n_classes = len(np.unique(predictions))
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, int(pred)] = 1.0
                return proba
            return predictions.reshape(-1, 1)


class ExplanationEngine:
    """Advanced explanation engine for AI model interpretability."""
    
    def __init__(self):
        """Initialize explanation engine."""
        # Initialize SHAP explainer (will be set per model)
        self.shap_explainer = None
        
        # Initialize LIME explainer for text
        self.lime_text_explainer = lime.lime_text.LimeTextExplainer(
            class_names=['negative', 'positive'],
            feature_selection='auto',
            split_expression=r'\W+',
        )
        
        # Surrogate models for explanation
        self.surrogate_models = {}
        
        # Visualization settings
        self.viz_config = {
            'figure_size': (10, 6),
            'color_positive': '#2ecc71',
            'color_negative': '#e74c3c',
            'color_neutral': '#95a5a6'
        }
        
        # Plain language templates
        self.explanation_templates = self._load_explanation_templates()
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load plain language explanation templates."""
        return {
            'high_risk': "This clause is considered high risk because {reasons}. The main factors are {factors}.",
            'arbitration_detected': "An arbitration clause was detected based on {evidence}. This means {implications}.",
            'favorable': "This clause appears favorable because {reasons}. Key positive aspects: {aspects}.",
            'unfavorable': "This clause may be unfavorable due to {reasons}. Main concerns: {concerns}.",
            'neutral': "This clause is relatively standard. {details}",
            'complex': "This is a complex clause with multiple components: {components}. Breaking it down: {breakdown}",
            'recommendation': "Based on the analysis, we recommend: {action}. This is because {reasoning}."
        }
    
    async def explain_prediction(
        self,
        model: Any,
        input_data: Union[str, np.ndarray],
        prediction: Any,
        feature_names: Optional[List[str]] = None,
        explanation_type: str = "comprehensive"
    ) -> Explanation:
        """
        Generate comprehensive explanation for a model prediction.
        
        Args:
            model: The model to explain
            input_data: Input data (text or features)
            prediction: Model's prediction
            feature_names: Names of features if applicable
            explanation_type: Type of explanation (comprehensive, simple, technical)
            
        Returns:
            Comprehensive explanation object
        """
        # Wrap model for compatibility
        explainable_model = ExplainableModel(model)
        
        # Get confidence score
        confidence = self._get_confidence(explainable_model, input_data)
        
        # Generate different types of explanations
        feature_importance = await self._get_feature_importance(
            explainable_model, input_data, feature_names
        )
        
        # Generate decision path
        decision_path = self._generate_decision_path(
            explainable_model, input_data, prediction
        )
        
        # Generate counterfactual
        counterfactual = self._generate_counterfactual(
            explainable_model, input_data, prediction
        )
        
        # Create visualizations
        visualizations = self._create_visualizations(
            feature_importance, decision_path, confidence
        )
        
        # Gather evidence
        evidence = self._gather_evidence(input_data, feature_importance, prediction)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(
            explainable_model, input_data, prediction
        )
        
        # Generate plain language explanation
        plain_language = self._generate_plain_language_explanation(
            prediction, confidence, feature_importance, evidence, explanation_type
        )
        
        return Explanation(
            prediction=prediction,
            confidence=confidence,
            feature_importance=feature_importance,
            decision_path=decision_path,
            counterfactual=counterfactual,
            plain_language=plain_language,
            visualizations=visualizations,
            evidence=evidence,
            alternatives=alternatives
        )
    
    def _get_confidence(self, model: ExplainableModel, input_data: Any) -> float:
        """Get confidence score for prediction."""
        try:
            if isinstance(input_data, str):
                # For text, create simple features
                features = self._text_to_features(input_data)
                proba = model.predict_proba(features)
            else:
                proba = model.predict_proba(input_data)
            
            # Return max probability as confidence
            return float(np.max(proba))
        except Exception as e:
            logger.error(f"Error getting confidence: {e}")
            return 0.5
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to simple numerical features."""
        features = []
        
        # Basic text statistics
        features.extend([
            len(text),
            len(text.split()),
            text.count('.'),
            text.count(','),
            len(re.findall(r'\b[A-Z]\w+', text)),  # Capitalized words
        ])
        
        # Keyword presence
        keywords = ['arbitration', 'liability', 'termination', 'confidential', 'payment']
        for keyword in keywords:
            features.append(1.0 if keyword in text.lower() else 0.0)
        
        return np.array(features).reshape(1, -1)
    
    async def _get_feature_importance(
        self,
        model: ExplainableModel,
        input_data: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance using SHAP or LIME."""
        importance = {}
        
        try:
            if isinstance(input_data, str):
                # Use LIME for text
                importance = self._lime_text_explanation(model, input_data)
            else:
                # Use SHAP for tabular data
                importance = self._shap_explanation(model, input_data, feature_names)
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            # Fallback to simple method
            importance = self._simple_feature_importance(model, input_data, feature_names)
        
        return importance
    
    def _lime_text_explanation(self, model: ExplainableModel, text: str) -> Dict[str, float]:
        """Generate LIME explanation for text."""
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                features = np.vstack([self._text_to_features(t) for t in texts])
                return model.predict_proba(features)
            
            # Generate explanation
            exp = self.lime_text_explainer.explain_instance(
                text,
                predict_fn,
                num_features=10,
                num_samples=100
            )
            
            # Convert to dictionary
            importance = {}
            for feature, weight in exp.as_list():
                importance[feature] = weight
            
            return importance
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {}
    
    def _shap_explanation(
        self,
        model: ExplainableModel,
        input_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Generate SHAP explanation for tabular data."""
        try:
            # Create SHAP explainer if not exists
            if self.shap_explainer is None:
                self.shap_explainer = shap.Explainer(model.predict, input_data)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer(input_data)
            
            # Convert to importance dictionary
            importance = {}
            if feature_names:
                for i, name in enumerate(feature_names):
                    if i < len(shap_values.values[0]):
                        importance[name] = float(shap_values.values[0][i])
            else:
                for i in range(len(shap_values.values[0])):
                    importance[f"feature_{i}"] = float(shap_values.values[0][i])
            
            return importance
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return {}
    
    def _simple_feature_importance(
        self,
        model: ExplainableModel,
        input_data: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Simple feature importance using perturbation."""
        importance = {}
        
        if isinstance(input_data, str):
            # For text, use word frequency
            words = input_data.lower().split()
            word_freq = Counter(words)
            total = sum(word_freq.values())
            
            for word, count in word_freq.most_common(10):
                importance[word] = count / total
        else:
            # For tabular, use feature variance
            if isinstance(input_data, np.ndarray):
                variances = np.var(input_data, axis=0) if input_data.ndim > 1 else [np.var(input_data)]
                
                if feature_names:
                    for i, name in enumerate(feature_names):
                        if i < len(variances):
                            importance[name] = float(variances[i])
                else:
                    for i, var in enumerate(variances):
                        importance[f"feature_{i}"] = float(var)
        
        return importance
    
    def _generate_decision_path(
        self,
        model: ExplainableModel,
        input_data: Any,
        prediction: Any
    ) -> List[str]:
        """Generate decision path explanation."""
        path = []
        
        # Create surrogate decision tree
        try:
            if isinstance(input_data, str):
                features = self._text_to_features(input_data)
            else:
                features = input_data
            
            # Train simple decision tree as surrogate
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            
            # Generate training data around the input
            X_train = self._generate_neighborhood_data(features, n_samples=100)
            y_train = model.predict(X_train)
            
            dt.fit(X_train, y_train)
            
            # Get decision path
            tree_rules = export_text(dt, feature_names=[f"f{i}" for i in range(X_train.shape[1])])
            
            # Parse rules into readable format
            for line in tree_rules.split('\n')[:10]:  # Limit to 10 steps
                if '|---' in line:
                    path.append(line.replace('|---', '').strip())
            
        except Exception as e:
            logger.error(f"Error generating decision path: {e}")
            path.append(f"Direct prediction: {prediction}")
        
        return path if path else ["Direct prediction based on input features"]
    
    def _generate_neighborhood_data(self, input_data: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Generate data points in neighborhood of input."""
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Add noise to create neighborhood
        noise_scale = 0.1
        neighborhood = []
        
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_scale, input_data.shape)
            neighbor = input_data + noise
            neighborhood.append(neighbor.flatten())
        
        return np.array(neighborhood)
    
    def _generate_counterfactual(
        self,
        model: ExplainableModel,
        input_data: Any,
        prediction: Any
    ) -> str:
        """Generate counterfactual explanation."""
        try:
            if isinstance(input_data, str):
                # For text, suggest word changes
                important_words = ['arbitration', 'binding', 'waive', 'liability', 'terminate']
                present_words = [w for w in important_words if w in input_data.lower()]
                
                if present_words:
                    return f"If the clause did not contain '{present_words[0]}', the prediction might be different"
                else:
                    return "Adding terms like 'mutual agreement' or 'reasonable notice' might change the assessment"
            else:
                # For tabular, find minimal change
                features = input_data.copy()
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                # Try flipping each feature
                for i in range(features.shape[1]):
                    modified = features.copy()
                    modified[0, i] = 1 - modified[0, i]  # Simple flip for binary
                    
                    new_pred = model.predict(modified)
                    if new_pred != prediction:
                        return f"Changing feature {i} would likely change the prediction"
                
                return "Multiple feature changes would be needed to alter the prediction"
                
        except Exception as e:
            logger.error(f"Error generating counterfactual: {e}")
            return "Alternative scenarios would require significant changes to the input"
    
    def _create_visualizations(
        self,
        feature_importance: Dict[str, float],
        decision_path: List[str],
        confidence: float
    ) -> Dict[str, str]:
        """Create visualization images encoded as base64."""
        visualizations = {}
        
        try:
            # Feature importance bar chart
            if feature_importance:
                fig, ax = plt.subplots(figsize=self.viz_config['figure_size'])
                
                features = list(feature_importance.keys())[:10]
                values = [feature_importance[f] for f in features]
                colors = [self.viz_config['color_positive'] if v > 0 else self.viz_config['color_negative'] 
                         for v in values]
                
                ax.barh(features, values, color=colors)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                # Save to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                visualizations['feature_importance'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # Confidence gauge
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Create gauge visualization
            theta = np.linspace(0, np.pi, 100)
            r = 1
            
            # Background arc
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=20)
            
            # Confidence arc
            conf_theta = theta[:int(confidence * 100)]
            color = self.viz_config['color_positive'] if confidence > 0.7 else \
                   self.viz_config['color_negative'] if confidence < 0.3 else \
                   self.viz_config['color_neutral']
            ax.plot(r * np.cos(conf_theta), r * np.sin(conf_theta), color, linewidth=20)
            
            # Add text
            ax.text(0, -0.3, f'{confidence:.1%}', ha='center', fontsize=24, fontweight='bold')
            ax.text(0, -0.5, 'Confidence', ha='center', fontsize=12)
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.6, 1.2)
            ax.axis('off')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            visualizations['confidence_gauge'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations
    
    def _gather_evidence(
        self,
        input_data: Any,
        feature_importance: Dict[str, float],
        prediction: Any
    ) -> List[Dict[str, Any]]:
        """Gather evidence supporting the prediction."""
        evidence = []
        
        # Top important features as evidence
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, importance in sorted_features[:5]:
            evidence.append({
                'type': 'feature',
                'name': feature,
                'importance': importance,
                'direction': 'supporting' if importance > 0 else 'opposing',
                'description': f"{feature} has {'high' if abs(importance) > 0.5 else 'moderate'} impact"
            })
        
        # Add text-based evidence if input is text
        if isinstance(input_data, str):
            # Key phrases
            key_phrases = self._extract_key_phrases(input_data)
            for phrase in key_phrases[:3]:
                evidence.append({
                    'type': 'phrase',
                    'text': phrase,
                    'relevance': 'high',
                    'description': f"Key phrase: '{phrase}'"
                })
        
        return evidence
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple extraction - in production, use NLP
        phrases = []
        
        # Legal terms
        legal_patterns = [
            r'binding arbitration',
            r'waive\s+\w+\s+rights?',
            r'limitation of liability',
            r'exclusive jurisdiction',
            r'governing law',
            r'confidential information'
        ]
        
        for pattern in legal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrases.append(match.group())
        
        return phrases
    
    def _generate_alternatives(
        self,
        model: ExplainableModel,
        input_data: Any,
        prediction: Any
    ) -> List[Dict[str, Any]]:
        """Generate alternative predictions with different inputs."""
        alternatives = []
        
        try:
            if isinstance(input_data, str):
                # Text alternatives
                modifications = [
                    ("removing key terms", self._remove_key_terms(input_data)),
                    ("simplifying language", self._simplify_text(input_data)),
                    ("adding balance", self._add_balance(input_data))
                ]
                
                for description, modified_text in modifications:
                    if modified_text != input_data:
                        features = self._text_to_features(modified_text)
                        alt_pred = model.predict(features)
                        alt_conf = self._get_confidence(model, modified_text)
                        
                        alternatives.append({
                            'description': description,
                            'prediction': alt_pred,
                            'confidence': alt_conf,
                            'change': 'significant' if alt_pred != prediction else 'minor'
                        })
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _remove_key_terms(self, text: str) -> str:
        """Remove key legal terms from text."""
        terms_to_remove = ['binding', 'waive', 'exclusive', 'sole', 'absolute']
        result = text
        for term in terms_to_remove:
            result = re.sub(r'\b' + term + r'\b', '', result, flags=re.IGNORECASE)
        return result
    
    def _simplify_text(self, text: str) -> str:
        """Simplify legal text."""
        simplifications = {
            'notwithstanding': 'despite',
            'pursuant to': 'according to',
            'heretofore': 'before',
            'whereas': 'since'
        }
        result = text
        for complex, simple in simplifications.items():
            result = result.replace(complex, simple)
        return result
    
    def _add_balance(self, text: str) -> str:
        """Add balancing language to text."""
        if 'shall' in text and 'both parties' not in text.lower():
            text = text.replace('shall', 'both parties shall', 1)
        return text
    
    def _generate_plain_language_explanation(
        self,
        prediction: Any,
        confidence: float,
        feature_importance: Dict[str, float],
        evidence: List[Dict[str, Any]],
        explanation_type: str
    ) -> str:
        """Generate plain language explanation."""
        parts = []
        
        # Prediction summary
        if hasattr(prediction, '__iter__') and not isinstance(prediction, str):
            pred_str = f"The model predicts class {prediction}"
        else:
            pred_str = f"The model predicts: {prediction}"
        
        parts.append(f"{pred_str} with {confidence:.1%} confidence.")
        
        # Key factors
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            if top_features:
                factors = [f"{feat[0]} ({feat[1]:.2f})" for feat in top_features]
                parts.append(f"Key factors: {', '.join(factors)}")
        
        # Evidence summary
        if evidence:
            evidence_summary = []
            for e in evidence[:2]:
                if e['type'] == 'phrase':
                    evidence_summary.append(f"contains '{e['text']}'")
                elif e['type'] == 'feature':
                    evidence_summary.append(f"{e['name']} is {e['direction']}")
            
            if evidence_summary:
                parts.append(f"The input {' and '.join(evidence_summary)}.")
        
        # Explanation type specific additions
        if explanation_type == "simple":
            parts = parts[:2]  # Keep it brief
        elif explanation_type == "technical":
            parts.append(f"Feature importance analysis shows {len(feature_importance)} contributing factors.")
            parts.append(f"Decision path involves {len(evidence)} key decision points.")
        
        return " ".join(parts)
    
    def explain_model_behavior(
        self,
        model: Any,
        test_data: List[Any],
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Explain overall model behavior across multiple inputs.
        
        Args:
            model: Model to explain
            test_data: List of test inputs
            model_name: Name of the model
            
        Returns:
            Overall model behavior analysis
        """
        behavior_analysis = {
            'model_name': model_name,
            'num_samples': len(test_data),
            'global_patterns': {},
            'consistency': 0.0,
            'bias_assessment': {},
            'recommendations': []
        }
        
        # Analyze predictions across test data
        predictions = []
        confidences = []
        
        explainable_model = ExplainableModel(model)
        
        for data in test_data:
            pred = explainable_model.predict(data if not isinstance(data, str) 
                                            else self._text_to_features(data))
            conf = self._get_confidence(explainable_model, data)
            predictions.append(pred)
            confidences.append(conf)
        
        # Calculate consistency
        if predictions:
            unique_preds = len(set(map(tuple, predictions) if isinstance(predictions[0], np.ndarray) 
                                     else predictions))
            behavior_analysis['consistency'] = 1.0 - (unique_preds - 1) / len(predictions)
        
        # Identify global patterns
        behavior_analysis['global_patterns'] = {
            'avg_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'prediction_distribution': dict(Counter(map(str, predictions)))
        }
        
        # Assess potential biases
        behavior_analysis['bias_assessment'] = self._assess_model_bias(
            model, test_data, predictions
        )
        
        # Generate recommendations
        if behavior_analysis['consistency'] < 0.5:
            behavior_analysis['recommendations'].append(
                "Model shows high variability - consider additional training"
            )
        
        if behavior_analysis['global_patterns']['avg_confidence'] < 0.6:
            behavior_analysis['recommendations'].append(
                "Low average confidence - model may need more diverse training data"
            )
        
        return behavior_analysis
    
    def _assess_model_bias(
        self,
        model: Any,
        test_data: List[Any],
        predictions: List[Any]
    ) -> Dict[str, Any]:
        """Assess potential biases in model."""
        bias_assessment = {
            'detected_biases': [],
            'fairness_score': 0.8,  # Default
            'recommendations': []
        }
        
        # Check for systematic patterns
        if isinstance(test_data[0], str):
            # Text bias detection
            positive_keywords = ['favorable', 'reasonable', 'mutual', 'fair']
            negative_keywords = ['binding', 'waive', 'exclusive', 'sole']
            
            positive_preds = []
            negative_preds = []
            
            for data, pred in zip(test_data, predictions):
                if any(word in data.lower() for word in positive_keywords):
                    positive_preds.append(pred)
                if any(word in data.lower() for word in negative_keywords):
                    negative_preds.append(pred)
            
            # Check for keyword bias
            if positive_preds and negative_preds:
                if np.mean(positive_preds) != np.mean(negative_preds):
                    bias_assessment['detected_biases'].append('keyword_bias')
                    bias_assessment['fairness_score'] -= 0.2
        
        return bias_assessment
    
    def generate_decision_tree_visualization(
        self,
        model: Any,
        feature_names: List[str],
        class_names: List[str]
    ) -> str:
        """Generate decision tree visualization."""
        try:
            from sklearn.tree import export_graphviz
            import graphviz
            
            # If model is a decision tree
            if hasattr(model, 'tree_'):
                dot_data = export_graphviz(
                    model,
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                
                graph = graphviz.Source(dot_data)
                
                # Convert to image
                buffer = BytesIO()
                graph.render(format='png', cleanup=True)
                
                with open('temp.png', 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                return image_data
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error generating decision tree visualization: {e}")
            return ""
    
    def compare_explanations(
        self,
        explanations: List[Explanation]
    ) -> Dict[str, Any]:
        """Compare multiple explanations to find patterns."""
        comparison = {
            'num_explanations': len(explanations),
            'common_features': {},
            'confidence_distribution': {},
            'decision_patterns': [],
            'consistency_score': 0.0
        }
        
        if not explanations:
            return comparison
        
        # Aggregate feature importance
        all_features = defaultdict(list)
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                all_features[feature].append(importance)
        
        # Find common important features
        for feature, importances in all_features.items():
            avg_importance = np.mean(importances)
            if abs(avg_importance) > 0.1:  # Threshold for importance
                comparison['common_features'][feature] = {
                    'avg_importance': float(avg_importance),
                    'consistency': float(1.0 - np.std(importances))
                }
        
        # Confidence distribution
        confidences = [exp.confidence for exp in explanations]
        comparison['confidence_distribution'] = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
        
        # Decision patterns
        decision_paths = [tuple(exp.decision_path) for exp in explanations]
        path_counts = Counter(decision_paths)
        
        for path, count in path_counts.most_common(3):
            comparison['decision_patterns'].append({
                'path': list(path)[:3],  # Limit to first 3 steps
                'frequency': count / len(explanations)
            })
        
        # Calculate consistency
        if len(set(exp.prediction for exp in explanations)) == 1:
            comparison['consistency_score'] = 1.0
        else:
            unique_preds = len(set(str(exp.prediction) for exp in explanations))
            comparison['consistency_score'] = 1.0 - (unique_preds - 1) / len(explanations)
        
        return comparison