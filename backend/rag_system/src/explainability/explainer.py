import numpy as np
import lime
import lime.lime_text
import sklearn
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ArbitrationExplainer:
    """
    Comprehensive explainability module for RAG system arbitration explanations
    
    Provides advanced text explanation, confidence breakdown, 
    and visualization capabilities.
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize the ArbitrationExplainer
        
        Args:
            model: The underlying machine learning model
            class_names: Optional list of class names for interpretation
        """
        self.model = model
        self.class_names = class_names or ['Positive', 'Negative']
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=self.class_names,
            split_expression=' ',
            bow=True
        )
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME-based explanation for a single text prediction
        
        Args:
            text: Input text to explain
            num_features: Number of top features to explain
        
        Returns:
            Comprehensive explanation dictionary
        """
        try:
            # Predict probability for each class
            probas = self.model.predict_proba([text])[0]
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                text, 
                self.model.predict_proba, 
                num_features=num_features
            )
            
            # Extract top features and their contributions
            top_features = explanation.as_list()
            feature_weights = {f: w for f, w in top_features}
            
            return {
                'prediction': self.class_names[np.argmax(probas)],
                'probabilities': dict(zip(self.class_names, probas)),
                'top_features': feature_weights,
                'feature_explanation': explanation
            }
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Failed to generate explanation'
            }
    
    def confidence_breakdown(self, texts: List[str]) -> Dict[str, float]:
        """
        Provide confidence breakdown across multiple texts
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            Confidence metrics dictionary
        """
        if not texts:
            return {}
        
        probas = self.model.predict_proba(texts)
        confidence_metrics = {
            'mean_confidence': np.mean(np.max(probas, axis=1)),
            'median_confidence': np.median(np.max(probas, axis=1)),
            'confidence_std': np.std(np.max(probas, axis=1)),
            'low_confidence_count': np.sum(np.max(probas, axis=1) < 0.5)
        }
        
        return confidence_metrics
    
    def generate_feature_importance_plot(self, explanation) -> Optional[str]:
        """
        Generate a feature importance visualization
        
        Args:
            explanation: LIME explanation object
        
        Returns:
            Path to saved plot or None
        """
        try:
            # Create a DataFrame from explanation
            features = [f for f, _ in explanation.as_list()]
            weights = [w for _, w in explanation.as_list()]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=weights, y=features)
            plt.title('Feature Importance in Prediction')
            plt.xlabel('Weight')
            plt.ylabel('Features')
            
            plot_path = '/tmp/feature_importance.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
        except Exception as e:
            print(f"Plot generation error: {e}")
            return None

class VisualExplainer:
    """
    Advanced visual explanation generation for RAG system
    """
    
    @staticmethod
    def create_confidence_heatmap(confidences: Dict[str, float]) -> Optional[str]:
        """
        Create a heatmap visualization of confidence metrics
        
        Args:
            confidences: Dictionary of confidence metrics
        
        Returns:
            Path to saved heatmap
        """
        try:
            # Convert confidence metrics to DataFrame
            df = pd.DataFrame.from_dict(confidences, orient='index', columns=['Value'])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.T, annot=True, cmap='YlGnBu', cbar=True)
            plt.title('Confidence Metrics Heatmap')
            
            plot_path = '/tmp/confidence_heatmap.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
        except Exception as e:
            print(f"Heatmap generation error: {e}")
            return None
    
    @staticmethod
    def generate_interpretability_report(explanation: Dict[str, Any]) -> str:
        """
        Generate a comprehensive human-readable interpretability report
        
        Args:
            explanation: Explanation dictionary from ArbitrationExplainer
        
        Returns:
            Formatted interpretability report
        """
        report = "RAG System Interpretability Report\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Prediction: {explanation.get('prediction', 'N/A')}\n"
        report += "Probability Breakdown:\n"
        for cls, prob in explanation.get('probabilities', {}).items():
            report += f"  {cls}: {prob:.2%}\n"
        
        report += "\nTop Contributing Features:\n"
        for feature, weight in explanation.get('top_features', {}).items():
            report += f"  {feature}: {weight:.4f}\n"
        
        return report