"""
Model Evaluation Script for Arbitration Clause Detection

This module provides comprehensive evaluation capabilities including:
- Cross-validation with multiple metrics
- ROC curves and precision-recall curves
- Confusion matrices and classification reports
- Feature importance analysis
- Model comparison and statistical significance testing
- Performance visualization and reporting
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML evaluation imports
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, 
    learning_curve, validation_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import joblib

# Statistical testing
from scipy import stats
from scipy.stats import chi2_contingency, mcnemar

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.app.ml.features import LegalFeatureExtractor, ArbitrationFeatureAnalyzer
from backend.app.ml.predictor import ArbitrationPredictor
from backend.data.training_data import TrainingDataGenerator


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis system
    """
    
    def __init__(self, models_dir: str = None, results_dir: str = None):
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent.parent / "evaluation_results"
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = None
        self.models = {}
        self.evaluation_results = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_models_and_data(self):
        """Load trained models and test data"""
        
        print("Loading models and data...")
        
        # Load feature extractor
        feature_path = self.models_dir / "feature_extractor.pkl"
        if feature_path.exists():
            self.feature_extractor = LegalFeatureExtractor.load(str(feature_path))
            print("Feature extractor loaded")
        else:
            raise FileNotFoundError(f"Feature extractor not found at {feature_path}")
        
        # Load models
        model_files = list(self.models_dir.glob("*_model.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
        
        if not self.models:
            raise ValueError("No models could be loaded")
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def prepare_test_data(self, test_size: int = 500) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Prepare test dataset"""
        
        print("Preparing test data...")
        
        # Generate test data
        generator = TrainingDataGenerator()
        df = generator.create_training_dataset(
            num_synthetic_positive=test_size//2,
            num_synthetic_negative=test_size//2,
            num_ambiguous=50,
            include_variations=True
        )
        
        texts = df['text'].tolist()
        labels = df['label'].values
        
        # Extract features
        X = self.feature_extractor.transform(texts)
        
        print(f"Test data prepared: {len(texts)} examples")
        print(f"Positive examples: {np.sum(labels)}")
        print(f"Negative examples: {len(labels) - np.sum(labels)}")
        
        return texts, X, labels
    
    def evaluate_single_model(self, model_name: str, model, X, y, 
                            cv_folds: int = 5) -> Dict[str, Any]:
        """Evaluate a single model with comprehensive metrics"""
        
        print(f"Evaluating {model_name}...")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall', 
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                  return_train_score=True, n_jobs=-1)
        
        # Calculate metrics
        metrics = {}
        for metric in scoring.keys():
            metrics[f'test_{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            metrics[f'test_{metric}_std'] = cv_results[f'test_{metric}'].std()
            metrics[f'train_{metric}_mean'] = cv_results[f'train_{metric}'].mean()
            metrics[f'train_{metric}_std'] = cv_results[f'train_{metric}'].std()
        
        # Fit model on full data for additional analysis
        model.fit(X, y)
        
        # Get predictions and probabilities
        y_pred = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred_proba = None
        
        # Detailed metrics on full data
        detailed_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y, y_pred)
        }
        
        if y_pred_proba is not None:
            detailed_metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            detailed_metrics['average_precision'] = average_precision_score(y, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_names = self.feature_extractor.get_feature_names()
            feature_importance = list(zip(feature_names, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('lr', model.named_steps.get('svm')), 'coef_'):
            # For linear models in pipelines
            coef = model.named_steps.get('lr', model.named_steps.get('svm')).coef_[0]
            feature_names = self.feature_extractor.get_feature_names()
            feature_importance = list(zip(feature_names, np.abs(coef)))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'cv_metrics': metrics,
            'detailed_metrics': detailed_metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
            'feature_importance': feature_importance[:50] if feature_importance else None  # Top 50 features
        }
    
    def evaluate_all_models(self, X, y) -> Dict[str, Any]:
        """Evaluate all loaded models"""
        
        self.evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Special handling for naive bayes
                if 'naive_bayes' in model_name:
                    X_eval = X.toarray() if hasattr(X, 'toarray') else X
                    X_eval = np.maximum(X_eval, 0)  # Ensure non-negative values
                else:
                    X_eval = X
                
                results = self.evaluate_single_model(model_name, model, X_eval, y)
                self.evaluation_results[model_name] = results
                
                # Print summary
                cv_f1 = results['cv_metrics']['test_f1_mean']
                cv_auc = results['cv_metrics'].get('test_roc_auc_mean', 0)
                print(f"{model_name}: CV F1={cv_f1:.4f}, CV AUC={cv_auc:.4f}")
                
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
        
        return self.evaluation_results
    
    def create_roc_curves(self, texts: List[str], y: np.ndarray):
        """Create ROC curves for all models"""
        
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            try:
                X = self.feature_extractor.transform(texts)
                
                if 'naive_bayes' in model_name:
                    X_eval = X.toarray() if hasattr(X, 'toarray') else X
                    X_eval = np.maximum(X_eval, 0)
                else:
                    X_eval = X
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_eval)[:, 1]
                    fpr, tpr, _ = roc_curve(y, y_proba)
                    auc = roc_auc_score(y, y_proba)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
                
            except Exception as e:
                print(f"Failed to create ROC curve for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_precision_recall_curves(self, texts: List[str], y: np.ndarray):
        """Create precision-recall curves for all models"""
        
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            try:
                X = self.feature_extractor.transform(texts)
                
                if 'naive_bayes' in model_name:
                    X_eval = X.toarray() if hasattr(X, 'toarray') else X
                    X_eval = np.maximum(X_eval, 0)
                else:
                    X_eval = X
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_eval)[:, 1]
                    precision, recall, _ = precision_recall_curve(y, y_proba)
                    ap = average_precision_score(y, y_proba)
                    
                    plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})', linewidth=2)
                
            except Exception as e:
                print(f"Failed to create PR curve for {model_name}: {e}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.results_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_confusion_matrices(self):
        """Create confusion matrix visualizations"""
        
        n_models = len(self.evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Arbitration', 'Arbitration'],
                       yticklabels=['No Arbitration', 'Arbitration'])
            
            ax.set_title(f'{model_name}\nAccuracy: {results["detailed_metrics"]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_plots(self):
        """Create feature importance visualizations"""
        
        models_with_importance = {name: results for name, results in self.evaluation_results.items()
                                if results['feature_importance'] is not None}
        
        if not models_with_importance:
            print("No models with feature importance available")
            return
        
        for model_name, results in models_with_importance.items():
            features, importances = zip(*results['feature_importance'][:20])  # Top 20
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            
            plt.barh(y_pos, importances)
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'feature_importance_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_model_comparison_chart(self):
        """Create model comparison chart"""
        
        metrics = ['test_accuracy_mean', 'test_precision_mean', 'test_recall_mean', 
                  'test_f1_mean', 'test_roc_auc_mean']
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            row = {'Model': model_name}
            for metric in metrics:
                row[metric.replace('test_', '').replace('_mean', '')] = results['cv_metrics'].get(metric, 0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        df_melted = df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def create_learning_curves(self, texts: List[str], y: np.ndarray):
        """Create learning curves for best models"""
        
        # Select top 3 models by F1 score
        top_models = sorted(self.evaluation_results.items(), 
                           key=lambda x: x[1]['cv_metrics']['test_f1_mean'], 
                           reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (model_name, _) in enumerate(top_models):
            model = self.models[model_name]
            X = self.feature_extractor.transform(texts)
            
            if 'naive_bayes' in model_name:
                X_eval = X.toarray() if hasattr(X, 'toarray') else X
                X_eval = np.maximum(X_eval, 0)
            else:
                X_eval = X
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_eval, y, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='f1'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[i].plot(train_sizes, train_mean, 'o-', label='Training Score')
            axes[i].plot(train_sizes, val_mean, 'o-', label='Validation Score')
            
            axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
            
            axes[i].set_title(f'Learning Curve - {model_name}')
            axes[i].set_xlabel('Training Set Size')
            axes[i].set_ylabel('F1 Score')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_significance_test(self) -> Dict[str, Any]:
        """Test statistical significance between models"""
        
        print("Performing statistical significance tests...")
        
        significance_results = {}
        
        # Get F1 scores from cross-validation
        model_scores = {}
        for model_name, results in self.evaluation_results.items():
            # We need to rerun CV to get individual fold scores
            # This is simplified - in practice you'd store CV fold results
            f1_mean = results['cv_metrics']['test_f1_mean']
            f1_std = results['cv_metrics']['test_f1_std']
            
            # Approximate individual scores (this is a simplification)
            model_scores[model_name] = {
                'mean': f1_mean,
                'std': f1_std
            }
        
        # Pairwise comparisons
        model_names = list(model_scores.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Simplified significance test using normal approximation
                mean1 = model_scores[model1]['mean']
                mean2 = model_scores[model2]['mean']
                std1 = model_scores[model1]['std']
                std2 = model_scores[model2]['std']
                
                # Two-sample t-test approximation
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(2/5))  # 5 CV folds
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), 8))  # df = 2*(5-1)
                
                significance_results[f"{model1}_vs_{model2}"] = {
                    'mean_diff': mean1 - mean2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return significance_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        print("Generating comprehensive evaluation report...")
        
        # Get best model
        best_model_name = max(self.evaluation_results.items(), 
                            key=lambda x: x[1]['cv_metrics']['test_f1_mean'])[0]
        best_results = self.evaluation_results[best_model_name]
        
        # Create summary
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': len(self.evaluation_results),
            'best_model': {
                'name': best_model_name,
                'f1_score': best_results['cv_metrics']['test_f1_mean'],
                'accuracy': best_results['cv_metrics']['test_accuracy_mean'],
                'precision': best_results['cv_metrics']['test_precision_mean'],
                'recall': best_results['cv_metrics']['test_recall_mean'],
                'roc_auc': best_results['cv_metrics'].get('test_roc_auc_mean', 0)
            },
            'model_rankings': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # Rank models by different metrics
        metrics = ['test_f1_mean', 'test_accuracy_mean', 'test_roc_auc_mean']
        
        for metric in metrics:
            ranked = sorted(self.evaluation_results.items(), 
                          key=lambda x: x[1]['cv_metrics'].get(metric, 0), 
                          reverse=True)
            report['model_rankings'][metric] = [
                {'model': name, 'score': results['cv_metrics'].get(metric, 0)}
                for name, results in ranked
            ]
        
        # Performance summary for all models
        for model_name, results in self.evaluation_results.items():
            report['performance_summary'][model_name] = {
                'cv_metrics': results['cv_metrics'],
                'detailed_metrics': results['detailed_metrics'],
                'top_features': results['feature_importance'][:10] if results['feature_importance'] else None
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Save report
        report_path = self.results_dir / 'comprehensive_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive report saved to {report_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Find best performing model
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['cv_metrics']['test_f1_mean'])
        
        recommendations.append(f"Use {best_model[0]} as the primary model (highest F1-score: {best_model[1]['cv_metrics']['test_f1_mean']:.4f})")
        
        # Check for overfitting
        for model_name, results in self.evaluation_results.items():
            train_f1 = results['cv_metrics']['train_f1_mean']
            test_f1 = results['cv_metrics']['test_f1_mean']
            
            if train_f1 - test_f1 > 0.1:
                recommendations.append(f"Consider regularization for {model_name} (potential overfitting)")
        
        # Check ensemble benefit
        if 'ensemble' in self.evaluation_results:
            ensemble_f1 = self.evaluation_results['ensemble']['cv_metrics']['test_f1_mean']
            individual_f1s = [results['cv_metrics']['test_f1_mean'] 
                            for name, results in self.evaluation_results.items() 
                            if name != 'ensemble']
            
            if ensemble_f1 > max(individual_f1s):
                recommendations.append("Ensemble model provides performance improvement over individual models")
            else:
                recommendations.append("Ensemble model does not improve performance - use best individual model")
        
        # Feature importance recommendations
        if best_model[1]['feature_importance']:
            top_feature = best_model[1]['feature_importance'][0][0]
            recommendations.append(f"Most important feature: {top_feature}")
        
        return recommendations
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        
        print("Starting complete model evaluation...")
        print("=" * 60)
        
        # Load models and data
        self.load_models_and_data()
        
        # Prepare test data
        texts, X, y = self.prepare_test_data()
        
        # Evaluate all models
        self.evaluate_all_models(X, y)
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_roc_curves(texts, y)
        self.create_precision_recall_curves(texts, y)
        self.create_confusion_matrices()
        self.create_feature_importance_plots()
        comparison_df = self.create_model_comparison_chart()
        self.create_learning_curves(texts, y)
        
        # Statistical significance testing
        significance_results = self.statistical_significance_test()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Best model: {report['best_model']['name']}")
        print(f"F1-score: {report['best_model']['f1_score']:.4f}")
        print(f"Accuracy: {report['best_model']['accuracy']:.4f}")
        print(f"Results saved to: {self.results_dir}")
        
        return report, comparison_df, significance_results


def main():
    """Main evaluation script"""
    
    evaluator = ModelEvaluator()
    
    try:
        report, comparison_df, significance_results = evaluator.run_complete_evaluation()
        
        # Print summary results
        print("\nModel Performance Summary:")
        print("-" * 40)
        print(comparison_df.round(4))
        
        print("\nStatistical Significance Tests:")
        print("-" * 40)
        for comparison, results in significance_results.items():
            significance = "Significant" if results['significant'] else "Not significant"
            print(f"{comparison}: {significance} (p={results['p_value']:.4f})")
        
        print("\nRecommendations:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        return evaluator, report
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Make sure models are trained first by running train.py")
        raise


if __name__ == "__main__":
    evaluator, report = main()