"""
Training Script for Arbitration Clause Detection Models

This module trains multiple machine learning models for detecting arbitration clauses:
- RandomForest classifier
- Support Vector Machine (SVM)
- Gradient Boosting
- Ensemble model combining multiple approaches
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import our modules
from backend.app.ml.features import LegalFeatureExtractor
from backend.data.training_data import TrainingDataGenerator


class ArbitrationModelTrainer:
    """
    Comprehensive training system for arbitration detection models
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = None
        self.models = {}
        self.ensemble_model = None
        self.training_history = {}
        self.model_performance = {}
        
    def prepare_data(self, data_path: str = None, generate_new: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare training data"""
        
        if generate_new or data_path is None:
            print("Generating new training data...")
            generator = TrainingDataGenerator()
            df = generator.create_training_dataset(
                num_synthetic_positive=500,
                num_synthetic_negative=500,
                num_ambiguous=150,
                include_variations=True
            )
            
            # Save the generated data
            data_path = generator.save_training_data(df, "ml_training_data.csv")
        else:
            print(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)
        
        print(f"Dataset loaded: {len(df)} examples")
        print(f"Positive examples: {len(df[df['label'] == 1])}")
        print(f"Negative examples: {len(df[df['label'] == 0])}")
        
        # Prepare features and labels
        texts = df['text'].tolist()
        labels = df['label'].values
        
        return df, texts, labels
    
    def create_feature_extractor(self, texts: List[str]) -> LegalFeatureExtractor:
        """Create and fit feature extractor"""
        
        print("Creating and fitting feature extractor...")
        self.feature_extractor = LegalFeatureExtractor(
            max_features_tfidf=3000,
            ngram_range=(1, 3),
            use_legal_keywords=True,
            use_structure_features=True,
            use_statistical_features=True
        )
        
        self.feature_extractor.fit(texts)
        
        # Save feature extractor
        feature_path = self.models_dir / "feature_extractor.pkl"
        self.feature_extractor.save(str(feature_path))
        
        return self.feature_extractor
    
    def define_models(self) -> Dict[str, Any]:
        """Define individual models to train"""
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'svm_linear': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('svm', SVC(
                    kernel='linear',
                    C=1.0,
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            
            'svm_rbf': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('svm', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            ),
            
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('lr', LogisticRegression(
                    C=1.0,
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]),
            
            'naive_bayes': MultinomialNB(alpha=1.0)
        }
        
        return models
    
    def train_individual_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train individual models"""
        
        models = self.define_models()
        trained_models = {}
        
        print("\nTraining individual models...")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Handle sparse matrices for naive bayes
            if name == 'naive_bayes':
                # Convert sparse matrix to dense for MultinomialNB
                X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
                
                # Ensure non-negative values for NB
                X_train_dense = np.maximum(X_train_dense, 0)
                X_test_dense = np.maximum(X_test_dense, 0)
                
                model.fit(X_train_dense, y_train)
                y_pred = model.predict(X_test_dense)
                y_pred_proba = model.predict_proba(X_test_dense)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Store model and metrics
            trained_models[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Cross-validation
            if name != 'naive_bayes':  # Skip CV for NB due to sparse matrix issues
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
                trained_models[name]['cv_f1_mean'] = cv_scores.mean()
                trained_models[name]['cv_f1_std'] = cv_scores.std()
                print(f"{name} - CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = trained_models
        return trained_models
    
    def create_ensemble_model(self, X_train, y_train, X_test, y_test) -> VotingClassifier:
        """Create ensemble model from best performing individual models"""
        
        print("\nCreating ensemble model...")
        
        # Select best models for ensemble (excluding those with low performance)
        ensemble_models = []
        min_f1_threshold = 0.7
        
        for name, model_info in self.models.items():
            if model_info['metrics']['f1_score'] >= min_f1_threshold:
                ensemble_models.append((name, model_info['model']))
                print(f"Including {name} in ensemble (F1: {model_info['metrics']['f1_score']:.4f})")
        
        if len(ensemble_models) < 2:
            print("Warning: Less than 2 models meet the threshold. Including top 3 models.")
            sorted_models = sorted(self.models.items(), 
                                 key=lambda x: x[1]['metrics']['f1_score'], 
                                 reverse=True)[:3]
            ensemble_models = [(name, info['model']) for name, info in sorted_models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use predicted probabilities
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1_score': f1_score(y_test, y_pred_ensemble),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble)
        }
        
        print(f"Ensemble - Accuracy: {ensemble_metrics['accuracy']:.4f}, F1: {ensemble_metrics['f1_score']:.4f}, ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        # Store ensemble results
        self.ensemble_model = ensemble
        self.models['ensemble'] = {
            'model': ensemble,
            'metrics': ensemble_metrics,
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        return ensemble
    
    def calibrate_models(self, X_train, y_train):
        """Calibrate model probabilities for better confidence estimates"""
        
        print("\nCalibrating model probabilities...")
        
        calibrated_models = {}
        
        for name, model_info in self.models.items():
            if name == 'ensemble':
                continue
                
            print(f"Calibrating {name}...")
            
            # Create calibrated version
            calibrated = CalibratedClassifierCV(
                model_info['model'], 
                method='platt',
                cv=3
            )
            
            if name == 'naive_bayes':
                X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                X_train_dense = np.maximum(X_train_dense, 0)
                calibrated.fit(X_train_dense, y_train)
            else:
                calibrated.fit(X_train, y_train)
            
            calibrated_models[f"{name}_calibrated"] = calibrated
        
        # Also calibrate ensemble
        if self.ensemble_model:
            calibrated_ensemble = CalibratedClassifierCV(
                self.ensemble_model,
                method='platt',
                cv=3
            )
            calibrated_ensemble.fit(X_train, y_train)
            calibrated_models['ensemble_calibrated'] = calibrated_ensemble
        
        return calibrated_models
    
    def save_models(self, calibrated_models: Dict = None):
        """Save all trained models"""
        
        print("\nSaving models...")
        
        # Save individual models
        for name, model_info in self.models.items():
            model_path = self.models_dir / f"{name}_model.pkl"
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save calibrated models
        if calibrated_models:
            for name, model in calibrated_models.items():
                model_path = self.models_dir / f"{name}_model.pkl"
                joblib.dump(model, model_path)
                print(f"Saved {name} to {model_path}")
        
        # Save performance metrics
        metrics_path = self.models_dir / "model_performance.json"
        performance_data = {}
        for name, model_info in self.models.items():
            performance_data[name] = model_info['metrics']
            if 'cv_f1_mean' in model_info:
                performance_data[name]['cv_f1_mean'] = model_info['cv_f1_mean']
                performance_data[name]['cv_f1_std'] = model_info['cv_f1_std']
        
        with open(metrics_path, 'w') as f:
            json.dump(performance_data, f, indent=2)
        print(f"Saved performance metrics to {metrics_path}")
        
        # Save training configuration
        config = {
            'training_date': datetime.now().isoformat(),
            'feature_extractor_config': {
                'max_features_tfidf': self.feature_extractor.max_features_tfidf,
                'ngram_range': self.feature_extractor.ngram_range,
                'use_legal_keywords': self.feature_extractor.use_legal_keywords,
                'use_structure_features': self.feature_extractor.use_structure_features,
                'use_statistical_features': self.feature_extractor.use_statistical_features
            },
            'models_trained': list(self.models.keys()),
            'best_model': max(self.models.items(), key=lambda x: x[1]['metrics']['f1_score'])[0]
        }
        
        config_path = self.models_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved training configuration to {config_path}")
    
    def generate_detailed_report(self, X_test, y_test) -> Dict[str, Any]:
        """Generate detailed performance report"""
        
        print("\nGenerating detailed performance report...")
        
        report = {
            'summary': {},
            'detailed_metrics': {},
            'model_comparison': {},
            'feature_importance': {}
        }
        
        # Summary statistics
        report['summary'] = {
            'total_models_trained': len(self.models),
            'best_model_by_f1': max(self.models.items(), key=lambda x: x[1]['metrics']['f1_score'])[0],
            'best_f1_score': max(model_info['metrics']['f1_score'] for model_info in self.models.values()),
            'best_model_by_auc': max(self.models.items(), key=lambda x: x[1]['metrics']['roc_auc'])[0],
            'best_auc_score': max(model_info['metrics']['roc_auc'] for model_info in self.models.values())
        }
        
        # Detailed metrics for each model
        for name, model_info in self.models.items():
            y_pred = model_info['predictions']
            
            # Classification report
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            report['detailed_metrics'][name] = {
                'classification_report': clf_report,
                'confusion_matrix': cm.tolist(),
                'basic_metrics': model_info['metrics']
            }
            
            if 'cv_f1_mean' in model_info:
                report['detailed_metrics'][name]['cross_validation'] = {
                    'cv_f1_mean': model_info['cv_f1_mean'],
                    'cv_f1_std': model_info['cv_f1_std']
                }
        
        # Model comparison
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in comparison_metrics:
            report['model_comparison'][metric] = {
                name: model_info['metrics'][metric] 
                for name, model_info in self.models.items()
            }
        
        # Feature importance (for tree-based models)
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Extract the actual model from pipeline if needed
            if hasattr(model, 'named_steps'):
                if 'svm' in model.named_steps:
                    actual_model = model.named_steps['svm']
                elif 'lr' in model.named_steps:
                    actual_model = model.named_steps['lr']
                else:
                    actual_model = model
            else:
                actual_model = model
            
            if hasattr(actual_model, 'feature_importances_'):
                feature_names = self.feature_extractor.get_feature_names()
                importances = actual_model.feature_importances_
                
                # Get top 20 features
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                report['feature_importance'][name] = {
                    'top_features': feature_importance[:20],
                    'total_features': len(feature_names)
                }
        
        # Save report
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved to {report_path}")
        
        return report
    
    def train_all_models(self, generate_new_data: bool = True) -> Dict[str, Any]:
        """Main training pipeline"""
        
        print("Starting comprehensive model training...")
        print("=" * 50)
        
        # Step 1: Prepare data
        df, texts, labels = self.prepare_data(generate_new=generate_new_data)
        
        # Step 2: Create feature extractor
        self.create_feature_extractor(texts)
        
        # Step 3: Extract features
        print("Extracting features...")
        X = self.feature_extractor.transform(texts)
        print(f"Feature matrix shape: {X.shape}")
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape[0]} examples")
        print(f"Test set: {X_test.shape[0]} examples")
        
        # Step 5: Train individual models
        self.train_individual_models(X_train, X_test, y_train, y_test)
        
        # Step 6: Create ensemble model
        self.create_ensemble_model(X_train, y_train, X_test, y_test)
        
        # Step 7: Calibrate models
        calibrated_models = self.calibrate_models(X_train, y_train)
        
        # Step 8: Save models
        self.save_models(calibrated_models)
        
        # Step 9: Generate report
        report = self.generate_detailed_report(X_test, y_test)
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Best model by F1-score: {report['summary']['best_model_by_f1']} ({report['summary']['best_f1_score']:.4f})")
        print(f"Best model by ROC-AUC: {report['summary']['best_model_by_auc']} ({report['summary']['best_auc_score']:.4f})")
        print(f"Models saved to: {self.models_dir}")
        
        return report


def main():
    """Main training script"""
    
    # Create trainer
    trainer = ArbitrationModelTrainer()
    
    # Train all models
    report = trainer.train_all_models(generate_new_data=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for model_name, metrics in report['model_comparison'].items():
        print(f"\n{model_name.upper()}:")
        for metric_name, scores in metrics.items():
            best_score = max(scores.values())
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            print(f"  {metric_name}: {best_score:.4f} ({best_model})")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()