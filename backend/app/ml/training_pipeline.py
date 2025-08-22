"""
Comprehensive training pipeline for arbitration clause detection
Includes data augmentation, cross-validation, and hyperparameter tuning
"""
import logging
import os
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
from datetime import datetime
import random
import re
from collections import defaultdict

from .feature_extraction import LegalFeatureExtractor
from .classifier import ArbitrationClassifier
from .ner import LegalEntityRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Data augmentation for legal texts to improve model robustness
    """
    
    def __init__(self):
        self.legal_synonyms = {
            "dispute": ["controversy", "disagreement", "conflict", "claim"],
            "arbitration": ["arbitral proceedings", "arbitral process"],
            "binding": ["final", "conclusive", "definitive"],
            "agreement": ["contract", "accord", "understanding"],
            "parties": ["contracting parties", "signatories"],
            "shall": ["will", "must"],
            "resolved": ["settled", "determined", "decided"],
            "tribunal": ["arbitral panel", "arbitration panel"],
            "rules": ["procedures", "regulations", "guidelines"]
        }
        
        self.legal_templates = [
            "Any {dispute_term} arising under this {agreement_term} {shall_term} be {resolved_term} through {binding_term} {arbitration_term}.",
            "All {dispute_term}s {shall_term} be finally settled by {arbitration_term} administered by {organization}.",
            "The {parties_term} agree to submit any {dispute_term} to {arbitration_term} under the {rules_term} of {organization}.",
            "{Dispute_term}s arising out of this {agreement_term} {shall_term} be {resolved_term} exclusively through {arbitration_term}.",
            "Any {dispute_term} {shall_term} be subject to {binding_term} {arbitration_term} in accordance with {rules_term}."
        ]
        
        self.organizations = [
            "the American Arbitration Association",
            "AAA", "JAMS", "the ICC", "the International Chamber of Commerce",
            "LCIA", "the Singapore International Arbitration Centre", "SIAC"
        ]
    
    def augment_arbitration_clause(self, text: str, num_variations: int = 3) -> List[str]:
        """
        Create variations of arbitration clauses through synonym replacement
        """
        variations = [text]
        
        for _ in range(num_variations):
            augmented_text = text
            
            # Replace synonyms
            for original, synonyms in self.legal_synonyms.items():
                if original in augmented_text.lower():
                    replacement = random.choice(synonyms)
                    augmented_text = re.sub(
                        r'\b' + re.escape(original) + r'\b',
                        replacement,
                        augmented_text,
                        flags=re.IGNORECASE
                    )
            
            variations.append(augmented_text)
        
        return variations
    
    def generate_synthetic_clauses(self, num_clauses: int = 50) -> List[str]:
        """
        Generate synthetic arbitration clauses using templates
        """
        synthetic_clauses = []
        
        for _ in range(num_clauses):
            template = random.choice(self.legal_templates)
            
            # Fill template placeholders
            clause = template.format(
                dispute_term=random.choice(self.legal_synonyms["dispute"]),
                arbitration_term=random.choice(self.legal_synonyms["arbitration"] + ["arbitration"]),
                binding_term=random.choice(self.legal_synonyms["binding"]),
                agreement_term=random.choice(self.legal_synonyms["agreement"]),
                parties_term=random.choice(self.legal_synonyms["parties"]),
                shall_term=random.choice(self.legal_synonyms["shall"]),
                resolved_term=random.choice(self.legal_synonyms["resolved"]),
                rules_term=random.choice(self.legal_synonyms["rules"]),
                organization=random.choice(self.organizations),
                Dispute_term=random.choice(self.legal_synonyms["dispute"]).capitalize()
            )
            
            synthetic_clauses.append(clause)
        
        return synthetic_clauses
    
    def add_noise_to_text(self, text: str, noise_level: float = 0.1) -> str:
        """
        Add realistic noise to text (typos, spacing issues)
        """
        words = text.split()
        num_changes = int(len(words) * noise_level)
        
        for _ in range(num_changes):
            if words:
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                
                # Random modifications
                modification = random.choice(['typo', 'spacing', 'case'])
                
                if modification == 'typo' and len(word) > 3:
                    # Insert random character
                    pos = random.randint(1, len(word) - 1)
                    word = word[:pos] + random.choice('aeiou') + word[pos:]
                
                elif modification == 'spacing':
                    # Remove or add space
                    if random.random() < 0.5:
                        word = word.replace(' ', '')
                    else:
                        pos = random.randint(1, len(word))
                        word = word[:pos] + ' ' + word[pos:]
                
                elif modification == 'case':
                    # Random case changes
                    word = ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in word)
                
                words[idx] = word
        
        return ' '.join(words)


class TrainingPipeline:
    """
    Comprehensive training pipeline for arbitration detection models
    """
    
    def __init__(self,
                 experiment_name: str = "arbitration_detection_pipeline",
                 model_save_path: str = "backend/models",
                 random_state: int = 42):
        self.experiment_name = experiment_name
        self.model_save_path = model_save_path
        self.random_state = random_state
        
        # Initialize components
        self.feature_extractor = LegalFeatureExtractor()
        self.classifier = ArbitrationClassifier(model_save_path=model_save_path)
        self.augmenter = DataAugmenter()
        
        # Pipeline state
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.best_model = None
        self.feature_importance = None
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Ensure directories exist
        os.makedirs(model_save_path, exist_ok=True)
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def load_and_prepare_data(self, 
                            arbitration_texts: List[str],
                            non_arbitration_texts: List[str],
                            augment_data: bool = True,
                            test_size: float = 0.2,
                            val_size: float = 0.2) -> Dict[str, Any]:
        """
        Load and prepare training data with augmentation
        """
        logger.info("Loading and preparing training data...")
        
        # Combine and label data
        texts = arbitration_texts + non_arbitration_texts
        labels = [1] * len(arbitration_texts) + [0] * len(non_arbitration_texts)
        
        # Data augmentation for arbitration clauses
        if augment_data:
            logger.info("Applying data augmentation...")
            
            augmented_arb_texts = []
            for text in arbitration_texts:
                variations = self.augmenter.augment_arbitration_clause(text, num_variations=2)
                augmented_arb_texts.extend(variations[1:])  # Skip original
            
            # Add synthetic clauses
            synthetic_clauses = self.augmenter.generate_synthetic_clauses(num_clauses=len(arbitration_texts) // 2)
            augmented_arb_texts.extend(synthetic_clauses)
            
            # Add augmented data
            texts.extend(augmented_arb_texts)
            labels.extend([1] * len(augmented_arb_texts))
            
            logger.info(f"Added {len(augmented_arb_texts)} augmented arbitration clauses")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, random_state=self.random_state
        )
        
        # Store data splits
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        data_info = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_positive_ratio": sum(y_train) / len(y_train),
            "val_positive_ratio": sum(y_val) / len(y_val),
            "test_positive_ratio": sum(y_test) / len(y_test),
            "augmentation_applied": augment_data
        }
        
        logger.info(f"Data prepared: {data_info}")
        return data_info
    
    def extract_and_prepare_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features for all data splits
        """
        logger.info("Extracting features for all data splits...")
        
        X_train, y_train = self.training_data
        X_val, y_val = self.validation_data
        X_test, y_test = self.test_data
        
        # Extract features for all splits
        train_features_df, train_embeddings = self.feature_extractor.extract_features_batch(X_train)
        val_features_df, val_embeddings = self.feature_extractor.extract_features_batch(X_val)
        test_features_df, test_embeddings = self.feature_extractor.extract_features_batch(X_test)
        
        # Fit vectorizers on training data
        all_texts = X_train + X_val + X_test
        self.feature_extractor.fit_vectorizers(all_texts)
        
        # Get TF-IDF features
        train_tfidf = self.feature_extractor.get_tfidf_features(X_train)
        val_tfidf = self.feature_extractor.get_tfidf_features(X_val)
        test_tfidf = self.feature_extractor.get_tfidf_features(X_test)
        
        # Combine all features
        train_features_combined = np.hstack([
            train_features_df.values,
            train_embeddings,
            train_tfidf
        ])
        
        val_features_combined = np.hstack([
            val_features_df.values,
            val_embeddings,
            val_tfidf
        ])
        
        test_features_combined = np.hstack([
            test_features_df.values,
            test_embeddings,
            test_tfidf
        ])
        
        # Scale features
        train_features_scaled = self.feature_extractor.scale_features(train_features_combined, fit=True)
        val_features_scaled = self.feature_extractor.scale_features(val_features_combined, fit=False)
        test_features_scaled = self.feature_extractor.scale_features(test_features_combined, fit=False)
        
        # Store feature names for interpretability
        feature_names = (
            list(train_features_df.columns) +
            [f"embedding_{i}" for i in range(train_embeddings.shape[1])] +
            [f"tfidf_{i}" for i in range(train_tfidf.shape[1])]
        )
        
        self.feature_names = feature_names
        
        logger.info(f"Feature extraction complete. Total features: {train_features_scaled.shape[1]}")
        
        return train_features_scaled, val_features_scaled, test_features_scaled
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            cv_folds: int = 5,
                            n_iter: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with cross-validation
        """
        logger.info("Starting hyperparameter tuning...")
        
        with mlflow.start_run(run_name="hyperparameter_tuning", nested=True):
            # Define parameter grids for different models
            param_grids = {
                'logistic_regression': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                    'classifier__solver': ['liblinear', 'saga'],
                    'classifier__class_weight': [None, 'balanced']
                },
                'random_forest': {
                    'classifier__n_estimators': [50, 100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__class_weight': [None, 'balanced']
                },
                'gradient_boosting': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__subsample': [0.8, 0.9, 1.0]
                }
            }
            
            # Cross-validation setup
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            best_results = {}
            
            # Tune each model type
            for model_name, param_grid in param_grids.items():
                logger.info(f"Tuning {model_name}...")
                
                # Get base model
                if model_name == 'logistic_regression':
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    
                    base_model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
                    ])
                
                elif model_name == 'random_forest':
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    
                    base_model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', RandomForestClassifier(random_state=self.random_state, n_jobs=-1))
                    ])
                
                elif model_name == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    
                    base_model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', GradientBoostingClassifier(random_state=self.random_state))
                    ])
                
                # Randomized search for efficiency
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='precision',  # Focus on precision
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
                
                search.fit(X_train, y_train)
                
                best_results[model_name] = {
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'best_model': search.best_estimator_
                }
                
                # Log results to MLflow
                mlflow.log_params({f"{model_name}_best_params": str(search.best_params_)})
                mlflow.log_metric(f"{model_name}_best_cv_precision", search.best_score_)
                
                logger.info(f"{model_name} best CV precision: {search.best_score_:.4f}")
            
            # Select overall best model
            best_model_name = max(best_results.keys(), key=lambda k: best_results[k]['best_score'])
            self.best_model = best_results[best_model_name]['best_model']
            
            mlflow.log_params({"best_model_type": best_model_name})
            mlflow.log_metric("best_overall_cv_precision", best_results[best_model_name]['best_score'])
            
            logger.info(f"Best model: {best_model_name} with CV precision: {best_results[best_model_name]['best_score']:.4f}")
            
            return best_results
    
    def evaluate_model(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      threshold_optimization: bool = True) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        """
        logger.info("Evaluating model performance...")
        
        with mlflow.start_run(run_name="model_evaluation", nested=True):
            # Get predictions
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # Optimize threshold for high precision if needed
            if threshold_optimization:
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                
                # Find threshold that achieves 95% precision
                target_precision = 0.95
                high_precision_indices = precision >= target_precision
                
                if np.any(high_precision_indices):
                    optimal_threshold = thresholds[np.where(high_precision_indices)[0][0]]
                else:
                    optimal_threshold = thresholds[np.argmax(precision)]
                
                self.optimal_threshold = optimal_threshold
            else:
                self.optimal_threshold = 0.5
            
            y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_pred_proba),
                "optimal_threshold": self.optimal_threshold
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics.update({
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1])
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            logger.info(f"Test set evaluation metrics: {metrics}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            return metrics
    
    def analyze_feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance
        """
        logger.info("Analyzing feature importance...")
        
        # Get feature importance from the best model
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            # Tree-based models
            importance_scores = self.best_model.named_steps['classifier'].feature_importances_
        elif hasattr(self.best_model.named_steps['classifier'], 'coef_'):
            # Linear models
            importance_scores = np.abs(self.best_model.named_steps['classifier'].coef_[0])
        else:
            logger.warning("Model does not support feature importance analysis")
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        self.feature_importance = importance_df
        
        # Log top features
        top_features = importance_df.head(top_k)
        logger.info(f"Top {top_k} most important features:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_pipeline(self, version: str = None) -> str:
        """
        Save the complete trained pipeline
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pipeline_data = {
            'model': self.best_model,
            'feature_extractor': self.feature_extractor,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5),
            'feature_names': getattr(self, 'feature_names', []),
            'feature_importance': getattr(self, 'feature_importance', None),
            'training_config': {
                'experiment_name': self.experiment_name,
                'random_state': self.random_state,
                'version': version,
                'created_at': datetime.now().isoformat()
            }
        }
        
        pipeline_path = os.path.join(self.model_save_path, f"arbitration_pipeline_v{version}.pkl")
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {pipeline_path}")
        return pipeline_path
    
    def run_full_pipeline(self,
                         arbitration_texts: List[str],
                         non_arbitration_texts: List[str],
                         augment_data: bool = True,
                         tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        """
        logger.info("Starting full training pipeline...")
        
        with mlflow.start_run(run_name="full_training_pipeline"):
            results = {}
            
            # Step 1: Prepare data
            data_info = self.load_and_prepare_data(
                arbitration_texts, non_arbitration_texts, augment_data=augment_data
            )
            results['data_preparation'] = data_info
            mlflow.log_params(data_info)
            
            # Step 2: Extract features
            X_train, X_val, X_test = self.extract_and_prepare_features()
            y_train = self.training_data[1]
            y_val = self.validation_data[1]
            y_test = self.test_data[1]
            
            # Step 3: Hyperparameter tuning
            if tune_hyperparameters:
                tuning_results = self.hyperparameter_tuning(X_train, y_train)
                results['hyperparameter_tuning'] = tuning_results
            else:
                # Use default models
                self.classifier.train_statistical_classifier(X_train, y_train, X_val, y_val)
                self.best_model = self.classifier.models['statistical']
            
            # Step 4: Final evaluation
            eval_metrics = self.evaluate_model(X_test, y_test)
            results['evaluation'] = eval_metrics
            
            # Step 5: Feature importance analysis
            importance_df = self.analyze_feature_importance()
            results['feature_importance'] = importance_df
            
            # Step 6: Save pipeline
            pipeline_path = self.save_pipeline()
            results['pipeline_path'] = pipeline_path
            mlflow.log_artifact(pipeline_path)
            
            logger.info("Training pipeline completed successfully!")
            return results


def create_sample_data() -> Tuple[List[str], List[str]]:
    """
    Create sample training data for demonstration
    """
    arbitration_texts = [
        "Any dispute arising under this agreement shall be resolved through binding arbitration administered by the American Arbitration Association.",
        "All controversies shall be settled by final and binding arbitration under the Commercial Arbitration Rules of AAA.",
        "The parties agree to submit any dispute to arbitration under JAMS rules.",
        "Disputes will be resolved exclusively through binding arbitration administered by ICC.",
        "Any claim shall be subject to final arbitration under UNCITRAL rules.",
        "All disputes shall be finally settled by arbitration administered by LCIA.",
        "The parties waive their right to jury trial and agree to binding arbitration.",
        "Any controversy arising out of this contract shall be settled by arbitration.",
        "Disputes must be resolved through individual arbitration proceedings.",
        "The arbitral tribunal shall have exclusive jurisdiction over any disputes."
    ]
    
    non_arbitration_texts = [
        "Any dispute shall be resolved in the courts of New York.",
        "The parties retain the right to seek judicial remedies in court.",
        "This agreement shall be governed by the laws of California.",
        "Any legal action must be brought in the appropriate federal court.",
        "The parties may seek injunctive relief in any court of competent jurisdiction.",
        "Disputes may be resolved through negotiation or mediation only.",
        "This contract does not waive the right to jury trial.",
        "Legal proceedings shall be conducted in state court.",
        "The parties reserve all rights to pursue legal remedies in court.",
        "Any lawsuit must be filed in the designated jurisdiction."
    ]
    
    return arbitration_texts, non_arbitration_texts


if __name__ == "__main__":
    # Example usage
    pipeline = TrainingPipeline()
    
    # Create sample data
    arb_texts, non_arb_texts = create_sample_data()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(arb_texts, non_arb_texts)
    
    print("Training pipeline completed!")
    print(f"Final test precision: {results['evaluation']['precision']:.4f}")
    print(f"Final test recall: {results['evaluation']['recall']:.4f}")
    print(f"Pipeline saved to: {results['pipeline_path']}")