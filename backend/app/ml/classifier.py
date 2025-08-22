"""
Binary classifier for arbitration clause detection
High precision focus for legal document analysis
"""
import logging
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArbitrationClassifier:
    """
    High-precision binary classifier for arbitration clause detection
    Combines multiple models for optimal performance
    """
    
    def __init__(self, 
                 model_save_path: str = "backend/models",
                 experiment_name: str = "arbitration_classification",
                 precision_threshold: float = 0.95):
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        self.precision_threshold = precision_threshold
        self.models = {}
        self.feature_extractors = {}
        self.scalers = {}
        self.best_threshold = 0.5
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Ensure model directory exists
        os.makedirs(model_save_path, exist_ok=True)
    
    def train_statistical_classifier(self, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_val: np.ndarray = None,
                                   y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train multiple statistical classifiers and select best for high precision
        """
        with mlflow.start_run(run_name="statistical_classifier_training"):
            results = {}
            
            # Define classifiers to test
            classifiers = {
                'logistic_regression': LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'svm': SVC(
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=100
                )
            }
            
            # Cross-validation setup
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, classifier in classifiers.items():
                logger.info(f"Training {name}...")
                
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', classifier)
                ])
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision')
                
                # Fit on full training set
                pipeline.fit(X_train, y_train)
                
                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                    
                    # Find optimal threshold for high precision
                    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
                    
                    # Find threshold that achieves desired precision
                    high_precision_indices = precision >= self.precision_threshold
                    if np.any(high_precision_indices):
                        optimal_threshold = thresholds[np.where(high_precision_indices)[0][0]]
                    else:
                        # Fallback to highest precision threshold
                        optimal_threshold = thresholds[np.argmax(precision)]
                    
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                    
                    # Calculate metrics
                    precision_val = precision_score(y_val, y_pred)
                    recall_val = recall_score(y_val, y_pred)
                    f1_val = f1_score(y_val, y_pred)
                    auc_val = roc_auc_score(y_val, y_pred_proba)
                    
                    results[name] = {
                        'model': pipeline,
                        'cv_precision_mean': cv_scores.mean(),
                        'cv_precision_std': cv_scores.std(),
                        'val_precision': precision_val,
                        'val_recall': recall_val,
                        'val_f1': f1_val,
                        'val_auc': auc_val,
                        'optimal_threshold': optimal_threshold
                    }
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        f"{name}_cv_precision_mean": cv_scores.mean(),
                        f"{name}_cv_precision_std": cv_scores.std(),
                        f"{name}_val_precision": precision_val,
                        f"{name}_val_recall": recall_val,
                        f"{name}_val_f1": f1_val,
                        f"{name}_val_auc": auc_val,
                        f"{name}_optimal_threshold": optimal_threshold
                    })
                    
                    logger.info(f"{name} - Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}")
                else:
                    results[name] = {
                        'model': pipeline,
                        'cv_precision_mean': cv_scores.mean(),
                        'cv_precision_std': cv_scores.std()
                    }
            
            # Select best model based on validation precision
            if X_val is not None:
                best_model_name = max(results.keys(), key=lambda k: results[k]['val_precision'])
                self.best_threshold = results[best_model_name]['optimal_threshold']
            else:
                best_model_name = max(results.keys(), key=lambda k: results[k]['cv_precision_mean'])
            
            self.models['statistical'] = results[best_model_name]['model']
            
            # Save best model
            model_path = os.path.join(self.model_save_path, f"statistical_classifier_{best_model_name}.pkl")
            joblib.dump(self.models['statistical'], model_path)
            
            # Log best model to MLflow
            mlflow.sklearn.log_model(
                self.models['statistical'],
                f"statistical_classifier_{best_model_name}",
                registered_model_name="arbitration_statistical_classifier"
            )
            
            logger.info(f"Best statistical model: {best_model_name}")
            return results
    
    def train_transformer_classifier(self,
                                   train_texts: List[str],
                                   train_labels: List[int],
                                   val_texts: List[str] = None,
                                   val_labels: List[int] = None,
                                   model_name: str = "nlpaueb/legal-bert-base-uncased",
                                   epochs: int = 3,
                                   batch_size: int = 16) -> Dict[str, float]:
        """
        Train transformer-based classifier for arbitration detection
        """
        with mlflow.start_run(run_name="transformer_classifier_training"):
            from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
            import torch
            from torch.utils.data import Dataset
            
            class ArbitrationDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=512):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = str(self.texts[idx])
                    label = self.labels[idx]
                    
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].flatten(),
                        'attention_mask': encoding['attention_mask'].flatten(),
                        'labels': torch.tensor(label, dtype=torch.long)
                    }
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                problem_type="single_label_classification"
            )
            
            # Create datasets
            train_dataset = ArbitrationDataset(train_texts, train_labels, tokenizer)
            val_dataset = None
            if val_texts and val_labels:
                val_dataset = ArbitrationDataset(val_texts, val_labels, tokenizer)
            
            # Training arguments optimized for high precision
            training_args = TrainingArguments(
                output_dir=os.path.join(self.model_save_path, "transformer_classifier"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=2e-5,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False,
                save_total_limit=2,
                seed=42
            )
            
            # Custom metrics for high precision focus
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                precision = precision_score(labels, predictions, zero_division=0)
                recall = recall_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics if val_dataset else None
            )
            
            # Train
            trainer.train()
            
            # Save model and tokenizer
            model_save_dir = os.path.join(self.model_save_path, "transformer_classifier")
            trainer.save_model(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            
            self.models['transformer'] = (model, tokenizer)
            
            # Evaluate and log metrics
            if val_dataset:
                eval_results = trainer.evaluate()
                
                # Log to MLflow
                mlflow.log_metrics({
                    "transformer_val_precision": eval_results.get("eval_precision", 0),
                    "transformer_val_recall": eval_results.get("eval_recall", 0),
                    "transformer_val_f1": eval_results.get("eval_f1", 0),
                    "transformer_val_loss": eval_results.get("eval_loss", 0)
                })
                
                # Log model to MLflow
                mlflow.pytorch.log_model(
                    model,
                    "transformer_classifier",
                    registered_model_name="arbitration_transformer_classifier"
                )
                
                return eval_results
            
            return {}
    
    def optimize_threshold_for_precision(self, 
                                       y_true: np.ndarray, 
                                       y_proba: np.ndarray,
                                       target_precision: float = None) -> float:
        """
        Find optimal threshold to achieve target precision
        """
        if target_precision is None:
            target_precision = self.precision_threshold
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that achieves target precision
        high_precision_indices = precision >= target_precision
        
        if np.any(high_precision_indices):
            # Choose threshold with highest recall among those meeting precision requirement
            valid_indices = np.where(high_precision_indices)[0]
            best_index = valid_indices[np.argmax(recall[valid_indices])]
            optimal_threshold = thresholds[best_index]
        else:
            # Fallback to highest precision threshold
            optimal_threshold = thresholds[np.argmax(precision)]
        
        logger.info(f"Optimal threshold for precision {target_precision}: {optimal_threshold}")
        return optimal_threshold
    
    def predict(self, 
                texts: List[str], 
                model_type: str = 'ensemble',
                return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions with high precision focus
        """
        if model_type == 'statistical' and 'statistical' in self.models:
            # For statistical models, we need features
            raise NotImplementedError("Feature extraction needed for statistical model predictions")
        
        elif model_type == 'transformer' and 'transformer' in self.models:
            model, tokenizer = self.models['transformer']
            
            # Tokenize texts
            inputs = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].numpy()
            
            if return_probabilities:
                return probabilities
            
            # Apply optimized threshold for high precision
            predictions = (probabilities >= self.best_threshold).astype(int)
            return predictions
        
        elif model_type == 'ensemble':
            # Combine multiple model predictions
            # Implementation would depend on available models
            raise NotImplementedError("Ensemble prediction requires all models to be trained")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def evaluate_model(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      model_type: str = 'statistical') -> Dict[str, float]:
        """
        Comprehensive evaluation focused on precision
        """
        if model_type == 'statistical' and 'statistical' in self.models:
            y_pred_proba = self.models['statistical'].predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        else:
            raise NotImplementedError(f"Evaluation for {model_type} not implemented")
        
        # Calculate comprehensive metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'optimal_threshold': self.best_threshold
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics.update({
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        })
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model evaluation metrics: {metrics}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return metrics
    
    def save_model(self, model_name: str, version: str = None) -> str:
        """
        Save trained model with versioning
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_filename = f"{model_name}_v{version}.pkl"
        model_path = os.path.join(self.model_save_path, model_filename)
        
        model_info = {
            'models': self.models,
            'best_threshold': self.best_threshold,
            'precision_threshold': self.precision_threshold,
            'version': version,
            'created_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load trained model
        """
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.models = model_info['models']
        self.best_threshold = model_info['best_threshold']
        self.precision_threshold = model_info['precision_threshold']
        
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    classifier = ArbitrationClassifier()
    
    # Example training data (replace with real data)
    # X_train, y_train = load_training_data()
    # X_val, y_val = load_validation_data()
    
    # Train statistical classifier
    # results = classifier.train_statistical_classifier(X_train, y_train, X_val, y_val)
    
    # Train transformer classifier
    # train_texts = ["Sample arbitration clause...", "Regular contract clause..."]
    # train_labels = [1, 0]
    # transformer_results = classifier.train_transformer_classifier(train_texts, train_labels)
    
    print("Arbitration classifier module loaded successfully")