"""
Model registry and versioning system with A/B testing capability
Manages model lifecycle, performance tracking, and deployment
"""
import logging
import os
import json
import pickle
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import random
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    SHADOW = "shadow"  # For A/B testing


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    evaluation_date: str
    dataset_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    version: str
    name: str
    description: str
    model_type: str  # ensemble, statistical, transformer, etc.
    status: ModelStatus
    created_at: str
    updated_at: str
    created_by: str
    file_path: str
    file_size: int
    file_hash: str
    training_config: Dict[str, Any]
    dependencies: Dict[str, str]
    metrics: ModelMetrics
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        return data


class ModelRegistry:
    """
    Centralized model registry for arbitration detection models
    """
    
    def __init__(self, 
                 registry_path: str = "backend/models/registry",
                 mlflow_tracking_uri: str = None):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "model_registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # MLflow setup
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load existing registry
        self.registry = self._load_registry()
        
        logger.info(f"Model registry initialized at {registry_path}")
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load existing model registry"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            registry = {}
            for model_id, model_data in data.items():
                # Convert status back to enum
                model_data['status'] = ModelStatus(model_data['status'])
                # Convert metrics dict back to ModelMetrics object
                metrics_data = model_data['metrics']
                model_data['metrics'] = ModelMetrics(**metrics_data)
                registry[model_id] = ModelMetadata(**model_data)
            
            return registry
        
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_data = {}
        for model_id, metadata in self.registry.items():
            registry_data[model_id] = metadata.to_dict()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of model file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_v{version}_{timestamp}"
    
    def register_model(self,
                      model_path: str,
                      name: str,
                      version: str,
                      description: str,
                      model_type: str,
                      metrics: ModelMetrics,
                      training_config: Dict[str, Any],
                      dependencies: Dict[str, str] = None,
                      tags: List[str] = None,
                      status: ModelStatus = ModelStatus.DEVELOPMENT,
                      created_by: str = "system") -> str:
        """
        Register a new model in the registry
        """
        logger.info(f"Registering model: {name} v{version}")
        
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Copy model file to registry
        model_filename = f"{model_id}.pkl"
        registry_model_path = self.models_dir / model_filename
        shutil.copy2(model_path, registry_model_path)
        
        # Calculate file metadata
        file_size = os.path.getsize(registry_model_path)
        file_hash = self._calculate_file_hash(registry_model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            model_type=model_type,
            status=status,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            created_by=created_by,
            file_path=str(registry_model_path),
            file_size=file_size,
            file_hash=file_hash,
            training_config=training_config,
            dependencies=dependencies or {},
            metrics=metrics,
            tags=tags or []
        )
        
        # Add to registry
        self.registry[model_id] = metadata
        self._save_registry()
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"model_registration_{model_id}"):
            mlflow.log_params({
                "model_id": model_id,
                "model_name": name,
                "model_version": version,
                "model_type": model_type,
                "status": status.value
            })
            mlflow.log_metrics(metrics.to_dict())
            mlflow.log_artifact(str(registry_model_path))
        
        logger.info(f"Model registered with ID: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.registry.get(model_id)
    
    def list_models(self,
                   name: str = None,
                   status: ModelStatus = None,
                   model_type: str = None,
                   tags: List[str] = None) -> List[ModelMetadata]:
        """
        List models with optional filtering
        """
        models = list(self.registry.values())
        
        if name:
            models = [m for m in models if m.name == name]
        
        if status:
            models = [m for m in models if m.status == status]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def promote_model(self, model_id: str, new_status: ModelStatus) -> bool:
        """
        Promote model to new status (development -> staging -> production)
        """
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found")
            return False
        
        current_status = self.registry[model_id].status
        
        # Validate promotion path
        valid_promotions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.STAGING, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.SHADOW, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.ARCHIVED],
            ModelStatus.SHADOW: [ModelStatus.PRODUCTION, ModelStatus.STAGING, ModelStatus.ARCHIVED]
        }
        
        if new_status not in valid_promotions.get(current_status, []):
            logger.error(f"Invalid promotion from {current_status.value} to {new_status.value}")
            return False
        
        # If promoting to production, demote current production models
        if new_status == ModelStatus.PRODUCTION:
            for mid, metadata in self.registry.items():
                if metadata.status == ModelStatus.PRODUCTION and metadata.name == self.registry[model_id].name:
                    metadata.status = ModelStatus.ARCHIVED
                    metadata.updated_at = datetime.now().isoformat()
        
        # Update model status
        self.registry[model_id].status = new_status
        self.registry[model_id].updated_at = datetime.now().isoformat()
        self._save_registry()
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"model_promotion_{model_id}"):
            mlflow.log_params({
                "model_id": model_id,
                "old_status": current_status.value,
                "new_status": new_status.value
            })
        
        logger.info(f"Model {model_id} promoted from {current_status.value} to {new_status.value}")
        return True
    
    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get current production model for a given name"""
        production_models = [
            m for m in self.registry.values()
            if m.name == model_name and m.status == ModelStatus.PRODUCTION
        ]
        
        return production_models[0] if production_models else None
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models side by side
        """
        comparison_data = []
        
        for model_id in model_ids:
            if model_id in self.registry:
                metadata = self.registry[model_id]
                row = {
                    'model_id': model_id,
                    'name': metadata.name,
                    'version': metadata.version,
                    'model_type': metadata.model_type,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at,
                    'precision': metadata.metrics.precision,
                    'recall': metadata.metrics.recall,
                    'f1_score': metadata.metrics.f1_score,
                    'auc_roc': metadata.metrics.auc_roc,
                    'file_size_mb': metadata.file_size / (1024 * 1024)
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model from registry (only if not in production unless forced)
        """
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found")
            return False
        
        metadata = self.registry[model_id]
        
        # Prevent deletion of production models unless forced
        if metadata.status == ModelStatus.PRODUCTION and not force:
            logger.error(f"Cannot delete production model {model_id} without force=True")
            return False
        
        # Delete model file
        if os.path.exists(metadata.file_path):
            os.remove(metadata.file_path)
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
        
        logger.info(f"Model {model_id} deleted from registry")
        return True


class ABTestManager:
    """
    A/B testing manager for model deployment
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.experiments = {}
        self.experiment_file = registry.registry_path / "ab_experiments.json"
        self._load_experiments()
    
    def _load_experiments(self):
        """Load existing A/B test experiments"""
        if self.experiment_file.exists():
            with open(self.experiment_file, 'r') as f:
                self.experiments = json.load(f)
    
    def _save_experiments(self):
        """Save A/B test experiments to disk"""
        with open(self.experiment_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def create_experiment(self,
                         experiment_name: str,
                         control_model_id: str,
                         treatment_model_id: str,
                         traffic_split: float = 0.5,
                         duration_days: int = 7,
                         success_metric: str = "precision") -> str:
        """
        Create a new A/B test experiment
        """
        # Validate models exist
        if control_model_id not in self.registry.registry:
            raise ValueError(f"Control model {control_model_id} not found")
        
        if treatment_model_id not in self.registry.registry:
            raise ValueError(f"Treatment model {treatment_model_id} not found")
        
        # Promote treatment model to shadow status
        self.registry.promote_model(treatment_model_id, ModelStatus.SHADOW)
        
        experiment_id = f"exp_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "control_model_id": control_model_id,
            "treatment_model_id": treatment_model_id,
            "traffic_split": traffic_split,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "status": "active",
            "success_metric": success_metric,
            "results": {
                "control_metrics": {},
                "treatment_metrics": {},
                "statistical_significance": None,
                "winner": None
            }
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        logger.info(f"A/B test experiment created: {experiment_id}")
        return experiment_id
    
    def route_request(self, experiment_id: str, user_id: str = None) -> str:
        """
        Route request to control or treatment model based on traffic split
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment["status"] != "active":
            # Return control model if experiment is not active
            return experiment["control_model_id"]
        
        # Check if experiment has expired
        if datetime.now() > datetime.fromisoformat(experiment["end_date"]):
            self.experiments[experiment_id]["status"] = "completed"
            self._save_experiments()
            return experiment["control_model_id"]
        
        # Determine routing based on traffic split
        if user_id:
            # Consistent routing based on user ID hash
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            split_point = hash_val % 100 / 100.0
        else:
            # Random routing
            split_point = random.random()
        
        if split_point < experiment["traffic_split"]:
            return experiment["treatment_model_id"]
        else:
            return experiment["control_model_id"]
    
    def record_prediction(self,
                         experiment_id: str,
                         model_id: str,
                         prediction: int,
                         actual_label: int = None,
                         confidence: float = None):
        """
        Record prediction results for A/B test analysis
        """
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        
        # Determine if this is control or treatment
        model_type = "control" if model_id == experiment["control_model_id"] else "treatment"
        
        # Initialize metrics if not exists
        if f"{model_type}_predictions" not in experiment["results"]:
            experiment["results"][f"{model_type}_predictions"] = []
        
        # Record prediction
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "actual_label": actual_label,
            "confidence": confidence
        }
        
        experiment["results"][f"{model_type}_predictions"].append(prediction_record)
        self._save_experiments()
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results and determine statistical significance
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Get predictions for both models
        control_predictions = experiment["results"].get("control_predictions", [])
        treatment_predictions = experiment["results"].get("treatment_predictions", [])
        
        if not control_predictions or not treatment_predictions:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate metrics for both models
        def calculate_metrics(predictions):
            if not predictions:
                return {}
            
            pred_values = [p["prediction"] for p in predictions if p["actual_label"] is not None]
            actual_values = [p["actual_label"] for p in predictions if p["actual_label"] is not None]
            
            if not pred_values:
                return {}
            
            return {
                "precision": precision_score(actual_values, pred_values, zero_division=0),
                "recall": recall_score(actual_values, pred_values, zero_division=0),
                "f1_score": f1_score(actual_values, pred_values, zero_division=0),
                "sample_size": len(pred_values)
            }
        
        control_metrics = calculate_metrics(control_predictions)
        treatment_metrics = calculate_metrics(treatment_predictions)
        
        # Statistical significance test (simplified)
        success_metric = experiment["success_metric"]
        if success_metric in control_metrics and success_metric in treatment_metrics:
            control_value = control_metrics[success_metric]
            treatment_value = treatment_metrics[success_metric]
            
            # Simple comparison (in practice, use proper statistical tests)
            improvement = (treatment_value - control_value) / control_value * 100
            
            # Determine winner (simplified)
            if abs(improvement) > 5:  # 5% threshold
                winner = "treatment" if improvement > 0 else "control"
                statistical_significance = True
            else:
                winner = "tie"
                statistical_significance = False
        else:
            improvement = 0
            winner = "insufficient_data"
            statistical_significance = False
        
        # Update experiment results
        experiment["results"]["control_metrics"] = control_metrics
        experiment["results"]["treatment_metrics"] = treatment_metrics
        experiment["results"]["statistical_significance"] = statistical_significance
        experiment["results"]["winner"] = winner
        experiment["results"]["improvement_percent"] = improvement
        
        self._save_experiments()
        
        return {
            "experiment_id": experiment_id,
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
            "statistical_significance": statistical_significance,
            "winner": winner,
            "improvement_percent": improvement
        }
    
    def end_experiment(self, experiment_id: str, promote_winner: bool = True) -> Dict[str, Any]:
        """
        End A/B test experiment and optionally promote winner
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Analyze final results
        results = self.analyze_experiment(experiment_id)
        
        # Mark experiment as completed
        self.experiments[experiment_id]["status"] = "completed"
        self.experiments[experiment_id]["end_date"] = datetime.now().isoformat()
        
        if promote_winner and results["winner"] == "treatment":
            # Promote treatment model to production
            treatment_model_id = self.experiments[experiment_id]["treatment_model_id"]
            self.registry.promote_model(treatment_model_id, ModelStatus.PRODUCTION)
            logger.info(f"Treatment model {treatment_model_id} promoted to production")
        
        self._save_experiments()
        
        logger.info(f"A/B test experiment {experiment_id} ended. Winner: {results['winner']}")
        return results


def demo_model_registry():
    """
    Demonstrate model registry and A/B testing functionality
    """
    # Initialize registry
    registry = ModelRegistry()
    
    # Create sample metrics
    metrics = ModelMetrics(
        precision=0.95,
        recall=0.85,
        f1_score=0.90,
        auc_roc=0.92,
        accuracy=0.88,
        true_positives=85,
        false_positives=5,
        true_negatives=95,
        false_negatives=15,
        evaluation_date=datetime.now().isoformat(),
        dataset_size=200
    )
    
    # Create dummy model file
    dummy_model_path = "dummy_model.pkl"
    with open(dummy_model_path, 'wb') as f:
        pickle.dump({"dummy": "model"}, f)
    
    try:
        # Register a model
        model_id = registry.register_model(
            model_path=dummy_model_path,
            name="arbitration_ensemble",
            version="1.0",
            description="Initial ensemble model for arbitration detection",
            model_type="ensemble",
            metrics=metrics,
            training_config={"epochs": 10, "batch_size": 32},
            tags=["production-ready", "ensemble"]
        )
        
        print(f"Model registered with ID: {model_id}")
        
        # List models
        models = registry.list_models()
        print(f"Total models in registry: {len(models)}")
        
        # Promote model
        registry.promote_model(model_id, ModelStatus.STAGING)
        registry.promote_model(model_id, ModelStatus.PRODUCTION)
        
        # Initialize A/B test manager
        ab_manager = ABTestManager(registry)
        
        print("Model registry demo completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)


if __name__ == "__main__":
    demo_model_registry()