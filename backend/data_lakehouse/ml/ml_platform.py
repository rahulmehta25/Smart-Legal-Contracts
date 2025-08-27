"""
ML Platform Manager

Comprehensive machine learning platform for the lakehouse architecture:
- Unified ML workflow orchestration
- Feature store integration
- Model lifecycle management
- Experiment tracking and versioning
- Automated training and deployment pipelines
- Model monitoring and governance
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.regression import *
from pyspark.ml.clustering import *
from pyspark.ml.evaluation import *


class MLTaskType(Enum):
    """Types of ML tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class ModelStatus(Enum):
    """Model lifecycle status"""
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentType(Enum):
    """Model deployment types"""
    BATCH = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"
    SERVERLESS = "serverless"


@dataclass
class MLExperiment:
    """ML experiment configuration"""
    experiment_id: str
    name: str
    description: Optional[str]
    task_type: MLTaskType
    
    # Data configuration
    training_data_path: str
    validation_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    feature_store_config: Optional[Dict[str, Any]] = None
    
    # Model configuration
    algorithm: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=list)
    
    # Experiment settings
    max_runtime_hours: int = 24
    early_stopping: bool = True
    cross_validation_folds: int = 5
    
    # Resource requirements
    executor_instances: int = 4
    executor_memory: str = "8g"
    
    # Metadata
    created_by: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MLModel:
    """ML model definition"""
    model_id: str
    name: str
    version: str
    experiment_id: str
    task_type: MLTaskType
    status: ModelStatus
    
    # Model artifacts
    model_path: Optional[str] = None
    pipeline_path: Optional[str] = None
    metadata_path: Optional[str] = None
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model schema
    input_schema: Optional[StructType] = None
    output_schema: Optional[StructType] = None
    feature_names: List[str] = field(default_factory=list)
    
    # Deployment configuration
    deployment_config: Optional[Dict[str, Any]] = None
    serving_endpoints: List[str] = field(default_factory=list)
    
    # Lineage and governance
    training_data_lineage: List[str] = field(default_factory=list)
    feature_dependencies: List[str] = field(default_factory=list)
    
    # Timestamps
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    deployed_date: Optional[datetime] = None
    
    # Custom properties
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Training job execution details"""
    job_id: str
    experiment_id: str
    model_id: Optional[str]
    status: str
    
    # Execution details
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Resource usage
    cpu_hours: float = 0.0
    memory_gb_hours: float = 0.0
    
    # Results
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Logs and artifacts
    log_path: Optional[str] = None
    artifact_paths: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class MLPlatformManager:
    """
    Comprehensive ML platform manager for lakehouse architecture.
    
    Features:
    - End-to-end ML workflow orchestration
    - Feature store integration and management
    - Experiment tracking and versioning
    - Automated model training and hyperparameter tuning
    - Model deployment and serving infrastructure
    - Model monitoring and drift detection
    - MLflow integration for experiment management
    - AutoML capabilities for citizen data scientists
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.feature_store = None
        self.model_training_engine = None
        self.model_serving_engine = None
        self.automl_engine = None
        self.monitoring_engine = None
        
        # Registries
        self.experiments: Dict[str, MLExperiment] = {}
        self.models: Dict[str, MLModel] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        # Performance tracking
        self.platform_metrics: Dict[str, Any] = {
            "total_experiments": 0,
            "total_models": 0,
            "models_in_production": 0,
            "training_jobs_completed": 0,
            "avg_training_time_hours": 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        # Setup MLflow integration
        self._setup_mlflow()
    
    def _initialize_components(self):
        """Initialize ML platform components"""
        try:
            # Feature Store
            from .feature_store import FeatureStore
            feature_store_config = self.config.get("feature_store", {})
            self.feature_store = FeatureStore(self.spark, feature_store_config)
            
            # Model Training Engine
            from .model_training import ModelTrainingEngine
            training_config = self.config.get("training", {})
            self.model_training_engine = ModelTrainingEngine(self.spark, training_config)
            
            # Model Serving Engine
            from .model_serving import ModelServingEngine
            serving_config = self.config.get("serving", {})
            self.model_serving_engine = ModelServingEngine(self.spark, serving_config)
            
            # AutoML Engine
            from .automl import AutoMLEngine
            automl_config = self.config.get("automl", {})
            self.automl_engine = AutoMLEngine(self.spark, automl_config)
            
            # Model Monitoring Engine
            from .monitoring import ModelMonitoringEngine
            monitoring_config = self.config.get("monitoring", {})
            self.monitoring_engine = ModelMonitoringEngine(self.spark, monitoring_config)
            
            self.logger.info("ML platform components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML platform components: {str(e)}")
            raise
    
    def _setup_mlflow(self):
        """Setup MLflow integration"""
        try:
            import mlflow
            import mlflow.spark
            
            # Configure MLflow tracking
            mlflow_config = self.config.get("mlflow", {})
            tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
            
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set default experiment
            experiment_name = mlflow_config.get("default_experiment", "Default")
            try:
                mlflow.set_experiment(experiment_name)
            except:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            
            self.logger.info(f"MLflow tracking initialized: {tracking_uri}")
            
        except ImportError:
            self.logger.warning("MLflow not available - experiment tracking will be limited")
        except Exception as e:
            self.logger.warning(f"Error setting up MLflow: {str(e)}")
    
    def create_experiment(
        self,
        name: str,
        task_type: MLTaskType,
        training_data_path: str,
        description: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a new ML experiment
        
        Args:
            name: Experiment name
            task_type: Type of ML task
            training_data_path: Path to training data
            description: Optional description
            **kwargs: Additional experiment configuration
            
        Returns:
            str: Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            experiment = MLExperiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                task_type=task_type,
                training_data_path=training_data_path,
                **kwargs
            )
            
            # Store experiment
            self.experiments[experiment_id] = experiment
            
            # Create MLflow experiment
            try:
                import mlflow
                mlflow.create_experiment(f"{name}_{experiment_id}")
            except:
                pass  # MLflow not available or experiment exists
            
            # Update metrics
            self.platform_metrics["total_experiments"] += 1
            
            self.logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Error creating experiment {name}: {str(e)}")
            raise
    
    def train_model(
        self,
        experiment_id: str,
        model_name: str,
        algorithm: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_automl: bool = False
    ) -> str:
        """
        Train a model for an experiment
        
        Args:
            experiment_id: Experiment ID
            model_name: Name for the model
            algorithm: ML algorithm to use
            hyperparameters: Algorithm hyperparameters
            use_automl: Whether to use AutoML for training
            
        Returns:
            str: Training job ID
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            job_id = str(uuid.uuid4())
            
            # Create training job record
            training_job = TrainingJob(
                job_id=job_id,
                experiment_id=experiment_id,
                model_id=None,  # Will be set after training
                status="starting",
                start_time=datetime.now()
            )
            
            self.training_jobs[job_id] = training_job
            
            self.logger.info(f"Starting model training for experiment {experiment_id}")
            
            # Load training data
            training_df = self.spark.read.format("delta").load(experiment.training_data_path)
            
            # Prepare features using feature store
            if experiment.feature_store_config:
                training_df = self.feature_store.get_features_for_training(
                    training_df, 
                    experiment.feature_store_config
                )
            
            # Train model
            if use_automl:
                model_result = self.automl_engine.train_model(
                    training_df,
                    experiment.task_type,
                    target_column=experiment.hyperparameters.get("target_column", "label"),
                    experiment_config=experiment
                )
            else:
                model_result = self.model_training_engine.train_model(
                    training_df,
                    algorithm or "auto",
                    hyperparameters or {},
                    experiment
                )
            
            # Create model record
            model_id = str(uuid.uuid4())
            model = MLModel(
                model_id=model_id,
                name=model_name,
                version="1.0.0",
                experiment_id=experiment_id,
                task_type=experiment.task_type,
                status=ModelStatus.TRAINED,
                model_path=model_result.get("model_path"),
                training_metrics=model_result.get("metrics", {}),
                input_schema=training_df.schema,
                feature_names=model_result.get("feature_names", []),
                training_data_lineage=[experiment.training_data_path]
            )
            
            # Store model
            self.models[model_id] = model
            
            # Update training job
            training_job.model_id = model_id
            training_job.status = "completed"
            training_job.end_time = datetime.now()
            training_job.duration_seconds = int((training_job.end_time - training_job.start_time).total_seconds())
            training_job.final_metrics = model_result.get("metrics", {})
            
            # Update platform metrics
            self.platform_metrics["total_models"] += 1
            self.platform_metrics["training_jobs_completed"] += 1
            
            self.logger.info(f"Model training completed: {model_id}")
            return job_id
            
        except Exception as e:
            # Update training job with error
            if job_id in self.training_jobs:
                training_job = self.training_jobs[job_id]
                training_job.status = "failed"
                training_job.end_time = datetime.now()
                training_job.error_message = str(e)
            
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate_model(
        self,
        model_id: str,
        test_data_path: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_id: Model ID to evaluate
            test_data_path: Path to test data (optional)
            metrics: Evaluation metrics to compute
            
        Returns:
            Dict[str, float]: Evaluation results
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Load test data
            if test_data_path:
                test_df = self.spark.read.format("delta").load(test_data_path)
            else:
                # Use experiment's test data
                experiment = self.experiments[model.experiment_id]
                if experiment.test_data_path:
                    test_df = self.spark.read.format("delta").load(experiment.test_data_path)
                else:
                    raise ValueError("No test data available for evaluation")
            
            # Load model for evaluation
            from pyspark.ml import PipelineModel
            pipeline_model = PipelineModel.load(model.model_path)
            
            # Make predictions
            predictions = pipeline_model.transform(test_df)
            
            # Calculate evaluation metrics
            evaluation_results = self._calculate_evaluation_metrics(
                predictions,
                model.task_type,
                metrics or ["accuracy", "f1", "precision", "recall"]
            )
            
            # Update model with test metrics
            model.test_metrics = evaluation_results
            model.last_modified = datetime.now()
            
            self.logger.info(f"Model evaluation completed for {model_id}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_id}: {str(e)}")
            raise
    
    def _calculate_evaluation_metrics(
        self,
        predictions: DataFrame,
        task_type: MLTaskType,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type"""
        try:
            results = {}
            
            if task_type == MLTaskType.CLASSIFICATION:
                # Classification metrics
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label", 
                    predictionCol="prediction"
                )
                
                if "accuracy" in metrics:
                    results["accuracy"] = evaluator.setMetricName("accuracy").evaluate(predictions)
                if "f1" in metrics:
                    results["f1"] = evaluator.setMetricName("f1").evaluate(predictions)
                if "precision" in metrics:
                    results["precision"] = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
                if "recall" in metrics:
                    results["recall"] = evaluator.setMetricName("weightedRecall").evaluate(predictions)
            
            elif task_type == MLTaskType.REGRESSION:
                # Regression metrics
                evaluator = RegressionEvaluator(
                    labelCol="label",
                    predictionCol="prediction"
                )
                
                if "rmse" in metrics:
                    results["rmse"] = evaluator.setMetricName("rmse").evaluate(predictions)
                if "mae" in metrics:
                    results["mae"] = evaluator.setMetricName("mae").evaluate(predictions)
                if "r2" in metrics:
                    results["r2"] = evaluator.setMetricName("r2").evaluate(predictions)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def deploy_model(
        self,
        model_id: str,
        deployment_type: DeploymentType,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Deploy a model for serving
        
        Args:
            model_id: Model ID to deploy
            deployment_type: Type of deployment
            deployment_config: Deployment configuration
            
        Returns:
            str: Deployment endpoint URL
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Deploy using model serving engine
            endpoint_url = self.model_serving_engine.deploy_model(
                model_id,
                model.model_path,
                deployment_type,
                deployment_config or {}
            )
            
            # Update model status and serving info
            model.status = ModelStatus.PRODUCTION
            model.deployment_config = deployment_config
            model.serving_endpoints.append(endpoint_url)
            model.deployed_date = datetime.now()
            model.last_modified = datetime.now()
            
            # Update platform metrics
            self.platform_metrics["models_in_production"] += 1
            
            self.logger.info(f"Model {model_id} deployed to {endpoint_url}")
            return endpoint_url
            
        except Exception as e:
            self.logger.error(f"Error deploying model {model_id}: {str(e)}")
            raise
    
    def predict(
        self,
        model_id: str,
        input_data: Union[DataFrame, Dict[str, Any]],
        endpoint_url: Optional[str] = None
    ) -> Union[DataFrame, Dict[str, Any]]:
        """
        Make predictions using a deployed model
        
        Args:
            model_id: Model ID
            input_data: Input data for prediction
            endpoint_url: Optional specific endpoint to use
            
        Returns:
            Union[DataFrame, Dict[str, Any]]: Predictions
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Use model serving engine for prediction
            predictions = self.model_serving_engine.predict(
                model_id,
                input_data,
                endpoint_url
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions with model {model_id}: {str(e)}")
            raise
    
    def monitor_model(
        self,
        model_id: str,
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Setup monitoring for a deployed model
        
        Args:
            model_id: Model ID to monitor
            monitoring_config: Monitoring configuration
            
        Returns:
            str: Monitoring job ID
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Setup monitoring using monitoring engine
            monitoring_job_id = self.monitoring_engine.setup_monitoring(
                model_id,
                model,
                monitoring_config or {}
            )
            
            self.logger.info(f"Model monitoring setup for {model_id}: {monitoring_job_id}")
            return monitoring_job_id
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring for model {model_id}: {str(e)}")
            raise
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results and metrics for an experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            # Get associated models
            experiment_models = [
                model for model in self.models.values()
                if model.experiment_id == experiment_id
            ]
            
            # Get training jobs
            training_jobs = [
                job for job in self.training_jobs.values()
                if job.experiment_id == experiment_id
            ]
            
            results = {
                "experiment": experiment,
                "models": len(experiment_models),
                "best_model": None,
                "training_jobs": len(training_jobs),
                "avg_training_time": sum(j.duration_seconds or 0 for j in training_jobs) / len(training_jobs) if training_jobs else 0
            }
            
            # Find best model based on validation metrics
            if experiment_models:
                best_model = max(
                    experiment_models,
                    key=lambda m: m.validation_metrics.get("accuracy", m.validation_metrics.get("f1", 0))
                )
                results["best_model"] = {
                    "model_id": best_model.model_id,
                    "name": best_model.name,
                    "metrics": best_model.validation_metrics
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting experiment results: {str(e)}")
            return {"error": str(e)}
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get lineage information for a model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            experiment = self.experiments[model.experiment_id]
            
            lineage = {
                "model_id": model_id,
                "experiment_id": model.experiment_id,
                "training_data": model.training_data_lineage,
                "feature_dependencies": model.feature_dependencies,
                "algorithm": experiment.algorithm,
                "hyperparameters": experiment.hyperparameters,
                "created_date": model.created_date.isoformat(),
                "last_modified": model.last_modified.isoformat()
            }
            
            # Add feature store lineage if available
            if self.feature_store:
                feature_lineage = self.feature_store.get_feature_lineage(model.feature_names)
                lineage["feature_lineage"] = feature_lineage
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Error getting model lineage: {str(e)}")
            return {"error": str(e)}
    
    def get_platform_metrics(self) -> Dict[str, Any]:
        """Get overall platform metrics and statistics"""
        try:
            # Update dynamic metrics
            models_in_production = len([
                m for m in self.models.values()
                if m.status == ModelStatus.PRODUCTION
            ])
            
            completed_jobs = [j for j in self.training_jobs.values() if j.status == "completed"]
            avg_training_time = sum(j.duration_seconds or 0 for j in completed_jobs) / len(completed_jobs) if completed_jobs else 0
            
            metrics = self.platform_metrics.copy()
            metrics.update({
                "models_in_production": models_in_production,
                "avg_training_time_hours": avg_training_time / 3600,
                "active_experiments": len(self.experiments),
                "total_training_jobs": len(self.training_jobs),
                "success_rate": len(completed_jobs) / len(self.training_jobs) if self.training_jobs else 0.0
            })
            
            # Add component-specific metrics
            if self.feature_store:
                metrics["feature_store"] = self.feature_store.get_metrics()
            
            if self.model_serving_engine:
                metrics["serving"] = self.model_serving_engine.get_metrics()
            
            if self.monitoring_engine:
                metrics["monitoring"] = self.monitoring_engine.get_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting platform metrics: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_experiments(self, retention_days: int = 90):
        """Cleanup old experiments and models"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Find old experiments
            old_experiments = [
                exp_id for exp_id, exp in self.experiments.items()
                if exp.created_date < cutoff_date
            ]
            
            for exp_id in old_experiments:
                # Archive associated models
                exp_models = [m_id for m_id, m in self.models.items() if m.experiment_id == exp_id]
                
                for model_id in exp_models:
                    model = self.models[model_id]
                    if model.status != ModelStatus.PRODUCTION:
                        model.status = ModelStatus.ARCHIVED
                
                # Keep experiment but mark as archived
                self.experiments[exp_id].tags["archived"] = "true"
            
            self.logger.info(f"Cleaned up {len(old_experiments)} old experiments")
            
        except Exception as e:
            self.logger.error(f"Error during experiment cleanup: {str(e)}")
    
    def cleanup(self):
        """Cleanup ML platform resources"""
        try:
            # Cleanup components
            if self.feature_store:
                self.feature_store.cleanup()
            
            if self.model_training_engine:
                self.model_training_engine.cleanup()
            
            if self.model_serving_engine:
                self.model_serving_engine.cleanup()
            
            if self.automl_engine:
                self.automl_engine.cleanup()
            
            if self.monitoring_engine:
                self.monitoring_engine.cleanup()
            
            self.logger.info("ML platform cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during ML platform cleanup: {str(e)}")


# Utility functions

def create_classification_experiment(
    name: str,
    training_data_path: str,
    target_column: str = "label",
    **kwargs
) -> MLExperiment:
    """Create a classification experiment configuration"""
    return MLExperiment(
        experiment_id=str(uuid.uuid4()),
        name=name,
        task_type=MLTaskType.CLASSIFICATION,
        training_data_path=training_data_path,
        hyperparameters={"target_column": target_column},
        evaluation_metrics=["accuracy", "f1", "precision", "recall"],
        **kwargs
    )


def create_regression_experiment(
    name: str,
    training_data_path: str,
    target_column: str = "label",
    **kwargs
) -> MLExperiment:
    """Create a regression experiment configuration"""
    return MLExperiment(
        experiment_id=str(uuid.uuid4()),
        name=name,
        task_type=MLTaskType.REGRESSION,
        training_data_path=training_data_path,
        hyperparameters={"target_column": target_column},
        evaluation_metrics=["rmse", "mae", "r2"],
        **kwargs
    )