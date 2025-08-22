"""
Feedback loop system for continuous model improvement
Collects user corrections, retrains models, and tracks performance drift
"""
import logging
import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import threading
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow

from .model_registry import ModelRegistry, ModelMetrics, ModelStatus
from .training_pipeline import TrainingPipeline
from .ensemble import ArbitrationEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Individual feedback record from user correction"""
    feedback_id: str
    text: str
    model_prediction: int
    model_confidence: float
    user_correction: int
    user_id: str
    timestamp: str
    model_id: str
    session_id: str
    additional_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class ModelDriftMetrics:
    """Model drift detection metrics"""
    model_id: str
    evaluation_date: str
    precision_drift: float
    recall_drift: float
    f1_drift: float
    prediction_distribution_drift: float
    confidence_distribution_drift: float
    sample_size: int
    drift_detected: bool
    drift_severity: str  # low, medium, high


class FeedbackDatabase:
    """
    SQLite database for storing feedback records
    """
    
    def __init__(self, db_path: str = "backend/models/feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feedback records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    model_prediction INTEGER NOT NULL,
                    model_confidence REAL NOT NULL,
                    user_correction INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    session_id TEXT,
                    additional_context TEXT
                )
            """)
            
            # Model performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    performance_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    accuracy REAL,
                    sample_size INTEGER,
                    drift_detected BOOLEAN,
                    drift_severity TEXT
                )
            """)
            
            # Retraining logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_logs (
                    retrain_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    new_model_id TEXT,
                    performance_improvement REAL,
                    training_data_size INTEGER
                )
            """)
            
            conn.commit()
    
    def store_feedback(self, feedback: FeedbackRecord):
        """Store feedback record in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO feedback_records 
                (feedback_id, text, model_prediction, model_confidence, user_correction,
                 user_id, timestamp, model_id, session_id, additional_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.text,
                feedback.model_prediction,
                feedback.model_confidence,
                feedback.user_correction,
                feedback.user_id,
                feedback.timestamp,
                feedback.model_id,
                feedback.session_id,
                json.dumps(feedback.additional_context)
            ))
            conn.commit()
    
    def get_feedback_records(self,
                           model_id: str = None,
                           start_date: str = None,
                           end_date: str = None,
                           limit: int = None) -> List[FeedbackRecord]:
        """Retrieve feedback records with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM feedback_records WHERE 1=1"
            params = []
            
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                additional_context = json.loads(row[9]) if row[9] else {}
                record = FeedbackRecord(
                    feedback_id=row[0],
                    text=row[1],
                    model_prediction=row[2],
                    model_confidence=row[3],
                    user_correction=row[4],
                    user_id=row[5],
                    timestamp=row[6],
                    model_id=row[7],
                    session_id=row[8],
                    additional_context=additional_context
                )
                records.append(record)
            
            return records
    
    def get_correction_stats(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get correction statistics for a model"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total corrections
            cursor.execute("""
                SELECT COUNT(*) FROM feedback_records 
                WHERE model_id = ? AND timestamp >= ?
            """, (model_id, cutoff_date))
            total_corrections = cursor.fetchone()[0]
            
            # Corrections by type
            cursor.execute("""
                SELECT model_prediction, user_correction, COUNT(*) 
                FROM feedback_records 
                WHERE model_id = ? AND timestamp >= ?
                GROUP BY model_prediction, user_correction
            """, (model_id, cutoff_date))
            correction_breakdown = cursor.fetchall()
            
            # Calculate error rates
            cursor.execute("""
                SELECT COUNT(*) FROM feedback_records 
                WHERE model_id = ? AND timestamp >= ? 
                AND model_prediction != user_correction
            """, (model_id, cutoff_date))
            errors = cursor.fetchone()[0]
            
            error_rate = errors / total_corrections if total_corrections > 0 else 0
            
            return {
                "total_corrections": total_corrections,
                "total_errors": errors,
                "error_rate": error_rate,
                "correction_breakdown": correction_breakdown,
                "evaluation_period_days": days
            }


class DriftDetector:
    """
    Detect model performance drift using feedback data
    """
    
    def __init__(self, 
                 feedback_db: FeedbackDatabase,
                 drift_threshold: float = 0.05,
                 min_samples: int = 100):
        self.feedback_db = feedback_db
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.baseline_metrics = {}
    
    def set_baseline_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Set baseline metrics for drift comparison"""
        self.baseline_metrics[model_id] = metrics
        logger.info(f"Baseline metrics set for model {model_id}: {metrics}")
    
    def detect_performance_drift(self, model_id: str, days: int = 7) -> ModelDriftMetrics:
        """
        Detect performance drift by comparing recent feedback to baseline
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get recent feedback
        feedback_records = self.feedback_db.get_feedback_records(
            model_id=model_id,
            start_date=cutoff_date
        )
        
        if len(feedback_records) < self.min_samples:
            logger.warning(f"Insufficient samples for drift detection: {len(feedback_records)}")
            return ModelDriftMetrics(
                model_id=model_id,
                evaluation_date=datetime.now().isoformat(),
                precision_drift=0.0,
                recall_drift=0.0,
                f1_drift=0.0,
                prediction_distribution_drift=0.0,
                confidence_distribution_drift=0.0,
                sample_size=len(feedback_records),
                drift_detected=False,
                drift_severity="insufficient_data"
            )
        
        # Calculate current metrics from feedback
        y_true = [r.user_correction for r in feedback_records]
        y_pred = [r.model_prediction for r in feedback_records]
        confidences = [r.model_confidence for r in feedback_records]
        
        current_metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Compare to baseline
        baseline = self.baseline_metrics.get(model_id, {})
        
        precision_drift = abs(current_metrics["precision"] - baseline.get("precision", 0))
        recall_drift = abs(current_metrics["recall"] - baseline.get("recall", 0))
        f1_drift = abs(current_metrics["f1"] - baseline.get("f1", 0))
        
        # Detect distribution drift
        pred_distribution_drift = self._calculate_distribution_drift(y_pred, baseline.get("pred_distribution", []))
        confidence_drift = self._calculate_distribution_drift(confidences, baseline.get("confidence_distribution", []))
        
        # Determine if drift detected
        max_drift = max(precision_drift, recall_drift, f1_drift)
        drift_detected = max_drift > self.drift_threshold
        
        # Classify drift severity
        if max_drift < self.drift_threshold * 0.5:
            severity = "low"
        elif max_drift < self.drift_threshold * 1.5:
            severity = "medium"
        else:
            severity = "high"
        
        drift_metrics = ModelDriftMetrics(
            model_id=model_id,
            evaluation_date=datetime.now().isoformat(),
            precision_drift=precision_drift,
            recall_drift=recall_drift,
            f1_drift=f1_drift,
            prediction_distribution_drift=pred_distribution_drift,
            confidence_distribution_drift=confidence_drift,
            sample_size=len(feedback_records),
            drift_detected=drift_detected,
            drift_severity=severity if drift_detected else "none"
        )
        
        logger.info(f"Drift detection for {model_id}: {drift_detected} (severity: {drift_metrics.drift_severity})")
        return drift_metrics
    
    def _calculate_distribution_drift(self, current_data: List[float], baseline_data: List[float]) -> float:
        """Calculate distribution drift using KL divergence (simplified)"""
        if not baseline_data:
            return 0.0
        
        # Simple histogram-based comparison
        try:
            current_hist, _ = np.histogram(current_data, bins=10, range=(0, 1))
            baseline_hist, _ = np.histogram(baseline_data, bins=10, range=(0, 1))
            
            # Normalize to probabilities
            current_prob = current_hist / np.sum(current_hist)
            baseline_prob = baseline_hist / np.sum(baseline_hist)
            
            # Calculate KL divergence
            epsilon = 1e-8
            kl_div = np.sum(current_prob * np.log((current_prob + epsilon) / (baseline_prob + epsilon)))
            
            return float(kl_div)
        
        except Exception as e:
            logger.warning(f"Error calculating distribution drift: {e}")
            return 0.0


class FeedbackLoop:
    """
    Main feedback loop coordinator for continuous model improvement
    """
    
    def __init__(self,
                 model_registry: ModelRegistry,
                 feedback_db: FeedbackDatabase = None,
                 retraining_threshold: int = 1000,
                 drift_check_interval_hours: int = 24,
                 auto_retrain: bool = True):
        self.model_registry = model_registry
        self.feedback_db = feedback_db or FeedbackDatabase()
        self.retraining_threshold = retraining_threshold
        self.drift_check_interval_hours = drift_check_interval_hours
        self.auto_retrain = auto_retrain
        
        self.drift_detector = DriftDetector(self.feedback_db)
        self.training_pipeline = TrainingPipeline()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info("Feedback loop system initialized")
    
    def collect_feedback(self,
                        text: str,
                        model_prediction: int,
                        model_confidence: float,
                        user_correction: int,
                        user_id: str,
                        model_id: str,
                        session_id: str = None,
                        additional_context: Dict[str, Any] = None) -> str:
        """
        Collect user feedback for a prediction
        """
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            text=text,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            user_correction=user_correction,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            model_id=model_id,
            session_id=session_id or "",
            additional_context=additional_context or {}
        )
        
        # Store feedback
        self.feedback_db.store_feedback(feedback)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"feedback_collection_{feedback_id}", nested=True):
            mlflow.log_params({
                "model_id": model_id,
                "user_id": user_id,
                "prediction_correct": model_prediction == user_correction
            })
            mlflow.log_metrics({
                "model_confidence": model_confidence,
                "prediction_error": abs(model_prediction - user_correction)
            })
        
        # Check if retraining threshold reached
        if self.auto_retrain:
            self._check_retraining_trigger(model_id)
        
        logger.info(f"Feedback collected: {feedback_id}")
        return feedback_id
    
    def _check_retraining_trigger(self, model_id: str):
        """
        Check if model should be retrained based on feedback volume or drift
        """
        # Get recent feedback count
        recent_feedback = self.feedback_db.get_feedback_records(
            model_id=model_id,
            start_date=(datetime.now() - timedelta(days=30)).isoformat()
        )
        
        feedback_count = len(recent_feedback)
        error_count = sum(1 for f in recent_feedback if f.model_prediction != f.user_correction)
        error_rate = error_count / feedback_count if feedback_count > 0 else 0
        
        # Trigger retraining if thresholds exceeded
        should_retrain = False
        trigger_reason = ""
        
        if feedback_count >= self.retraining_threshold:
            should_retrain = True
            trigger_reason = f"feedback_volume_threshold_reached_{feedback_count}"
        
        elif error_rate > 0.2 and feedback_count >= 100:  # 20% error rate with minimum samples
            should_retrain = True
            trigger_reason = f"high_error_rate_{error_rate:.3f}"
        
        # Check for drift
        drift_metrics = self.drift_detector.detect_performance_drift(model_id)
        if drift_metrics.drift_detected and drift_metrics.drift_severity in ["medium", "high"]:
            should_retrain = True
            trigger_reason = f"performance_drift_{drift_metrics.drift_severity}"
        
        if should_retrain:
            logger.info(f"Triggering retraining for {model_id}: {trigger_reason}")
            self._trigger_retraining(model_id, trigger_reason)
    
    def _trigger_retraining(self, model_id: str, trigger_reason: str):
        """
        Trigger model retraining using feedback data
        """
        retrain_id = f"retrain_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log retraining start
        with sqlite3.connect(self.feedback_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retraining_logs 
                (retrain_id, model_id, trigger_reason, start_time, status, training_data_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                retrain_id,
                model_id,
                trigger_reason,
                datetime.now().isoformat(),
                "started",
                0
            ))
            conn.commit()
        
        try:
            # Get feedback data for retraining
            feedback_records = self.feedback_db.get_feedback_records(
                model_id=model_id,
                start_date=(datetime.now() - timedelta(days=90)).isoformat()
            )
            
            if len(feedback_records) < 50:
                logger.warning(f"Insufficient feedback data for retraining: {len(feedback_records)}")
                return
            
            # Prepare training data
            texts = [f.text for f in feedback_records]
            labels = [f.user_correction for f in feedback_records]
            
            # Split into arbitration and non-arbitration
            arbitration_texts = [t for t, l in zip(texts, labels) if l == 1]
            non_arbitration_texts = [t for t, l in zip(texts, labels) if l == 0]
            
            # Retrain model
            logger.info(f"Starting retraining with {len(feedback_records)} feedback samples")
            
            results = self.training_pipeline.run_full_pipeline(
                arbitration_texts=arbitration_texts,
                non_arbitration_texts=non_arbitration_texts,
                augment_data=True,
                tune_hyperparameters=True
            )
            
            # Register new model
            current_model = self.model_registry.get_model(model_id)
            new_version = f"{current_model.version}_retrained_{datetime.now().strftime('%Y%m%d')}"
            
            new_metrics = ModelMetrics(
                precision=results['evaluation']['precision'],
                recall=results['evaluation']['recall'],
                f1_score=results['evaluation']['f1_score'],
                auc_roc=results['evaluation']['auc_roc'],
                accuracy=0.0,  # Calculate if needed
                true_positives=results['evaluation']['true_positives'],
                false_positives=results['evaluation']['false_positives'],
                true_negatives=results['evaluation']['true_negatives'],
                false_negatives=results['evaluation']['false_negatives'],
                evaluation_date=datetime.now().isoformat(),
                dataset_size=len(feedback_records)
            )
            
            new_model_id = self.model_registry.register_model(
                model_path=results['pipeline_path'],
                name=current_model.name,
                version=new_version,
                description=f"Retrained model based on user feedback. Trigger: {trigger_reason}",
                model_type=current_model.model_type,
                metrics=new_metrics,
                training_config={"retrain_trigger": trigger_reason, "feedback_samples": len(feedback_records)},
                tags=current_model.tags + ["retrained", "feedback-based"]
            )
            
            # Promote to staging for evaluation
            self.model_registry.promote_model(new_model_id, ModelStatus.STAGING)
            
            # Update retraining log
            performance_improvement = new_metrics.precision - current_model.metrics.precision
            
            with sqlite3.connect(self.feedback_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE retraining_logs 
                    SET end_time = ?, status = ?, new_model_id = ?, 
                        performance_improvement = ?, training_data_size = ?
                    WHERE retrain_id = ?
                """, (
                    datetime.now().isoformat(),
                    "completed",
                    new_model_id,
                    performance_improvement,
                    len(feedback_records),
                    retrain_id
                ))
                conn.commit()
            
            logger.info(f"Retraining completed. New model: {new_model_id}, Performance improvement: {performance_improvement:.4f}")
        
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            
            # Update retraining log with failure
            with sqlite3.connect(self.feedback_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE retraining_logs 
                    SET end_time = ?, status = ?
                    WHERE retrain_id = ?
                """, (
                    datetime.now().isoformat(),
                    f"failed: {str(e)}",
                    retrain_id
                ))
                conn.commit()
    
    def start_monitoring(self):
        """
        Start background monitoring for drift detection and automatic retraining
        """
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """
        Stop background monitoring
        """
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Background monitoring loop
        """
        while self._monitoring_active:
            try:
                # Check drift for all production models
                production_models = self.model_registry.list_models(status=ModelStatus.PRODUCTION)
                
                for model in production_models:
                    drift_metrics = self.drift_detector.detect_performance_drift(model.model_id)
                    
                    if drift_metrics.drift_detected:
                        logger.warning(f"Drift detected for model {model.model_id}: {drift_metrics.drift_severity}")
                        
                        if self.auto_retrain and drift_metrics.drift_severity in ["medium", "high"]:
                            self._trigger_retraining(
                                model.model_id,
                                f"automatic_drift_detection_{drift_metrics.drift_severity}"
                            )
                
                # Sleep until next check
                time.sleep(self.drift_check_interval_hours * 3600)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def get_feedback_summary(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive feedback summary for a model
        """
        correction_stats = self.feedback_db.get_correction_stats(model_id, days)
        drift_metrics = self.drift_detector.detect_performance_drift(model_id, days)
        
        recent_feedback = self.feedback_db.get_feedback_records(
            model_id=model_id,
            start_date=(datetime.now() - timedelta(days=days)).isoformat()
        )
        
        # Confidence analysis
        if recent_feedback:
            confidences = [f.model_confidence for f in recent_feedback]
            avg_confidence = np.mean(confidences)
            
            # Low confidence predictions
            low_confidence_threshold = 0.7
            low_confidence_count = sum(1 for c in confidences if c < low_confidence_threshold)
            low_confidence_rate = low_confidence_count / len(confidences)
        else:
            avg_confidence = 0
            low_confidence_rate = 0
        
        return {
            "model_id": model_id,
            "evaluation_period_days": days,
            "correction_stats": correction_stats,
            "drift_metrics": asdict(drift_metrics),
            "confidence_analysis": {
                "average_confidence": avg_confidence,
                "low_confidence_rate": low_confidence_rate,
                "low_confidence_threshold": 0.7
            },
            "feedback_volume": len(recent_feedback)
        }


def demo_feedback_loop():
    """
    Demonstrate feedback loop functionality
    """
    # Initialize components
    registry = ModelRegistry()
    feedback_db = FeedbackDatabase()
    feedback_loop = FeedbackLoop(registry, feedback_db, auto_retrain=False)
    
    # Simulate model feedback
    sample_texts = [
        "Any dispute arising under this agreement shall be resolved through binding arbitration.",
        "The parties retain the right to seek judicial remedies in court.",
        "All controversies shall be settled by final arbitration under AAA rules."
    ]
    
    model_id = "demo_model_123"
    
    # Collect feedback
    for i, text in enumerate(sample_texts):
        model_pred = 1 if "arbitration" in text else 0
        user_correction = model_pred  # Simulate correct predictions
        
        feedback_id = feedback_loop.collect_feedback(
            text=text,
            model_prediction=model_pred,
            model_confidence=0.85,
            user_correction=user_correction,
            user_id=f"user_{i}",
            model_id=model_id
        )
        
        print(f"Collected feedback: {feedback_id}")
    
    # Get feedback summary
    summary = feedback_loop.get_feedback_summary(model_id)
    print(f"Feedback summary: {summary}")
    
    print("Feedback loop demo completed!")


if __name__ == "__main__":
    demo_feedback_loop()