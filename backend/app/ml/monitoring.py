"""
Model monitoring and evaluation metrics for production ML systems
Tracks performance, latency, resource usage, and data quality
"""
import logging
import os
import json
import time
import psutil
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import threading
import queue
import warnings
from collections import defaultdict, deque
import mlflow
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    """Individual prediction log entry"""
    prediction_id: str
    model_id: str
    timestamp: str
    input_text: str
    input_hash: str
    prediction: int
    confidence: float
    latency_ms: float
    model_version: str
    user_id: str
    session_id: str
    request_metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: str
    model_id: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: Optional[float]
    prediction_count: int
    error_rate: float
    avg_confidence: float
    avg_latency_ms: float


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    gpu_usage_percent: Optional[float]
    gpu_memory_usage_mb: Optional[float]
    active_requests: int
    queue_size: int


@dataclass
class DataQualityMetrics:
    """Input data quality metrics"""
    timestamp: str
    model_id: str
    avg_text_length: float
    text_length_std: float
    null_inputs: int
    empty_inputs: int
    duplicate_inputs: int
    unusual_characters: int
    encoding_issues: int
    total_inputs: int


class MetricsDatabase:
    """
    SQLite database for storing monitoring metrics
    """
    
    def __init__(self, db_path: str = "backend/models/monitoring.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prediction logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    prediction_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    input_text TEXT,
                    input_hash TEXT,
                    prediction INTEGER,
                    confidence REAL,
                    latency_ms REAL,
                    model_version TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    request_metadata TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    accuracy REAL,
                    auc_roc REAL,
                    prediction_count INTEGER,
                    error_rate REAL,
                    avg_confidence REAL,
                    avg_latency_ms REAL
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage_percent REAL,
                    memory_usage_percent REAL,
                    memory_usage_mb REAL,
                    disk_usage_percent REAL,
                    gpu_usage_percent REAL,
                    gpu_memory_usage_mb REAL,
                    active_requests INTEGER,
                    queue_size INTEGER
                )
            """)
            
            # Data quality metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    avg_text_length REAL,
                    text_length_std REAL,
                    null_inputs INTEGER,
                    empty_inputs INTEGER,
                    duplicate_inputs INTEGER,
                    unusual_characters INTEGER,
                    encoding_issues INTEGER,
                    total_inputs INTEGER
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    model_id TEXT,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            """)
            
            conn.commit()
    
    def log_prediction(self, log_entry: PredictionLog):
        """Store prediction log"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_logs
                (prediction_id, model_id, timestamp, input_text, input_hash,
                 prediction, confidence, latency_ms, model_version, user_id,
                 session_id, request_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.prediction_id,
                log_entry.model_id,
                log_entry.timestamp,
                log_entry.input_text,
                log_entry.input_hash,
                log_entry.prediction,
                log_entry.confidence,
                log_entry.latency_ms,
                log_entry.model_version,
                log_entry.user_id,
                log_entry.session_id,
                json.dumps(log_entry.request_metadata)
            ))
            conn.commit()
    
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics
                (timestamp, model_id, precision_score, recall, f1_score, accuracy,
                 auc_roc, prediction_count, error_rate, avg_confidence, avg_latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.model_id,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.accuracy,
                metrics.auc_roc,
                metrics.prediction_count,
                metrics.error_rate,
                metrics.avg_confidence,
                metrics.avg_latency_ms
            ))
            conn.commit()
    
    def log_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics
                (timestamp, cpu_usage_percent, memory_usage_percent, memory_usage_mb,
                 disk_usage_percent, gpu_usage_percent, gpu_memory_usage_mb,
                 active_requests, queue_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.cpu_usage_percent,
                metrics.memory_usage_percent,
                metrics.memory_usage_mb,
                metrics.disk_usage_percent,
                metrics.gpu_usage_percent,
                metrics.gpu_memory_usage_mb,
                metrics.active_requests,
                metrics.queue_size
            ))
            conn.commit()
    
    def get_recent_metrics(self, 
                          metric_type: str,
                          hours: int = 24,
                          model_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve recent metrics"""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if metric_type == "performance":
                query = """
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= ?
                """
                params = [cutoff_time]
                
                if model_id:
                    query += " AND model_id = ?"
                    params.append(model_id)
                
                query += " ORDER BY timestamp DESC"
                
            elif metric_type == "system":
                query = "SELECT * FROM system_metrics WHERE timestamp >= ? ORDER BY timestamp DESC"
                params = [cutoff_time]
                
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]


class AlertManager:
    """
    Alert management system for monitoring thresholds
    """
    
    def __init__(self, metrics_db: MetricsDatabase):
        self.metrics_db = metrics_db
        self.alert_rules = {}
        self.alert_callbacks = []
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules = {
            "low_precision": {
                "metric": "precision",
                "threshold": 0.85,
                "operator": "lt",
                "severity": "warning",
                "message": "Model precision below threshold"
            },
            "high_latency": {
                "metric": "avg_latency_ms",
                "threshold": 1000,
                "operator": "gt",
                "severity": "warning",
                "message": "High prediction latency detected"
            },
            "high_error_rate": {
                "metric": "error_rate",
                "threshold": 0.1,
                "operator": "gt",
                "severity": "critical",
                "message": "High error rate detected"
            },
            "high_cpu_usage": {
                "metric": "cpu_usage_percent",
                "threshold": 90.0,
                "operator": "gt",
                "severity": "warning",
                "message": "High CPU usage"
            },
            "high_memory_usage": {
                "metric": "memory_usage_percent",
                "threshold": 85.0,
                "operator": "gt",
                "severity": "warning",
                "message": "High memory usage"
            }
        }
    
    def add_alert_rule(self, 
                      rule_name: str,
                      metric: str,
                      threshold: float,
                      operator: str,
                      severity: str,
                      message: str):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = {
            "metric": metric,
            "threshold": threshold,
            "operator": operator,
            "severity": severity,
            "message": message
        }
    
    def add_alert_callback(self, callback: Callable[[str, str, str, str], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Any], model_id: str = None):
        """Check metrics against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            metric_value = metrics.get(rule["metric"])
            
            if metric_value is None:
                continue
            
            # Evaluate alert condition
            triggered = False
            if rule["operator"] == "gt" and metric_value > rule["threshold"]:
                triggered = True
            elif rule["operator"] == "lt" and metric_value < rule["threshold"]:
                triggered = True
            elif rule["operator"] == "eq" and metric_value == rule["threshold"]:
                triggered = True
            
            if triggered:
                self._trigger_alert(rule_name, rule, metric_value, model_id)
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metric_value: float, model_id: str):
        """Trigger alert and notify callbacks"""
        alert_id = f"alert_{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert_details = {
            "rule_name": rule_name,
            "metric": rule["metric"],
            "threshold": rule["threshold"],
            "actual_value": metric_value,
            "model_id": model_id
        }
        
        # Store alert in database
        with sqlite3.connect(self.metrics_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts
                (alert_id, timestamp, alert_type, severity, model_id, message, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                datetime.now().isoformat(),
                rule_name,
                rule["severity"],
                model_id,
                rule["message"],
                json.dumps(alert_details)
            ))
            conn.commit()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_id, rule["severity"], rule["message"], json.dumps(alert_details))
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert triggered: {rule_name} - {rule['message']} (value: {metric_value})")


class ModelMonitor:
    """
    Comprehensive model monitoring system
    """
    
    def __init__(self,
                 metrics_db: MetricsDatabase = None,
                 monitoring_interval_seconds: int = 60,
                 max_prediction_logs: int = 10000):
        self.metrics_db = metrics_db or MetricsDatabase()
        self.monitoring_interval = monitoring_interval_seconds
        self.max_prediction_logs = max_prediction_logs
        
        # Alert manager
        self.alert_manager = AlertManager(self.metrics_db)
        
        # In-memory caches for real-time metrics
        self.prediction_cache = deque(maxlen=max_prediction_logs)
        self.recent_predictions = defaultdict(list)
        self.system_metrics_cache = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._request_queue = queue.Queue()
        self._active_requests = 0
        
        logger.info("Model monitor initialized")
    
    def log_prediction(self,
                      model_id: str,
                      input_text: str,
                      prediction: int,
                      confidence: float,
                      latency_ms: float,
                      model_version: str = "unknown",
                      user_id: str = "anonymous",
                      session_id: str = "",
                      request_metadata: Dict[str, Any] = None):
        """Log a prediction for monitoring"""
        import hashlib
        
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id}_{user_id}"
        input_hash = hashlib.md5(input_text.encode()).hexdigest()
        
        log_entry = PredictionLog(
            prediction_id=prediction_id,
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            input_text=input_text,
            input_hash=input_hash,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            model_version=model_version,
            user_id=user_id,
            session_id=session_id,
            request_metadata=request_metadata or {}
        )
        
        # Add to cache
        self.prediction_cache.append(log_entry)
        self.recent_predictions[model_id].append(log_entry)
        
        # Limit recent predictions per model
        if len(self.recent_predictions[model_id]) > 1000:
            self.recent_predictions[model_id] = self.recent_predictions[model_id][-1000:]
        
        # Store in database
        self.metrics_db.log_prediction(log_entry)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"prediction_log_{prediction_id}", nested=True):
            mlflow.log_params({
                "model_id": model_id,
                "model_version": model_version,
                "user_id": user_id
            })
            mlflow.log_metrics({
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms
            })
    
    def calculate_performance_metrics(self, model_id: str, window_hours: int = 1) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics for a model"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        # Get recent predictions for this model
        recent_preds = [
            pred for pred in self.recent_predictions[model_id]
            if datetime.fromisoformat(pred.timestamp) >= cutoff_time
        ]
        
        if not recent_preds:
            return None
        
        # Calculate metrics (would need ground truth for accuracy - using synthetic here)
        predictions = [pred.prediction for pred in recent_preds]
        confidences = [pred.confidence for pred in recent_preds]
        latencies = [pred.latency_ms for pred in recent_preds]
        
        # For demo purposes, simulate some errors
        error_count = sum(1 for conf in confidences if conf < 0.7)  # Low confidence as proxy for errors
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            model_id=model_id,
            precision=0.95 - (error_count / len(recent_preds)) * 0.1,  # Simulated
            recall=0.90,  # Simulated
            f1_score=0.92,  # Simulated
            accuracy=1.0 - (error_count / len(recent_preds)),  # Simulated
            auc_roc=0.93,  # Simulated
            prediction_count=len(recent_preds),
            error_rate=error_count / len(recent_preds),
            avg_confidence=np.mean(confidences),
            avg_latency_ms=np.mean(latencies)
        )
        
        # Store metrics
        self.metrics_db.log_performance_metrics(metrics)
        
        # Check alerts
        self.alert_manager.check_alerts(asdict(metrics), model_id)
        
        return metrics
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
                gpu_memory = gpus[0].memoryUsed
        except ImportError:
            pass  # GPU monitoring not available
        
        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            memory_usage_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_usage_mb=gpu_memory,
            active_requests=self._active_requests,
            queue_size=self._request_queue.qsize()
        )
        
        # Add to cache
        self.system_metrics_cache.append(metrics)
        
        # Store in database
        self.metrics_db.log_system_metrics(metrics)
        
        # Check alerts
        self.alert_manager.check_alerts(asdict(metrics))
        
        return metrics
    
    def analyze_data_quality(self, model_id: str, window_hours: int = 1) -> DataQualityMetrics:
        """Analyze input data quality"""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        recent_preds = [
            pred for pred in self.recent_predictions[model_id]
            if datetime.fromisoformat(pred.timestamp) >= cutoff_time
        ]
        
        if not recent_preds:
            return DataQualityMetrics(
                timestamp=datetime.now().isoformat(),
                model_id=model_id,
                avg_text_length=0,
                text_length_std=0,
                null_inputs=0,
                empty_inputs=0,
                duplicate_inputs=0,
                unusual_characters=0,
                encoding_issues=0,
                total_inputs=0
            )
        
        texts = [pred.input_text for pred in recent_preds]
        text_lengths = [len(text) if text else 0 for text in texts]
        
        # Analyze quality issues
        null_inputs = sum(1 for text in texts if text is None)
        empty_inputs = sum(1 for text in texts if text == "")
        
        # Find duplicates
        text_hashes = [pred.input_hash for pred in recent_preds]
        unique_hashes = set(text_hashes)
        duplicate_inputs = len(text_hashes) - len(unique_hashes)
        
        # Check for unusual characters
        unusual_chars = 0
        encoding_issues = 0
        for text in texts:
            if text:
                # Count non-ASCII characters as unusual (simplified)
                unusual_chars += sum(1 for char in text if ord(char) > 127)
                
                # Check for encoding issues (simplified)
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    encoding_issues += 1
        
        metrics = DataQualityMetrics(
            timestamp=datetime.now().isoformat(),
            model_id=model_id,
            avg_text_length=np.mean(text_lengths) if text_lengths else 0,
            text_length_std=np.std(text_lengths) if text_lengths else 0,
            null_inputs=null_inputs,
            empty_inputs=empty_inputs,
            duplicate_inputs=duplicate_inputs,
            unusual_characters=unusual_chars,
            encoding_issues=encoding_issues,
            total_inputs=len(recent_preds)
        )
        
        return metrics
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                self.collect_system_metrics()
                
                # Calculate performance metrics for all active models
                active_models = set(pred.model_id for pred in self.prediction_cache)
                for model_id in active_models:
                    self.calculate_performance_metrics(model_id)
                
                # Sleep until next iteration
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def generate_dashboard_data(self, model_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        dashboard_data = {}
        
        # Performance metrics
        perf_metrics = self.metrics_db.get_recent_metrics("performance", hours, model_id)
        dashboard_data["performance_metrics"] = perf_metrics
        
        # System metrics
        sys_metrics = self.metrics_db.get_recent_metrics("system", hours)
        dashboard_data["system_metrics"] = sys_metrics
        
        # Current status
        if model_id:
            current_perf = self.calculate_performance_metrics(model_id, window_hours=1)
            dashboard_data["current_performance"] = asdict(current_perf) if current_perf else None
            
            data_quality = self.analyze_data_quality(model_id, window_hours=1)
            dashboard_data["data_quality"] = asdict(data_quality)
        
        current_system = self.collect_system_metrics()
        dashboard_data["current_system"] = asdict(current_system)
        
        return dashboard_data
    
    def create_monitoring_charts(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Create monitoring charts using Plotly"""
        dashboard_data = self.generate_dashboard_data(model_id, hours)
        
        charts = {}
        
        # Performance metrics chart
        if dashboard_data["performance_metrics"]:
            perf_df = pd.DataFrame(dashboard_data["performance_metrics"])
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision/Recall', 'Latency', 'Prediction Count', 'Error Rate'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Precision/Recall
            fig.add_trace(
                go.Scatter(x=perf_df['timestamp'], y=perf_df['precision_score'], 
                          name='Precision', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=perf_df['timestamp'], y=perf_df['recall'], 
                          name='Recall', line=dict(color='green')),
                row=1, col=1
            )
            
            # Latency
            fig.add_trace(
                go.Scatter(x=perf_df['timestamp'], y=perf_df['avg_latency_ms'], 
                          name='Avg Latency (ms)', line=dict(color='orange')),
                row=1, col=2
            )
            
            # Prediction Count
            fig.add_trace(
                go.Scatter(x=perf_df['timestamp'], y=perf_df['prediction_count'], 
                          name='Predictions', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Error Rate
            fig.add_trace(
                go.Scatter(x=perf_df['timestamp'], y=perf_df['error_rate'], 
                          name='Error Rate', line=dict(color='red')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text=f"Model Performance Metrics - {model_id}")
            charts["performance"] = fig.to_json()
        
        # System metrics chart
        if dashboard_data["system_metrics"]:
            sys_df = pd.DataFrame(dashboard_data["system_metrics"])
            sys_df['timestamp'] = pd.to_datetime(sys_df['timestamp'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Active Requests', 'Queue Size')
            )
            
            fig.add_trace(
                go.Scatter(x=sys_df['timestamp'], y=sys_df['cpu_usage_percent'], 
                          name='CPU %', line=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=sys_df['timestamp'], y=sys_df['memory_usage_percent'], 
                          name='Memory %', line=dict(color='blue')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=sys_df['timestamp'], y=sys_df['active_requests'], 
                          name='Active Requests', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=sys_df['timestamp'], y=sys_df['queue_size'], 
                          name='Queue Size', line=dict(color='orange')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="System Metrics")
            charts["system"] = fig.to_json()
        
        return charts


def demo_monitoring():
    """
    Demonstrate monitoring functionality
    """
    # Initialize monitor
    monitor = ModelMonitor(monitoring_interval_seconds=5)
    
    # Add alert callback
    def alert_callback(alert_id, severity, message, details):
        print(f"ALERT [{severity}]: {message} (ID: {alert_id})")
    
    monitor.alert_manager.add_alert_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some predictions
    model_id = "demo_model_123"
    
    for i in range(10):
        monitor.log_prediction(
            model_id=model_id,
            input_text=f"Sample legal text {i}",
            prediction=1 if i % 2 == 0 else 0,
            confidence=0.85 + (i % 3) * 0.05,
            latency_ms=100 + i * 10,
            user_id=f"user_{i}"
        )
        time.sleep(1)
    
    # Generate dashboard data
    dashboard_data = monitor.generate_dashboard_data(model_id)
    print(f"Dashboard data generated with {len(dashboard_data)} sections")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("Monitoring demo completed!")


if __name__ == "__main__":
    demo_monitoring()