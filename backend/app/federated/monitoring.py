"""
Comprehensive Monitoring System for Federated Learning

This module implements real-time monitoring, metrics collection, alerting,
and performance analysis for federated learning systems.
"""

import asyncio
import logging
import time
import json
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import uuid
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Individual metric value with timestamp"""
    value: float
    timestamp: float
    metadata: Optional[Dict] = None


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    metric_name: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: float
    client_id: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ClientMetrics:
    """Metrics for individual client"""
    client_id: str
    is_active: bool = True
    last_seen: float = 0.0
    
    # Training metrics
    training_loss: deque = field(default_factory=lambda: deque(maxlen=100))
    training_accuracy: deque = field(default_factory=lambda: deque(maxlen=100))
    training_time: deque = field(default_factory=lambda: deque(maxlen=100))
    communication_time: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Resource metrics
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    network_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Privacy metrics
    privacy_budget_used: float = 0.0
    privacy_budget_remaining: float = 10.0
    
    # Quality metrics
    data_quality_score: float = 1.0
    model_contribution_score: float = 1.0
    participation_rate: float = 1.0


@dataclass
class SystemMetrics:
    """Global system metrics"""
    timestamp: float = 0.0
    
    # Training metrics
    global_loss: float = 0.0
    convergence_rate: float = 0.0
    rounds_completed: int = 0
    clients_participated: int = 0
    
    # Performance metrics
    aggregation_time: float = 0.0
    communication_overhead: float = 0.0
    system_throughput: float = 0.0
    
    # Resource metrics
    server_cpu_usage: float = 0.0
    server_memory_usage: float = 0.0
    server_disk_usage: float = 0.0
    server_network_io: Tuple[float, float] = (0.0, 0.0)
    
    # Privacy metrics
    total_privacy_spent: float = 0.0
    privacy_efficiency: float = 1.0
    
    # Quality metrics
    model_quality_score: float = 1.0
    data_heterogeneity: float = 0.0
    client_reliability: float = 1.0


class FederatedMonitor:
    """
    Comprehensive monitoring system for federated learning
    
    Features:
    - Real-time metrics collection
    - Client and system performance monitoring
    - Privacy budget tracking
    - Anomaly detection
    - Alert management
    - Performance analytics and visualization
    - Resource optimization recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Configuration
        self.metrics_retention_days = self.config.get("metrics_retention_days", 30)
        self.alert_cooldown_seconds = self.config.get("alert_cooldown_seconds", 300)
        self.monitoring_interval = self.config.get("monitoring_interval", 5)
        self.anomaly_detection_enabled = self.config.get("anomaly_detection", True)
        
        # State
        self.client_metrics: Dict[str, ClientMetrics] = {}
        self.system_metrics: deque = deque(maxlen=10000)
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Database for persistence
        self.db_path = self.config.get("db_path", "federated_metrics.db")
        self._initialize_database()
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_monitoring = False
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Anomaly detection
        self.anomaly_thresholds = {
            "high_loss": self.config.get("high_loss_threshold", 10.0),
            "low_accuracy": self.config.get("low_accuracy_threshold", 0.3),
            "high_cpu": self.config.get("high_cpu_threshold", 90.0),
            "high_memory": self.config.get("high_memory_threshold", 90.0),
            "long_training_time": self.config.get("long_training_threshold", 300),
            "privacy_budget_low": self.config.get("privacy_budget_threshold", 0.1)
        }
        
        # Performance baseline
        self.performance_baseline = {}
        
        logger.info("Federated learning monitor initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS client_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    client_id TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_client_metrics_timestamp ON client_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_client_metrics_client_id ON client_metrics(client_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Monitoring database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    async def start_monitoring(self):
        """Start monitoring tasks"""
        self.is_monitoring = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._anomaly_detector()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._database_persister()),
            asyncio.create_task(self._metrics_cleaner())
        ]
        
        logger.info("Federated learning monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        self.is_monitoring = False
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Federated learning monitoring stopped")
    
    def register_client(self, client_id: str, initial_metrics: Optional[Dict] = None):
        """Register a new client for monitoring"""
        self.client_metrics[client_id] = ClientMetrics(
            client_id=client_id,
            last_seen=time.time()
        )
        
        if initial_metrics:
            self.update_client_metrics(client_id, initial_metrics)
        
        logger.info(f"Client {client_id} registered for monitoring")
    
    def unregister_client(self, client_id: str):
        """Unregister a client from monitoring"""
        if client_id in self.client_metrics:
            self.client_metrics[client_id].is_active = False
            logger.info(f"Client {client_id} unregistered from monitoring")
    
    def update_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Update metrics for a specific client"""
        if client_id not in self.client_metrics:
            self.register_client(client_id)
        
        client_metrics = self.client_metrics[client_id]
        client_metrics.last_seen = time.time()
        
        # Update various metrics
        if "training_loss" in metrics:
            client_metrics.training_loss.append(MetricValue(
                value=metrics["training_loss"],
                timestamp=time.time()
            ))
        
        if "training_accuracy" in metrics:
            client_metrics.training_accuracy.append(MetricValue(
                value=metrics["training_accuracy"],
                timestamp=time.time()
            ))
        
        if "training_time" in metrics:
            client_metrics.training_time.append(MetricValue(
                value=metrics["training_time"],
                timestamp=time.time()
            ))
        
        if "privacy_budget_used" in metrics:
            client_metrics.privacy_budget_used = metrics["privacy_budget_used"]
        
        if "privacy_budget_remaining" in metrics:
            client_metrics.privacy_budget_remaining = metrics["privacy_budget_remaining"]
        
        # Resource metrics
        if "cpu_usage" in metrics:
            client_metrics.cpu_usage.append(MetricValue(
                value=metrics["cpu_usage"],
                timestamp=time.time()
            ))
        
        if "memory_usage" in metrics:
            client_metrics.memory_usage.append(MetricValue(
                value=metrics["memory_usage"],
                timestamp=time.time()
            ))
    
    async def log_training_round(self, round_info: Any, aggregated_weights: Dict):
        """Log information from a completed training round"""
        try:
            # Calculate system metrics
            system_metric = SystemMetrics(
                timestamp=time.time(),
                rounds_completed=round_info.round_id,
                clients_participated=len(round_info.participating_clients),
                aggregation_time=getattr(round_info, 'aggregation_time', 0.0),
                global_loss=getattr(round_info, 'convergence_metric', 0.0)
            )
            
            # Add resource metrics
            system_metric.server_cpu_usage = psutil.cpu_percent()
            system_metric.server_memory_usage = psutil.virtual_memory().percent
            system_metric.server_disk_usage = psutil.disk_usage('/').percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            system_metric.server_network_io = (net_io.bytes_sent, net_io.bytes_recv)
            
            # Calculate quality metrics
            system_metric.model_quality_score = self._calculate_model_quality_score()
            system_metric.data_heterogeneity = self._calculate_data_heterogeneity()
            system_metric.client_reliability = self._calculate_client_reliability()
            
            self.system_metrics.append(system_metric)
            
            logger.debug(f"Logged training round {round_info.round_id} metrics")
            
        except Exception as e:
            logger.error(f"Failed to log training round: {e}")
    
    def _calculate_model_quality_score(self) -> float:
        """Calculate overall model quality score"""
        try:
            if not self.system_metrics:
                return 1.0
            
            recent_metrics = list(self.system_metrics)[-5:]  # Last 5 rounds
            
            # Use convergence trend as quality indicator
            if len(recent_metrics) >= 2:
                losses = [m.global_loss for m in recent_metrics if m.global_loss > 0]
                if len(losses) >= 2:
                    # Improvement trend
                    improvement = (losses[0] - losses[-1]) / losses[0]
                    return min(1.0, max(0.0, improvement))
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Model quality calculation failed: {e}")
            return 1.0
    
    def _calculate_data_heterogeneity(self) -> float:
        """Calculate data heterogeneity across clients"""
        try:
            active_clients = [
                client for client in self.client_metrics.values()
                if client.is_active and client.training_loss
            ]
            
            if len(active_clients) < 2:
                return 0.0
            
            # Use loss variance as heterogeneity measure
            recent_losses = []
            for client in active_clients:
                if client.training_loss:
                    recent_losses.append(client.training_loss[-1].value)
            
            if len(recent_losses) >= 2:
                return float(np.std(recent_losses) / (np.mean(recent_losses) + 1e-6))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Data heterogeneity calculation failed: {e}")
            return 0.0
    
    def _calculate_client_reliability(self) -> float:
        """Calculate client reliability score"""
        try:
            if not self.client_metrics:
                return 1.0
            
            total_clients = len(self.client_metrics)
            active_clients = sum(1 for c in self.client_metrics.values() if c.is_active)
            
            return active_clients / total_clients if total_clients > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Client reliability calculation failed: {e}")
            return 1.0
    
    async def _system_metrics_collector(self):
        """Background task to collect system metrics"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                current_time = time.time()
                
                # Create system metric snapshot
                system_metric = SystemMetrics(timestamp=current_time)
                
                # System resources
                system_metric.server_cpu_usage = psutil.cpu_percent()
                system_metric.server_memory_usage = psutil.virtual_memory().percent
                system_metric.server_disk_usage = psutil.disk_usage('/').percent
                
                # Network I/O
                net_io = psutil.net_io_counters()
                system_metric.server_network_io = (net_io.bytes_sent, net_io.bytes_recv)
                
                # Client status
                active_clients = sum(1 for c in self.client_metrics.values() if c.is_active)
                system_metric.clients_participated = active_clients
                
                # Add to metrics
                self.system_metrics.append(system_metric)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _anomaly_detector(self):
        """Background task for anomaly detection"""
        while self.is_monitoring:
            try:
                # Check client anomalies
                for client_id, client_metrics in self.client_metrics.items():
                    if not client_metrics.is_active:
                        continue
                    
                    await self._check_client_anomalies(client_id, client_metrics)
                
                # Check system anomalies
                await self._check_system_anomalies()
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Check less frequently
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_client_anomalies(self, client_id: str, client_metrics: ClientMetrics):
        """Check for client-specific anomalies"""
        try:
            current_time = time.time()
            
            # Check training loss anomaly
            if client_metrics.training_loss and len(client_metrics.training_loss) >= 2:
                recent_losses = [m.value for m in list(client_metrics.training_loss)[-5:]]
                avg_loss = np.mean(recent_losses)
                
                if avg_loss > self.anomaly_thresholds["high_loss"]:
                    await self._create_alert(
                        f"high_loss_{client_id}",
                        "training_loss",
                        "high",
                        f"Client {client_id} has high training loss: {avg_loss:.4f}",
                        client_id
                    )
            
            # Check accuracy anomaly
            if client_metrics.training_accuracy and len(client_metrics.training_accuracy) >= 2:
                recent_accuracy = [m.value for m in list(client_metrics.training_accuracy)[-5:]]
                avg_accuracy = np.mean(recent_accuracy)
                
                if avg_accuracy < self.anomaly_thresholds["low_accuracy"]:
                    await self._create_alert(
                        f"low_accuracy_{client_id}",
                        "training_accuracy", 
                        "medium",
                        f"Client {client_id} has low accuracy: {avg_accuracy:.4f}",
                        client_id
                    )
            
            # Check resource usage
            if client_metrics.cpu_usage:
                recent_cpu = client_metrics.cpu_usage[-1].value
                if recent_cpu > self.anomaly_thresholds["high_cpu"]:
                    await self._create_alert(
                        f"high_cpu_{client_id}",
                        "cpu_usage",
                        "medium",
                        f"Client {client_id} has high CPU usage: {recent_cpu:.1f}%",
                        client_id
                    )
            
            # Check privacy budget
            if client_metrics.privacy_budget_remaining < self.anomaly_thresholds["privacy_budget_low"]:
                await self._create_alert(
                    f"low_privacy_{client_id}",
                    "privacy_budget",
                    "high",
                    f"Client {client_id} has low privacy budget: {client_metrics.privacy_budget_remaining:.4f}",
                    client_id
                )
            
            # Check client connectivity
            if current_time - client_metrics.last_seen > 300:  # 5 minutes
                await self._create_alert(
                    f"disconnected_{client_id}",
                    "connectivity",
                    "medium",
                    f"Client {client_id} has been disconnected for {current_time - client_metrics.last_seen:.0f} seconds",
                    client_id
                )
        
        except Exception as e:
            logger.warning(f"Client anomaly check failed for {client_id}: {e}")
    
    async def _check_system_anomalies(self):
        """Check for system-wide anomalies"""
        try:
            if not self.system_metrics:
                return
            
            recent_metrics = list(self.system_metrics)[-5:]
            
            # Check system resource usage
            latest_metric = recent_metrics[-1]
            
            if latest_metric.server_cpu_usage > self.anomaly_thresholds["high_cpu"]:
                await self._create_alert(
                    "system_high_cpu",
                    "server_cpu",
                    "high",
                    f"Server CPU usage is high: {latest_metric.server_cpu_usage:.1f}%"
                )
            
            if latest_metric.server_memory_usage > self.anomaly_thresholds["high_memory"]:
                await self._create_alert(
                    "system_high_memory",
                    "server_memory",
                    "high",
                    f"Server memory usage is high: {latest_metric.server_memory_usage:.1f}%"
                )
            
            # Check convergence stagnation
            if len(recent_metrics) >= 5:
                losses = [m.global_loss for m in recent_metrics if m.global_loss > 0]
                if len(losses) >= 5:
                    loss_variance = np.var(losses)
                    if loss_variance < 1e-6:  # Very low variance indicates stagnation
                        await self._create_alert(
                            "convergence_stagnation",
                            "convergence",
                            "medium",
                            f"Model convergence appears to have stagnated (loss variance: {loss_variance:.2e})"
                        )
        
        except Exception as e:
            logger.warning(f"System anomaly check failed: {e}")
    
    async def _create_alert(
        self,
        alert_id: str,
        metric_name: str,
        severity: str,
        message: str,
        client_id: Optional[str] = None
    ):
        """Create a new alert"""
        try:
            # Check if alert already exists and is recent
            if alert_id in self.alerts:
                existing_alert = self.alerts[alert_id]
                if (time.time() - existing_alert.timestamp) < self.alert_cooldown_seconds:
                    return  # Skip duplicate alert within cooldown period
            
            # Create new alert
            alert = Alert(
                alert_id=alert_id,
                metric_name=metric_name,
                severity=severity,
                message=message,
                timestamp=time.time(),
                client_id=client_id
            )
            
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.warning(f"Alert handler failed: {e}")
            
            logger.warning(f"ALERT [{severity.upper()}] {message}")
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def _alert_processor(self):
        """Background task to process and manage alerts"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                resolved_alerts = []
                
                # Auto-resolve old alerts
                for alert_id, alert in self.alerts.items():
                    if current_time - alert.timestamp > 3600:  # 1 hour
                        alert.resolved = True
                        resolved_alerts.append(alert_id)
                
                # Remove resolved alerts
                for alert_id in resolved_alerts:
                    del self.alerts[alert_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert processing failed: {e}")
                await asyncio.sleep(60)
    
    async def _database_persister(self):
        """Background task to persist metrics to database"""
        while self.is_monitoring:
            try:
                await self._persist_metrics()
                await asyncio.sleep(60)  # Persist every minute
                
            except Exception as e:
                logger.error(f"Database persistence failed: {e}")
                await asyncio.sleep(60)
    
    async def _persist_metrics(self):
        """Persist current metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Persist system metrics
            if self.system_metrics:
                latest_system_metric = self.system_metrics[-1]
                cursor.execute(
                    'INSERT INTO system_metrics (timestamp, metrics_json) VALUES (?, ?)',
                    (latest_system_metric.timestamp, json.dumps(asdict(latest_system_metric)))
                )
            
            # Persist client metrics
            for client_id, client_metrics in self.client_metrics.items():
                if client_metrics.training_loss:
                    latest_loss = client_metrics.training_loss[-1]
                    cursor.execute(
                        'INSERT INTO client_metrics (client_id, timestamp, metric_type, metric_value) VALUES (?, ?, ?, ?)',
                        (client_id, latest_loss.timestamp, 'training_loss', latest_loss.value)
                    )
                
                if client_metrics.training_accuracy:
                    latest_accuracy = client_metrics.training_accuracy[-1]
                    cursor.execute(
                        'INSERT INTO client_metrics (client_id, timestamp, metric_type, metric_value) VALUES (?, ?, ?, ?)',
                        (client_id, latest_accuracy.timestamp, 'training_accuracy', latest_accuracy.value)
                    )
            
            # Persist new alerts
            for alert in self.alerts.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, metric_name, severity, message, timestamp, client_id, acknowledged, resolved) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.metric_name, alert.severity, alert.message,
                    alert.timestamp, alert.client_id, alert.acknowledged, alert.resolved
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Metrics persistence failed: {e}")
    
    async def _metrics_cleaner(self):
        """Background task to clean old metrics"""
        while self.is_monitoring:
            try:
                await self._clean_old_metrics()
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                logger.error(f"Metrics cleaning failed: {e}")
                await asyncio.sleep(3600)
    
    async def _clean_old_metrics(self):
        """Clean old metrics from database"""
        try:
            cutoff_time = time.time() - (self.metrics_retention_days * 24 * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old metrics
            cursor.execute('DELETE FROM client_metrics WHERE timestamp < ?', (cutoff_time,))
            cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_time,))
            cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = 1', (cutoff_time,))
            
            conn.commit()
            conn.close()
            
            logger.info("Cleaned old metrics from database")
            
        except Exception as e:
            logger.error(f"Metrics cleaning failed: {e}")
    
    def get_client_status(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current client status and metrics"""
        if client_id:
            if client_id not in self.client_metrics:
                return {"error": f"Client {client_id} not found"}
            
            client = self.client_metrics[client_id]
            return {
                "client_id": client_id,
                "is_active": client.is_active,
                "last_seen": client.last_seen,
                "privacy_budget_remaining": client.privacy_budget_remaining,
                "recent_loss": client.training_loss[-1].value if client.training_loss else None,
                "recent_accuracy": client.training_accuracy[-1].value if client.training_accuracy else None,
                "participation_rate": client.participation_rate
            }
        else:
            return {
                "total_clients": len(self.client_metrics),
                "active_clients": sum(1 for c in self.client_metrics.values() if c.is_active),
                "clients": {cid: self.get_client_status(cid) for cid in self.client_metrics.keys()}
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest = self.system_metrics[-1]
        return {
            "timestamp": latest.timestamp,
            "rounds_completed": latest.rounds_completed,
            "clients_participated": latest.clients_participated,
            "server_cpu_usage": latest.server_cpu_usage,
            "server_memory_usage": latest.server_memory_usage,
            "global_loss": latest.global_loss,
            "model_quality_score": latest.model_quality_score,
            "total_active_alerts": len([a for a in self.alerts.values() if not a.resolved])
        }
    
    def get_alerts(self, severity: Optional[str] = None, resolved: Optional[bool] = None) -> List[Dict]:
        """Get current alerts"""
        alerts = list(self.alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return [asdict(alert) for alert in alerts]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """Register alert handler"""
        self.alert_handlers.append(handler)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        try:
            # System overview
            system_status = self.get_system_status()
            
            # Client summary
            client_summary = {
                "total_clients": len(self.client_metrics),
                "active_clients": sum(1 for c in self.client_metrics.values() if c.is_active),
                "avg_privacy_budget": np.mean([
                    c.privacy_budget_remaining for c in self.client_metrics.values()
                ]) if self.client_metrics else 0
            }
            
            # Training progress
            training_progress = []
            if self.system_metrics:
                recent_metrics = list(self.system_metrics)[-20:]  # Last 20 data points
                training_progress = [
                    {
                        "round": int(m.rounds_completed),
                        "loss": m.global_loss,
                        "participants": m.clients_participated,
                        "timestamp": m.timestamp
                    }
                    for m in recent_metrics
                ]
            
            # Resource usage
            resource_usage = []
            if self.system_metrics:
                recent_metrics = list(self.system_metrics)[-10:]
                resource_usage = [
                    {
                        "timestamp": m.timestamp,
                        "cpu": m.server_cpu_usage,
                        "memory": m.server_memory_usage,
                        "disk": m.server_disk_usage
                    }
                    for m in recent_metrics
                ]
            
            # Active alerts
            active_alerts = self.get_alerts(resolved=False)
            
            return {
                "system_status": system_status,
                "client_summary": client_summary,
                "training_progress": training_progress,
                "resource_usage": resource_usage,
                "active_alerts": active_alerts,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                "generated_at": time.time(),
                "time_period": f"Last {self.metrics_retention_days} days",
                "summary": {},
                "client_analysis": {},
                "system_analysis": {},
                "recommendations": []
            }
            
            # System summary
            if self.system_metrics:
                metrics_list = list(self.system_metrics)
                report["summary"] = {
                    "total_rounds": max(m.rounds_completed for m in metrics_list),
                    "avg_clients_per_round": np.mean([m.clients_participated for m in metrics_list]),
                    "avg_aggregation_time": np.mean([m.aggregation_time for m in metrics_list if m.aggregation_time > 0]),
                    "avg_system_cpu": np.mean([m.server_cpu_usage for m in metrics_list]),
                    "avg_system_memory": np.mean([m.server_memory_usage for m in metrics_list])
                }
            
            # Client analysis
            for client_id, client in self.client_metrics.items():
                if client.training_loss and len(client.training_loss) >= 2:
                    losses = [m.value for m in client.training_loss]
                    report["client_analysis"][client_id] = {
                        "avg_loss": np.mean(losses),
                        "loss_trend": (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0,
                        "participation_rate": client.participation_rate,
                        "privacy_budget_used": client.privacy_budget_used
                    }
            
            # Generate recommendations
            report["recommendations"] = self._generate_recommendations()
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            # System resource recommendations
            if self.system_metrics:
                recent_metrics = list(self.system_metrics)[-10:]
                avg_cpu = np.mean([m.server_cpu_usage for m in recent_metrics])
                avg_memory = np.mean([m.server_memory_usage for m in recent_metrics])
                
                if avg_cpu > 80:
                    recommendations.append("Consider scaling server resources - high CPU usage detected")
                
                if avg_memory > 85:
                    recommendations.append("Memory usage is high - consider optimizing model size or increasing RAM")
            
            # Client participation recommendations
            active_rate = sum(1 for c in self.client_metrics.values() if c.is_active) / len(self.client_metrics) if self.client_metrics else 1
            
            if active_rate < 0.7:
                recommendations.append("Low client participation rate - investigate connectivity issues")
            
            # Privacy budget recommendations
            low_budget_clients = sum(
                1 for c in self.client_metrics.values()
                if c.privacy_budget_remaining < 1.0
            )
            
            if low_budget_clients > 0:
                recommendations.append(f"{low_budget_clients} clients have low privacy budget - consider budget reallocation")
            
            # Training performance recommendations
            stagnant_clients = []
            for client_id, client in self.client_metrics.items():
                if client.training_loss and len(client.training_loss) >= 5:
                    recent_losses = [m.value for m in list(client.training_loss)[-5:]]
                    if np.var(recent_losses) < 1e-6:
                        stagnant_clients.append(client_id)
            
            if stagnant_clients:
                recommendations.append(f"Clients showing training stagnation: {', '.join(stagnant_clients[:3])}...")
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
        
        return recommendations