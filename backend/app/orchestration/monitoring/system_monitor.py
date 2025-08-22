"""
System Monitor Implementation

Comprehensive monitoring system that tracks health, performance, 
and usage metrics across all services and components.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import psutil
import aiohttp

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Represents a health check configuration"""
    name: str
    endpoint: str
    interval: int = 30
    timeout: int = 5
    expected_status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    custom_checker: Optional[callable] = None


@dataclass
class MetricData:
    """Represents a metric data point"""
    metric_name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Represents a system alert"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class SystemMonitor:
    """
    Comprehensive system monitoring for all services and infrastructure
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_health: Dict[str, HealthStatus] = {}
        self.metrics_buffer: List[MetricData] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Component references (will be injected)
        self.service_mesh = None
        self.api_gateway = None
        self.event_bus = None
        self.saga_orchestrator = None
        self.service_discovery = None
        
        # Configuration
        self.metrics_retention_days = self.config.get('metrics_retention_days', 7)
        self.alert_cooldown_minutes = self.config.get('alert_cooldown_minutes', 5)
        self.health_check_concurrency = self.config.get('health_check_concurrency', 10)
        
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # System metrics tracking
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_recv": 0},
            "active_connections": 0
        }
    
    async def start(self):
        """Start the system monitor"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting System Monitor")
        
        # Register default health checks
        await self._register_default_health_checks()
        
        # Start monitoring tasks
        health_task = asyncio.create_task(self._health_monitoring_loop())
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        system_task = asyncio.create_task(self._system_metrics_loop())
        alert_task = asyncio.create_task(self._alert_processing_loop())
        
        self._tasks.update([health_task, metrics_task, system_task, alert_task])
        
        logger.info("System Monitor started successfully")
    
    async def stop(self):
        """Stop the system monitor"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping System Monitor")
        
        # Cancel all monitoring tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("System Monitor stopped")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.service_health[health_check.name] = HealthStatus.HEALTHY
        logger.info(f"Registered health check: {health_check.name}")
    
    def record_metric(
        self, 
        metric_name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """Record a metric data point"""
        metric = MetricData(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        
        self.metrics_buffer.append(metric)
        
        # Keep buffer size manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000:]
    
    async def trigger_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Trigger a new alert"""
        alert_id = f"{source}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {title} ({severity.value})")
        
        # Send alert through event bus if available
        if self.event_bus:
            await self.event_bus.publish({
                "type": "system.alert.triggered",
                "source": "system-monitor",
                "data": {
                    "alert_id": alert_id,
                    "title": title,
                    "severity": severity.value,
                    "source": source
                }
            })
        
        return alert_id
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title}")
            
            # Send resolution event
            if self.event_bus:
                await self.event_bus.publish({
                    "type": "system.alert.resolved",
                    "source": "system-monitor",
                    "data": {
                        "alert_id": alert_id,
                        "resolved_at": alert.resolved_at.isoformat()
                    }
                })
            
            return True
        
        return False
    
    async def _health_monitoring_loop(self):
        """Background loop for health monitoring"""
        while self._running:
            try:
                # Run health checks with concurrency limit
                semaphore = asyncio.Semaphore(self.health_check_concurrency)
                health_tasks = []
                
                for name, check in self.health_checks.items():
                    task = asyncio.create_task(
                        self._run_health_check(name, check, semaphore)
                    )
                    health_tasks.append(task)
                
                if health_tasks:
                    await asyncio.gather(*health_tasks, return_exceptions=True)
                
                # Check overall system health
                await self._assess_system_health()
                
                await asyncio.sleep(30)  # Run health checks every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_health_check(self, name: str, check: HealthCheck, semaphore: asyncio.Semaphore):
        """Run a single health check"""
        async with semaphore:
            try:
                if check.custom_checker:
                    # Custom health checker
                    result = await check.custom_checker()
                    is_healthy = result.get("healthy", False)
                else:
                    # HTTP health check
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            check.endpoint,
                            headers=check.headers,
                            timeout=check.timeout
                        ) as response:
                            is_healthy = response.status == check.expected_status
                
                # Update health status
                old_status = self.service_health.get(name, HealthStatus.HEALTHY)
                
                if is_healthy:
                    new_status = HealthStatus.HEALTHY
                else:
                    # Determine degradation level based on consecutive failures
                    new_status = HealthStatus.UNHEALTHY
                
                self.service_health[name] = new_status
                
                # Record health metric
                self.record_metric(
                    f"service_health_{name}",
                    1.0 if is_healthy else 0.0,
                    {"service": name, "status": new_status.value}
                )
                
                # Trigger alert on status change
                if old_status != new_status and new_status != HealthStatus.HEALTHY:
                    await self.trigger_alert(
                        f"Service Health Alert: {name}",
                        f"Service {name} status changed from {old_status.value} to {new_status.value}",
                        AlertSeverity.WARNING if new_status == HealthStatus.DEGRADED else AlertSeverity.ERROR,
                        f"health_check_{name}",
                        {"service": name, "endpoint": check.endpoint}
                    )
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.service_health[name] = HealthStatus.UNHEALTHY
                
                # Record failure metric
                self.record_metric(
                    f"service_health_{name}",
                    0.0,
                    {"service": name, "status": "error", "error": str(e)}
                )
    
    async def _metrics_collection_loop(self):
        """Background loop for collecting metrics from all components"""
        while self._running:
            try:
                # Collect metrics from all system components
                await self._collect_component_metrics()
                
                # Persist metrics (in a real system, send to time series DB)
                await self._persist_metrics()
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)
    
    async def _system_metrics_loop(self):
        """Background loop for collecting system resource metrics"""
        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_metrics["cpu_usage"] = cpu_percent
                self.record_metric("system_cpu_usage_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.system_metrics["memory_usage"] = memory_percent
                self.record_metric("system_memory_usage_percent", memory_percent)
                self.record_metric("system_memory_available_bytes", memory.available)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.system_metrics["disk_usage"] = disk_percent
                self.record_metric("system_disk_usage_percent", disk_percent)
                self.record_metric("system_disk_free_bytes", disk.free)
                
                # Network I/O
                network = psutil.net_io_counters()
                self.system_metrics["network_io"] = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                }
                self.record_metric("system_network_bytes_sent", network.bytes_sent)
                self.record_metric("system_network_bytes_recv", network.bytes_recv)
                
                # Check for resource alerts
                await self._check_resource_alerts(cpu_percent, memory_percent, disk_percent)
                
                await asyncio.sleep(30)  # Collect system metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processing_loop(self):
        """Background loop for processing and managing alerts"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Auto-resolve stale alerts
                stale_threshold = timedelta(hours=24)
                stale_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if current_time - alert.timestamp > stale_threshold
                ]
                
                for alert_id in stale_alerts:
                    await self.resolve_alert(alert_id)
                    logger.info(f"Auto-resolved stale alert: {alert_id}")
                
                # Clean up old alert history
                history_threshold = timedelta(days=30)
                self.alert_history = [
                    alert for alert in self.alert_history
                    if current_time - alert.timestamp < history_threshold
                ]
                
                await asyncio.sleep(300)  # Process alerts every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_component_metrics(self):
        """Collect metrics from all system components"""
        # Service mesh metrics
        if self.service_mesh:
            try:
                topology = self.service_mesh.get_service_topology()
                self.record_metric("service_mesh_total_services", len(topology.get("services", {})))
                self.record_metric("service_mesh_healthy_endpoints", 
                                 topology.get("health_summary", {}).get("healthy_endpoints", 0))
                self.record_metric("service_mesh_total_endpoints", 
                                 topology.get("health_summary", {}).get("total_endpoints", 0))
            except Exception as e:
                logger.warning(f"Failed to collect service mesh metrics: {e}")
        
        # API Gateway metrics
        if self.api_gateway:
            try:
                gateway_status = self.api_gateway.get_gateway_status()
                stats = gateway_status.get("gateway_stats", {})
                
                self.record_metric("api_gateway_total_requests", stats.get("total_requests", 0))
                self.record_metric("api_gateway_successful_requests", stats.get("successful_requests", 0))
                self.record_metric("api_gateway_failed_requests", stats.get("failed_requests", 0))
                self.record_metric("api_gateway_average_response_time", stats.get("average_response_time", 0))
                self.record_metric("api_gateway_active_connections", stats.get("active_connections", 0))
            except Exception as e:
                logger.warning(f"Failed to collect API gateway metrics: {e}")
        
        # Event bus metrics  
        if self.event_bus:
            try:
                event_stats = self.event_bus.get_stats()
                self.record_metric("event_bus_events_published", event_stats.get("events_published", 0))
                self.record_metric("event_bus_events_processed", event_stats.get("events_processed", 0))
                self.record_metric("event_bus_events_failed", event_stats.get("events_failed", 0))
                self.record_metric("event_bus_dead_letter_size", event_stats.get("dead_letter_size", 0))
            except Exception as e:
                logger.warning(f"Failed to collect event bus metrics: {e}")
        
        # Saga orchestrator metrics
        if self.saga_orchestrator:
            try:
                saga_stats = self.saga_orchestrator.get_orchestrator_stats()
                self.record_metric("saga_orchestrator_sagas_running", saga_stats.get("running_sagas", 0))
                self.record_metric("saga_orchestrator_sagas_completed", saga_stats.get("sagas_completed", 0))
                self.record_metric("saga_orchestrator_sagas_failed", saga_stats.get("sagas_failed", 0))
            except Exception as e:
                logger.warning(f"Failed to collect saga orchestrator metrics: {e}")
        
        # Service discovery metrics
        if self.service_discovery:
            try:
                discovery_status = self.service_discovery.get_discovery_status()
                self.record_metric("service_discovery_total_services", discovery_status.get("total_services", 0))
                
                # Status breakdown
                status_counts = discovery_status.get("status_counts", {})
                for status, count in status_counts.items():
                    self.record_metric(f"service_discovery_services_{status}", count)
            except Exception as e:
                logger.warning(f"Failed to collect service discovery metrics: {e}")
    
    async def _persist_metrics(self):
        """Persist collected metrics to storage"""
        if not self.metrics_buffer:
            return
        
        # In a real system, this would send metrics to a time series database
        # like Prometheus, InfluxDB, or CloudWatch
        
        # For now, we'll just log a summary
        recent_metrics = self.metrics_buffer[-100:]  # Last 100 metrics
        
        metrics_summary = {}
        for metric in recent_metrics:
            key = f"{metric.metric_name}_{metric.unit}".strip("_")
            if key not in metrics_summary:
                metrics_summary[key] = {"count": 0, "sum": 0, "last_value": 0}
            
            metrics_summary[key]["count"] += 1
            metrics_summary[key]["sum"] += metric.value
            metrics_summary[key]["last_value"] = metric.value
        
        logger.debug(f"Metrics summary: {json.dumps(metrics_summary, indent=2)}")
    
    async def _check_resource_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check system resources and trigger alerts if thresholds exceeded"""
        # CPU alerts
        if cpu_percent > 90:
            await self.trigger_alert(
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                AlertSeverity.CRITICAL,
                "system_resources",
                {"cpu_percent": cpu_percent}
            )
        elif cpu_percent > 80:
            await self.trigger_alert(
                "Elevated CPU Usage", 
                f"CPU usage is {cpu_percent:.1f}%",
                AlertSeverity.WARNING,
                "system_resources",
                {"cpu_percent": cpu_percent}
            )
        
        # Memory alerts
        if memory_percent > 95:
            await self.trigger_alert(
                "Critical Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                AlertSeverity.CRITICAL,
                "system_resources",
                {"memory_percent": memory_percent}
            )
        elif memory_percent > 85:
            await self.trigger_alert(
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%", 
                AlertSeverity.WARNING,
                "system_resources",
                {"memory_percent": memory_percent}
            )
        
        # Disk alerts
        if disk_percent > 95:
            await self.trigger_alert(
                "Critical Disk Usage",
                f"Disk usage is {disk_percent:.1f}%",
                AlertSeverity.CRITICAL,
                "system_resources",
                {"disk_percent": disk_percent}
            )
        elif disk_percent > 85:
            await self.trigger_alert(
                "High Disk Usage",
                f"Disk usage is {disk_percent:.1f}%",
                AlertSeverity.WARNING,
                "system_resources", 
                {"disk_percent": disk_percent}
            )
    
    async def _assess_system_health(self):
        """Assess overall system health based on component health"""
        if not self.service_health:
            return
        
        healthy_count = len([s for s in self.service_health.values() if s == HealthStatus.HEALTHY])
        total_count = len(self.service_health)
        health_percentage = (healthy_count / total_count) * 100
        
        self.record_metric("system_overall_health_percentage", health_percentage)
        
        # Trigger system-wide alerts based on overall health
        if health_percentage < 50:
            await self.trigger_alert(
                "System Health Critical",
                f"Only {health_percentage:.1f}% of services are healthy",
                AlertSeverity.CRITICAL,
                "system_health",
                {"healthy_services": healthy_count, "total_services": total_count}
            )
        elif health_percentage < 80:
            await self.trigger_alert(
                "System Health Degraded", 
                f"{health_percentage:.1f}% of services are healthy",
                AlertSeverity.WARNING,
                "system_health",
                {"healthy_services": healthy_count, "total_services": total_count}
            )
    
    async def _register_default_health_checks(self):
        """Register default health checks for core services"""
        default_checks = [
            HealthCheck("document-service", "http://localhost:8001/health"),
            HealthCheck("analysis-service", "http://localhost:8002/health"),
            HealthCheck("ml-service", "http://localhost:8003/health"),
            HealthCheck("legal-service", "http://localhost:8004/health"),
            HealthCheck("blockchain-service", "http://localhost:8005/health"),
            HealthCheck("user-service", "http://localhost:8006/health"),
            HealthCheck("payment-service", "http://localhost:8007/health"),
            HealthCheck("notification-service", "http://localhost:8008/health"),
            
            # New feature services
            HealthCheck("voice-interface", "http://localhost:8009/health"),
            HealthCheck("document-comparison", "http://localhost:8010/health"),
            HealthCheck("whitelabel-service", "http://localhost:8011/health"),
            HealthCheck("compliance-automation", "http://localhost:8012/health"),
            HealthCheck("visualization-engine", "http://localhost:8013/health"),
        ]
        
        for check in default_checks:
            self.register_health_check(check)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "overall_health": {
                "healthy_services": len([s for s in self.service_health.values() if s == HealthStatus.HEALTHY]),
                "total_services": len(self.service_health),
                "health_percentage": (len([s for s in self.service_health.values() if s == HealthStatus.HEALTHY]) / len(self.service_health) * 100) if self.service_health else 0
            },
            "service_health": {name: status.value for name, status in self.service_health.items()},
            "system_resources": self.system_metrics,
            "active_alerts": len(self.active_alerts),
            "total_metrics": len(self.metrics_buffer),
            "alerts_by_severity": {
                severity.value: len([a for a in self.active_alerts.values() if a.severity == severity])
                for severity in AlertSeverity
            }
        }
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics for a specific metric name"""
        matching_metrics = [
            {
                "timestamp": m.timestamp.isoformat(),
                "value": m.value,
                "labels": m.labels,
                "unit": m.unit
            }
            for m in self.metrics_buffer[-limit:]
            if m.metric_name == metric_name
        ]
        
        return matching_metrics[-limit:]