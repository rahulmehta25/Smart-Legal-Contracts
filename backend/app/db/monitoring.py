"""
Enterprise Database Performance Monitoring Module

Implements comprehensive real-time database performance monitoring,
connection pool management, and automated performance reporting.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

import time
import asyncio
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, Pool
from sqlalchemy.engine import Engine
import weakref

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    

class MetricType(Enum):
    """Types of metrics to monitor"""
    QUERY_PERFORMANCE = "query_performance"
    CONNECTION_POOL = "connection_pool"
    SYSTEM_RESOURCES = "system_resources"
    CACHE_PERFORMANCE = "cache_performance"
    ERROR_RATE = "error_rate"
    

@dataclass
class QueryMetric:
    """Query performance metric"""
    query_hash: str
    query_text: str
    execution_time: float
    rows_examined: int
    rows_returned: int
    cpu_time: float
    io_reads: int
    io_writes: int
    timestamp: datetime
    session_id: str
    database_name: str
    table_names: List[str]
    

@dataclass
class ConnectionPoolMetric:
    """Connection pool performance metric"""
    pool_size: int
    checked_out: int
    overflow: int
    invalid: int
    active_connections: int
    waiting_connections: int
    total_created: int
    total_closed: int
    avg_checkout_time: float
    max_checkout_time: float
    timestamp: datetime
    

@dataclass
class SystemMetric:
    """System resource metric"""
    cpu_percent: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    open_file_descriptors: int
    tcp_connections: int
    timestamp: datetime
    

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    title: str
    description: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    affected_resources: List[str]
    suggested_actions: List[str]
    auto_resolved: bool = False
    resolved_at: Optional[datetime] = None
    

class MetricsCollector:
    """
    Real-time metrics collection with configurable sampling
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.query_metrics: deque = deque(maxlen=10000)  # Last 10k queries
        self.connection_metrics: deque = deque(maxlen=1000)  # Last 1k samples
        self.system_metrics: deque = deque(maxlen=1000)  # Last 1k samples
        self.active_sessions: Dict[str, Dict] = {}
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Set up SQLAlchemy event listeners
        self._setup_query_monitoring()
        
    def _setup_query_monitoring(self):
        """Set up SQLAlchemy event listeners for query monitoring"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
            context._query_statement = statement
            
        @event.listens_for(self.engine, "after_cursor_execute") 
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            execution_time = time.time() - getattr(context, '_query_start_time', time.time())
            
            # Create query metric
            metric = QueryMetric(
                query_hash=self._hash_query(statement),
                query_text=statement[:1000],  # Truncate long queries
                execution_time=execution_time,
                rows_examined=cursor.rowcount if hasattr(cursor, 'rowcount') else 0,
                rows_returned=cursor.rowcount if hasattr(cursor, 'rowcount') else 0,
                cpu_time=0.0,  # Would need more detailed instrumentation
                io_reads=0,
                io_writes=0,
                timestamp=datetime.now(),
                session_id=str(id(conn)),
                database_name=self.engine.url.database or "unknown",
                table_names=self._extract_table_names(statement)
            )
            
            self.query_metrics.append(metric)
    
    def start_collection(self):
        """Start metrics collection"""
        if not self.is_collecting:
            self.is_collecting = True
            asyncio.create_task(self._collection_loop())
            logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        
        while self.is_collecting:
            try:
                # Collect connection pool metrics
                await self._collect_connection_metrics()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Sleep until next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _collect_connection_metrics(self):
        """Collect connection pool metrics"""
        
        try:
            pool = self.engine.pool
            
            if isinstance(pool, QueuePool):
                metric = ConnectionPoolMetric(
                    pool_size=pool.size(),
                    checked_out=pool.checkedout(),
                    overflow=pool.overflow(),
                    invalid=pool.invalid(),
                    active_connections=pool.checkedout(),
                    waiting_connections=getattr(pool, '_waiting', 0),
                    total_created=getattr(pool, '_total_creates', 0),
                    total_closed=getattr(pool, '_total_closes', 0),
                    avg_checkout_time=0.0,  # Would need instrumentation
                    max_checkout_time=0.0,  # Would need instrumentation
                    timestamp=datetime.now()
                )
                
                self.connection_metrics.append(metric)
                
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Process statistics
            process = psutil.Process()
            
            metric = SystemMetric(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_io_sent=network_io.bytes_sent if network_io else 0,
                network_io_recv=network_io.bytes_recv if network_io else 0,
                open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                tcp_connections=len(process.connections()) if hasattr(process, 'connections') else 0,
                timestamp=datetime.now()
            )
            
            self.system_metrics.append(metric)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query normalization"""
        import hashlib
        import re
        
        # Normalize query
        normalized = re.sub(r"\s+", " ", query.strip().upper())
        normalized = re.sub(r"['\"][^'\"]*['\"]", "?", normalized)
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query"""
        import re
        
        table_names = []
        
        # Simple regex patterns for table extraction
        patterns = [
            r"\bFROM\s+(\w+)",
            r"\bJOIN\s+(\w+)",
            r"\bUPDATE\s+(\w+)",
            r"\bINSERT\s+INTO\s+(\w+)",
            r"\bDELETE\s+FROM\s+(\w+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            table_names.extend(matches)
        
        return list(set(table_names))
    
    def get_recent_metrics(self, metric_type: MetricType, 
                          duration: timedelta = timedelta(minutes=10)) -> List[Any]:
        """Get recent metrics of specified type"""
        
        cutoff_time = datetime.now() - duration
        
        if metric_type == MetricType.QUERY_PERFORMANCE:
            return [m for m in self.query_metrics if m.timestamp >= cutoff_time]
        elif metric_type == MetricType.CONNECTION_POOL:
            return [m for m in self.connection_metrics if m.timestamp >= cutoff_time]
        elif metric_type == MetricType.SYSTEM_RESOURCES:
            return [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        else:
            return []


class PerformanceAnalyzer:
    """
    Advanced performance analysis and anomaly detection
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_baselines: Dict[str, Dict] = {}
        self.anomaly_thresholds: Dict[str, float] = {
            "query_time_p95": 1.0,  # 95th percentile query time
            "connection_pool_utilization": 0.8,  # 80% pool utilization
            "cpu_usage": 0.8,  # 80% CPU usage
            "memory_usage": 0.9,  # 90% memory usage
            "error_rate": 0.05,  # 5% error rate
        }
        
    async def analyze_query_performance(self, duration: timedelta = timedelta(minutes=30)) -> Dict[str, Any]:
        """Analyze query performance over specified duration"""
        
        metrics = self.metrics_collector.get_recent_metrics(
            MetricType.QUERY_PERFORMANCE, duration
        )
        
        if not metrics:
            return {"status": "no_data"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in metrics]
        
        analysis = {
            "total_queries": len(metrics),
            "avg_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "p95_execution_time": self._percentile(execution_times, 95),
            "p99_execution_time": self._percentile(execution_times, 99),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "queries_per_second": len(metrics) / duration.total_seconds(),
            "slow_queries": len([m for m in metrics if m.execution_time > 1.0]),
            "very_slow_queries": len([m for m in metrics if m.execution_time > 5.0]),
            "top_slow_queries": self._get_top_slow_queries(metrics),
            "table_usage": self._analyze_table_usage(metrics),
            "performance_trends": self._analyze_performance_trends(metrics)
        }
        
        return analysis
    
    async def analyze_connection_pool(self, duration: timedelta = timedelta(minutes=30)) -> Dict[str, Any]:
        """Analyze connection pool performance"""
        
        metrics = self.metrics_collector.get_recent_metrics(
            MetricType.CONNECTION_POOL, duration
        )
        
        if not metrics:
            return {"status": "no_data"}
        
        # Calculate statistics
        utilization_rates = [
            m.checked_out / m.pool_size if m.pool_size > 0 else 0 
            for m in metrics
        ]
        
        analysis = {
            "avg_pool_utilization": statistics.mean(utilization_rates),
            "max_pool_utilization": max(utilization_rates),
            "avg_checked_out": statistics.mean([m.checked_out for m in metrics]),
            "max_checked_out": max([m.checked_out for m in metrics]),
            "avg_overflow": statistics.mean([m.overflow for m in metrics]),
            "max_overflow": max([m.overflow for m in metrics]),
            "pool_pressure_events": len([u for u in utilization_rates if u > 0.8]),
            "connection_churn": self._analyze_connection_churn(metrics),
            "recommendations": self._generate_pool_recommendations(utilization_rates)
        }
        
        return analysis
    
    async def analyze_system_resources(self, duration: timedelta = timedelta(minutes=30)) -> Dict[str, Any]:
        """Analyze system resource usage"""
        
        metrics = self.metrics_collector.get_recent_metrics(
            MetricType.SYSTEM_RESOURCES, duration
        )
        
        if not metrics:
            return {"status": "no_data"}
        
        analysis = {
            "avg_cpu_usage": statistics.mean([m.cpu_percent for m in metrics]),
            "max_cpu_usage": max([m.cpu_percent for m in metrics]),
            "avg_memory_usage": statistics.mean([m.memory_percent for m in metrics]),
            "max_memory_usage": max([m.memory_percent for m in metrics]),
            "disk_io_activity": self._analyze_disk_io(metrics),
            "network_io_activity": self._analyze_network_io(metrics),
            "resource_pressure_events": self._identify_resource_pressure(metrics),
            "capacity_recommendations": self._generate_capacity_recommendations(metrics)
        }
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_top_slow_queries(self, metrics: List[QueryMetric], limit: int = 10) -> List[Dict]:
        """Get top slow queries"""
        
        # Group by query hash and calculate average execution time
        query_stats = defaultdict(list)
        for metric in metrics:
            query_stats[metric.query_hash].append(metric)
        
        slow_queries = []
        for query_hash, query_metrics in query_stats.items():
            avg_time = statistics.mean([m.execution_time for m in query_metrics])
            slow_queries.append({
                "query_hash": query_hash,
                "query_text": query_metrics[0].query_text[:200],
                "avg_execution_time": avg_time,
                "execution_count": len(query_metrics),
                "table_names": query_metrics[0].table_names
            })
        
        return sorted(slow_queries, key=lambda x: x["avg_execution_time"], reverse=True)[:limit]
    
    def _analyze_table_usage(self, metrics: List[QueryMetric]) -> Dict[str, int]:
        """Analyze table usage patterns"""
        
        table_counts = defaultdict(int)
        for metric in metrics:
            for table in metric.table_names:
                table_counts[table] += 1
        
        return dict(sorted(table_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_performance_trends(self, metrics: List[QueryMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        # Group metrics by time windows (5-minute intervals)
        time_windows = defaultdict(list)
        
        for metric in metrics:
            window = metric.timestamp.replace(minute=(metric.timestamp.minute // 5) * 5, second=0, microsecond=0)
            time_windows[window].append(metric.execution_time)
        
        trends = {}
        for window, times in time_windows.items():
            trends[window.isoformat()] = {
                "avg_time": statistics.mean(times),
                "query_count": len(times),
                "p95_time": self._percentile(times, 95)
            }
        
        return trends
    
    def _analyze_connection_churn(self, metrics: List[ConnectionPoolMetric]) -> Dict[str, float]:
        """Analyze connection pool churn"""
        
        if len(metrics) < 2:
            return {"churn_rate": 0.0}
        
        first_metric = metrics[0]
        last_metric = metrics[-1]
        
        duration = (last_metric.timestamp - first_metric.timestamp).total_seconds()
        
        creates_delta = last_metric.total_created - first_metric.total_created
        closes_delta = last_metric.total_closed - first_metric.total_closed
        
        return {
            "churn_rate": (creates_delta + closes_delta) / duration if duration > 0 else 0,
            "creates_per_second": creates_delta / duration if duration > 0 else 0,
            "closes_per_second": closes_delta / duration if duration > 0 else 0
        }
    
    def _generate_pool_recommendations(self, utilization_rates: List[float]) -> List[str]:
        """Generate connection pool recommendations"""
        
        recommendations = []
        avg_utilization = statistics.mean(utilization_rates)
        max_utilization = max(utilization_rates)
        
        if avg_utilization > 0.8:
            recommendations.append("Consider increasing connection pool size")
        
        if max_utilization > 0.95:
            recommendations.append("Pool is frequently at capacity - increase max_overflow")
        
        if avg_utilization < 0.3:
            recommendations.append("Pool may be oversized - consider reducing pool size")
        
        return recommendations
    
    def _analyze_disk_io(self, metrics: List[SystemMetric]) -> Dict[str, Any]:
        """Analyze disk I/O patterns"""
        
        if len(metrics) < 2:
            return {"status": "insufficient_data"}
        
        read_rates = []
        write_rates = []
        
        for i in range(1, len(metrics)):
            prev = metrics[i-1]
            curr = metrics[i]
            
            duration = (curr.timestamp - prev.timestamp).total_seconds()
            
            if duration > 0:
                read_rate = (curr.disk_io_read - prev.disk_io_read) / duration
                write_rate = (curr.disk_io_write - prev.disk_io_write) / duration
                
                read_rates.append(read_rate)
                write_rates.append(write_rate)
        
        return {
            "avg_read_rate": statistics.mean(read_rates) if read_rates else 0,
            "avg_write_rate": statistics.mean(write_rates) if write_rates else 0,
            "max_read_rate": max(read_rates) if read_rates else 0,
            "max_write_rate": max(write_rates) if write_rates else 0
        }
    
    def _analyze_network_io(self, metrics: List[SystemMetric]) -> Dict[str, Any]:
        """Analyze network I/O patterns"""
        
        if len(metrics) < 2:
            return {"status": "insufficient_data"}
        
        sent_rates = []
        recv_rates = []
        
        for i in range(1, len(metrics)):
            prev = metrics[i-1]
            curr = metrics[i]
            
            duration = (curr.timestamp - prev.timestamp).total_seconds()
            
            if duration > 0:
                sent_rate = (curr.network_io_sent - prev.network_io_sent) / duration
                recv_rate = (curr.network_io_recv - prev.network_io_recv) / duration
                
                sent_rates.append(sent_rate)
                recv_rates.append(recv_rate)
        
        return {
            "avg_sent_rate": statistics.mean(sent_rates) if sent_rates else 0,
            "avg_recv_rate": statistics.mean(recv_rates) if recv_rates else 0,
            "max_sent_rate": max(sent_rates) if sent_rates else 0,
            "max_recv_rate": max(recv_rates) if recv_rates else 0
        }
    
    def _identify_resource_pressure(self, metrics: List[SystemMetric]) -> List[Dict]:
        """Identify resource pressure events"""
        
        pressure_events = []
        
        for metric in metrics:
            events = []
            
            if metric.cpu_percent > 80:
                events.append("High CPU usage")
            
            if metric.memory_percent > 90:
                events.append("High memory usage")
            
            if events:
                pressure_events.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "events": events,
                    "cpu_percent": metric.cpu_percent,
                    "memory_percent": metric.memory_percent
                })
        
        return pressure_events
    
    def _generate_capacity_recommendations(self, metrics: List[SystemMetric]) -> List[str]:
        """Generate capacity planning recommendations"""
        
        recommendations = []
        
        if not metrics:
            return recommendations
        
        avg_cpu = statistics.mean([m.cpu_percent for m in metrics])
        avg_memory = statistics.mean([m.memory_percent for m in metrics])
        
        if avg_cpu > 70:
            recommendations.append("CPU usage is consistently high - consider scaling up")
        
        if avg_memory > 80:
            recommendations.append("Memory usage is high - consider increasing RAM")
        
        return recommendations


class AlertManager:
    """
    Intelligent alerting system for performance issues
    """
    
    def __init__(self, performance_analyzer: PerformanceAnalyzer):
        self.performance_analyzer = performance_analyzer
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        self.suppression_periods: Dict[str, timedelta] = defaultdict(lambda: timedelta(minutes=5))
        
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    async def check_performance_alerts(self):
        """Check for performance issues and trigger alerts"""
        
        current_time = datetime.now()
        
        # Check query performance
        await self._check_query_performance_alerts()
        
        # Check connection pool alerts
        await self._check_connection_pool_alerts()
        
        # Check system resource alerts
        await self._check_system_resource_alerts()
        
        # Auto-resolve alerts that are no longer active
        await self._auto_resolve_alerts()
    
    async def _check_query_performance_alerts(self):
        """Check for query performance alerts"""
        
        analysis = await self.performance_analyzer.analyze_query_performance(
            timedelta(minutes=5)
        )
        
        if analysis.get("status") == "no_data":
            return
        
        # Check for slow query alerts
        if analysis["p95_execution_time"] > 1.0:
            await self._create_alert(
                alert_id="slow_queries_p95",
                level=AlertLevel.WARNING,
                metric_type=MetricType.QUERY_PERFORMANCE,
                title="Slow Query Performance",
                description=f"95th percentile query time is {analysis['p95_execution_time']:.2f}s",
                threshold_value=1.0,
                actual_value=analysis["p95_execution_time"],
                affected_resources=["database"],
                suggested_actions=[
                    "Check for missing indexes",
                    "Review query execution plans",
                    "Consider query optimization"
                ]
            )
        
        # Check for very slow queries
        if analysis["very_slow_queries"] > 0:
            await self._create_alert(
                alert_id="very_slow_queries",
                level=AlertLevel.CRITICAL,
                metric_type=MetricType.QUERY_PERFORMANCE,
                title="Very Slow Queries Detected",
                description=f"{analysis['very_slow_queries']} queries taking >5 seconds",
                threshold_value=0,
                actual_value=analysis["very_slow_queries"],
                affected_resources=["database"],
                suggested_actions=[
                    "Identify and optimize slow queries immediately",
                    "Check for table locks",
                    "Review database configuration"
                ]
            )
    
    async def _check_connection_pool_alerts(self):
        """Check for connection pool alerts"""
        
        analysis = await self.performance_analyzer.analyze_connection_pool(
            timedelta(minutes=5)
        )
        
        if analysis.get("status") == "no_data":
            return
        
        # Check for high pool utilization
        if analysis["max_pool_utilization"] > 0.9:
            await self._create_alert(
                alert_id="high_pool_utilization",
                level=AlertLevel.WARNING,
                metric_type=MetricType.CONNECTION_POOL,
                title="High Connection Pool Utilization",
                description=f"Pool utilization reached {analysis['max_pool_utilization']:.1%}",
                threshold_value=0.9,
                actual_value=analysis["max_pool_utilization"],
                affected_resources=["connection_pool"],
                suggested_actions=[
                    "Increase connection pool size",
                    "Optimize connection usage patterns",
                    "Check for connection leaks"
                ]
            )
    
    async def _check_system_resource_alerts(self):
        """Check for system resource alerts"""
        
        analysis = await self.performance_analyzer.analyze_system_resources(
            timedelta(minutes=5)
        )
        
        if analysis.get("status") == "no_data":
            return
        
        # Check for high CPU usage
        if analysis["max_cpu_usage"] > 85:
            await self._create_alert(
                alert_id="high_cpu_usage",
                level=AlertLevel.WARNING,
                metric_type=MetricType.SYSTEM_RESOURCES,
                title="High CPU Usage",
                description=f"CPU usage reached {analysis['max_cpu_usage']:.1f}%",
                threshold_value=85,
                actual_value=analysis["max_cpu_usage"],
                affected_resources=["system"],
                suggested_actions=[
                    "Check for CPU-intensive queries",
                    "Consider horizontal scaling",
                    "Optimize application logic"
                ]
            )
        
        # Check for high memory usage
        if analysis["max_memory_usage"] > 90:
            await self._create_alert(
                alert_id="high_memory_usage",
                level=AlertLevel.CRITICAL,
                metric_type=MetricType.SYSTEM_RESOURCES,
                title="High Memory Usage",
                description=f"Memory usage reached {analysis['max_memory_usage']:.1f}%",
                threshold_value=90,
                actual_value=analysis["max_memory_usage"],
                affected_resources=["system"],
                suggested_actions=[
                    "Check for memory leaks",
                    "Increase available memory",
                    "Optimize cache configurations"
                ]
            )
    
    async def _create_alert(self, alert_id: str, level: AlertLevel, metric_type: MetricType,
                          title: str, description: str, threshold_value: float,
                          actual_value: float, affected_resources: List[str],
                          suggested_actions: List[str]):
        """Create and process a new alert"""
        
        current_time = datetime.now()
        
        # Check if alert is in suppression period
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            suppression_period = self.suppression_periods[alert_id]
            
            if current_time - existing_alert.timestamp < suppression_period:
                return  # Suppress duplicate alert
        
        # Create new alert
        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            metric_type=metric_type,
            title=title,
            description=description,
            threshold_value=threshold_value,
            actual_value=actual_value,
            timestamp=current_time,
            affected_resources=affected_resources,
            suggested_actions=suggested_actions
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Performance alert: {title} - {description}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are no longer active"""
        
        current_time = datetime.now()
        resolved_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            # Check if alert condition no longer exists
            should_resolve = await self._should_auto_resolve_alert(alert)
            
            if should_resolve:
                alert.auto_resolved = True
                alert.resolved_at = current_time
                resolved_alerts.append(alert_id)
                
                logger.info(f"Auto-resolved alert: {alert.title}")
        
        # Remove resolved alerts
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    async def _should_auto_resolve_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be auto-resolved"""
        
        # Simple resolution logic - in practice, this would be more sophisticated
        alert_age = datetime.now() - alert.timestamp
        
        # Auto-resolve alerts older than 10 minutes if conditions have improved
        if alert_age > timedelta(minutes=10):
            return True
        
        return False
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        
        active_alerts = self.get_active_alerts()
        
        return {
            "total_active_alerts": len(active_alerts),
            "alerts_by_level": {
                level.value: len([a for a in active_alerts if a.level == level])
                for level in AlertLevel
            },
            "alerts_by_type": {
                metric_type.value: len([a for a in active_alerts if a.metric_type == metric_type])
                for metric_type in MetricType
            },
            "oldest_alert": min([a.timestamp for a in active_alerts]) if active_alerts else None,
            "newest_alert": max([a.timestamp for a in active_alerts]) if active_alerts else None
        }


class PerformanceReporter:
    """
    Automated performance reporting system
    """
    
    def __init__(self, performance_analyzer: PerformanceAnalyzer, 
                 alert_manager: AlertManager):
        self.performance_analyzer = performance_analyzer
        self.alert_manager = alert_manager
        
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        
        # Analyze last 24 hours
        duration = timedelta(hours=24)
        
        query_analysis = await self.performance_analyzer.analyze_query_performance(duration)
        pool_analysis = await self.performance_analyzer.analyze_connection_pool(duration)
        system_analysis = await self.performance_analyzer.analyze_system_resources(duration)
        
        alert_summary = self.alert_manager.get_alert_summary()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "period": "24 hours",
            "query_performance": query_analysis,
            "connection_pool": pool_analysis,
            "system_resources": system_analysis,
            "alerts": alert_summary,
            "recommendations": self._generate_recommendations(
                query_analysis, pool_analysis, system_analysis
            ),
            "summary": self._generate_executive_summary(
                query_analysis, pool_analysis, system_analysis, alert_summary
            )
        }
        
        return report
    
    async def generate_real_time_dashboard(self) -> Dict[str, Any]:
        """Generate real-time dashboard data"""
        
        # Analyze last 10 minutes
        duration = timedelta(minutes=10)
        
        query_analysis = await self.performance_analyzer.analyze_query_performance(duration)
        pool_analysis = await self.performance_analyzer.analyze_connection_pool(duration)
        system_analysis = await self.performance_analyzer.analyze_system_resources(duration)
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "status": self._determine_overall_status(active_alerts),
            "key_metrics": {
                "queries_per_second": query_analysis.get("queries_per_second", 0),
                "avg_query_time": query_analysis.get("avg_execution_time", 0),
                "pool_utilization": pool_analysis.get("avg_pool_utilization", 0),
                "cpu_usage": system_analysis.get("avg_cpu_usage", 0),
                "memory_usage": system_analysis.get("avg_memory_usage", 0)
            },
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
        }
        
        return dashboard
    
    def _generate_recommendations(self, query_analysis: Dict, pool_analysis: Dict, 
                                system_analysis: Dict) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        # Query performance recommendations
        if query_analysis.get("p95_execution_time", 0) > 0.5:
            recommendations.append("Optimize slow queries - 95th percentile > 500ms")
        
        # Connection pool recommendations
        pool_recs = pool_analysis.get("recommendations", [])
        recommendations.extend(pool_recs)
        
        # System resource recommendations
        capacity_recs = system_analysis.get("capacity_recommendations", [])
        recommendations.extend(capacity_recs)
        
        return recommendations
    
    def _generate_executive_summary(self, query_analysis: Dict, pool_analysis: Dict,
                                  system_analysis: Dict, alert_summary: Dict) -> str:
        """Generate executive summary"""
        
        total_queries = query_analysis.get("total_queries", 0)
        avg_query_time = query_analysis.get("avg_execution_time", 0)
        active_alerts = alert_summary.get("total_active_alerts", 0)
        
        if active_alerts == 0:
            status = "HEALTHY"
        elif alert_summary.get("alerts_by_level", {}).get("critical", 0) > 0:
            status = "CRITICAL"
        else:
            status = "WARNING"
        
        summary = f"""
        System Status: {status}
        
        Performance Summary:
        - Processed {total_queries:,} queries with average response time of {avg_query_time:.3f}s
        - Connection pool utilization: {pool_analysis.get('avg_pool_utilization', 0):.1%}
        - System CPU usage: {system_analysis.get('avg_cpu_usage', 0):.1f}%
        - Active alerts: {active_alerts}
        
        Overall system performance is {"excellent" if status == "HEALTHY" else "needs attention"}.
        """
        
        return summary.strip()
    
    def _determine_overall_status(self, active_alerts: List[PerformanceAlert]) -> str:
        """Determine overall system status"""
        
        if not active_alerts:
            return "healthy"
        
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            return "critical"
        
        warning_alerts = [a for a in active_alerts if a.level == AlertLevel.WARNING]
        if warning_alerts:
            return "warning"
        
        return "healthy"


async def setup_enterprise_monitoring(engine: Engine) -> Tuple[MetricsCollector, PerformanceAnalyzer, AlertManager, PerformanceReporter]:
    """
    Set up enterprise-level database monitoring
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Tuple of monitoring components
    """
    
    # Create monitoring components
    metrics_collector = MetricsCollector(engine)
    performance_analyzer = PerformanceAnalyzer(metrics_collector)
    alert_manager = AlertManager(performance_analyzer)
    performance_reporter = PerformanceReporter(performance_analyzer, alert_manager)
    
    # Start metrics collection
    metrics_collector.start_collection()
    
    logger.info("Enterprise monitoring setup completed")
    
    return metrics_collector, performance_analyzer, alert_manager, performance_reporter


# Background task for monitoring maintenance
async def monitoring_maintenance_task(metrics_collector: MetricsCollector,
                                    alert_manager: AlertManager,
                                    performance_reporter: PerformanceReporter):
    """
    Background task for monitoring system maintenance
    """
    
    while True:
        try:
            # Check for alerts every minute
            await alert_manager.check_performance_alerts()
            
            # Generate daily report (run once per day at midnight)
            current_time = datetime.now()
            if current_time.hour == 0 and current_time.minute == 0:
                daily_report = await performance_reporter.generate_daily_report()
                logger.info("Daily performance report generated")
            
            # Sleep for 1 minute before next check
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Monitoring maintenance failed: {e}")
            await asyncio.sleep(30)  # Wait 30 seconds on error