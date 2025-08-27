"""
Custom metrics collection system with Prometheus integration.
Tracks business metrics, performance indicators, and system health.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    push_to_gateway, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily


class MetricType(Enum):
    """Types of metrics supported by the system"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PrometheusMetrics:
    """
    Prometheus metrics collector for comprehensive application monitoring.
    """
    
    def __init__(self, app_name: str = "arbitration_detector", 
                 push_gateway: Optional[str] = None):
        """
        Initialize Prometheus metrics with custom registry.
        
        Args:
            app_name: Application name for metric labeling
            push_gateway: Optional Prometheus push gateway URL
        """
        self.app_name = app_name
        self.push_gateway = push_gateway
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'application_errors_total',
            'Total application errors',
            ['error_type', 'service'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query execution time',
            ['query_type', 'table'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        self.db_connection_pool = Gauge(
            'database_connection_pool_size',
            'Database connection pool metrics',
            ['pool_name', 'state'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hit count',
            ['cache_name'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache miss count',
            ['cache_name'],
            registry=self.registry
        )
        
        self.cache_evictions = Counter(
            'cache_evictions_total',
            'Cache eviction count',
            ['cache_name', 'reason'],
            registry=self.registry
        )
        
        # Business metrics
        self.documents_processed = Counter(
            'documents_processed_total',
            'Total documents processed',
            ['document_type', 'status'],
            registry=self.registry
        )
        
        self.arbitration_clauses_detected = Counter(
            'arbitration_clauses_detected_total',
            'Arbitration clauses detected',
            ['confidence_level', 'document_type'],
            registry=self.registry
        )
        
        self.ai_model_predictions = Histogram(
            'ai_model_prediction_time_seconds',
            'AI model prediction latency',
            ['model_name', 'model_version'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.api_usage = Counter(
            'api_usage_total',
            'API usage by client',
            ['client_id', 'api_key', 'endpoint'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['core'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point', 'type'],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'queue_size',
            'Message queue size',
            ['queue_name'],
            registry=self.registry
        )
        
        self.queue_processing_time = Histogram(
            'queue_message_processing_seconds',
            'Queue message processing time',
            ['queue_name', 'message_type'],
            registry=self.registry
        )
        
        # Feature usage metrics
        self.feature_usage = Counter(
            'feature_usage_total',
            'Feature usage tracking',
            ['feature_name', 'user_segment'],
            registry=self.registry
        )
        
        # SLA metrics
        self.sla_violations = Counter(
            'sla_violations_total',
            'SLA violation count',
            ['sla_type', 'severity'],
            registry=self.registry
        )
        
        self.availability = Gauge(
            'service_availability_percent',
            'Service availability percentage',
            ['service_name'],
            registry=self.registry
        )
    
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics"""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def track_error(self, error_type: str, service: str):
        """Track application errors"""
        self.error_count.labels(
            error_type=error_type,
            service=service
        ).inc()
    
    def track_db_query(self, query_type: str, table: str, duration: float):
        """Track database query metrics"""
        self.db_query_duration.labels(
            query_type=query_type,
            table=table
        ).observe(duration)
    
    def update_db_pool(self, pool_name: str, active: int, idle: int):
        """Update database connection pool metrics"""
        self.db_connection_pool.labels(
            pool_name=pool_name,
            state='active'
        ).set(active)
        
        self.db_connection_pool.labels(
            pool_name=pool_name,
            state='idle'
        ).set(idle)
    
    def track_cache_hit(self, cache_name: str):
        """Track cache hit"""
        self.cache_hits.labels(cache_name=cache_name).inc()
    
    def track_cache_miss(self, cache_name: str):
        """Track cache miss"""
        self.cache_misses.labels(cache_name=cache_name).inc()
    
    def track_cache_eviction(self, cache_name: str, reason: str):
        """Track cache eviction"""
        self.cache_evictions.labels(
            cache_name=cache_name,
            reason=reason
        ).inc()
    
    def track_document_processed(self, doc_type: str, status: str):
        """Track document processing"""
        self.documents_processed.labels(
            document_type=doc_type,
            status=status
        ).inc()
    
    def track_arbitration_detection(self, confidence: str, doc_type: str):
        """Track arbitration clause detection"""
        self.arbitration_clauses_detected.labels(
            confidence_level=confidence,
            document_type=doc_type
        ).inc()
    
    def track_model_prediction(self, model_name: str, version: str, duration: float):
        """Track AI model prediction metrics"""
        self.ai_model_predictions.labels(
            model_name=model_name,
            model_version=version
        ).observe(duration)
    
    def track_api_usage(self, client_id: str, api_key: str, endpoint: str):
        """Track API usage by client"""
        self.api_usage.labels(
            client_id=client_id,
            api_key=api_key[:8] + "...",  # Mask API key
            endpoint=endpoint
        ).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # CPU metrics
        cpu_percents = psutil.cpu_percent(percpu=True)
        for i, percent in enumerate(cpu_percents):
            self.cpu_usage.labels(core=str(i)).set(percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_usage.labels(type='used').set(memory.used)
        self.memory_usage.labels(type='available').set(memory.available)
        self.memory_usage.labels(type='cached').set(memory.cached)
        
        # Disk metrics
        for partition in psutil.disk_partitions():
            usage = psutil.disk_usage(partition.mountpoint)
            self.disk_usage.labels(
                mount_point=partition.mountpoint,
                type='used'
            ).set(usage.used)
            self.disk_usage.labels(
                mount_point=partition.mountpoint,
                type='free'
            ).set(usage.free)
    
    def track_queue_metrics(self, queue_name: str, size: int):
        """Track message queue metrics"""
        self.queue_size.labels(queue_name=queue_name).set(size)
    
    def track_queue_processing(self, queue_name: str, message_type: str, duration: float):
        """Track queue message processing time"""
        self.queue_processing_time.labels(
            queue_name=queue_name,
            message_type=message_type
        ).observe(duration)
    
    def track_feature_usage(self, feature: str, user_segment: str):
        """Track feature usage"""
        self.feature_usage.labels(
            feature_name=feature,
            user_segment=user_segment
        ).inc()
    
    def track_sla_violation(self, sla_type: str, severity: str):
        """Track SLA violations"""
        self.sla_violations.labels(
            sla_type=sla_type,
            severity=severity
        ).inc()
    
    def update_availability(self, service: str, availability_percent: float):
        """Update service availability metric"""
        self.availability.labels(service_name=service).set(availability_percent)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    async def push_metrics(self):
        """Push metrics to Prometheus push gateway"""
        if self.push_gateway:
            try:
                push_to_gateway(
                    self.push_gateway,
                    job=self.app_name,
                    registry=self.registry
                )
            except Exception as e:
                print(f"Failed to push metrics: {e}")


class MetricsCollector:
    """
    Advanced metrics collection with aggregation and analysis capabilities.
    """
    
    def __init__(self, window_size: int = 3600):
        """
        Initialize metrics collector with time window.
        
        Args:
            window_size: Time window in seconds for metric aggregation
        """
        self.window_size = window_size
        self.metrics_store = defaultdict(lambda: deque(maxlen=window_size))
        self.prometheus = PrometheusMetrics()
        self._aggregation_cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
        self._last_cache_update = {}
    
    def record_metric(self, metric_name: str, value: float, 
                     labels: Optional[Dict[str, str]] = None,
                     timestamp: Optional[float] = None):
        """
        Record a metric value with optional labels.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        key = self._generate_key(metric_name, labels)
        self.metrics_store[key].append({
            'value': value,
            'timestamp': timestamp,
            'labels': labels or {}
        })
    
    def _generate_key(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate unique key for metric with labels"""
        if not labels:
            return metric_name
        
        label_str = "_".join(f"{k}:{v}" for k, v in sorted(labels.items()))
        return f"{metric_name}_{label_str}"
    
    def get_metric_stats(self, metric_name: str, 
                        labels: Optional[Dict[str, str]] = None,
                        time_range: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistical summary of a metric.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels to filter by
            time_range: Time range in seconds (defaults to window_size)
        
        Returns:
            Dictionary with statistical summary
        """
        key = self._generate_key(metric_name, labels)
        
        # Check cache
        cache_key = f"{key}_{time_range}"
        if cache_key in self._aggregation_cache:
            if time.time() - self._last_cache_update.get(cache_key, 0) < self._cache_ttl:
                return self._aggregation_cache[cache_key]
        
        # Get metrics within time range
        current_time = time.time()
        if time_range is None:
            time_range = self.window_size
        
        cutoff_time = current_time - time_range
        
        values = []
        for metric in self.metrics_store[key]:
            if metric['timestamp'] >= cutoff_time:
                values.append(metric['value'])
        
        if not values:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
        
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
        # Update cache
        self._aggregation_cache[cache_key] = stats
        self._last_cache_update[cache_key] = current_time
        
        return stats
    
    def get_rate(self, metric_name: str, 
                 labels: Optional[Dict[str, str]] = None,
                 interval: int = 60) -> float:
        """
        Calculate rate of change for a counter metric.
        
        Args:
            metric_name: Name of the counter metric
            labels: Optional labels to filter by
            interval: Time interval in seconds
        
        Returns:
            Rate per second
        """
        key = self._generate_key(metric_name, labels)
        current_time = time.time()
        cutoff_time = current_time - interval
        
        values = []
        for metric in self.metrics_store[key]:
            if metric['timestamp'] >= cutoff_time:
                values.append(metric['value'])
        
        if len(values) < 2:
            return 0.0
        
        # Calculate rate
        time_diff = values[-1]['timestamp'] - values[0]['timestamp']
        value_diff = values[-1]['value'] - values[0]['value']
        
        if time_diff == 0:
            return 0.0
        
        return value_diff / time_diff
    
    def calculate_sli(self, success_metric: str, total_metric: str,
                     labels: Optional[Dict[str, str]] = None) -> float:
        """
        Calculate Service Level Indicator (SLI).
        
        Args:
            success_metric: Name of success metric
            total_metric: Name of total metric
            labels: Optional labels to filter by
        
        Returns:
            SLI percentage
        """
        success_stats = self.get_metric_stats(success_metric, labels)
        total_stats = self.get_metric_stats(total_metric, labels)
        
        if total_stats['count'] == 0:
            return 100.0
        
        return (success_stats['count'] / total_stats['count']) * 100
    
    def detect_anomalies(self, metric_name: str,
                         labels: Optional[Dict[str, str]] = None,
                         threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric values using statistical methods.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels to filter by
            threshold: Z-score threshold for anomaly detection
        
        Returns:
            List of anomalous data points
        """
        key = self._generate_key(metric_name, labels)
        values = [m['value'] for m in self.metrics_store[key]]
        
        if len(values) < 10:
            return []
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for metric in self.metrics_store[key]:
            z_score = abs((metric['value'] - mean) / std)
            if z_score > threshold:
                anomalies.append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value'],
                    'z_score': z_score,
                    'labels': metric['labels']
                })
        
        return anomalies
    
    async def run_metrics_aggregator(self, interval: int = 60):
        """
        Background task to aggregate and push metrics.
        
        Args:
            interval: Aggregation interval in seconds
        """
        while True:
            try:
                # Update system metrics
                self.prometheus.update_system_metrics()
                
                # Push metrics to Prometheus
                await self.prometheus.push_metrics()
                
                # Clear old cache entries
                current_time = time.time()
                keys_to_remove = []
                for key, last_update in self._last_cache_update.items():
                    if current_time - last_update > self._cache_ttl * 2:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._aggregation_cache.pop(key, None)
                    self._last_cache_update.pop(key, None)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error in metrics aggregator: {e}")
                await asyncio.sleep(interval)


class CustomBusinessMetrics:
    """
    Custom business metrics specific to the legal document processing platform.
    """
    
    def __init__(self, collector: MetricsCollector):
        """
        Initialize custom business metrics.
        
        Args:
            collector: MetricsCollector instance
        """
        self.collector = collector
    
    def track_document_processing(self, doc_id: str, doc_type: str,
                                 processing_time: float, success: bool,
                                 arbitration_found: bool, confidence: float):
        """Track document processing metrics"""
        
        # Record processing time
        self.collector.record_metric(
            'document_processing_time',
            processing_time,
            {'type': doc_type, 'status': 'success' if success else 'failed'}
        )
        
        # Track success/failure
        self.collector.record_metric(
            'document_processing_total',
            1,
            {'type': doc_type, 'status': 'success' if success else 'failed'}
        )
        
        # Track arbitration detection
        if arbitration_found:
            self.collector.record_metric(
                'arbitration_detections',
                1,
                {'type': doc_type, 'confidence': self._categorize_confidence(confidence)}
            )
        
        # Update Prometheus metrics
        self.collector.prometheus.track_document_processed(
            doc_type,
            'success' if success else 'failed'
        )
        
        if arbitration_found:
            self.collector.prometheus.track_arbitration_detection(
                self._categorize_confidence(confidence),
                doc_type
            )
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def track_user_activity(self, user_id: str, action: str,
                           feature: str, segment: str):
        """Track user activity and engagement"""
        
        self.collector.record_metric(
            'user_actions',
            1,
            {'action': action, 'feature': feature, 'segment': segment}
        )
        
        self.collector.prometheus.track_feature_usage(feature, segment)
    
    def track_conversion_funnel(self, user_id: str, step: str,
                               funnel_name: str, completed: bool):
        """Track conversion funnel metrics"""
        
        self.collector.record_metric(
            'funnel_steps',
            1,
            {'funnel': funnel_name, 'step': step, 'completed': str(completed)}
        )
    
    def track_revenue_metrics(self, amount: float, currency: str,
                             product: str, customer_segment: str):
        """Track revenue and monetization metrics"""
        
        self.collector.record_metric(
            'revenue',
            amount,
            {'currency': currency, 'product': product, 'segment': customer_segment}
        )
    
    def calculate_business_kpis(self) -> Dict[str, Any]:
        """Calculate key business performance indicators"""
        
        # Document processing KPIs
        doc_success_rate = self.collector.calculate_sli(
            'document_processing_total',
            'document_processing_total',
            {'status': 'success'}
        )
        
        doc_processing_stats = self.collector.get_metric_stats(
            'document_processing_time'
        )
        
        # User engagement KPIs
        user_action_stats = self.collector.get_metric_stats('user_actions')
        
        # Revenue KPIs
        revenue_stats = self.collector.get_metric_stats('revenue')
        
        return {
            'document_processing': {
                'success_rate': doc_success_rate,
                'avg_processing_time': doc_processing_stats['mean'],
                'p95_processing_time': doc_processing_stats['p95']
            },
            'user_engagement': {
                'total_actions': user_action_stats['count'],
                'actions_per_minute': self.collector.get_rate('user_actions')
            },
            'revenue': {
                'total': revenue_stats['count'] * revenue_stats['mean'],
                'average_transaction': revenue_stats['mean']
            }
        }