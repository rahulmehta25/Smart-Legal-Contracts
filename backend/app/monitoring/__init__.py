"""
Production-grade monitoring and observability system for legal document processing platform.

This module provides comprehensive monitoring capabilities including:
- Custom metrics collection with Prometheus
- Health check endpoints
- Performance profiling
- Distributed tracing with OpenTelemetry
- Real-time alerting
- SLA tracking
"""

from .metrics import MetricsCollector, PrometheusMetrics
from .health import HealthChecker, HealthStatus
from .profiling import PerformanceProfiler, ProfileResult
from .tracing import TracingService, SpanContext

__all__ = [
    'MetricsCollector',
    'PrometheusMetrics',
    'HealthChecker',
    'HealthStatus',
    'PerformanceProfiler',
    'ProfileResult',
    'TracingService',
    'SpanContext'
]