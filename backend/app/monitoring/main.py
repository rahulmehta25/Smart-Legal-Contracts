"""
Main monitoring integration module that ties together all monitoring components.
Provides FastAPI integration and background monitoring tasks.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import time

from .metrics import MetricsCollector, PrometheusMetrics, CustomBusinessMetrics
from .health import HealthChecker, HealthEndpoint
from .profiling import PerformanceProfiler
from .tracing import TracingService, DistributedTracer, TracingBackend
from .logging import StructuredLogger, LogAggregator
from ..analytics.user_behavior import UserBehaviorAnalytics, UserEvent, EventType


class MonitoringSystem:
    """
    Comprehensive monitoring system integration for FastAPI applications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            window_size=config.get('metrics_window', 3600)
        )
        self.prometheus = PrometheusMetrics(
            app_name=config.get('app_name', 'arbitration_detector'),
            push_gateway=config.get('prometheus_push_gateway')
        )
        
        self.health_checker = HealthChecker(
            db_url=config.get('db_url'),
            redis_url=config.get('redis_url'),
            mongodb_url=config.get('mongodb_url'),
            external_services=config.get('external_services', {})
        )
        
        self.profiler = PerformanceProfiler(
            enable_auto_profiling=config.get('auto_profiling', False)
        )
        
        self.tracer = DistributedTracer(
            service_name=config.get('app_name', 'arbitration_detector'),
            config={
                'backend': config.get('tracing_backend', 'jaeger'),
                'endpoint': config.get('tracing_endpoint'),
                'sample_rate': config.get('tracing_sample_rate', 1.0)
            }
        )
        
        self.logger = StructuredLogger(
            service_name=config.get('app_name', 'arbitration_detector'),
            log_dir=config.get('log_dir', '/var/log/app'),
            enable_sentry=config.get('enable_sentry', False),
            sentry_dsn=config.get('sentry_dsn'),
            log_level=config.get('log_level', 'INFO')
        )
        
        self.user_analytics = UserBehaviorAnalytics(
            session_timeout_minutes=config.get('session_timeout', 30)
        )
        
        self.business_metrics = CustomBusinessMetrics(self.metrics_collector)
        
        # Background tasks
        self.background_tasks = []
    
    def setup_fastapi(self, app: FastAPI):
        """
        Set up monitoring for FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add monitoring middleware
        @app.middleware("http")
        async def monitoring_middleware(request: Request, call_next):
            # Start timing
            start_time = time.time()
            
            # Extract trace context from headers
            trace_context = {}
            if 'x-trace-id' in request.headers:
                trace_context['trace_id'] = request.headers['x-trace-id']
            if 'x-request-id' in request.headers:
                trace_context['request_id'] = request.headers['x-request-id']
            
            # Set logging context
            self.logger.set_context(**trace_context)
            
            # Start span for request
            with self.tracer.tracing_service.tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    'http.method': request.method,
                    'http.url': str(request.url),
                    'http.scheme': request.url.scheme,
                    'http.host': request.url.hostname,
                    'http.target': request.url.path,
                    'user_agent': request.headers.get('user-agent', '')
                }
            ) as span:
                try:
                    # Process request
                    response = await call_next(request)
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track metrics
                    self.prometheus.track_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status=response.status_code,
                        duration=duration
                    )
                    
                    # Log request
                    self.logger.info({
                        'type': 'http_request',
                        'method': request.method,
                        'path': request.url.path,
                        'status': response.status_code,
                        'duration_seconds': duration,
                        'client_ip': request.client.host if request.client else None
                    })
                    
                    # Add trace ID to response headers
                    response.headers['X-Trace-Id'] = trace_context.get('trace_id', '')
                    
                    # Set span status
                    span.set_attribute('http.status_code', response.status_code)
                    
                    return response
                    
                except Exception as e:
                    # Track error
                    self.prometheus.track_error(
                        error_type=type(e).__name__,
                        service='api'
                    )
                    
                    # Log error
                    self.logger.error(
                        f"Request failed: {request.url.path}",
                        exception=e
                    )
                    
                    # Record exception in span
                    span.record_exception(e)
                    
                    raise
        
        # Add health endpoints
        health_endpoint = HealthEndpoint(self.health_checker)
        
        @app.get("/health")
        async def health():
            """Basic health check"""
            return await health_endpoint.health()
        
        @app.get("/health/detailed")
        async def health_detailed():
            """Detailed health check"""
            return await health_endpoint.health_detailed()
        
        @app.get("/health/ready")
        async def health_ready():
            """Kubernetes readiness probe"""
            return await health_endpoint.health_ready()
        
        @app.get("/health/live")
        async def health_live():
            """Kubernetes liveness probe"""
            return await health_endpoint.health_live()
        
        # Add metrics endpoint
        @app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            """Prometheus metrics endpoint"""
            return self.prometheus.export_metrics()
        
        # Add profiling endpoints
        @app.get("/debug/profile/cpu")
        async def profile_cpu():
            """Get CPU profiling data"""
            if not self.config.get('enable_debug_endpoints', False):
                raise HTTPException(status_code=403, detail="Debug endpoints disabled")
            
            self.profiler.start_cpu_profiling()
            await asyncio.sleep(10)  # Profile for 10 seconds
            result = self.profiler.stop_cpu_profiling()
            
            return {
                'profile_type': result.profile_type,
                'duration_seconds': result.duration_seconds,
                'summary': result.summary,
                'recommendations': result.recommendations
            }
        
        @app.get("/debug/profile/memory")
        async def profile_memory():
            """Analyze memory usage"""
            if not self.config.get('enable_debug_endpoints', False):
                raise HTTPException(status_code=403, detail="Debug endpoints disabled")
            
            result = self.profiler.analyze_memory_leaks()
            
            return {
                'profile_type': result.profile_type,
                'summary': result.summary,
                'recommendations': result.recommendations
            }
        
        # Add analytics endpoints
        @app.get("/analytics/user-behavior")
        async def get_user_behavior():
            """Get user behavior analytics"""
            return {
                'segments': self.user_analytics.get_user_segments(),
                'feature_adoption': self.user_analytics.get_feature_adoption(),
                'session_metrics': self.user_analytics.get_session_metrics()
            }
        
        @app.get("/analytics/business-kpis")
        async def get_business_kpis():
            """Get business KPIs"""
            return self.business_metrics.calculate_business_kpis()
        
        @app.get("/analytics/funnel/{funnel_name}")
        async def get_funnel_metrics(funnel_name: str):
            """Get conversion funnel metrics"""
            return self.user_analytics.get_funnel_metrics(funnel_name)
        
        # Add tracing analysis endpoint
        @app.get("/tracing/analysis")
        async def get_trace_analysis(trace_id: Optional[str] = None):
            """Get trace analysis"""
            return self.tracer.tracing_service.get_trace_analysis(trace_id)
    
    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        # Metrics aggregation
        self.background_tasks.append(
            asyncio.create_task(
                self.metrics_collector.run_metrics_aggregator(interval=60)
            )
        )
        
        # System metrics update
        self.background_tasks.append(
            asyncio.create_task(
                self._update_system_metrics(interval=30)
            )
        )
        
        # Health check monitoring
        self.background_tasks.append(
            asyncio.create_task(
                self._monitor_health(interval=60)
            )
        )
        
        # Log compression
        self.background_tasks.append(
            asyncio.create_task(
                self._compress_logs(interval=86400)  # Daily
            )
        )
    
    async def stop_background_tasks(self):
        """Stop background monitoring tasks"""
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
    
    async def _update_system_metrics(self, interval: int):
        """Background task to update system metrics"""
        while True:
            try:
                self.prometheus.update_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Failed to update system metrics", exception=e)
                await asyncio.sleep(interval)
    
    async def _monitor_health(self, interval: int):
        """Background task to monitor health"""
        while True:
            try:
                health_status = await self.health_checker.check_health()
                
                # Update availability metric
                health_score = health_status.get('health_score', 0)
                self.prometheus.update_availability(
                    self.config.get('app_name', 'arbitration_detector'),
                    health_score
                )
                
                # Log if unhealthy
                if health_status['status'] != 'healthy':
                    self.logger.warning(
                        "Health check failed",
                        status=health_status['status'],
                        health_score=health_score
                    )
                
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health monitoring failed", exception=e)
                await asyncio.sleep(interval)
    
    async def _compress_logs(self, interval: int):
        """Background task to compress old logs"""
        while True:
            try:
                self.logger.compress_old_logs(days_old=7)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Log compression failed", exception=e)
                await asyncio.sleep(interval)
    
    def track_document_processing(self, doc_id: str, doc_type: str,
                                 processing_time: float, success: bool,
                                 arbitration_found: bool, confidence: float):
        """
        Track document processing metrics.
        
        Args:
            doc_id: Document identifier
            doc_type: Type of document
            processing_time: Processing time in seconds
            success: Whether processing was successful
            arbitration_found: Whether arbitration clause was found
            confidence: Confidence score
        """
        self.business_metrics.track_document_processing(
            doc_id=doc_id,
            doc_type=doc_type,
            processing_time=processing_time,
            success=success,
            arbitration_found=arbitration_found,
            confidence=confidence
        )
        
        # Log audit event
        self.logger.audit(
            audit_type='document_processing',
            user='system',
            action='process_document',
            resource=doc_id,
            result='success' if success else 'failure',
            metadata={
                'doc_type': doc_type,
                'processing_time': processing_time,
                'arbitration_found': arbitration_found,
                'confidence': confidence
            }
        )
    
    def track_user_activity(self, user_id: str, event_type: str,
                          event_name: str, properties: Optional[Dict] = None):
        """
        Track user activity for analytics.
        
        Args:
            user_id: User identifier
            event_type: Type of event
            event_name: Name of the event
            properties: Event properties
        """
        from datetime import datetime
        
        event = UserEvent(
            user_id=user_id,
            event_type=EventType[event_type.upper()],
            event_name=event_name,
            timestamp=datetime.utcnow(),
            properties=properties or {}
        )
        
        self.user_analytics.track_event(event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown of monitoring system.
    """
    # Load configuration
    config = {
        'app_name': 'arbitration_detector',
        'metrics_window': 3600,
        'prometheus_push_gateway': None,  # Set if using push gateway
        'db_url': 'postgresql://user:pass@localhost/db',
        'redis_url': 'redis://localhost:6379',
        'mongodb_url': 'mongodb://localhost:27017',
        'external_services': {
            'auth_service': 'http://auth-service:8000/health',
            'ml_service': 'http://ml-service:8000/health'
        },
        'auto_profiling': True,
        'tracing_backend': 'jaeger',
        'tracing_endpoint': 'localhost:6831',
        'tracing_sample_rate': 0.1,
        'log_dir': '/var/log/app',
        'enable_sentry': False,
        'sentry_dsn': None,
        'log_level': 'INFO',
        'session_timeout': 30,
        'enable_debug_endpoints': True  # Disable in production
    }
    
    # Initialize monitoring system
    monitoring = MonitoringSystem(config)
    
    # Setup FastAPI integration
    monitoring.setup_fastapi(app)
    
    # Start background tasks
    await monitoring.start_background_tasks()
    
    # Make monitoring available to app
    app.state.monitoring = monitoring
    
    monitoring.logger.info("Monitoring system started successfully")
    
    yield
    
    # Shutdown
    monitoring.logger.info("Shutting down monitoring system")
    
    # Stop background tasks
    await monitoring.stop_background_tasks()
    
    monitoring.logger.info("Monitoring system shutdown complete")


def create_monitored_app() -> FastAPI:
    """
    Create a FastAPI app with monitoring enabled.
    
    Returns:
        FastAPI application with monitoring
    """
    app = FastAPI(
        title="Arbitration Detector API",
        description="Legal document arbitration detection system with comprehensive monitoring",
        version="1.0.0",
        lifespan=lifespan
    )
    
    return app