"""
Distributed tracing system using OpenTelemetry.
Provides end-to-end visibility across microservices and components.
"""

import time
import asyncio
import functools
import contextvars
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid

from opentelemetry import trace, metrics, baggage
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter
)
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# Context variable for current span
current_span_context = contextvars.ContextVar('current_span', default=None)


class TracingBackend(Enum):
    """Supported tracing backends"""
    CONSOLE = "console"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"


@dataclass
class SpanContext:
    """Context information for a span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)


class TracingService:
    """
    Comprehensive distributed tracing service.
    """
    
    def __init__(self, 
                 service_name: str,
                 backend: TracingBackend = TracingBackend.JAEGER,
                 endpoint: Optional[str] = None,
                 sample_rate: float = 1.0):
        """
        Initialize tracing service.
        
        Args:
            service_name: Name of the service for tracing
            backend: Tracing backend to use
            endpoint: Endpoint for the tracing backend
            sample_rate: Sampling rate for traces (0.0 to 1.0)
        """
        self.service_name = service_name
        self.backend = backend
        self.endpoint = endpoint
        self.sample_rate = sample_rate
        
        # Initialize tracer provider
        self.resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        self.tracer_provider = TracerProvider(resource=self.resource)
        
        # Set up exporter based on backend
        self._setup_exporter()
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Initialize propagator
        self.propagator = TraceContextTextMapPropagator()
        
        # Span storage for analysis
        self.span_storage: List[Dict[str, Any]] = []
        self.max_stored_spans = 1000
        
        # Initialize instrumentations
        self._setup_auto_instrumentation()
    
    def _setup_exporter(self):
        """Set up the appropriate span exporter based on backend"""
        if self.backend == TracingBackend.CONSOLE:
            exporter = ConsoleSpanExporter()
        elif self.backend == TracingBackend.JAEGER:
            exporter = JaegerExporter(
                agent_host_name=self.endpoint or "localhost",
                agent_port=6831,
                collector_endpoint=f"http://{self.endpoint or 'localhost'}:14268/api/traces" if self.endpoint else None
            )
        elif self.backend == TracingBackend.ZIPKIN:
            exporter = ZipkinExporter(
                endpoint=f"http://{self.endpoint or 'localhost'}:9411/api/v2/spans"
            )
        elif self.backend == TracingBackend.OTLP:
            exporter = OTLPSpanExporter(
                endpoint=self.endpoint or "localhost:4317",
                insecure=True
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        self.tracer_provider.add_span_processor(span_processor)
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries"""
        try:
            # HTTP requests
            RequestsInstrumentor().instrument()
            
            # Database
            # SQLAlchemyInstrumentor().instrument()
            
            # Redis
            # RedisInstrumentor().instrument()
            
            # FastAPI
            # This would be done in the FastAPI app initialization
            
        except Exception as e:
            print(f"Failed to set up auto-instrumentation: {e}")
    
    def start_span(self, name: str, 
                   kind: SpanKind = SpanKind.INTERNAL,
                   attributes: Optional[Dict[str, Any]] = None) -> Span:
        """
        Start a new span.
        
        Args:
            name: Name of the span
            kind: Type of span (INTERNAL, SERVER, CLIENT, etc.)
            attributes: Initial span attributes
        
        Returns:
            Started span
        """
        span = self.tracer.start_span(
            name,
            kind=kind,
            attributes=attributes or {}
        )
        
        # Store in context
        current_span_context.set(span)
        
        # Store span info for analysis
        self._store_span_info(span, name, attributes)
        
        return span
    
    def trace_function(self, name: Optional[str] = None,
                      kind: SpanKind = SpanKind.INTERNAL,
                      attributes: Optional[Dict[str, Any]] = None):
        """
        Decorator to trace function execution.
        
        Args:
            name: Span name (defaults to function name)
            kind: Type of span
            attributes: Additional span attributes
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes or {}
                ) as span:
                    try:
                        # Add function arguments as span attributes
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_count", len(kwargs))
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record exception
                        span.record_exception(e)
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        raise
            
            return wrapper
        return decorator
    
    def trace_async_function(self, name: Optional[str] = None,
                            kind: SpanKind = SpanKind.INTERNAL,
                            attributes: Optional[Dict[str, Any]] = None):
        """
        Decorator to trace async function execution.
        
        Args:
            name: Span name (defaults to function name)
            kind: Type of span
            attributes: Additional span attributes
        
        Returns:
            Decorated async function
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes or {}
                ) as span:
                    try:
                        # Add function arguments as span attributes
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_count", len(kwargs))
                        span.set_attribute("function.async", True)
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record exception
                        span.record_exception(e)
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        raise
            
            return wrapper
        return decorator
    
    def trace_http_request(self, method: str, url: str,
                          headers: Optional[Dict[str, str]] = None):
        """
        Trace an HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
        
        Returns:
            Span for the HTTP request
        """
        span = self.tracer.start_span(
            f"HTTP {method}",
            kind=SpanKind.CLIENT,
            attributes={
                "http.method": method,
                "http.url": url,
                "http.scheme": url.split("://")[0] if "://" in url else "http"
            }
        )
        
        # Inject trace context into headers
        if headers is not None:
            inject(headers)
        
        return span
    
    def trace_database_query(self, query: str, operation: str,
                            table: Optional[str] = None):
        """
        Trace a database query.
        
        Args:
            query: SQL query
            operation: Database operation (SELECT, INSERT, etc.)
            table: Table name
        
        Returns:
            Span for the database query
        """
        span_name = f"DB {operation}"
        if table:
            span_name += f" {table}"
        
        span = self.tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                "db.statement": query[:1000],  # Truncate long queries
                "db.operation": operation,
                "db.table": table or "unknown"
            }
        )
        
        return span
    
    def trace_message_queue(self, operation: str, queue_name: str,
                          message_id: Optional[str] = None):
        """
        Trace message queue operations.
        
        Args:
            operation: Queue operation (send, receive, ack, etc.)
            queue_name: Name of the queue
            message_id: Message identifier
        
        Returns:
            Span for the queue operation
        """
        span = self.tracer.start_span(
            f"Queue {operation}",
            kind=SpanKind.PRODUCER if operation == "send" else SpanKind.CONSUMER,
            attributes={
                "messaging.system": "queue",
                "messaging.destination": queue_name,
                "messaging.operation": operation,
                "messaging.message_id": message_id or str(uuid.uuid4())
            }
        )
        
        return span
    
    def trace_cache_operation(self, operation: str, key: str,
                            cache_name: str = "default"):
        """
        Trace cache operations.
        
        Args:
            operation: Cache operation (get, set, delete, etc.)
            key: Cache key
            cache_name: Name of the cache
        
        Returns:
            Span for the cache operation
        """
        span = self.tracer.start_span(
            f"Cache {operation}",
            kind=SpanKind.CLIENT,
            attributes={
                "cache.operation": operation,
                "cache.key": key,
                "cache.name": cache_name
            }
        )
        
        return span
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to the current span.
        
        Args:
            name: Event name
            attributes: Event attributes
        """
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})
    
    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)
    
    def set_baggage(self, key: str, value: str):
        """
        Set baggage item for context propagation.
        
        Args:
            key: Baggage key
            value: Baggage value
        """
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """
        Get baggage item from context.
        
        Args:
            key: Baggage key
        
        Returns:
            Baggage value or None
        """
        return baggage.get_baggage(key)
    
    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """
        Extract trace context from carrier.
        
        Args:
            carrier: Dictionary containing trace context
        
        Returns:
            Extracted context
        """
        return extract(carrier)
    
    def inject_context(self, carrier: Dict[str, str]):
        """
        Inject trace context into carrier.
        
        Args:
            carrier: Dictionary to inject context into
        """
        inject(carrier)
    
    def _store_span_info(self, span: Span, name: str,
                        attributes: Optional[Dict[str, Any]] = None):
        """Store span information for analysis"""
        span_info = {
            'name': name,
            'trace_id': format(span.get_span_context().trace_id, '032x'),
            'span_id': format(span.get_span_context().span_id, '016x'),
            'start_time': datetime.utcnow().isoformat(),
            'attributes': attributes or {}
        }
        
        self.span_storage.append(span_info)
        
        # Trim storage if needed
        if len(self.span_storage) > self.max_stored_spans:
            self.span_storage = self.span_storage[-self.max_stored_spans:]
    
    def get_trace_analysis(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trace data.
        
        Args:
            trace_id: Specific trace ID to analyze (analyzes all if None)
        
        Returns:
            Trace analysis results
        """
        if trace_id:
            spans = [s for s in self.span_storage if s['trace_id'] == trace_id]
        else:
            spans = self.span_storage
        
        if not spans:
            return {'message': 'No trace data available'}
        
        # Group spans by trace
        traces = {}
        for span in spans:
            tid = span['trace_id']
            if tid not in traces:
                traces[tid] = []
            traces[tid].append(span)
        
        # Analyze each trace
        trace_analysis = []
        for tid, trace_spans in traces.items():
            trace_analysis.append({
                'trace_id': tid,
                'span_count': len(trace_spans),
                'start_time': min(s['start_time'] for s in trace_spans),
                'services': list(set(s.get('attributes', {}).get('service', 'unknown') 
                                    for s in trace_spans))
            })
        
        return {
            'total_traces': len(traces),
            'total_spans': len(spans),
            'traces': trace_analysis[:10]  # Return top 10 traces
        }
    
    def get_span_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about spans.
        
        Returns:
            Span metrics
        """
        if not self.span_storage:
            return {'message': 'No span data available'}
        
        # Count spans by name
        span_counts = {}
        for span in self.span_storage:
            name = span['name']
            span_counts[name] = span_counts.get(name, 0) + 1
        
        # Get top operations
        top_operations = sorted(
            span_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_spans': len(self.span_storage),
            'unique_operations': len(span_counts),
            'top_operations': dict(top_operations)
        }


class DistributedTracer:
    """
    High-level distributed tracing interface for the application.
    """
    
    def __init__(self, service_name: str, config: Dict[str, Any]):
        """
        Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
            config: Tracing configuration
        """
        backend = TracingBackend(config.get('backend', 'jaeger'))
        endpoint = config.get('endpoint')
        sample_rate = config.get('sample_rate', 1.0)
        
        self.tracing_service = TracingService(
            service_name=service_name,
            backend=backend,
            endpoint=endpoint,
            sample_rate=sample_rate
        )
    
    def trace_document_processing(self, doc_id: str, doc_type: str):
        """
        Create a trace for document processing.
        
        Args:
            doc_id: Document identifier
            doc_type: Type of document
        
        Returns:
            Span for document processing
        """
        return self.tracing_service.tracer.start_span(
            "document.process",
            kind=SpanKind.SERVER,
            attributes={
                "document.id": doc_id,
                "document.type": doc_type,
                "processing.stage": "start"
            }
        )
    
    def trace_ai_inference(self, model_name: str, model_version: str,
                          input_size: int):
        """
        Create a trace for AI model inference.
        
        Args:
            model_name: Name of the AI model
            model_version: Model version
            input_size: Size of input data
        
        Returns:
            Span for AI inference
        """
        return self.tracing_service.tracer.start_span(
            "ai.inference",
            kind=SpanKind.INTERNAL,
            attributes={
                "ai.model": model_name,
                "ai.version": model_version,
                "ai.input_size": input_size
            }
        )
    
    def trace_api_call(self, endpoint: str, method: str,
                      client_id: Optional[str] = None):
        """
        Create a trace for API calls.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            client_id: Client identifier
        
        Returns:
            Span for API call
        """
        return self.tracing_service.tracer.start_span(
            f"api.{method.lower()}",
            kind=SpanKind.SERVER,
            attributes={
                "api.endpoint": endpoint,
                "api.method": method,
                "api.client_id": client_id or "anonymous"
            }
        )
    
    def create_trace_context(self, operation: str) -> SpanContext:
        """
        Create a new trace context.
        
        Args:
            operation: Name of the operation
        
        Returns:
            SpanContext object
        """
        span = self.tracing_service.start_span(operation)
        context = span.get_span_context()
        
        return SpanContext(
            trace_id=format(context.trace_id, '032x'),
            span_id=format(context.span_id, '016x'),
            attributes={'operation': operation}
        )
    
    def correlate_logs(self, trace_id: str, span_id: str) -> Dict[str, str]:
        """
        Create correlation IDs for log correlation.
        
        Args:
            trace_id: Trace identifier
            span_id: Span identifier
        
        Returns:
            Dictionary with correlation IDs
        """
        return {
            'trace_id': trace_id,
            'span_id': span_id,
            'correlation_id': f"{trace_id[:8]}-{span_id[:8]}"
        }