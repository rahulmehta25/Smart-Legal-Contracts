"""
Production-grade logging system with structured logging, audit trails, and log aggregation.
Integrates with Sentry for error tracking and provides log rotation and archival.
"""

import os
import sys
import json
import logging
import logging.handlers
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import gzip
import shutil

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from pythonjsonlogger import jsonlogger
from loguru import logger as loguru_logger


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"


@dataclass
class LogContext:
    """Context information for structured logging"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredLogger:
    """
    Structured logging with JSON output and multiple handlers.
    """
    
    def __init__(self, 
                 service_name: str,
                 log_dir: str = "/var/log/app",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_sentry: bool = False,
                 sentry_dsn: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize structured logger.
        
        Args:
            service_name: Name of the service
            log_dir: Directory for log files
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_sentry: Enable Sentry integration
            sentry_dsn: Sentry DSN for error tracking
            log_level: Minimum log level
        """
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize main logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # JSON formatter
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )
        
        # Add handlers
        if enable_console:
            self._add_console_handler(json_formatter)
        
        if enable_file:
            self._add_file_handlers(json_formatter)
        
        if enable_sentry and sentry_dsn:
            self._setup_sentry(sentry_dsn)
        
        # Initialize audit logger
        self.audit_logger = self._setup_audit_logger()
        
        # Initialize performance logger
        self.perf_logger = self._setup_performance_logger()
        
        # Log context storage
        self.context = LogContext(service=service_name)
    
    def _add_console_handler(self, formatter):
        """Add console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handlers(self, formatter):
        """Add rotating file handlers"""
        # General log file
        general_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.service_name}.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        general_handler.setFormatter(formatter)
        self.logger.addHandler(general_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.service_name}.error.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def _setup_sentry(self, dsn: str):
        """Setup Sentry error tracking"""
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
        
        sentry_sdk.init(
            dsn=dsn,
            integrations=[sentry_logging],
            traces_sample_rate=0.1,
            environment=self.context.environment or "production"
        )
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup dedicated audit logger"""
        audit_logger = logging.getLogger(f"{self.service_name}.audit")
        audit_logger.setLevel(logging.INFO)
        
        # Audit log handler with daily rotation
        audit_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / f"{self.service_name}.audit.log",
            when='midnight',
            interval=1,
            backupCount=365  # Keep 1 year of audit logs
        )
        
        # Custom audit formatter
        audit_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(audit_type)s %(user)s %(action)s %(resource)s %(result)s',
            timestamp=True
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        
        return audit_logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance logging"""
        perf_logger = logging.getLogger(f"{self.service_name}.performance")
        perf_logger.setLevel(logging.INFO)
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.service_name}.performance.log",
            maxBytes=100 * 1024 * 1024,
            backupCount=5
        )
        
        perf_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(operation)s %(duration_ms)s %(metadata)s',
            timestamp=True
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        return perf_logger
    
    def set_context(self, **kwargs):
        """
        Set logging context.
        
        Args:
            **kwargs: Context fields to set
        """
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def clear_context(self):
        """Clear logging context"""
        self.context = LogContext(service=self.service_name)
    
    def _enrich_message(self, message: Union[str, Dict], 
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enrich log message with context"""
        if isinstance(message, str):
            log_data = {'message': message}
        else:
            log_data = message.copy()
        
        # Add context
        log_data.update(self.context.to_dict())
        
        # Add extra data
        if extra:
            log_data.update(extra)
        
        # Add timestamp if not present
        if 'timestamp' not in log_data:
            log_data['timestamp'] = datetime.utcnow().isoformat()
        
        return log_data
    
    def debug(self, message: Union[str, Dict], **kwargs):
        """Log debug message"""
        log_data = self._enrich_message(message, kwargs)
        self.logger.debug(json.dumps(log_data))
    
    def info(self, message: Union[str, Dict], **kwargs):
        """Log info message"""
        log_data = self._enrich_message(message, kwargs)
        self.logger.info(json.dumps(log_data))
    
    def warning(self, message: Union[str, Dict], **kwargs):
        """Log warning message"""
        log_data = self._enrich_message(message, kwargs)
        self.logger.warning(json.dumps(log_data))
    
    def error(self, message: Union[str, Dict], exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        log_data = self._enrich_message(message, kwargs)
        
        if exception:
            log_data['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
            
            # Send to Sentry if configured
            if sentry_sdk.Hub.current.client:
                sentry_sdk.capture_exception(exception)
        
        self.logger.error(json.dumps(log_data))
    
    def critical(self, message: Union[str, Dict], **kwargs):
        """Log critical message"""
        log_data = self._enrich_message(message, kwargs)
        self.logger.critical(json.dumps(log_data))
        
        # Send immediate alert if critical
        if sentry_sdk.Hub.current.client:
            sentry_sdk.capture_message(json.dumps(log_data), level="fatal")
    
    def audit(self, audit_type: str, user: str, action: str, 
             resource: str, result: str, metadata: Optional[Dict] = None):
        """
        Log audit event.
        
        Args:
            audit_type: Type of audit event (access, modification, deletion, etc.)
            user: User performing the action
            action: Action performed
            resource: Resource affected
            result: Result of the action (success, failure, etc.)
            metadata: Additional metadata
        """
        audit_data = {
            'audit_type': audit_type,
            'user': user,
            'action': action,
            'resource': resource,
            'result': result,
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name
        }
        
        if metadata:
            audit_data['metadata'] = metadata
        
        # Add context
        audit_data.update(self.context.to_dict())
        
        self.audit_logger.info(json.dumps(audit_data))
    
    def performance(self, operation: str, duration_ms: float, 
                   metadata: Optional[Dict] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Additional performance metadata
        """
        perf_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name
        }
        
        if metadata:
            perf_data['metadata'] = metadata
        
        # Add context
        perf_data.update(self.context.to_dict())
        
        self.perf_logger.info(json.dumps(perf_data))
    
    def rotate_logs(self):
        """Manually trigger log rotation"""
        for handler in self.logger.handlers:
            if isinstance(handler, (logging.handlers.RotatingFileHandler,
                                  logging.handlers.TimedRotatingFileHandler)):
                handler.doRollover()
    
    def compress_old_logs(self, days_old: int = 7):
        """
        Compress logs older than specified days.
        
        Args:
            days_old: Compress logs older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for log_file in self.log_dir.glob("*.log.*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                # Compress the file
                compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original file
                log_file.unlink()
                
                self.info(f"Compressed old log file: {log_file.name}")


class LogAggregator:
    """
    Log aggregation and analysis system.
    """
    
    def __init__(self, log_dir: str = "/var/log/app"):
        """
        Initialize log aggregator.
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
        self.parsed_logs: List[Dict[str, Any]] = []
        self.max_logs = 10000
    
    async def parse_log_files(self, file_pattern: str = "*.log") -> List[Dict[str, Any]]:
        """
        Parse log files matching pattern.
        
        Args:
            file_pattern: Glob pattern for log files
        
        Returns:
            List of parsed log entries
        """
        self.parsed_logs = []
        
        for log_file in self.log_dir.glob(file_pattern):
            if log_file.suffix == '.gz':
                await self._parse_compressed_log(log_file)
            else:
                await self._parse_log(log_file)
        
        return self.parsed_logs
    
    async def _parse_log(self, log_file: Path):
        """Parse a regular log file"""
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        self.parsed_logs.append(log_entry)
                        
                        if len(self.parsed_logs) >= self.max_logs:
                            return
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
    
    async def _parse_compressed_log(self, log_file: Path):
        """Parse a compressed log file"""
        try:
            with gzip.open(log_file, 'rt') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        self.parsed_logs.append(log_entry)
                        
                        if len(self.parsed_logs) >= self.max_logs:
                            return
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error parsing compressed log file {log_file}: {e}")
    
    def analyze_errors(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze error patterns in logs.
        
        Args:
            time_range_hours: Hours to look back
        
        Returns:
            Error analysis results
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        errors = []
        error_types = {}
        error_services = {}
        
        for log in self.parsed_logs:
            if log.get('level') in ['ERROR', 'CRITICAL']:
                log_time = datetime.fromisoformat(log.get('timestamp', ''))
                
                if log_time > cutoff_time:
                    errors.append(log)
                    
                    # Count by error type
                    error_type = log.get('exception', {}).get('type', 'Unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    # Count by service
                    service = log.get('service', 'unknown')
                    error_services[service] = error_services.get(service, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'error_by_service': error_services,
            'recent_errors': errors[:10],
            'time_range_hours': time_range_hours
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance metrics from logs.
        
        Returns:
            Performance analysis results
        """
        operations = {}
        
        for log in self.parsed_logs:
            if 'duration_ms' in log:
                operation = log.get('operation', 'unknown')
                
                if operation not in operations:
                    operations[operation] = []
                
                operations[operation].append(log['duration_ms'])
        
        # Calculate statistics
        operation_stats = {}
        for operation, durations in operations.items():
            if durations:
                operation_stats[operation] = {
                    'count': len(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'p50_ms': sorted(durations)[len(durations) // 2],
                    'p95_ms': sorted(durations)[int(len(durations) * 0.95)]
                }
        
        return {
            'operations': operation_stats,
            'slowest_operations': sorted(
                operation_stats.items(),
                key=lambda x: x[1]['avg_ms'],
                reverse=True
            )[:10]
        }
    
    def get_audit_trail(self, user: Optional[str] = None,
                       resource: Optional[str] = None,
                       days: int = 7) -> List[Dict[str, Any]]:
        """
        Get audit trail from logs.
        
        Args:
            user: Filter by user
            resource: Filter by resource
            days: Days to look back
        
        Returns:
            List of audit events
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        audit_events = []
        
        for log in self.parsed_logs:
            if 'audit_type' in log:
                log_time = datetime.fromisoformat(log.get('timestamp', ''))
                
                if log_time > cutoff_time:
                    if user and log.get('user') != user:
                        continue
                    if resource and log.get('resource') != resource:
                        continue
                    
                    audit_events.append(log)
        
        # Sort by timestamp
        audit_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return audit_events
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies in log patterns.
        
        Returns:
            Detected anomalies
        """
        anomalies = []
        
        # Detect high error rates
        error_count_by_minute = {}
        
        for log in self.parsed_logs:
            if log.get('level') == 'ERROR':
                timestamp = datetime.fromisoformat(log.get('timestamp', ''))
                minute_key = timestamp.strftime('%Y-%m-%d %H:%M')
                error_count_by_minute[minute_key] = error_count_by_minute.get(minute_key, 0) + 1
        
        # Find minutes with high error rates
        if error_count_by_minute:
            avg_errors = sum(error_count_by_minute.values()) / len(error_count_by_minute)
            threshold = avg_errors * 3  # 3x average
            
            for minute, count in error_count_by_minute.items():
                if count > threshold:
                    anomalies.append({
                        'type': 'high_error_rate',
                        'minute': minute,
                        'error_count': count,
                        'threshold': threshold
                    })
        
        # Detect unusual user activity
        user_activity = {}
        for log in self.parsed_logs:
            if 'user' in log:
                user = log['user']
                user_activity[user] = user_activity.get(user, 0) + 1
        
        if user_activity:
            avg_activity = sum(user_activity.values()) / len(user_activity)
            threshold = avg_activity * 10  # 10x average
            
            for user, count in user_activity.items():
                if count > threshold:
                    anomalies.append({
                        'type': 'unusual_user_activity',
                        'user': user,
                        'activity_count': count,
                        'threshold': threshold
                    })
        
        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }


class LogRouter:
    """
    Route logs to different destinations based on rules.
    """
    
    def __init__(self):
        """Initialize log router"""
        self.routes: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Any] = {}
    
    def add_route(self, name: str, filter_func: Callable[[Dict], bool],
                 handler: Any):
        """
        Add a routing rule.
        
        Args:
            name: Name of the route
            filter_func: Function to filter logs
            handler: Handler for matching logs
        """
        self.routes.append({
            'name': name,
            'filter': filter_func,
            'handler': handler
        })
        self.handlers[name] = handler
    
    async def route_log(self, log_entry: Dict[str, Any]):
        """
        Route a log entry to appropriate handlers.
        
        Args:
            log_entry: Log entry to route
        """
        for route in self.routes:
            if route['filter'](log_entry):
                await route['handler'].handle(log_entry)
    
    def create_level_filter(self, level: str) -> Callable:
        """Create filter for log level"""
        return lambda log: log.get('level') == level
    
    def create_service_filter(self, service: str) -> Callable:
        """Create filter for service"""
        return lambda log: log.get('service') == service
    
    def create_pattern_filter(self, pattern: str) -> Callable:
        """Create filter for message pattern"""
        import re
        regex = re.compile(pattern)
        return lambda log: regex.search(log.get('message', ''))