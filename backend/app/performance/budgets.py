"""
Performance Budget System
Enforces performance limits and detects regressions automatically
"""

import time
import functools
import warnings
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from pathlib import Path
import traceback

class BudgetType(Enum):
    """Types of performance budgets"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    DATABASE_QUERIES = "database_queries"
    BUNDLE_SIZE = "bundle_size"
    API_CALLS = "api_calls"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"

class ViolationSeverity(Enum):
    """Severity levels for budget violations"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceBudget:
    """Individual performance budget definition"""
    name: str
    type: BudgetType
    limit: float
    unit: str
    severity: ViolationSeverity = ViolationSeverity.WARNING
    description: str = ""
    enabled: bool = True
    
@dataclass
class BudgetViolation:
    """Record of a budget violation"""
    budget: PerformanceBudget
    actual_value: float
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    
@dataclass
class PerformanceMetric:
    """Performance measurement"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceBudgetManager:
    """Manages and enforces performance budgets across the application"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.budgets: Dict[str, PerformanceBudget] = {}
        self.violations: List[BudgetViolation] = []
        self.metrics: List[PerformanceMetric] = []
        self.baseline_metrics: Dict[str, float] = {}
        
        # Load configuration
        if config_file:
            self.load_config(config_file)
        else:
            self._load_default_budgets()
    
    def _load_default_budgets(self):
        """Load default performance budgets"""
        default_budgets = [
            # API Response Times
            PerformanceBudget(
                name="api_response_p50",
                type=BudgetType.RESPONSE_TIME,
                limit=100,
                unit="ms",
                severity=ViolationSeverity.WARNING,
                description="API response time 50th percentile"
            ),
            PerformanceBudget(
                name="api_response_p95",
                type=BudgetType.RESPONSE_TIME,
                limit=500,
                unit="ms",
                severity=ViolationSeverity.ERROR,
                description="API response time 95th percentile"
            ),
            PerformanceBudget(
                name="api_response_p99",
                type=BudgetType.RESPONSE_TIME,
                limit=2000,
                unit="ms",
                severity=ViolationSeverity.CRITICAL,
                description="API response time 99th percentile"
            ),
            
            # Database Performance
            PerformanceBudget(
                name="db_query_time",
                type=BudgetType.DATABASE_QUERIES,
                limit=100,
                unit="ms",
                severity=ViolationSeverity.WARNING,
                description="Maximum database query time"
            ),
            PerformanceBudget(
                name="db_queries_per_request",
                type=BudgetType.DATABASE_QUERIES,
                limit=10,
                unit="queries",
                severity=ViolationSeverity.WARNING,
                description="Maximum queries per request"
            ),
            
            # Memory Usage
            PerformanceBudget(
                name="memory_per_request",
                type=BudgetType.MEMORY_USAGE,
                limit=50,
                unit="MB",
                severity=ViolationSeverity.WARNING,
                description="Memory usage per request"
            ),
            PerformanceBudget(
                name="total_memory",
                type=BudgetType.MEMORY_USAGE,
                limit=4096,
                unit="MB",
                severity=ViolationSeverity.CRITICAL,
                description="Total application memory"
            ),
            
            # Frontend Performance
            PerformanceBudget(
                name="bundle_size_js",
                type=BudgetType.BUNDLE_SIZE,
                limit=500,
                unit="KB",
                severity=ViolationSeverity.WARNING,
                description="JavaScript bundle size"
            ),
            PerformanceBudget(
                name="bundle_size_css",
                type=BudgetType.BUNDLE_SIZE,
                limit=100,
                unit="KB",
                severity=ViolationSeverity.WARNING,
                description="CSS bundle size"
            ),
            
            # Cache Performance
            PerformanceBudget(
                name="cache_hit_rate",
                type=BudgetType.CACHE_HIT_RATE,
                limit=80,
                unit="%",
                severity=ViolationSeverity.WARNING,
                description="Minimum cache hit rate"
            ),
            
            # Error Rates
            PerformanceBudget(
                name="error_rate",
                type=BudgetType.ERROR_RATE,
                limit=1,
                unit="%",
                severity=ViolationSeverity.ERROR,
                description="Maximum error rate"
            ),
            
            # Throughput
            PerformanceBudget(
                name="requests_per_second",
                type=BudgetType.THROUGHPUT,
                limit=1000,
                unit="RPS",
                severity=ViolationSeverity.WARNING,
                description="Minimum throughput"
            )
        ]
        
        for budget in default_budgets:
            self.add_budget(budget)
    
    def load_config(self, config_file: str):
        """Load budgets from configuration file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        for budget_config in config.get('budgets', []):
            budget = PerformanceBudget(
                name=budget_config['name'],
                type=BudgetType(budget_config['type']),
                limit=budget_config['limit'],
                unit=budget_config['unit'],
                severity=ViolationSeverity(budget_config.get('severity', 'warning')),
                description=budget_config.get('description', ''),
                enabled=budget_config.get('enabled', True)
            )
            self.add_budget(budget)
    
    def add_budget(self, budget: PerformanceBudget):
        """Add a performance budget"""
        self.budgets[budget.name] = budget
    
    def check_budget(self, 
                     budget_name: str, 
                     value: float,
                     context: Optional[Dict] = None) -> Optional[BudgetViolation]:
        """Check if a value violates a budget"""
        if budget_name not in self.budgets:
            return None
        
        budget = self.budgets[budget_name]
        
        if not budget.enabled:
            return None
        
        # Check for violation
        violated = False
        
        if budget.type == BudgetType.CACHE_HIT_RATE:
            # For cache hit rate, lower is bad
            violated = value < budget.limit
        elif budget.type == BudgetType.THROUGHPUT:
            # For throughput, lower is bad
            violated = value < budget.limit
        else:
            # For most metrics, higher is bad
            violated = value > budget.limit
        
        if violated:
            violation = BudgetViolation(
                budget=budget,
                actual_value=value,
                timestamp=datetime.now(),
                context=context or {},
                stack_trace=traceback.format_stack()
            )
            
            self.violations.append(violation)
            self._handle_violation(violation)
            
            return violation
        
        return None
    
    def _handle_violation(self, violation: BudgetViolation):
        """Handle a budget violation"""
        # Log violation
        message = (
            f"Performance budget violation: {violation.budget.name}\n"
            f"  Limit: {violation.budget.limit} {violation.budget.unit}\n"
            f"  Actual: {violation.actual_value} {violation.budget.unit}\n"
            f"  Severity: {violation.budget.severity.value}"
        )
        
        if violation.budget.severity == ViolationSeverity.WARNING:
            warnings.warn(message, PerformanceWarning)
        elif violation.budget.severity == ViolationSeverity.ERROR:
            print(f"ERROR: {message}")
            # In production, might send to monitoring service
        elif violation.budget.severity == ViolationSeverity.CRITICAL:
            print(f"CRITICAL: {message}")
            # In production, might trigger alerts
            
            # Could optionally raise exception for critical violations
            if self.fail_on_critical:
                raise PerformanceBudgetExceeded(message)
    
    def enforce_response_time(self, max_time: float = None) -> Callable:
        """Decorator to enforce response time budget"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start_time) * 1000  # ms
                    
                    # Check budget
                    budget_name = f"{func.__name__}_response_time"
                    limit = max_time or 100  # Default 100ms
                    
                    if budget_name not in self.budgets:
                        self.add_budget(PerformanceBudget(
                            name=budget_name,
                            type=BudgetType.RESPONSE_TIME,
                            limit=limit,
                            unit="ms",
                            severity=ViolationSeverity.WARNING
                        ))
                    
                    self.check_budget(budget_name, elapsed, {
                        'function': func.__name__,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100]
                    })
                    
                    # Record metric
                    self.record_metric(
                        name=budget_name,
                        value=elapsed,
                        unit="ms",
                        metadata={'function': func.__name__}
                    )
                
                return result
            return wrapper
        return decorator
    
    def enforce_memory_usage(self, max_memory_mb: float = None) -> Callable:
        """Decorator to enforce memory usage budget"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import psutil
                process = psutil.Process()
                
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_used = end_memory - start_memory
                    
                    # Check budget
                    budget_name = f"{func.__name__}_memory_usage"
                    limit = max_memory_mb or 50  # Default 50MB
                    
                    if budget_name not in self.budgets:
                        self.add_budget(PerformanceBudget(
                            name=budget_name,
                            type=BudgetType.MEMORY_USAGE,
                            limit=limit,
                            unit="MB",
                            severity=ViolationSeverity.WARNING
                        ))
                    
                    self.check_budget(budget_name, memory_used, {
                        'function': func.__name__,
                        'start_memory': start_memory,
                        'end_memory': end_memory
                    })
                    
                    # Record metric
                    self.record_metric(
                        name=budget_name,
                        value=memory_used,
                        unit="MB",
                        metadata={'function': func.__name__}
                    )
                
                return result
            return wrapper
        return decorator
    
    def enforce_query_count(self, max_queries: int = None) -> Callable:
        """Decorator to enforce database query count budget"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # This would integrate with your ORM/database layer
                query_count = 0
                
                # Mock implementation - replace with actual query counting
                import random
                query_count = random.randint(1, 20)
                
                result = func(*args, **kwargs)
                
                # Check budget
                budget_name = f"{func.__name__}_query_count"
                limit = max_queries or 10  # Default 10 queries
                
                if budget_name not in self.budgets:
                    self.add_budget(PerformanceBudget(
                        name=budget_name,
                        type=BudgetType.DATABASE_QUERIES,
                        limit=limit,
                        unit="queries",
                        severity=ViolationSeverity.WARNING
                    ))
                
                self.check_budget(budget_name, query_count, {
                    'function': func.__name__,
                    'queries': query_count
                })
                
                return result
            return wrapper
        return decorator
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     unit: str,
                     metadata: Optional[Dict] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
    
    def calculate_percentiles(self, metric_name: str) -> Dict[str, float]:
        """Calculate percentiles for a metric"""
        values = [m.value for m in self.metrics if m.name == metric_name]
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        
        return {
            'p50': sorted_values[int(len(sorted_values) * 0.5)],
            'p75': sorted_values[int(len(sorted_values) * 0.75)],
            'p90': sorted_values[int(len(sorted_values) * 0.9)],
            'p95': sorted_values[int(len(sorted_values) * 0.95)],
            'p99': sorted_values[int(len(sorted_values) * 0.99)]
        }
    
    def detect_regression(self, metric_name: str, threshold: float = 0.1) -> bool:
        """Detect performance regression by comparing to baseline"""
        if metric_name not in self.baseline_metrics:
            return False
        
        current_metrics = [m.value for m in self.metrics 
                          if m.name == metric_name and 
                          m.timestamp > datetime.now() - timedelta(hours=1)]
        
        if not current_metrics:
            return False
        
        current_avg = statistics.mean(current_metrics)
        baseline = self.baseline_metrics[metric_name]
        
        # Check if current is worse than baseline by threshold
        regression = (current_avg - baseline) / baseline > threshold
        
        if regression:
            print(f"Performance regression detected for {metric_name}:")
            print(f"  Baseline: {baseline}")
            print(f"  Current: {current_avg}")
            print(f"  Degradation: {((current_avg - baseline) / baseline) * 100:.1f}%")
        
        return regression
    
    def set_baseline(self, metric_name: str, value: Optional[float] = None):
        """Set baseline for a metric"""
        if value is not None:
            self.baseline_metrics[metric_name] = value
        else:
            # Calculate baseline from recent metrics
            recent_values = [m.value for m in self.metrics 
                           if m.name == metric_name]
            if recent_values:
                self.baseline_metrics[metric_name] = statistics.median(recent_values)
    
    def generate_report(self) -> Dict:
        """Generate performance budget report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'budgets': {},
            'violations': [],
            'metrics_summary': {},
            'regression_detection': {}
        }
        
        # Budget status
        for name, budget in self.budgets.items():
            recent_metrics = [m for m in self.metrics 
                            if m.name == name and 
                            m.timestamp > datetime.now() - timedelta(hours=1)]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                report['budgets'][name] = {
                    'limit': budget.limit,
                    'unit': budget.unit,
                    'current_avg': statistics.mean(values),
                    'current_max': max(values),
                    'violations_count': len([v for v in self.violations 
                                           if v.budget.name == name])
                }
        
        # Recent violations
        recent_violations = [v for v in self.violations 
                           if v.timestamp > datetime.now() - timedelta(hours=1)]
        
        for violation in recent_violations[:10]:  # Last 10 violations
            report['violations'].append({
                'budget': violation.budget.name,
                'limit': violation.budget.limit,
                'actual': violation.actual_value,
                'severity': violation.budget.severity.value,
                'timestamp': violation.timestamp.isoformat()
            })
        
        # Metrics summary
        unique_metrics = set(m.name for m in self.metrics)
        for metric_name in unique_metrics:
            percentiles = self.calculate_percentiles(metric_name)
            if percentiles:
                report['metrics_summary'][metric_name] = percentiles
        
        # Regression detection
        for metric_name in self.baseline_metrics:
            has_regression = self.detect_regression(metric_name)
            report['regression_detection'][metric_name] = {
                'has_regression': has_regression,
                'baseline': self.baseline_metrics[metric_name]
            }
        
        return report
    
    def export_violations_csv(self, filepath: str):
        """Export violations to CSV file"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Budget', 'Type', 'Limit', 'Actual', 
                'Unit', 'Severity', 'Context'
            ])
            
            for violation in self.violations:
                writer.writerow([
                    violation.timestamp.isoformat(),
                    violation.budget.name,
                    violation.budget.type.value,
                    violation.budget.limit,
                    violation.actual_value,
                    violation.budget.unit,
                    violation.budget.severity.value,
                    json.dumps(violation.context)
                ])
    
    def enforce_all(self, func: Callable) -> Callable:
        """Decorator to enforce all applicable budgets"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Combine multiple enforcement decorators
            enforced = func
            enforced = self.enforce_response_time()(enforced)
            enforced = self.enforce_memory_usage()(enforced)
            enforced = self.enforce_query_count()(enforced)
            
            return enforced(*args, **kwargs)
        
        return wrapper

class PerformanceWarning(UserWarning):
    """Warning for performance budget violations"""
    pass

class PerformanceBudgetExceeded(Exception):
    """Exception for critical performance budget violations"""
    pass

# Global budget manager instance
budget_manager = PerformanceBudgetManager()

# Example usage
if __name__ == "__main__":
    # Create budget manager
    manager = PerformanceBudgetManager()
    
    # Example function with performance budgets
    @manager.enforce_response_time(max_time=50)
    @manager.enforce_memory_usage(max_memory_mb=10)
    def example_operation(n: int) -> int:
        """Example operation with performance budgets"""
        import time
        time.sleep(0.01 * n)  # Simulate work
        
        # Allocate some memory
        data = [0] * (n * 100000)
        
        return sum(data)
    
    # Test with different inputs
    for i in range(1, 10):
        try:
            result = example_operation(i)
            print(f"Operation {i} completed: {result}")
        except PerformanceBudgetExceeded as e:
            print(f"Operation {i} failed: {e}")
    
    # Generate report
    report = manager.generate_report()
    print("\nPerformance Budget Report:")
    print(json.dumps(report, indent=2, default=str))