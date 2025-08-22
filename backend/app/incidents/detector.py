"""
Incident Detection System

Real-time anomaly detection with multiple detection strategies:
- Statistical anomaly detection
- Pattern-based detection
- Threshold monitoring
- Predictive detection
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Detection methods for anomalies"""
    THRESHOLD = "threshold"
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    PREDICTIVE = "predictive"
    COMPOSITE = "composite"


class MetricType(Enum):
    """Types of metrics to monitor"""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    BUSINESS_METRIC = "business_metric"
    CUSTOM = "custom"


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    id: str
    timestamp: datetime
    service: str
    metric_type: MetricType
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    confidence: float
    detection_method: DetectionMethod
    context: Dict[str, Any] = field(default_factory=dict)
    related_anomalies: List[str] = field(default_factory=list)


@dataclass
class DetectionRule:
    """Rule for detecting anomalies"""
    name: str
    metric_type: MetricType
    detection_method: DetectionMethod
    threshold: Optional[float] = None
    sensitivity: float = 0.95
    window_size: int = 60  # seconds
    min_occurrences: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class IncidentDetector:
    """
    Main incident detection system
    Monitors metrics and detects anomalies in real-time
    """
    
    def __init__(self):
        self.rules: Dict[str, DetectionRule] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomalies: Dict[str, Anomaly] = {}
        self.active_monitoring: Set[str] = set()
        self.baseline_models: Dict[str, Any] = {}
        self.detection_stats: Dict[str, Dict] = defaultdict(dict)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default detection rules"""
        
        # Latency detection
        self.add_rule(DetectionRule(
            name="high_latency",
            metric_type=MetricType.LATENCY,
            detection_method=DetectionMethod.STATISTICAL,
            threshold=1000,  # ms
            sensitivity=0.99,
            window_size=300,
            min_occurrences=3
        ))
        
        # Error rate detection
        self.add_rule(DetectionRule(
            name="high_error_rate",
            metric_type=MetricType.ERROR_RATE,
            detection_method=DetectionMethod.THRESHOLD,
            threshold=0.05,  # 5% error rate
            sensitivity=0.98,
            window_size=120,
            min_occurrences=2
        ))
        
        # CPU usage detection
        self.add_rule(DetectionRule(
            name="high_cpu",
            metric_type=MetricType.CPU_USAGE,
            detection_method=DetectionMethod.THRESHOLD,
            threshold=0.85,  # 85% CPU
            sensitivity=0.95,
            window_size=180,
            min_occurrences=5
        ))
        
        # Memory usage detection
        self.add_rule(DetectionRule(
            name="high_memory",
            metric_type=MetricType.MEMORY_USAGE,
            detection_method=DetectionMethod.THRESHOLD,
            threshold=0.90,  # 90% memory
            sensitivity=0.95,
            window_size=300,
            min_occurrences=5
        ))
        
        # Throughput anomaly detection
        self.add_rule(DetectionRule(
            name="throughput_anomaly",
            metric_type=MetricType.THROUGHPUT,
            detection_method=DetectionMethod.STATISTICAL,
            sensitivity=0.97,
            window_size=600,
            min_occurrences=3
        ))
        
    def add_rule(self, rule: DetectionRule):
        """Add a detection rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added detection rule: {rule.name}")
        
    def remove_rule(self, rule_name: str):
        """Remove a detection rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed detection rule: {rule_name}")
            
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if self._monitoring_task:
            return
            
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started incident detection monitoring")
        
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)
            self._monitoring_task = None
            logger.info("Stopped incident detection monitoring")
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Process each metric
                for metric_name, metric_data in metrics.items():
                    # Store in history
                    self.metric_history[metric_name].append({
                        'timestamp': datetime.utcnow(),
                        'value': metric_data['value'],
                        'metadata': metric_data.get('metadata', {})
                    })
                    
                    # Check for anomalies
                    anomalies = await self._detect_anomalies(
                        metric_name, 
                        metric_data
                    )
                    
                    # Process detected anomalies
                    for anomaly in anomalies:
                        await self._process_anomaly(anomaly)
                        
                # Correlate anomalies
                await self._correlate_anomalies()
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
                
    async def _collect_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from various sources"""
        metrics = {}
        
        # This would integrate with actual monitoring systems
        # For now, simulating metric collection
        
        # Simulated metrics (replace with actual data sources)
        metrics['api_latency'] = {
            'value': np.random.normal(100, 20),  # ms
            'type': MetricType.LATENCY,
            'service': 'api',
            'metadata': {'endpoint': '/api/v1/users'}
        }
        
        metrics['api_error_rate'] = {
            'value': np.random.beta(2, 100),  # Error rate
            'type': MetricType.ERROR_RATE,
            'service': 'api',
            'metadata': {'status_codes': {'500': 5, '502': 2}}
        }
        
        metrics['cpu_usage'] = {
            'value': np.random.beta(5, 3),  # CPU percentage
            'type': MetricType.CPU_USAGE,
            'service': 'backend',
            'metadata': {'node': 'backend-1'}
        }
        
        metrics['memory_usage'] = {
            'value': np.random.beta(7, 3),  # Memory percentage
            'type': MetricType.MEMORY_USAGE,
            'service': 'backend',
            'metadata': {'node': 'backend-1'}
        }
        
        metrics['throughput'] = {
            'value': np.random.normal(1000, 100),  # requests/sec
            'type': MetricType.THROUGHPUT,
            'service': 'api',
            'metadata': {}
        }
        
        return metrics
        
    async def _detect_anomalies(
        self, 
        metric_name: str, 
        metric_data: Dict[str, Any]
    ) -> List[Anomaly]:
        """Detect anomalies in a metric"""
        anomalies = []
        metric_type = metric_data.get('type', MetricType.CUSTOM)
        
        # Check each applicable rule
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            if rule.metric_type != metric_type:
                continue
                
            # Apply detection method
            is_anomaly, confidence, expected = await self._apply_detection_method(
                rule,
                metric_name,
                metric_data['value']
            )
            
            if is_anomaly:
                anomaly = Anomaly(
                    id=f"{metric_name}_{int(time.time()*1000)}",
                    timestamp=datetime.utcnow(),
                    service=metric_data.get('service', 'unknown'),
                    metric_type=metric_type,
                    metric_name=metric_name,
                    current_value=metric_data['value'],
                    expected_value=expected,
                    deviation=abs(metric_data['value'] - expected) / max(expected, 0.001),
                    confidence=confidence,
                    detection_method=rule.detection_method,
                    context=metric_data.get('metadata', {})
                )
                anomalies.append(anomaly)
                
        return anomalies
        
    async def _apply_detection_method(
        self,
        rule: DetectionRule,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float, float]:
        """Apply detection method to check for anomaly"""
        
        if rule.detection_method == DetectionMethod.THRESHOLD:
            return self._threshold_detection(rule, value)
            
        elif rule.detection_method == DetectionMethod.STATISTICAL:
            return self._statistical_detection(rule, metric_name, value)
            
        elif rule.detection_method == DetectionMethod.PATTERN:
            return self._pattern_detection(rule, metric_name, value)
            
        elif rule.detection_method == DetectionMethod.PREDICTIVE:
            return await self._predictive_detection(rule, metric_name, value)
            
        elif rule.detection_method == DetectionMethod.COMPOSITE:
            return await self._composite_detection(rule, metric_name, value)
            
        return False, 0.0, value
        
    def _threshold_detection(
        self,
        rule: DetectionRule,
        value: float
    ) -> Tuple[bool, float, float]:
        """Simple threshold-based detection"""
        if rule.threshold is None:
            return False, 0.0, value
            
        is_anomaly = value > rule.threshold
        confidence = min(1.0, abs(value - rule.threshold) / rule.threshold) if is_anomaly else 0.0
        
        return is_anomaly, confidence, rule.threshold
        
    def _statistical_detection(
        self,
        rule: DetectionRule,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float, float]:
        """Statistical anomaly detection using z-score"""
        history = self.metric_history.get(metric_name, deque())
        
        if len(history) < rule.window_size:
            return False, 0.0, value
            
        # Get recent values
        recent_values = [
            h['value'] for h in list(history)[-rule.window_size:]
        ]
        
        # Calculate statistics
        mean = statistics.mean(recent_values)
        stdev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        if stdev == 0:
            return False, 0.0, mean
            
        # Calculate z-score
        z_score = abs((value - mean) / stdev)
        
        # Determine if anomaly based on sensitivity
        z_threshold = {
            0.99: 3.0,  # 99% confidence
            0.98: 2.5,
            0.97: 2.3,
            0.95: 2.0,
            0.90: 1.65
        }.get(rule.sensitivity, 2.0)
        
        is_anomaly = z_score > z_threshold
        confidence = min(1.0, z_score / 4.0) if is_anomaly else 0.0
        
        return is_anomaly, confidence, mean
        
    def _pattern_detection(
        self,
        rule: DetectionRule,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float, float]:
        """Pattern-based anomaly detection"""
        history = self.metric_history.get(metric_name, deque())
        
        if len(history) < rule.window_size:
            return False, 0.0, value
            
        # Look for specific patterns
        recent_values = [
            h['value'] for h in list(history)[-rule.window_size:]
        ]
        
        # Detect sudden spikes
        if len(recent_values) >= 3:
            avg_recent = statistics.mean(recent_values[-3:])
            avg_previous = statistics.mean(recent_values[:-3])
            
            if avg_previous > 0:
                spike_ratio = avg_recent / avg_previous
                if spike_ratio > 2.0:  # Sudden 2x increase
                    return True, min(1.0, spike_ratio / 3.0), avg_previous
                    
        # Detect gradual increase
        if len(recent_values) >= 10:
            # Check if values are consistently increasing
            increasing_count = sum(
                1 for i in range(1, len(recent_values))
                if recent_values[i] > recent_values[i-1]
            )
            
            if increasing_count > len(recent_values) * 0.8:  # 80% increasing
                expected = recent_values[0]
                confidence = increasing_count / len(recent_values)
                return True, confidence, expected
                
        return False, 0.0, value
        
    async def _predictive_detection(
        self,
        rule: DetectionRule,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float, float]:
        """Predictive anomaly detection using forecasting"""
        history = self.metric_history.get(metric_name, deque())
        
        if len(history) < rule.window_size * 2:
            return False, 0.0, value
            
        # Simple linear regression for prediction
        recent_data = list(history)[-rule.window_size*2:]
        times = list(range(len(recent_data)))
        values = [d['value'] for d in recent_data]
        
        # Calculate linear regression
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(times, values))
        sum_x2 = sum(x * x for x in times)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return False, 0.0, value
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        predicted = slope * n + intercept
        
        # Calculate prediction error
        error = abs(value - predicted)
        avg_value = statistics.mean(values)
        
        if avg_value > 0:
            error_ratio = error / avg_value
            is_anomaly = error_ratio > (1 - rule.sensitivity)
            confidence = min(1.0, error_ratio)
            return is_anomaly, confidence, predicted
            
        return False, 0.0, predicted
        
    async def _composite_detection(
        self,
        rule: DetectionRule,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float, float]:
        """Composite detection using multiple methods"""
        results = []
        
        # Apply multiple detection methods
        methods = [
            self._threshold_detection,
            self._statistical_detection,
            self._pattern_detection,
        ]
        
        for method in methods:
            if method == self._threshold_detection:
                result = method(rule, value)
            else:
                result = method(rule, metric_name, value)
            results.append(result)
            
        # Also apply predictive
        predictive_result = await self._predictive_detection(rule, metric_name, value)
        results.append(predictive_result)
        
        # Combine results (majority voting)
        anomaly_votes = sum(1 for r in results if r[0])
        is_anomaly = anomaly_votes >= len(results) / 2
        
        # Average confidence from methods that detected anomaly
        confidences = [r[1] for r in results if r[0]]
        confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Use the most common expected value
        expected_values = [r[2] for r in results]
        expected = statistics.median(expected_values)
        
        return is_anomaly, confidence, expected
        
    async def _process_anomaly(self, anomaly: Anomaly):
        """Process a detected anomaly"""
        # Store anomaly
        self.anomalies[anomaly.id] = anomaly
        
        # Update detection stats
        stats = self.detection_stats[anomaly.metric_name]
        stats['total_anomalies'] = stats.get('total_anomalies', 0) + 1
        stats['last_anomaly'] = anomaly.timestamp
        stats['avg_confidence'] = (
            stats.get('avg_confidence', 0) * 0.9 + anomaly.confidence * 0.1
        )
        
        logger.warning(
            f"Anomaly detected: {anomaly.metric_name} "
            f"(value={anomaly.current_value:.2f}, "
            f"expected={anomaly.expected_value:.2f}, "
            f"confidence={anomaly.confidence:.2f})"
        )
        
    async def _correlate_anomalies(self):
        """Correlate related anomalies"""
        # Time window for correlation (5 minutes)
        correlation_window = timedelta(minutes=5)
        current_time = datetime.utcnow()
        
        # Get recent anomalies
        recent_anomalies = [
            a for a in self.anomalies.values()
            if current_time - a.timestamp < correlation_window
        ]
        
        # Group by service
        service_anomalies = defaultdict(list)
        for anomaly in recent_anomalies:
            service_anomalies[anomaly.service].append(anomaly)
            
        # Find correlations
        for service, anomalies in service_anomalies.items():
            if len(anomalies) > 1:
                # Link anomalies from same service
                for i, a1 in enumerate(anomalies):
                    for a2 in anomalies[i+1:]:
                        if a2.id not in a1.related_anomalies:
                            a1.related_anomalies.append(a2.id)
                        if a1.id not in a2.related_anomalies:
                            a2.related_anomalies.append(a1.id)
                            
    def get_current_anomalies(self, max_age_minutes: int = 5) -> List[Anomaly]:
        """Get current active anomalies"""
        current_time = datetime.utcnow()
        max_age = timedelta(minutes=max_age_minutes)
        
        return [
            a for a in self.anomalies.values()
            if current_time - a.timestamp < max_age
        ]
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'total_rules': len(self.rules),
            'active_rules': sum(1 for r in self.rules.values() if r.enabled),
            'total_anomalies': len(self.anomalies),
            'current_anomalies': len(self.get_current_anomalies()),
            'metric_stats': dict(self.detection_stats)
        }