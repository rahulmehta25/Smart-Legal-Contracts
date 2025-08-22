"""
AI-Driven Performance Testing System

This module provides intelligent performance testing with AI-driven load generation,
anomaly detection, bottleneck identification, and predictive scaling analysis.
"""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import statistics
import psutil
import requests
import aiohttp
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class LoadPattern(Enum):
    CONSTANT = "constant"
    LINEAR_RAMP = "linear_ramp"
    SPIKE = "spike"
    STEP = "step"
    WAVE = "wave"
    RANDOM = "random"
    REALISTIC = "realistic"


class MetricType(Enum):
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE_CONNECTIONS = "database_connections"
    QUEUE_DEPTH = "queue_depth"


class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    TIMEOUT_INCREASE = "timeout_increase"
    CAPACITY_LIMIT = "capacity_limit"


@dataclass
class LoadGenerationConfig:
    """Configuration for load generation."""
    base_url: str
    endpoints: List[Dict[str, Any]]
    load_pattern: LoadPattern
    duration_seconds: int
    max_concurrent_users: int
    ramp_up_time: int = 60
    ramp_down_time: int = 30
    think_time_range: Tuple[float, float] = (1.0, 3.0)
    user_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    auth_config: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific measurement."""
    timestamp: datetime
    response_time: float
    throughput: float
    error_rate: float
    concurrent_users: int
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly."""
    anomaly_type: AnomalyType
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    detected_at: datetime
    affected_metrics: List[str]
    confidence: float
    root_cause_analysis: Dict[str, Any]
    recommendations: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestResult:
    """Complete load test results."""
    test_name: str
    config: LoadGenerationConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    percentiles: Dict[str, float]
    peak_throughput: float
    detected_anomalies: List[PerformanceAnomaly]
    resource_utilization: Dict[str, List[float]]
    bottlenecks: List[Dict[str, Any]]
    scalability_analysis: Dict[str, Any]


class IntelligentLoadGenerator:
    """AI-driven load generation system."""
    
    def __init__(self):
        self.user_behavior_models = {}
        self.historical_patterns = {}
        self.adaptive_algorithms = {
            'response_time_based': self._adapt_by_response_time,
            'error_rate_based': self._adapt_by_error_rate,
            'resource_based': self._adapt_by_resource_usage,
            'ml_prediction': self._adapt_by_ml_prediction
        }
        
    def generate_load_pattern(self, config: LoadGenerationConfig) -> List[int]:
        """Generate intelligent load pattern based on configuration."""
        
        total_seconds = config.duration_seconds
        max_users = config.max_concurrent_users
        
        if config.load_pattern == LoadPattern.CONSTANT:
            return [max_users] * total_seconds
            
        elif config.load_pattern == LoadPattern.LINEAR_RAMP:
            ramp_up = config.ramp_up_time
            steady_state = total_seconds - ramp_up - config.ramp_down_time
            ramp_down = config.ramp_down_time
            
            pattern = []
            # Ramp up
            for i in range(ramp_up):
                users = int((i / ramp_up) * max_users)
                pattern.append(users)
            
            # Steady state
            pattern.extend([max_users] * steady_state)
            
            # Ramp down
            for i in range(ramp_down):
                users = int(max_users * (1 - i / ramp_down))
                pattern.append(users)
                
            return pattern
            
        elif config.load_pattern == LoadPattern.SPIKE:
            base_load = max_users // 4
            spike_duration = min(60, total_seconds // 10)
            spike_start = total_seconds // 2
            
            pattern = [base_load] * total_seconds
            
            for i in range(spike_start, min(spike_start + spike_duration, total_seconds)):
                pattern[i] = max_users
                
            return pattern
            
        elif config.load_pattern == LoadPattern.STEP:
            step_duration = total_seconds // 5
            steps = [max_users // 5 * i for i in range(1, 6)]
            
            pattern = []
            for step in steps:
                pattern.extend([step] * step_duration)
                
            # Trim to exact duration
            return pattern[:total_seconds]
            
        elif config.load_pattern == LoadPattern.WAVE:
            pattern = []
            for i in range(total_seconds):
                # Sine wave pattern
                users = int(max_users * (0.5 + 0.5 * np.sin(2 * np.pi * i / 300)))
                pattern.append(users)
                
            return pattern
            
        elif config.load_pattern == LoadPattern.RANDOM:
            return [random.randint(1, max_users) for _ in range(total_seconds)]
            
        elif config.load_pattern == LoadPattern.REALISTIC:
            return self._generate_realistic_pattern(config)
            
        return [max_users] * total_seconds
        
    def _generate_realistic_pattern(self, config: LoadGenerationConfig) -> List[int]:
        """Generate realistic user load pattern based on typical usage."""
        
        # Simulate realistic daily usage patterns
        total_seconds = config.duration_seconds
        max_users = config.max_concurrent_users
        
        pattern = []
        
        # Business hours simulation (9 AM to 6 PM peak)
        for i in range(total_seconds):
            # Simulate hour of day (0-23)
            hour = (i // 3600) % 24
            
            # Peak hours multiplier
            if 9 <= hour <= 17:  # Business hours
                peak_multiplier = 1.0
                if 11 <= hour <= 14:  # Lunch time peak
                    peak_multiplier = 1.5
            elif 18 <= hour <= 21:  # Evening usage
                peak_multiplier = 0.7
            else:  # Off hours
                peak_multiplier = 0.2
                
            # Add randomness
            noise = random.uniform(0.8, 1.2)
            
            users = int(max_users * peak_multiplier * noise)
            pattern.append(max(1, min(users, max_users)))
            
        return pattern
        
    def create_user_scenarios(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create intelligent user scenarios based on endpoints."""
        
        scenarios = []
        
        # Basic scenario: Browse and interact
        browse_scenario = {
            'name': 'browser_user',
            'weight': 0.6,  # 60% of users
            'actions': []
        }
        
        # Add browsing actions
        for endpoint in endpoints:
            if endpoint.get('method', 'GET') == 'GET':
                browse_scenario['actions'].append({
                    'action': 'request',
                    'endpoint': endpoint,
                    'think_time': random.uniform(2, 5)
                })
                
        scenarios.append(browse_scenario)
        
        # Power user scenario: More interactions
        power_user_scenario = {
            'name': 'power_user',
            'weight': 0.3,  # 30% of users
            'actions': []
        }
        
        # Add all endpoint interactions
        for endpoint in endpoints:
            power_user_scenario['actions'].append({
                'action': 'request',
                'endpoint': endpoint,
                'think_time': random.uniform(1, 3)
            })
            
        scenarios.append(power_user_scenario)
        
        # API user scenario: Fast, no think time
        api_scenario = {
            'name': 'api_user',
            'weight': 0.1,  # 10% of users
            'actions': []
        }
        
        for endpoint in endpoints:
            api_scenario['actions'].append({
                'action': 'request',
                'endpoint': endpoint,
                'think_time': random.uniform(0.1, 0.5)
            })
            
        scenarios.append(api_scenario)
        
        return scenarios
        
    def _adapt_by_response_time(self, current_metrics: PerformanceMetrics, 
                              current_load: int) -> int:
        """Adapt load based on response time."""
        
        target_response_time = 1000  # 1 second
        
        if current_metrics.response_time > target_response_time * 2:
            # Reduce load significantly if response time is too high
            return max(1, int(current_load * 0.7))
        elif current_metrics.response_time > target_response_time:
            # Reduce load slightly
            return max(1, int(current_load * 0.9))
        elif current_metrics.response_time < target_response_time * 0.5:
            # Increase load if response time is very good
            return int(current_load * 1.1)
            
        return current_load  # Keep current load
        
    def _adapt_by_error_rate(self, current_metrics: PerformanceMetrics, 
                           current_load: int) -> int:
        """Adapt load based on error rate."""
        
        target_error_rate = 0.01  # 1% error rate threshold
        
        if current_metrics.error_rate > target_error_rate * 5:
            return max(1, int(current_load * 0.5))  # Reduce load dramatically
        elif current_metrics.error_rate > target_error_rate:
            return max(1, int(current_load * 0.8))  # Reduce load
        elif current_metrics.error_rate < target_error_rate * 0.1:
            return int(current_load * 1.05)  # Slight increase
            
        return current_load
        
    def _adapt_by_resource_usage(self, current_metrics: PerformanceMetrics, 
                               current_load: int) -> int:
        """Adapt load based on resource usage."""
        
        cpu_threshold = 80.0  # 80% CPU usage
        memory_threshold = 85.0  # 85% memory usage
        
        if (current_metrics.cpu_usage > cpu_threshold or 
            current_metrics.memory_usage > memory_threshold):
            return max(1, int(current_load * 0.8))
        elif (current_metrics.cpu_usage < cpu_threshold * 0.5 and 
              current_metrics.memory_usage < memory_threshold * 0.5):
            return int(current_load * 1.1)
            
        return current_load
        
    def _adapt_by_ml_prediction(self, current_metrics: PerformanceMetrics, 
                              current_load: int) -> int:
        """Adapt load using ML predictions."""
        
        # This would use a trained model to predict optimal load
        # For now, use a simple heuristic
        
        # Composite score based on multiple metrics
        response_score = min(1.0, 1000 / max(current_metrics.response_time, 1))
        error_score = max(0.0, 1.0 - current_metrics.error_rate * 100)
        resource_score = max(0.0, 1.0 - (current_metrics.cpu_usage + 
                                        current_metrics.memory_usage) / 200)
        
        composite_score = (response_score + error_score + resource_score) / 3
        
        if composite_score > 0.8:
            return int(current_load * 1.1)  # Increase load
        elif composite_score < 0.5:
            return max(1, int(current_load * 0.8))  # Decrease load
            
        return current_load


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_baseline_established = False
        self.baseline_metrics = {}
        
    def collect_metrics(self, session_stats: Dict[str, Any], 
                       system_stats: Optional[Dict[str, float]] = None) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        
        # System metrics
        if system_stats is None:
            system_stats = self._collect_system_metrics()
            
        # Calculate response time statistics
        response_times = session_stats.get('response_times', [])
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Calculate throughput
        duration = session_stats.get('duration', 1)
        total_requests = session_stats.get('total_requests', 0)
        throughput = total_requests / duration if duration > 0 else 0
        
        # Calculate error rate
        failed_requests = session_stats.get('failed_requests', 0)
        error_rate = failed_requests / max(total_requests, 1)
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            concurrent_users=session_stats.get('concurrent_users', 0),
            cpu_usage=system_stats.get('cpu_percent', 0),
            memory_usage=system_stats.get('memory_percent', 0),
            network_io=system_stats.get('network_io', 0),
            disk_io=system_stats.get('disk_io', 0)
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Establish baseline if not done
        if not self.is_baseline_established and len(self.metrics_history) > 100:
            self._establish_baseline()
            
        return metrics
        
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_total = (disk_io.read_bytes + disk_io.write_bytes) if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_total = (network_io.bytes_sent + network_io.bytes_recv) if network_io else 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_io': disk_io_total,
                'network_io': network_io_total
            }
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return {'cpu_percent': 0, 'memory_percent': 0, 'disk_io': 0, 'network_io': 0}
            
    def _establish_baseline(self):
        """Establish performance baseline from initial metrics."""
        
        if len(self.metrics_history) < 50:
            return
            
        # Use first 50 metrics as baseline
        baseline_data = list(self.metrics_history)[:50]
        
        self.baseline_metrics = {
            'response_time': {
                'mean': np.mean([m.response_time for m in baseline_data]),
                'std': np.std([m.response_time for m in baseline_data]),
                'percentiles': np.percentile([m.response_time for m in baseline_data], [50, 90, 95, 99])
            },
            'throughput': {
                'mean': np.mean([m.throughput for m in baseline_data]),
                'std': np.std([m.throughput for m in baseline_data])
            },
            'error_rate': {
                'mean': np.mean([m.error_rate for m in baseline_data]),
                'std': np.std([m.error_rate for m in baseline_data])
            },
            'cpu_usage': {
                'mean': np.mean([m.cpu_usage for m in baseline_data]),
                'std': np.std([m.cpu_usage for m in baseline_data])
            },
            'memory_usage': {
                'mean': np.mean([m.memory_usage for m in baseline_data]),
                'std': np.std([m.memory_usage for m in baseline_data])
            }
        }
        
        self.is_baseline_established = True
        logging.info("Performance baseline established")
        
    def detect_anomalies(self, current_metrics: PerformanceMetrics) -> List[PerformanceAnomaly]:
        """Detect performance anomalies using multiple techniques."""
        
        anomalies = []
        
        # Statistical anomaly detection
        stat_anomalies = self._detect_statistical_anomalies(current_metrics)
        anomalies.extend(stat_anomalies)
        
        # ML-based anomaly detection
        if len(self.metrics_history) > 100:
            ml_anomalies = self._detect_ml_anomalies(current_metrics)
            anomalies.extend(ml_anomalies)
            
        # Rule-based anomaly detection
        rule_anomalies = self._detect_rule_based_anomalies(current_metrics)
        anomalies.extend(rule_anomalies)
        
        # Trend-based anomaly detection
        trend_anomalies = self._detect_trend_anomalies()
        anomalies.extend(trend_anomalies)
        
        return anomalies
        
    def _detect_statistical_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAnomaly]:
        """Detect anomalies using statistical methods."""
        
        anomalies = []
        
        if not self.is_baseline_established:
            return anomalies
            
        # Response time anomalies
        rt_mean = self.baseline_metrics['response_time']['mean']
        rt_std = self.baseline_metrics['response_time']['std']
        
        if metrics.response_time > rt_mean + 3 * rt_std:
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity='high',
                description=f"Response time {metrics.response_time:.2f}ms significantly higher than baseline {rt_mean:.2f}ms",
                detected_at=metrics.timestamp,
                affected_metrics=['response_time'],
                confidence=0.9,
                root_cause_analysis={'metric': 'response_time', 'deviation': 'high'},
                recommendations=['Investigate server performance', 'Check resource utilization']
            ))
            
        # Error rate anomalies
        er_mean = self.baseline_metrics['error_rate']['mean']
        er_std = self.baseline_metrics['error_rate']['std']
        
        if metrics.error_rate > er_mean + 3 * er_std:
            severity = 'critical' if metrics.error_rate > 0.1 else 'high'
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.ERROR_SPIKE,
                severity=severity,
                description=f"Error rate {metrics.error_rate:.3f} significantly higher than baseline {er_mean:.3f}",
                detected_at=metrics.timestamp,
                affected_metrics=['error_rate'],
                confidence=0.95,
                root_cause_analysis={'metric': 'error_rate', 'deviation': 'high'},
                recommendations=['Check application logs', 'Review recent deployments']
            ))
            
        return anomalies
        
    def _detect_ml_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAnomaly]:
        """Detect anomalies using machine learning."""
        
        anomalies = []
        
        # Prepare feature vector
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        if len(recent_metrics) < 50:
            return anomalies
            
        # Create feature matrix
        features = []
        for m in recent_metrics:
            feature_vector = [
                m.response_time,
                m.throughput,
                m.error_rate,
                m.cpu_usage,
                m.memory_usage
            ]
            features.append(feature_vector)
            
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit anomaly detector
        self.anomaly_detector.fit(features_scaled)
        
        # Check current metrics
        current_features = np.array([[
            metrics.response_time,
            metrics.throughput,
            metrics.error_rate,
            metrics.cpu_usage,
            metrics.memory_usage
        ]])
        
        current_features_scaled = scaler.transform(current_features)
        
        # Predict anomaly
        is_anomaly = self.anomaly_detector.predict(current_features_scaled)[0] == -1
        
        if is_anomaly:
            anomaly_score = self.anomaly_detector.decision_function(current_features_scaled)[0]
            
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity='medium',
                description=f"ML model detected performance anomaly (score: {anomaly_score:.3f})",
                detected_at=metrics.timestamp,
                affected_metrics=['multiple'],
                confidence=0.7,
                root_cause_analysis={'model': 'isolation_forest', 'score': anomaly_score},
                recommendations=['Review all performance metrics', 'Compare with baseline']
            ))
            
        return anomalies
        
    def _detect_rule_based_anomalies(self, metrics: PerformanceMetrics) -> List[PerformanceAnomaly]:
        """Detect anomalies using predefined rules."""
        
        anomalies = []
        
        # High resource utilization
        if metrics.cpu_usage > 90 or metrics.memory_usage > 95:
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.RESOURCE_EXHAUSTION,
                severity='critical',
                description=f"High resource utilization: CPU {metrics.cpu_usage:.1f}%, Memory {metrics.memory_usage:.1f}%",
                detected_at=metrics.timestamp,
                affected_metrics=['cpu_usage', 'memory_usage'],
                confidence=1.0,
                root_cause_analysis={'cpu': metrics.cpu_usage, 'memory': metrics.memory_usage},
                recommendations=['Scale up resources', 'Optimize resource usage']
            ))
            
        # Very high error rate
        if metrics.error_rate > 0.2:  # 20% error rate
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.ERROR_SPIKE,
                severity='critical',
                description=f"Critical error rate: {metrics.error_rate:.1%}",
                detected_at=metrics.timestamp,
                affected_metrics=['error_rate'],
                confidence=1.0,
                root_cause_analysis={'error_rate': metrics.error_rate},
                recommendations=['Immediate investigation required', 'Check system health']
            ))
            
        # Very slow response time
        if metrics.response_time > 10000:  # 10 seconds
            anomalies.append(PerformanceAnomaly(
                anomaly_type=AnomalyType.TIMEOUT_INCREASE,
                severity='high',
                description=f"Extremely slow response time: {metrics.response_time:.0f}ms",
                detected_at=metrics.timestamp,
                affected_metrics=['response_time'],
                confidence=1.0,
                root_cause_analysis={'response_time': metrics.response_time},
                recommendations=['Check database performance', 'Review application bottlenecks']
            ))
            
        return anomalies
        
    def _detect_trend_anomalies(self) -> List[PerformanceAnomaly]:
        """Detect anomalies based on trends."""
        
        anomalies = []
        
        if len(self.metrics_history) < 30:
            return anomalies
            
        recent_metrics = list(self.metrics_history)[-30:]  # Last 30 measurements
        
        # Memory leak detection (increasing memory usage)
        memory_usage = [m.memory_usage for m in recent_metrics]
        if len(memory_usage) > 10:
            # Calculate linear regression slope
            x = np.arange(len(memory_usage))
            slope, _, r_value, _, _ = stats.linregress(x, memory_usage)
            
            # If memory is consistently increasing
            if slope > 1.0 and r_value > 0.8:  # Strong positive correlation
                anomalies.append(PerformanceAnomaly(
                    anomaly_type=AnomalyType.MEMORY_LEAK,
                    severity='high',
                    description=f"Potential memory leak detected: Memory usage increasing at {slope:.2f}% per measurement",
                    detected_at=datetime.now(),
                    affected_metrics=['memory_usage'],
                    confidence=r_value,
                    root_cause_analysis={'slope': slope, 'correlation': r_value},
                    recommendations=['Check for memory leaks', 'Monitor garbage collection']
                ))
                
        # Response time degradation
        response_times = [m.response_time for m in recent_metrics]
        if len(response_times) > 10:
            x = np.arange(len(response_times))
            slope, _, r_value, _, _ = stats.linregress(x, response_times)
            
            # If response time is consistently increasing
            if slope > 50 and r_value > 0.7:  # Increasing by 50ms per measurement
                anomalies.append(PerformanceAnomaly(
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    severity='medium',
                    description=f"Response time degradation trend detected: Increasing by {slope:.2f}ms per measurement",
                    detected_at=datetime.now(),
                    affected_metrics=['response_time'],
                    confidence=r_value,
                    root_cause_analysis={'slope': slope, 'correlation': r_value},
                    recommendations=['Investigate performance bottlenecks', 'Check resource scaling']
                ))
                
        return anomalies


class LoadTestExecutor:
    """Executes load tests with real-time monitoring and adaptation."""
    
    def __init__(self):
        self.session = None
        self.metrics_collector = PerformanceMonitor()
        self.load_generator = IntelligentLoadGenerator()
        self.is_running = False
        
    async def execute_load_test(self, config: LoadGenerationConfig) -> LoadTestResult:
        """Execute a complete load test with AI-driven adaptation."""
        
        logging.info(f"Starting load test for {config.base_url}")
        start_time = datetime.now()
        
        # Generate load pattern
        load_pattern = self.load_generator.generate_load_pattern(config)
        
        # Create user scenarios if not provided
        if not config.user_scenarios:
            config.user_scenarios = self.load_generator.create_user_scenarios(config.endpoints)
            
        # Initialize session statistics
        session_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'start_time': start_time,
            'duration': 0,
            'concurrent_users': 0
        }
        
        # Track all metrics
        all_metrics = []
        detected_anomalies = []
        resource_utilization = defaultdict(list)
        
        self.is_running = True
        
        try:
            # Execute load test
            async with aiohttp.ClientSession() as session:
                self.session = session
                
                for second, target_users in enumerate(load_pattern):
                    if not self.is_running:
                        break
                        
                    # Collect current metrics
                    session_stats['duration'] = second + 1
                    session_stats['concurrent_users'] = target_users
                    
                    current_metrics = self.metrics_collector.collect_metrics(session_stats)
                    all_metrics.append(current_metrics)
                    
                    # Store resource utilization
                    resource_utilization['cpu'].append(current_metrics.cpu_usage)
                    resource_utilization['memory'].append(current_metrics.memory_usage)
                    resource_utilization['network'].append(current_metrics.network_io)
                    
                    # Detect anomalies
                    anomalies = self.metrics_collector.detect_anomalies(current_metrics)
                    detected_anomalies.extend(anomalies)
                    
                    # Log critical anomalies
                    for anomaly in anomalies:
                        if anomaly.severity == 'critical':
                            logging.warning(f"Critical anomaly detected: {anomaly.description}")
                            
                    # Execute requests for this second
                    await self._execute_requests_for_second(
                        session, config, target_users, session_stats
                    )
                    
                    # Adaptive load adjustment (optional)
                    if len(all_metrics) > 10:  # Have some history
                        adapted_users = self._adapt_load_based_on_metrics(
                            current_metrics, target_users
                        )
                        if adapted_users != target_users:
                            logging.info(f"Adapted load from {target_users} to {adapted_users} users")
                            # Update remaining pattern (simplified)
                            for i in range(second + 1, len(load_pattern)):
                                load_pattern[i] = adapted_users
                                
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                    
        except Exception as e:
            logging.error(f"Error during load test execution: {e}")
            
        finally:
            self.is_running = False
            
        end_time = datetime.now()
        
        # Calculate final statistics
        if session_stats['total_requests'] > 0:
            avg_response_time = np.mean(session_stats['response_times'])
            percentiles = {
                'p50': np.percentile(session_stats['response_times'], 50),
                'p90': np.percentile(session_stats['response_times'], 90),
                'p95': np.percentile(session_stats['response_times'], 95),
                'p99': np.percentile(session_stats['response_times'], 99)
            }
        else:
            avg_response_time = 0
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
            
        # Calculate peak throughput
        if all_metrics:
            peak_throughput = max(m.throughput for m in all_metrics)
        else:
            peak_throughput = 0
            
        # Analyze bottlenecks
        bottlenecks = self._analyze_bottlenecks(all_metrics, detected_anomalies)
        
        # Scalability analysis
        scalability_analysis = self._analyze_scalability(all_metrics, load_pattern)
        
        return LoadTestResult(
            test_name=f"load_test_{int(start_time.timestamp())}",
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_requests=session_stats['total_requests'],
            successful_requests=session_stats['successful_requests'],
            failed_requests=session_stats['failed_requests'],
            average_response_time=avg_response_time,
            percentiles=percentiles,
            peak_throughput=peak_throughput,
            detected_anomalies=detected_anomalies,
            resource_utilization=dict(resource_utilization),
            bottlenecks=bottlenecks,
            scalability_analysis=scalability_analysis
        )
        
    async def _execute_requests_for_second(self, session: aiohttp.ClientSession,
                                         config: LoadGenerationConfig,
                                         target_users: int,
                                         session_stats: Dict[str, Any]):
        """Execute requests for a single second of the load test."""
        
        tasks = []
        
        # Create user tasks
        for user_id in range(target_users):
            # Select user scenario
            scenario = self._select_user_scenario(config.user_scenarios)
            task = asyncio.create_task(
                self._simulate_user_session(session, config, scenario, session_stats)
            )
            tasks.append(task)
            
        # Execute all user sessions concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    def _select_user_scenario(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a user scenario based on weights."""
        
        if not scenarios:
            return {'actions': []}
            
        # Select based on weights
        weights = [s.get('weight', 1.0) for s in scenarios]
        total_weight = sum(weights)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return scenarios[i]
                
        return scenarios[0]  # Fallback
        
    async def _simulate_user_session(self, session: aiohttp.ClientSession,
                                   config: LoadGenerationConfig,
                                   scenario: Dict[str, Any],
                                   session_stats: Dict[str, Any]):
        """Simulate a single user session."""
        
        try:
            for action in scenario.get('actions', []):
                if not self.is_running:
                    break
                    
                if action['action'] == 'request':
                    await self._execute_request(session, config, action['endpoint'], session_stats)
                    
                    # Think time
                    think_time = action.get('think_time', random.uniform(*config.think_time_range))
                    await asyncio.sleep(think_time)
                    
        except Exception as e:
            logging.debug(f"Error in user session: {e}")
            session_stats['failed_requests'] += 1
            
    async def _execute_request(self, session: aiohttp.ClientSession,
                             config: LoadGenerationConfig,
                             endpoint: Dict[str, Any],
                             session_stats: Dict[str, Any]):
        """Execute a single HTTP request."""
        
        url = config.base_url.rstrip('/') + '/' + endpoint.get('path', '').lstrip('/')
        method = endpoint.get('method', 'GET').upper()
        headers = {**config.headers, **endpoint.get('headers', {})}
        
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
                json=endpoint.get('data') if method in ['POST', 'PUT', 'PATCH'] else None
            ) as response:
                await response.read()  # Consume response body
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                session_stats['total_requests'] += 1
                session_stats['response_times'].append(response_time)
                
                if response.status < 400:
                    session_stats['successful_requests'] += 1
                else:
                    session_stats['failed_requests'] += 1
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            session_stats['total_requests'] += 1
            session_stats['failed_requests'] += 1
            session_stats['response_times'].append(response_time)
            logging.debug(f"Request failed: {e}")
            
    def _adapt_load_based_on_metrics(self, current_metrics: PerformanceMetrics,
                                   current_load: int) -> int:
        """Adapt load based on current performance metrics."""
        
        # Use multiple adaptation strategies
        adaptations = []
        
        for strategy_name, strategy_func in self.load_generator.adaptive_algorithms.items():
            try:
                adapted_load = strategy_func(current_metrics, current_load)
                adaptations.append(adapted_load)
            except Exception as e:
                logging.debug(f"Error in adaptation strategy {strategy_name}: {e}")
                
        if adaptations:
            # Use median of all adaptations
            return int(np.median(adaptations))
        else:
            return current_load
            
    def _analyze_bottlenecks(self, metrics: List[PerformanceMetrics],
                           anomalies: List[PerformanceAnomaly]) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks."""
        
        bottlenecks = []
        
        if not metrics:
            return bottlenecks
            
        # CPU bottleneck
        avg_cpu = np.mean([m.cpu_usage for m in metrics])
        max_cpu = max([m.cpu_usage for m in metrics])
        
        if avg_cpu > 70 or max_cpu > 90:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high' if max_cpu > 90 else 'medium',
                'description': f'CPU usage: avg {avg_cpu:.1f}%, max {max_cpu:.1f}%',
                'recommendation': 'Consider CPU scaling or optimization'
            })
            
        # Memory bottleneck
        avg_memory = np.mean([m.memory_usage for m in metrics])
        max_memory = max([m.memory_usage for m in metrics])
        
        if avg_memory > 80 or max_memory > 95:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high' if max_memory > 95 else 'medium',
                'description': f'Memory usage: avg {avg_memory:.1f}%, max {max_memory:.1f}%',
                'recommendation': 'Consider memory scaling or optimization'
            })
            
        # Response time bottleneck
        avg_response_time = np.mean([m.response_time for m in metrics])
        p95_response_time = np.percentile([m.response_time for m in metrics], 95)
        
        if avg_response_time > 2000 or p95_response_time > 5000:
            bottlenecks.append({
                'type': 'response_time',
                'severity': 'high' if p95_response_time > 5000 else 'medium',
                'description': f'Response time: avg {avg_response_time:.0f}ms, p95 {p95_response_time:.0f}ms',
                'recommendation': 'Optimize application performance and database queries'
            })
            
        # Error rate bottleneck
        avg_error_rate = np.mean([m.error_rate for m in metrics])
        max_error_rate = max([m.error_rate for m in metrics])
        
        if avg_error_rate > 0.05 or max_error_rate > 0.2:
            bottlenecks.append({
                'type': 'error_rate',
                'severity': 'critical' if max_error_rate > 0.2 else 'high',
                'description': f'Error rate: avg {avg_error_rate:.1%}, max {max_error_rate:.1%}',
                'recommendation': 'Investigate and fix application errors'
            })
            
        return bottlenecks
        
    def _analyze_scalability(self, metrics: List[PerformanceMetrics],
                           load_pattern: List[int]) -> Dict[str, Any]:
        """Analyze scalability characteristics."""
        
        if len(metrics) < 10:
            return {}
            
        # Correlation between load and performance
        loads = [m.concurrent_users for m in metrics]
        response_times = [m.response_time for m in metrics]
        throughputs = [m.throughput for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        
        analysis = {}
        
        # Response time vs load correlation
        if len(loads) > 1 and len(response_times) > 1:
            rt_correlation = np.corrcoef(loads, response_times)[0, 1]
            analysis['response_time_scalability'] = {
                'correlation_with_load': rt_correlation,
                'assessment': self._assess_scalability(rt_correlation, 'response_time')
            }
            
        # Throughput vs load correlation
        if len(loads) > 1 and len(throughputs) > 1:
            tp_correlation = np.corrcoef(loads, throughputs)[0, 1]
            analysis['throughput_scalability'] = {
                'correlation_with_load': tp_correlation,
                'assessment': self._assess_scalability(tp_correlation, 'throughput')
            }
            
        # Error rate vs load correlation
        if len(loads) > 1 and len(error_rates) > 1:
            er_correlation = np.corrcoef(loads, error_rates)[0, 1]
            analysis['error_rate_scalability'] = {
                'correlation_with_load': er_correlation,
                'assessment': self._assess_scalability(er_correlation, 'error_rate')
            }
            
        # Breaking point analysis
        breaking_point = self._find_breaking_point(metrics)
        if breaking_point:
            analysis['breaking_point'] = breaking_point
            
        return analysis
        
    def _assess_scalability(self, correlation: float, metric_type: str) -> str:
        """Assess scalability based on correlation."""
        
        if metric_type == 'response_time':
            if correlation < 0.3:
                return 'excellent'  # Low correlation with load
            elif correlation < 0.6:
                return 'good'
            elif correlation < 0.8:
                return 'fair'
            else:
                return 'poor'  # High correlation means response time increases with load
                
        elif metric_type == 'throughput':
            if correlation > 0.8:
                return 'excellent'  # High positive correlation
            elif correlation > 0.5:
                return 'good'
            elif correlation > 0.2:
                return 'fair'
            else:
                return 'poor'
                
        elif metric_type == 'error_rate':
            if correlation < 0.2:
                return 'excellent'  # Low correlation with load
            elif correlation < 0.5:
                return 'good'
            elif correlation < 0.8:
                return 'fair'
            else:
                return 'poor'
                
        return 'unknown'
        
    def _find_breaking_point(self, metrics: List[PerformanceMetrics]) -> Optional[Dict[str, Any]]:
        """Find the point where performance significantly degrades."""
        
        if len(metrics) < 20:
            return None
            
        # Look for sudden increases in response time or error rate
        for i in range(10, len(metrics) - 5):
            window_before = metrics[i-10:i]
            window_after = metrics[i:i+5]
            
            # Response time increase
            avg_rt_before = np.mean([m.response_time for m in window_before])
            avg_rt_after = np.mean([m.response_time for m in window_after])
            
            if avg_rt_after > avg_rt_before * 2:  # 2x increase
                return {
                    'load_level': metrics[i].concurrent_users,
                    'metric': 'response_time',
                    'before_value': avg_rt_before,
                    'after_value': avg_rt_after,
                    'degradation_factor': avg_rt_after / avg_rt_before
                }
                
            # Error rate increase
            avg_er_before = np.mean([m.error_rate for m in window_before])
            avg_er_after = np.mean([m.error_rate for m in window_after])
            
            if avg_er_after > avg_er_before + 0.1:  # 10% increase
                return {
                    'load_level': metrics[i].concurrent_users,
                    'metric': 'error_rate',
                    'before_value': avg_er_before,
                    'after_value': avg_er_after,
                    'degradation_factor': (avg_er_after - avg_er_before)
                }
                
        return None


class PerformanceReportGenerator:
    """Generates comprehensive performance test reports."""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'detailed_analysis': self._generate_detailed_analysis,
            'charts_and_graphs': self._generate_visualizations,
            'recommendations': self._generate_recommendations
        }
        
    def generate_report(self, result: LoadTestResult, output_dir: str = "performance_reports/"):
        """Generate comprehensive performance test report."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate JSON report
        self._save_json_report(result, output_path / "test_results.json")
        
        # Generate visualizations
        self._generate_visualizations(result, output_path)
        
        # Generate HTML report
        self._generate_html_report(result, output_path / "performance_report.html")
        
        logging.info(f"Performance report generated in {output_path}")
        
    def _save_json_report(self, result: LoadTestResult, filepath: Path):
        """Save test results as JSON."""
        
        report_data = {
            'test_summary': {
                'test_name': result.test_name,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration': (result.end_time - result.start_time).total_seconds(),
                'total_requests': result.total_requests,
                'successful_requests': result.successful_requests,
                'failed_requests': result.failed_requests,
                'success_rate': result.successful_requests / max(result.total_requests, 1),
                'average_response_time': result.average_response_time,
                'peak_throughput': result.peak_throughput
            },
            'performance_metrics': {
                'percentiles': result.percentiles,
                'resource_utilization': result.resource_utilization
            },
            'anomalies': [
                {
                    'type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity,
                    'description': anomaly.description,
                    'detected_at': anomaly.detected_at.isoformat(),
                    'confidence': anomaly.confidence,
                    'recommendations': anomaly.recommendations
                }
                for anomaly in result.detected_anomalies
            ],
            'bottlenecks': result.bottlenecks,
            'scalability_analysis': result.scalability_analysis
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
            
    def _generate_visualizations(self, result: LoadTestResult, output_dir: Path):
        """Generate performance visualizations."""
        
        # Set up matplotlib
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Test Results - {result.test_name}', fontsize=16)
        
        # Response time percentiles
        if result.percentiles:
            percentiles = list(result.percentiles.keys())
            values = list(result.percentiles.values())
            
            axes[0, 0].bar(percentiles, values)
            axes[0, 0].set_title('Response Time Percentiles')
            axes[0, 0].set_ylabel('Response Time (ms)')
            
        # Resource utilization over time
        if result.resource_utilization:
            time_points = range(len(result.resource_utilization.get('cpu', [])))
            
            if 'cpu' in result.resource_utilization:
                axes[0, 1].plot(time_points, result.resource_utilization['cpu'], label='CPU %')
            if 'memory' in result.resource_utilization:
                axes[0, 1].plot(time_points, result.resource_utilization['memory'], label='Memory %')
                
            axes[0, 1].set_title('Resource Utilization Over Time')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Utilization %')
            axes[0, 1].legend()
            
        # Success rate pie chart
        success_rate = result.successful_requests / max(result.total_requests, 1)
        failure_rate = 1 - success_rate
        
        axes[1, 0].pie([success_rate, failure_rate], 
                      labels=['Success', 'Failure'],
                      colors=['green', 'red'],
                      autopct='%1.1f%%')
        axes[1, 0].set_title('Request Success Rate')
        
        # Bottlenecks
        if result.bottlenecks:
            bottleneck_types = [b['type'] for b in result.bottlenecks]
            bottleneck_counts = {}
            for bt in bottleneck_types:
                bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1
                
            axes[1, 1].bar(bottleneck_counts.keys(), bottleneck_counts.values())
            axes[1, 1].set_title('Detected Bottlenecks')
            axes[1, 1].set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_html_report(self, result: LoadTestResult, filepath: Path):
        """Generate HTML performance report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report - {result.test_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
                .anomaly {{ margin: 10px 0; padding: 10px; }}
                .critical {{ background-color: #ffebee; border-left: 5px solid red; }}
                .high {{ background-color: #fff3e0; border-left: 5px solid orange; }}
                .medium {{ background-color: #f3e5f5; border-left: 5px solid purple; }}
                .low {{ background-color: #e8f5e8; border-left: 5px solid green; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <h2>{result.test_name}</h2>
                <p>Test Period: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <h3>Total Requests</h3>
                    <p>{result.total_requests:,}</p>
                </div>
                <div class="metric">
                    <h3>Success Rate</h3>
                    <p>{result.successful_requests/max(result.total_requests, 1):.1%}</p>
                </div>
                <div class="metric">
                    <h3>Average Response Time</h3>
                    <p>{result.average_response_time:.0f} ms</p>
                </div>
                <div class="metric">
                    <h3>Peak Throughput</h3>
                    <p>{result.peak_throughput:.1f} req/s</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        # Add percentiles
        for percentile, value in result.percentiles.items():
            html_content += f"<tr><td>Response Time {percentile}</td><td>{value:.0f} ms</td></tr>"
            
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Detected Anomalies</h2>
        """
        
        if result.detected_anomalies:
            for anomaly in result.detected_anomalies:
                css_class = anomaly.severity.lower()
                html_content += f"""
                <div class="anomaly {css_class}">
                    <h3>{anomaly.anomaly_type.value.replace('_', ' ').title()}</h3>
                    <p><strong>Severity:</strong> {anomaly.severity.title()}</p>
                    <p><strong>Description:</strong> {anomaly.description}</p>
                    <p><strong>Confidence:</strong> {anomaly.confidence:.1%}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                """
                
                for recommendation in anomaly.recommendations:
                    html_content += f"<li>{recommendation}</li>"
                    
                html_content += "</ul></div>"
        else:
            html_content += "<p>No significant anomalies detected.</p>"
            
        html_content += """
            </div>
            
            <div class="section">
                <h2>Bottlenecks Analysis</h2>
        """
        
        if result.bottlenecks:
            html_content += "<table><tr><th>Type</th><th>Severity</th><th>Description</th><th>Recommendation</th></tr>"
            
            for bottleneck in result.bottlenecks:
                html_content += f"""
                <tr>
                    <td>{bottleneck['type'].title()}</td>
                    <td>{bottleneck['severity'].title()}</td>
                    <td>{bottleneck['description']}</td>
                    <td>{bottleneck['recommendation']}</td>
                </tr>
                """
                
            html_content += "</table>"
        else:
            html_content += "<p>No significant bottlenecks identified.</p>"
            
        html_content += """
            </div>
            
            <div class="section">
                <h2>Scalability Analysis</h2>
        """
        
        if result.scalability_analysis:
            for metric, analysis in result.scalability_analysis.items():
                if isinstance(analysis, dict) and 'assessment' in analysis:
                    html_content += f"<p><strong>{metric.replace('_', ' ').title()}:</strong> {analysis['assessment'].title()}</p>"
                    
        html_content += """
            </div>
            
            <div class="section">
                <h2>Performance Charts</h2>
                <img src="performance_charts.png" alt="Performance Charts" style="max-width: 100%; height: auto;">
            </div>
            
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)


# Example usage and main execution
if __name__ == "__main__":
    # Example configuration
    config = LoadGenerationConfig(
        base_url="http://localhost:8000",
        endpoints=[
            {"path": "/api/users", "method": "GET"},
            {"path": "/api/products", "method": "GET"},
            {"path": "/api/orders", "method": "POST", "data": {"product_id": 1, "quantity": 2}},
        ],
        load_pattern=LoadPattern.LINEAR_RAMP,
        duration_seconds=300,  # 5 minutes
        max_concurrent_users=100,
        ramp_up_time=60,
        ramp_down_time=30
    )
    
    # Execute load test
    async def run_test():
        executor = LoadTestExecutor()
        result = await executor.execute_load_test(config)
        
        # Generate report
        report_generator = PerformanceReportGenerator()
        report_generator.generate_report(result)
        
        print(f"Load test completed:")
        print(f"  Total requests: {result.total_requests:,}")
        print(f"  Success rate: {result.successful_requests/max(result.total_requests, 1):.1%}")
        print(f"  Average response time: {result.average_response_time:.0f} ms")
        print(f"  Peak throughput: {result.peak_throughput:.1f} req/s")
        print(f"  Anomalies detected: {len(result.detected_anomalies)}")
        print(f"  Bottlenecks identified: {len(result.bottlenecks)}")
        
        return result
    
    # Run the test
    # asyncio.run(run_test())