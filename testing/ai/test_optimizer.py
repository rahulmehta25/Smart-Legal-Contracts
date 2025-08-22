"""
AI-Powered Test Suite Optimizer

This module provides intelligent test suite optimization using machine learning
to prioritize test execution, reduce execution time, and maximize bug detection.
"""

import ast
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import networkx as nx
import pytest
import coverage


class TestPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"


class OptimizationStrategy(Enum):
    TIME_BASED = "time_based"
    COVERAGE_BASED = "coverage_based"
    RISK_BASED = "risk_based"
    FAILURE_BASED = "failure_based"
    HYBRID = "hybrid"


@dataclass
class TestMetrics:
    """Metrics for a single test."""
    name: str
    file_path: str
    execution_time: float
    success_rate: float
    failure_count: int
    last_failure_date: Optional[datetime]
    code_coverage: float
    lines_covered: Set[str]
    dependencies: Set[str]
    tags: Set[str]
    complexity_score: float
    flakiness_score: float
    maintenance_cost: float


@dataclass
class TestExecution:
    """Result of a test execution."""
    test_name: str
    duration: float
    status: str  # 'passed', 'failed', 'skipped', 'error'
    error_message: Optional[str]
    coverage_data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of test suite optimization."""
    optimized_order: List[str]
    estimated_time_saved: float
    expected_coverage: float
    risk_coverage: float
    strategy_used: OptimizationStrategy
    confidence_score: float
    recommendations: List[str]
    skipped_tests: List[str]


class TestDependencyAnalyzer:
    """Analyzes dependencies between tests."""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.shared_fixtures = defaultdict(set)
        self.data_dependencies = defaultdict(set)
        
    def analyze_dependencies(self, test_files: List[str]) -> Dict[str, Set[str]]:
        """Analyze dependencies between tests."""
        dependencies = defaultdict(set)
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                # Find test functions and their dependencies
                test_functions = self._extract_test_functions(tree)
                file_dependencies = self._analyze_file_dependencies(tree, test_file)
                
                for test_func in test_functions:
                    test_name = f"{test_file}::{test_func}"
                    dependencies[test_name].update(file_dependencies)
                    
            except Exception as e:
                logging.warning(f"Error analyzing dependencies in {test_file}: {e}")
                continue
                
        return dict(dependencies)
        
    def _extract_test_functions(self, tree: ast.AST) -> List[str]:
        """Extract test function names from AST."""
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_functions.append(node.name)
                
        return test_functions
        
    def _analyze_file_dependencies(self, tree: ast.AST, file_path: str) -> Set[str]:
        """Analyze dependencies within a test file."""
        dependencies = set()
        
        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module)
                    
        # Analyze pytest fixtures
        fixtures = self._find_pytest_fixtures(tree)
        dependencies.update(fixtures)
        
        return dependencies
        
    def _find_pytest_fixtures(self, tree: ast.AST) -> Set[str]:
        """Find pytest fixtures used in tests."""
        fixtures = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for pytest.fixture decorator
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == 'fixture') or \
                       (isinstance(decorator, ast.Attribute) and decorator.attr == 'fixture'):
                        fixtures.add(node.name)
                        
                # Check for fixture parameters in test functions
                if node.name.startswith('test_'):
                    for arg in node.args.args:
                        if arg.arg != 'self':  # Skip self parameter
                            fixtures.add(arg.arg)
                            
        return fixtures
        
    def build_dependency_graph(self, dependencies: Dict[str, Set[str]]) -> nx.DiGraph:
        """Build a dependency graph from test dependencies."""
        graph = nx.DiGraph()
        
        # Add nodes
        for test_name in dependencies:
            graph.add_node(test_name)
            
        # Add edges based on dependencies
        for test_name, deps in dependencies.items():
            for dep in deps:
                # Find tests that provide this dependency
                providers = [t for t in dependencies if dep in t or dep.endswith(t.split('::')[-1])]
                for provider in providers:
                    if provider != test_name:
                        graph.add_edge(provider, test_name)
                        
        return graph
        
    def find_optimal_execution_order(self, dependencies: Dict[str, Set[str]]) -> List[str]:
        """Find optimal test execution order based on dependencies."""
        graph = self.build_dependency_graph(dependencies)
        
        try:
            # Topological sort to respect dependencies
            return list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # If there are cycles, use approximation
            logging.warning("Circular dependencies detected, using approximation")
            return list(graph.nodes())


class TestCoverageAnalyzer:
    """Analyzes test coverage and code impact."""
    
    def __init__(self):
        self.coverage_data = {}
        self.code_change_impact = defaultdict(set)
        
    def analyze_coverage(self, test_command: str = "python -m pytest --cov=.") -> Dict[str, float]:
        """Analyze code coverage for tests."""
        try:
            # Initialize coverage
            cov = coverage.Coverage()
            cov.start()
            
            # This would integrate with actual test execution
            # For now, return mock data
            mock_coverage = {
                'test_user_validation': 0.85,
                'test_payment_processing': 0.92,
                'test_data_analysis': 0.78,
                'test_ui_components': 0.65,
                'test_api_endpoints': 0.88
            }
            
            return mock_coverage
            
        except Exception as e:
            logging.error(f"Error analyzing coverage: {e}")
            return {}
            
    def calculate_test_value(self, test_name: str, coverage: float, 
                           execution_time: float, failure_rate: float) -> float:
        """Calculate the value score of a test."""
        # Value = (Coverage * Reliability) / Time
        reliability = 1.0 - failure_rate
        time_factor = 1.0 / (1.0 + execution_time)  # Normalize time impact
        
        value_score = (coverage * reliability * time_factor) * 100
        return min(value_score, 100.0)
        
    def identify_redundant_tests(self, test_metrics: List[TestMetrics]) -> List[str]:
        """Identify potentially redundant tests based on coverage overlap."""
        redundant_tests = []
        
        # Group tests by similar coverage patterns
        coverage_vectors = {}
        for test in test_metrics:
            # Convert coverage to vector (simplified)
            coverage_vector = hash(tuple(sorted(test.lines_covered)))
            
            if coverage_vector not in coverage_vectors:
                coverage_vectors[coverage_vector] = []
            coverage_vectors[coverage_vector].append(test.name)
            
        # Identify groups with multiple tests (potential redundancy)
        for tests in coverage_vectors.values():
            if len(tests) > 1:
                # Keep the fastest/most reliable test, mark others as redundant
                sorted_tests = sorted(tests, key=lambda t: (
                    next(tm.execution_time for tm in test_metrics if tm.name == t),
                    -next(tm.success_rate for tm in test_metrics if tm.name == t)
                ))
                redundant_tests.extend(sorted_tests[1:])
                
        return redundant_tests


class TestPerformancePredictor:
    """Predicts test execution times and failure probabilities."""
    
    def __init__(self):
        self.time_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.failure_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, test_metrics: TestMetrics) -> np.ndarray:
        """Extract features for prediction models."""
        features = [
            test_metrics.execution_time,
            test_metrics.success_rate,
            test_metrics.failure_count,
            test_metrics.code_coverage,
            len(test_metrics.lines_covered),
            len(test_metrics.dependencies),
            len(test_metrics.tags),
            test_metrics.complexity_score,
            test_metrics.flakiness_score,
            test_metrics.maintenance_cost,
            # Time-based features
            (datetime.now() - test_metrics.last_failure_date).days if test_metrics.last_failure_date else 365,
        ]
        
        return np.array(features).reshape(1, -1)
        
    def train(self, historical_data: List[Tuple[TestMetrics, float, float]]):
        """Train prediction models on historical test data."""
        if not historical_data:
            return
            
        X = []
        y_time = []
        y_failure = []
        
        for test_metrics, actual_time, failure_prob in historical_data:
            features = self.extract_features(test_metrics)
            X.append(features.flatten())
            y_time.append(actual_time)
            y_failure.append(failure_prob)
            
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.time_model.fit(X_scaled, y_time)
        self.failure_model.fit(X_scaled, y_failure)
        self.is_trained = True
        
        logging.info("Performance prediction models trained successfully")
        
    def predict_execution_time(self, test_metrics: TestMetrics) -> float:
        """Predict test execution time."""
        if not self.is_trained:
            return test_metrics.execution_time  # Fallback to historical average
            
        features = self.extract_features(test_metrics)
        features_scaled = self.scaler.transform(features)
        
        predicted_time = self.time_model.predict(features_scaled)[0]
        return max(predicted_time, 0.1)  # Minimum 0.1 seconds
        
    def predict_failure_probability(self, test_metrics: TestMetrics) -> float:
        """Predict probability of test failure."""
        if not self.is_trained:
            return 1.0 - test_metrics.success_rate  # Fallback to historical rate
            
        features = self.extract_features(test_metrics)
        features_scaled = self.scaler.transform(features)
        
        predicted_prob = self.failure_model.predict(features_scaled)[0]
        return max(0.0, min(predicted_prob, 1.0))  # Clamp to [0, 1]


class FlakinessDetector:
    """Detects and analyzes flaky tests."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.execution_history = defaultdict(deque)
        
    def record_execution(self, test_name: str, execution: TestExecution):
        """Record a test execution result."""
        if len(self.execution_history[test_name]) >= self.window_size:
            self.execution_history[test_name].popleft()
            
        self.execution_history[test_name].append(execution)
        
    def calculate_flakiness_score(self, test_name: str) -> float:
        """Calculate flakiness score for a test (0 = stable, 1 = very flaky)."""
        if test_name not in self.execution_history:
            return 0.0
            
        executions = list(self.execution_history[test_name])
        if len(executions) < 5:  # Need sufficient data
            return 0.0
            
        # Analyze pattern of successes and failures
        results = [1 if exec.status == 'passed' else 0 for exec in executions]
        
        if len(set(results)) == 1:  # All same result
            return 0.0
            
        # Calculate transition frequency
        transitions = sum(1 for i in range(len(results)-1) if results[i] != results[i+1])
        max_transitions = len(results) - 1
        
        if max_transitions == 0:
            return 0.0
            
        transition_rate = transitions / max_transitions
        
        # Consider failure rate
        failure_rate = 1.0 - np.mean(results)
        
        # Flakiness is high when there are many transitions and moderate failure rate
        flakiness = transition_rate * (1.0 - abs(failure_rate - 0.5) * 2)
        
        return min(flakiness, 1.0)
        
    def identify_flaky_tests(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Identify flaky tests above threshold."""
        flaky_tests = []
        
        for test_name in self.execution_history:
            flakiness = self.calculate_flakiness_score(test_name)
            if flakiness > threshold:
                flaky_tests.append((test_name, flakiness))
                
        return sorted(flaky_tests, key=lambda x: x[1], reverse=True)


class TestSuiteOptimizer:
    """Main test suite optimizer that coordinates all optimization strategies."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.dependency_analyzer = TestDependencyAnalyzer()
        self.coverage_analyzer = TestCoverageAnalyzer()
        self.performance_predictor = TestPerformancePredictor()
        self.flakiness_detector = FlakinessDetector()
        
        self.test_metrics = {}
        self.execution_history = []
        self.optimization_cache = {}
        
    def collect_test_metrics(self, test_pattern: str = "test_*.py") -> Dict[str, TestMetrics]:
        """Collect comprehensive metrics for all tests."""
        test_files = list(self.repo_path.glob(f"**/{test_pattern}"))
        
        metrics = {}
        
        for test_file in test_files:
            try:
                file_metrics = self._analyze_test_file(test_file)
                metrics.update(file_metrics)
            except Exception as e:
                logging.error(f"Error analyzing {test_file}: {e}")
                continue
                
        self.test_metrics = metrics
        return metrics
        
    def _analyze_test_file(self, test_file: Path) -> Dict[str, TestMetrics]:
        """Analyze a single test file and extract metrics."""
        metrics = {}
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Find test functions
            test_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node)
                    
            # Analyze each test function
            for func_node in test_functions:
                test_name = f"{test_file.name}::{func_node.name}"
                
                # Calculate complexity
                complexity = self._calculate_test_complexity(func_node)
                
                # Mock other metrics (would be collected from actual runs)
                metrics[test_name] = TestMetrics(
                    name=test_name,
                    file_path=str(test_file),
                    execution_time=np.random.uniform(0.1, 10.0),  # Mock data
                    success_rate=np.random.uniform(0.8, 1.0),
                    failure_count=np.random.randint(0, 5),
                    last_failure_date=None,
                    code_coverage=np.random.uniform(0.5, 1.0),
                    lines_covered=set(f"line_{i}" for i in range(np.random.randint(10, 100))),
                    dependencies=set(),  # Would be filled by dependency analyzer
                    tags=set(['unit', 'integration'][np.random.randint(0, 2)]),
                    complexity_score=complexity,
                    flakiness_score=0.0,  # Would be filled by flakiness detector
                    maintenance_cost=complexity * 0.1
                )
                
        except Exception as e:
            logging.error(f"Error analyzing test file {test_file}: {e}")
            
        return metrics
        
    def _calculate_test_complexity(self, func_node: ast.FunctionDef) -> float:
        """Calculate complexity score for a test function."""
        # Count different types of complexity indicators
        assertions = len([n for n in ast.walk(func_node) if isinstance(n, ast.Assert)])
        calls = len([n for n in ast.walk(func_node) if isinstance(n, ast.Call)])
        branches = len([n for n in ast.walk(func_node) if isinstance(n, (ast.If, ast.For, ast.While))])
        
        # Simple complexity score
        complexity = (assertions * 0.5) + (calls * 0.3) + (branches * 2.0)
        return min(complexity, 100.0)
        
    def optimize_test_suite(self, strategy: OptimizationStrategy = OptimizationStrategy.HYBRID,
                          time_budget: Optional[float] = None,
                          coverage_target: float = 0.8) -> OptimizationResult:
        """Optimize test suite based on specified strategy."""
        
        if not self.test_metrics:
            self.collect_test_metrics()
            
        if strategy == OptimizationStrategy.TIME_BASED:
            return self._optimize_by_time(time_budget)
        elif strategy == OptimizationStrategy.COVERAGE_BASED:
            return self._optimize_by_coverage(coverage_target)
        elif strategy == OptimizationStrategy.RISK_BASED:
            return self._optimize_by_risk()
        elif strategy == OptimizationStrategy.FAILURE_BASED:
            return self._optimize_by_failure_probability()
        else:  # HYBRID
            return self._optimize_hybrid(time_budget, coverage_target)
            
    def _optimize_by_time(self, time_budget: Optional[float]) -> OptimizationResult:
        """Optimize test suite to fit within time budget."""
        tests = list(self.test_metrics.values())
        
        if time_budget is None:
            time_budget = sum(t.execution_time for t in tests) * 0.7  # 30% time savings
            
        # Sort by value per time unit
        value_scores = []
        for test in tests:
            coverage_value = test.code_coverage
            reliability_value = test.success_rate
            time_cost = test.execution_time
            
            value_per_time = (coverage_value * reliability_value) / max(time_cost, 0.1)
            value_scores.append((test.name, value_per_time, time_cost))
            
        # Greedy selection within time budget
        value_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_tests = []
        current_time = 0.0
        
        for test_name, value, time_cost in value_scores:
            if current_time + time_cost <= time_budget:
                selected_tests.append(test_name)
                current_time += time_cost
                
        # Calculate metrics
        total_original_time = sum(t.execution_time for t in tests)
        time_saved = total_original_time - current_time
        coverage = np.mean([self.test_metrics[t].code_coverage for t in selected_tests])
        
        skipped_tests = [t.name for t in tests if t.name not in selected_tests]
        
        return OptimizationResult(
            optimized_order=selected_tests,
            estimated_time_saved=time_saved,
            expected_coverage=coverage,
            risk_coverage=0.8,  # Placeholder
            strategy_used=OptimizationStrategy.TIME_BASED,
            confidence_score=0.85,
            recommendations=[
                f"Selected {len(selected_tests)} tests out of {len(tests)}",
                f"Estimated time savings: {time_saved:.1f} seconds",
                "Consider running skipped tests in nightly builds"
            ],
            skipped_tests=skipped_tests
        )
        
    def _optimize_by_coverage(self, coverage_target: float) -> OptimizationResult:
        """Optimize test suite to achieve target coverage with minimal tests."""
        tests = list(self.test_metrics.values())
        
        # Greedy set cover approximation
        covered_lines = set()
        selected_tests = []
        target_lines = set()
        
        # Calculate total lines to cover
        for test in tests:
            target_lines.update(test.lines_covered)
            
        target_line_count = int(len(target_lines) * coverage_target)
        
        # Greedy selection
        remaining_tests = tests.copy()
        
        while len(covered_lines) < target_line_count and remaining_tests:
            # Find test that covers most uncovered lines
            best_test = None
            best_new_coverage = 0
            
            for test in remaining_tests:
                new_lines = test.lines_covered - covered_lines
                if len(new_lines) > best_new_coverage:
                    best_new_coverage = len(new_lines)
                    best_test = test
                    
            if best_test:
                selected_tests.append(best_test.name)
                covered_lines.update(best_test.lines_covered)
                remaining_tests.remove(best_test)
            else:
                break
                
        # Calculate results
        total_time = sum(self.test_metrics[t].execution_time for t in selected_tests)
        original_time = sum(t.execution_time for t in tests)
        time_saved = original_time - total_time
        actual_coverage = len(covered_lines) / len(target_lines) if target_lines else 0
        
        skipped_tests = [t.name for t in tests if t.name not in selected_tests]
        
        return OptimizationResult(
            optimized_order=selected_tests,
            estimated_time_saved=time_saved,
            expected_coverage=actual_coverage,
            risk_coverage=0.9,
            strategy_used=OptimizationStrategy.COVERAGE_BASED,
            confidence_score=0.9,
            recommendations=[
                f"Achieved {actual_coverage:.1%} coverage with {len(selected_tests)} tests",
                f"Time savings: {time_saved:.1f} seconds",
                "Consider optimizing skipped tests for future inclusion"
            ],
            skipped_tests=skipped_tests
        )
        
    def _optimize_by_risk(self) -> OptimizationResult:
        """Optimize test suite based on risk-weighted priorities."""
        tests = list(self.test_metrics.values())
        
        # Calculate risk scores
        risk_scores = []
        for test in tests:
            # Risk factors
            failure_risk = 1.0 - test.success_rate
            complexity_risk = test.complexity_score / 100.0
            maintenance_risk = test.maintenance_cost
            flakiness_risk = test.flakiness_score
            
            # Weighted risk score
            total_risk = (failure_risk * 0.4 + 
                         complexity_risk * 0.3 + 
                         maintenance_risk * 0.2 + 
                         flakiness_risk * 0.1)
                         
            risk_scores.append((test.name, total_risk))
            
        # Sort by risk (highest first)
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 80% by risk
        num_to_select = int(len(risk_scores) * 0.8)
        selected_tests = [name for name, risk in risk_scores[:num_to_select]]
        
        # Calculate metrics
        selected_time = sum(self.test_metrics[t].execution_time for t in selected_tests)
        total_time = sum(t.execution_time for t in tests)
        time_saved = total_time - selected_time
        
        coverage = np.mean([self.test_metrics[t].code_coverage for t in selected_tests])
        
        skipped_tests = [name for name, risk in risk_scores[num_to_select:]]
        
        return OptimizationResult(
            optimized_order=selected_tests,
            estimated_time_saved=time_saved,
            expected_coverage=coverage,
            risk_coverage=0.95,
            strategy_used=OptimizationStrategy.RISK_BASED,
            confidence_score=0.88,
            recommendations=[
                "Prioritized high-risk tests for early detection",
                f"Focused on {num_to_select} highest-risk tests",
                "Consider addressing root causes of high-risk tests"
            ],
            skipped_tests=skipped_tests
        )
        
    def _optimize_by_failure_probability(self) -> OptimizationResult:
        """Optimize based on predicted failure probability."""
        tests = list(self.test_metrics.values())
        
        # Predict failure probabilities
        failure_probs = []
        for test in tests:
            if self.performance_predictor.is_trained:
                failure_prob = self.performance_predictor.predict_failure_probability(test)
            else:
                failure_prob = 1.0 - test.success_rate
                
            failure_probs.append((test.name, failure_prob))
            
        # Sort by failure probability (highest first)
        failure_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Select tests with significant failure probability
        selected_tests = [name for name, prob in failure_probs if prob > 0.1]
        
        # If too few tests, add stable tests for coverage
        if len(selected_tests) < len(tests) * 0.5:
            stable_tests = [name for name, prob in failure_probs if prob <= 0.1]
            selected_tests.extend(stable_tests[:len(tests)//3])
            
        # Calculate metrics
        selected_time = sum(self.test_metrics[t].execution_time for t in selected_tests)
        total_time = sum(t.execution_time for t in tests)
        time_saved = total_time - selected_time
        
        coverage = np.mean([self.test_metrics[t].code_coverage for t in selected_tests])
        
        skipped_tests = [t.name for t in tests if t.name not in selected_tests]
        
        return OptimizationResult(
            optimized_order=selected_tests,
            estimated_time_saved=time_saved,
            expected_coverage=coverage,
            risk_coverage=0.92,
            strategy_used=OptimizationStrategy.FAILURE_BASED,
            confidence_score=0.82,
            recommendations=[
                "Prioritized tests with high failure probability",
                "Added stable tests for coverage balance",
                "Consider investigating root causes of frequent failures"
            ],
            skipped_tests=skipped_tests
        )
        
    def _optimize_hybrid(self, time_budget: Optional[float], coverage_target: float) -> OptimizationResult:
        """Hybrid optimization combining multiple strategies."""
        tests = list(self.test_metrics.values())
        
        # Calculate combined scores
        combined_scores = []
        
        for test in tests:
            # Time efficiency
            time_score = 1.0 / (1.0 + test.execution_time)
            
            # Coverage value
            coverage_score = test.code_coverage
            
            # Risk/failure importance
            failure_prob = 1.0 - test.success_rate
            risk_score = failure_prob + (test.complexity_score / 100.0)
            
            # Stability (inverse of flakiness)
            stability_score = 1.0 - test.flakiness_score
            
            # Combined weighted score
            combined_score = (time_score * 0.2 + 
                            coverage_score * 0.3 + 
                            risk_score * 0.3 + 
                            stability_score * 0.2)
                            
            combined_scores.append((test.name, combined_score, test.execution_time))
            
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply time budget if specified
        if time_budget:
            selected_tests = []
            current_time = 0.0
            
            for test_name, score, time_cost in combined_scores:
                if current_time + time_cost <= time_budget:
                    selected_tests.append(test_name)
                    current_time += time_cost
        else:
            # Select top 80% by score
            num_to_select = int(len(combined_scores) * 0.8)
            selected_tests = [name for name, score, time in combined_scores[:num_to_select]]
            current_time = sum(time for name, score, time in combined_scores[:num_to_select])
            
        # Calculate metrics
        total_time = sum(t.execution_time for t in tests)
        time_saved = total_time - current_time
        
        coverage = np.mean([self.test_metrics[t].code_coverage for t in selected_tests])
        
        skipped_tests = [t.name for t in tests if t.name not in selected_tests]
        
        return OptimizationResult(
            optimized_order=selected_tests,
            estimated_time_saved=time_saved,
            expected_coverage=coverage,
            risk_coverage=0.9,
            strategy_used=OptimizationStrategy.HYBRID,
            confidence_score=0.9,
            recommendations=[
                "Used hybrid optimization balancing multiple factors",
                f"Achieved {coverage:.1%} coverage with {len(selected_tests)} tests",
                f"Estimated time savings: {time_saved:.1f} seconds",
                "Balanced time, coverage, risk, and stability"
            ],
            skipped_tests=skipped_tests
        )
        
    def generate_execution_plan(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """Generate detailed execution plan from optimization result."""
        plan = {
            'strategy': optimization_result.strategy_used.value,
            'total_tests': len(optimization_result.optimized_order),
            'estimated_duration': sum(self.test_metrics[t].execution_time 
                                    for t in optimization_result.optimized_order),
            'expected_coverage': optimization_result.expected_coverage,
            'confidence': optimization_result.confidence_score,
            'phases': []
        }
        
        # Group tests into phases for parallel execution
        test_groups = self._group_tests_for_parallel_execution(optimization_result.optimized_order)
        
        for i, group in enumerate(test_groups):
            phase = {
                'phase': i + 1,
                'tests': group,
                'estimated_duration': max(self.test_metrics[t].execution_time for t in group),
                'parallel_execution': len(group) > 1
            }
            plan['phases'].append(phase)
            
        return plan
        
    def _group_tests_for_parallel_execution(self, test_order: List[str]) -> List[List[str]]:
        """Group tests for parallel execution considering dependencies."""
        # Simplified grouping - in practice would consider actual dependencies
        groups = []
        current_group = []
        max_group_size = 4  # Maximum parallel tests
        
        for test in test_order:
            current_group.append(test)
            
            if len(current_group) >= max_group_size:
                groups.append(current_group)
                current_group = []
                
        if current_group:
            groups.append(current_group)
            
        return groups
        
    def save_optimization_results(self, result: OptimizationResult, filepath: str):
        """Save optimization results for future reference."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': result.strategy_used.value,
            'optimized_order': result.optimized_order,
            'estimated_time_saved': result.estimated_time_saved,
            'expected_coverage': result.expected_coverage,
            'risk_coverage': result.risk_coverage,
            'confidence_score': result.confidence_score,
            'recommendations': result.recommendations,
            'skipped_tests': result.skipped_tests
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Optimization results saved to {filepath}")
        
    def load_optimization_results(self, filepath: str) -> OptimizationResult:
        """Load previously saved optimization results."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return OptimizationResult(
            optimized_order=data['optimized_order'],
            estimated_time_saved=data['estimated_time_saved'],
            expected_coverage=data['expected_coverage'],
            risk_coverage=data['risk_coverage'],
            strategy_used=OptimizationStrategy(data['strategy']),
            confidence_score=data['confidence_score'],
            recommendations=data['recommendations'],
            skipped_tests=data['skipped_tests']
        )


if __name__ == "__main__":
    # Example usage
    optimizer = TestSuiteOptimizer("/path/to/repo")
    
    # Collect test metrics
    metrics = optimizer.collect_test_metrics()
    print(f"Analyzed {len(metrics)} tests")
    
    # Optimize test suite
    result = optimizer.optimize_test_suite(
        strategy=OptimizationStrategy.HYBRID,
        time_budget=300.0,  # 5 minutes
        coverage_target=0.85
    )
    
    print(f"Optimization completed:")
    print(f"  Strategy: {result.strategy_used.value}")
    print(f"  Selected tests: {len(result.optimized_order)}")
    print(f"  Time saved: {result.estimated_time_saved:.1f} seconds")
    print(f"  Expected coverage: {result.expected_coverage:.1%}")
    print(f"  Confidence: {result.confidence_score:.1%}")
    
    # Generate execution plan
    plan = optimizer.generate_execution_plan(result)
    print(f"Execution plan: {len(plan['phases'])} phases")
    
    # Save results
    optimizer.save_optimization_results(result, "optimization_results.json")