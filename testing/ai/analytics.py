"""
AI-Powered Test Analytics System

This module provides comprehensive test analytics including coverage prediction,
failure pattern analysis, test effectiveness scoring, and quality gates automation.
"""

import ast
import json
import logging
import os
import pickle
import re
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TestOutcome(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityGate(Enum):
    COVERAGE_THRESHOLD = "coverage_threshold"
    FAILURE_RATE = "failure_rate"
    PERFORMANCE_REGRESSION = "performance_regression"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    CODE_QUALITY = "code_quality"


@dataclass
class TestExecution:
    """Single test execution record."""
    test_id: str
    test_name: str
    test_file: str
    outcome: TestOutcome
    duration: float
    timestamp: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    coverage_data: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteMetrics:
    """Test suite execution metrics."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    coverage_percentage: float
    timestamp: datetime
    execution_id: str


@dataclass
class CoveragePrediction:
    """Coverage prediction for code changes."""
    file_path: str
    predicted_coverage: float
    confidence: float
    risk_assessment: RiskLevel
    recommended_tests: List[str]
    coverage_gaps: List[Dict[str, Any]]
    analysis_metadata: Dict[str, Any]


@dataclass
class FailurePattern:
    """Identified failure pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    affected_tests: List[str]
    root_causes: List[str]
    suggested_fixes: List[str]
    confidence: float
    first_seen: datetime
    last_seen: datetime


@dataclass
class QualityGateResult:
    """Quality gate evaluation result."""
    gate_name: str
    gate_type: QualityGate
    passed: bool
    current_value: float
    threshold: float
    severity: str
    message: str
    recommendations: List[str]


class TestDataCollector:
    """Collects and stores test execution data."""
    
    def __init__(self, db_path: str = "test_analytics.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for test data storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                test_file TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration REAL NOT NULL,
                timestamp TEXT NOT NULL,
                error_message TEXT,
                stack_trace TEXT,
                coverage_data TEXT,
                metadata TEXT
            )
        ''')
        
        # Test suite metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suite_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_name TEXT NOT NULL,
                execution_id TEXT NOT NULL,
                total_tests INTEGER NOT NULL,
                passed_tests INTEGER NOT NULL,
                failed_tests INTEGER NOT NULL,
                skipped_tests INTEGER NOT NULL,
                error_tests INTEGER NOT NULL,
                total_duration REAL NOT NULL,
                coverage_percentage REAL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # Coverage data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                lines_covered INTEGER NOT NULL,
                total_lines INTEGER NOT NULL,
                coverage_percentage REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_test_execution(self, execution: TestExecution):
        """Store a test execution record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_executions (
                test_id, test_name, test_file, outcome, duration, timestamp,
                error_message, stack_trace, coverage_data, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.test_id,
            execution.test_name,
            execution.test_file,
            execution.outcome.value,
            execution.duration,
            execution.timestamp.isoformat(),
            execution.error_message,
            execution.stack_trace,
            json.dumps(execution.coverage_data) if execution.coverage_data else None,
            json.dumps(execution.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def store_suite_metrics(self, metrics: TestSuiteMetrics):
        """Store test suite metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO suite_metrics (
                suite_name, execution_id, total_tests, passed_tests, failed_tests,
                skipped_tests, error_tests, total_duration, coverage_percentage, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.suite_name,
            metrics.execution_id,
            metrics.total_tests,
            metrics.passed_tests,
            metrics.failed_tests,
            metrics.skipped_tests,
            metrics.error_tests,
            metrics.total_duration,
            metrics.coverage_percentage,
            metrics.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_test_history(self, test_id: str = None, 
                        days: int = 30) -> List[TestExecution]:
        """Get test execution history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        if test_id:
            query = '''
                SELECT * FROM test_executions 
                WHERE test_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            '''
            cursor.execute(query, (test_id, cutoff_date))
        else:
            query = '''
                SELECT * FROM test_executions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            '''
            cursor.execute(query, (cutoff_date,))
            
        rows = cursor.fetchall()
        conn.close()
        
        executions = []
        for row in rows:
            executions.append(TestExecution(
                test_id=row[1],
                test_name=row[2],
                test_file=row[3],
                outcome=TestOutcome(row[4]),
                duration=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                error_message=row[7],
                stack_trace=row[8],
                coverage_data=json.loads(row[9]) if row[9] else None,
                metadata=json.loads(row[10]) if row[10] else {}
            ))
            
        return executions
        
    def import_junit_results(self, junit_xml_path: str, execution_id: str):
        """Import test results from JUnit XML format."""
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()
        
        # Parse test cases
        for testcase in root.findall('.//testcase'):
            test_name = testcase.get('name', '')
            classname = testcase.get('classname', '')
            duration = float(testcase.get('time', 0))
            
            # Determine outcome
            if testcase.find('failure') is not None:
                outcome = TestOutcome.FAILED
                error_element = testcase.find('failure')
                error_message = error_element.get('message', '') if error_element is not None else ''
                stack_trace = error_element.text if error_element is not None else ''
            elif testcase.find('error') is not None:
                outcome = TestOutcome.ERROR
                error_element = testcase.find('error')
                error_message = error_element.get('message', '') if error_element is not None else ''
                stack_trace = error_element.text if error_element is not None else ''
            elif testcase.find('skipped') is not None:
                outcome = TestOutcome.SKIPPED
                error_message = None
                stack_trace = None
            else:
                outcome = TestOutcome.PASSED
                error_message = None
                stack_trace = None
                
            # Create test execution record
            execution = TestExecution(
                test_id=f"{classname}.{test_name}",
                test_name=test_name,
                test_file=classname,
                outcome=outcome,
                duration=duration,
                timestamp=datetime.now(),
                error_message=error_message,
                stack_trace=stack_trace,
                metadata={'execution_id': execution_id}
            )
            
            self.store_test_execution(execution)
            
    def import_coverage_report(self, coverage_xml_path: str, execution_id: str):
        """Import coverage data from XML report."""
        tree = ET.parse(coverage_xml_path)
        root = tree.getroot()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Parse coverage data
        for package in root.findall('.//package'):
            for class_elem in package.findall('classes/class'):
                filename = class_elem.get('filename', '')
                
                # Calculate coverage from lines
                lines = class_elem.findall('lines/line')
                total_lines = len(lines)
                covered_lines = len([line for line in lines if line.get('hits', '0') != '0'])
                
                coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                
                cursor.execute('''
                    INSERT INTO coverage_data (
                        execution_id, file_path, lines_covered, total_lines, 
                        coverage_percentage, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    execution_id,
                    filename,
                    covered_lines,
                    total_lines,
                    coverage_percentage,
                    datetime.now().isoformat()
                ))
                
        conn.commit()
        conn.close()


class CoveragePredictor:
    """Predicts test coverage for code changes using ML."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def extract_code_features(self, file_path: str) -> Dict[str, float]:
        """Extract features from a code file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Basic metrics
            features = {
                'file_size': len(content),
                'line_count': len(content.split('\n')),
                'function_count': 0,
                'class_count': 0,
                'import_count': 0,
                'complexity_score': 0,
                'comment_ratio': 0,
                'docstring_ratio': 0,
                'test_assertions': 0
            }
            
            # AST-based analysis
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features['function_count'] += 1
                    # Simple complexity (number of branches)
                    features['complexity_score'] += len([
                        n for n in ast.walk(node) 
                        if isinstance(n, (ast.If, ast.While, ast.For))
                    ])
                elif isinstance(node, ast.ClassDef):
                    features['class_count'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features['import_count'] += 1
                elif isinstance(node, ast.Assert):
                    features['test_assertions'] += 1
                    
            # Text-based analysis
            lines = content.split('\n')
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            features['comment_ratio'] = comment_lines / max(len(lines), 1)
            
            # Docstring analysis
            docstring_count = len([
                node for node in ast.walk(tree) 
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) 
                and ast.get_docstring(node) is not None
            ])
            features['docstring_ratio'] = docstring_count / max(
                features['function_count'] + features['class_count'], 1
            )
            
            return features
            
        except Exception as e:
            logging.warning(f"Error extracting features from {file_path}: {e}")
            return {
                'file_size': 0, 'line_count': 0, 'function_count': 0,
                'class_count': 0, 'import_count': 0, 'complexity_score': 0,
                'comment_ratio': 0, 'docstring_ratio': 0, 'test_assertions': 0
            }
            
    def train(self, training_data: List[Tuple[str, float]]):
        """Train the coverage prediction model."""
        if not training_data:
            logging.warning("No training data provided")
            return
            
        features = []
        targets = []
        
        for file_path, coverage in training_data:
            file_features = self.extract_code_features(file_path)
            feature_vector = list(file_features.values())
            
            features.append(feature_vector)
            targets.append(coverage)
            
        if not features:
            logging.error("No valid features extracted")
            return
            
        # Store feature names
        sample_features = self.extract_code_features(training_data[0][0])
        self.feature_names = list(sample_features.keys())
        
        # Prepare data
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Evaluate model
        scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        logging.info(f"Coverage prediction model trained. CV RMSE: {np.sqrt(-scores.mean()):.3f}")
        
    def predict_coverage(self, file_path: str) -> CoveragePrediction:
        """Predict coverage for a file."""
        if not self.is_trained:
            return CoveragePrediction(
                file_path=file_path,
                predicted_coverage=0.5,
                confidence=0.1,
                risk_assessment=RiskLevel.MEDIUM,
                recommended_tests=[],
                coverage_gaps=[],
                analysis_metadata={'model_status': 'not_trained'}
            )
            
        # Extract features
        features = self.extract_code_features(file_path)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict coverage
        predicted_coverage = self.model.predict(feature_vector_scaled)[0]
        predicted_coverage = max(0, min(predicted_coverage, 1))  # Clamp to [0, 1]
        
        # Calculate confidence (based on feature importance and model uncertainty)
        confidence = self._calculate_prediction_confidence(features)
        
        # Assess risk level
        risk_level = self._assess_risk_level(predicted_coverage, features)
        
        # Generate recommendations
        recommended_tests = self._generate_test_recommendations(features, predicted_coverage)
        
        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(features, predicted_coverage)
        
        return CoveragePrediction(
            file_path=file_path,
            predicted_coverage=predicted_coverage,
            confidence=confidence,
            risk_assessment=risk_level,
            recommended_tests=recommended_tests,
            coverage_gaps=coverage_gaps,
            analysis_metadata={
                'feature_scores': dict(zip(self.feature_names, feature_vector[0])),
                'model_type': 'RandomForest'
            }
        )
        
    def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in prediction."""
        # Simple heuristic based on feature completeness and model reliability
        base_confidence = 0.7
        
        # Adjust based on code complexity
        complexity = features.get('complexity_score', 0)
        if complexity > 20:  # High complexity reduces confidence
            base_confidence -= 0.2
            
        # Adjust based on file size
        line_count = features.get('line_count', 0)
        if line_count < 10 or line_count > 1000:  # Very small or very large files
            base_confidence -= 0.1
            
        return max(0.1, min(base_confidence, 0.95))
        
    def _assess_risk_level(self, predicted_coverage: float, 
                          features: Dict[str, float]) -> RiskLevel:
        """Assess risk level based on predicted coverage and code features."""
        
        # Base risk from coverage
        if predicted_coverage < 0.3:
            base_risk = RiskLevel.CRITICAL
        elif predicted_coverage < 0.5:
            base_risk = RiskLevel.HIGH
        elif predicted_coverage < 0.7:
            base_risk = RiskLevel.MEDIUM
        else:
            base_risk = RiskLevel.LOW
            
        # Adjust based on complexity
        complexity = features.get('complexity_score', 0)
        if complexity > 15 and base_risk == RiskLevel.LOW:
            base_risk = RiskLevel.MEDIUM
        elif complexity > 25 and base_risk == RiskLevel.MEDIUM:
            base_risk = RiskLevel.HIGH
            
        return base_risk
        
    def _generate_test_recommendations(self, features: Dict[str, float], 
                                     predicted_coverage: float) -> List[str]:
        """Generate test recommendations."""
        recommendations = []
        
        # Based on code characteristics
        if features.get('function_count', 0) > 5 and predicted_coverage < 0.6:
            recommendations.append("Add unit tests for individual functions")
            
        if features.get('class_count', 0) > 0 and predicted_coverage < 0.7:
            recommendations.append("Add integration tests for class interactions")
            
        if features.get('complexity_score', 0) > 20:
            recommendations.append("Add edge case tests for complex logic")
            
        if features.get('test_assertions', 0) == 0:
            recommendations.append("This file appears to have no tests - start with basic functionality tests")
            
        return recommendations
        
    def _identify_coverage_gaps(self, features: Dict[str, float], 
                              predicted_coverage: float) -> List[Dict[str, Any]]:
        """Identify potential coverage gaps."""
        gaps = []
        
        if features.get('complexity_score', 0) > 10 and predicted_coverage < 0.8:
            gaps.append({
                'type': 'complex_logic',
                'description': 'Complex logic may not be fully tested',
                'severity': 'high' if predicted_coverage < 0.5 else 'medium'
            })
            
        if features.get('function_count', 0) > 3 and predicted_coverage < 0.6:
            gaps.append({
                'type': 'function_coverage',
                'description': 'Multiple functions may lack individual testing',
                'severity': 'medium'
            })
            
        return gaps


class FailureAnalyzer:
    """Analyzes test failure patterns and root causes."""
    
    def __init__(self):
        self.failure_patterns = {}
        self.pattern_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.vectorizer = None
        
    def analyze_failures(self, test_executions: List[TestExecution]) -> List[FailurePattern]:
        """Analyze failure patterns from test executions."""
        
        failed_tests = [
            execution for execution in test_executions 
            if execution.outcome in [TestOutcome.FAILED, TestOutcome.ERROR]
        ]
        
        if not failed_tests:
            return []
            
        # Group failures by error patterns
        error_groups = self._group_failures_by_error(failed_tests)
        
        patterns = []
        for pattern_id, group in error_groups.items():
            pattern = self._analyze_failure_group(pattern_id, group)
            patterns.append(pattern)
            
        return patterns
        
    def _group_failures_by_error(self, failed_tests: List[TestExecution]) -> Dict[str, List[TestExecution]]:
        """Group failures by similar error messages."""
        groups = defaultdict(list)
        
        for test in failed_tests:
            error_signature = self._extract_error_signature(test)
            groups[error_signature].append(test)
            
        return dict(groups)
        
    def _extract_error_signature(self, test_execution: TestExecution) -> str:
        """Extract a signature from error message for grouping."""
        error_msg = test_execution.error_message or "unknown_error"
        
        # Common error patterns
        patterns = [
            (r'AssertionError.*expected.*but was.*', 'assertion_mismatch'),
            (r'NoSuchElementException', 'element_not_found'),
            (r'TimeoutException', 'timeout'),
            (r'ConnectionError', 'connection_error'),
            (r'AttributeError.*has no attribute', 'attribute_error'),
            (r'KeyError.*', 'key_error'),
            (r'IndexError.*', 'index_error'),
            (r'ValueError.*', 'value_error'),
            (r'TypeError.*', 'type_error'),
        ]
        
        for pattern, signature in patterns:
            if re.search(pattern, error_msg, re.IGNORECASE):
                return signature
                
        # Fallback: use first few words of error
        words = error_msg.split()[:3]
        return '_'.join(words).lower().replace(':', '').replace('.', '')
        
    def _analyze_failure_group(self, pattern_id: str, 
                             failed_tests: List[TestExecution]) -> FailurePattern:
        """Analyze a group of similar failures."""
        
        if not failed_tests:
            return None
            
        # Basic statistics
        frequency = len(failed_tests)
        first_seen = min(test.timestamp for test in failed_tests)
        last_seen = max(test.timestamp for test in failed_tests)
        affected_tests = [test.test_name for test in failed_tests]
        
        # Analyze error messages for common patterns
        error_messages = [test.error_message for test in failed_tests if test.error_message]
        pattern_type = self._classify_pattern_type(pattern_id, error_messages)
        
        # Generate description
        description = self._generate_pattern_description(pattern_type, error_messages)
        
        # Identify root causes
        root_causes = self._identify_root_causes(pattern_type, failed_tests)
        
        # Generate suggested fixes
        suggested_fixes = self._generate_suggested_fixes(pattern_type, root_causes)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(frequency, failed_tests)
        
        return FailurePattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            frequency=frequency,
            affected_tests=affected_tests,
            root_causes=root_causes,
            suggested_fixes=suggested_fixes,
            confidence=confidence,
            first_seen=first_seen,
            last_seen=last_seen
        )
        
    def _classify_pattern_type(self, pattern_id: str, error_messages: List[str]) -> str:
        """Classify the type of failure pattern."""
        
        # Map pattern IDs to types
        type_mapping = {
            'assertion_mismatch': 'Test Logic Error',
            'element_not_found': 'UI Element Issue',
            'timeout': 'Performance/Timing Issue',
            'connection_error': 'Infrastructure Issue',
            'attribute_error': 'Code Structure Change',
            'key_error': 'Data Issue',
            'index_error': 'Data Boundary Issue',
            'value_error': 'Input Validation Issue',
            'type_error': 'API Contract Change'
        }
        
        return type_mapping.get(pattern_id, 'Unknown Pattern')
        
    def _generate_pattern_description(self, pattern_type: str, 
                                    error_messages: List[str]) -> str:
        """Generate a description of the failure pattern."""
        
        if not error_messages:
            return f"Multiple tests failing with {pattern_type}"
            
        # Take a representative error message
        sample_error = error_messages[0][:100] + "..." if len(error_messages[0]) > 100 else error_messages[0]
        
        return f"{pattern_type}: {sample_error}"
        
    def _identify_root_causes(self, pattern_type: str, 
                            failed_tests: List[TestExecution]) -> List[str]:
        """Identify potential root causes."""
        
        root_causes = []
        
        # Pattern-specific root cause analysis
        if pattern_type == 'Test Logic Error':
            root_causes.extend([
                'Expected behavior changed in application',
                'Test assertions need updating',
                'Test data inconsistency'
            ])
        elif pattern_type == 'UI Element Issue':
            root_causes.extend([
                'UI element locators changed',
                'Page loading timing issues',
                'Dynamic content rendering'
            ])
        elif pattern_type == 'Performance/Timing Issue':
            root_causes.extend([
                'Application performance degradation',
                'Network latency issues',
                'Resource contention'
            ])
        elif pattern_type == 'Infrastructure Issue':
            root_causes.extend([
                'Service unavailability',
                'Network connectivity problems',
                'Environment configuration issues'
            ])
        elif pattern_type == 'Code Structure Change':
            root_causes.extend([
                'API changes without test updates',
                'Refactoring broke test assumptions',
                'Missing dependencies'
            ])
        else:
            root_causes.append('Pattern requires manual investigation')
            
        # Check for temporal patterns
        if len(failed_tests) > 1:
            timestamps = [test.timestamp for test in failed_tests]
            time_span = max(timestamps) - min(timestamps)
            
            if time_span.total_seconds() < 3600:  # Within 1 hour
                root_causes.append('Recent deployment or configuration change')
                
        return root_causes
        
    def _generate_suggested_fixes(self, pattern_type: str, 
                                root_causes: List[str]) -> List[str]:
        """Generate suggested fixes."""
        
        fixes = []
        
        if pattern_type == 'Test Logic Error':
            fixes.extend([
                'Review and update test assertions',
                'Verify test data accuracy',
                'Check for application behavior changes'
            ])
        elif pattern_type == 'UI Element Issue':
            fixes.extend([
                'Update element locators',
                'Add explicit waits',
                'Use more robust locator strategies'
            ])
        elif pattern_type == 'Performance/Timing Issue':
            fixes.extend([
                'Increase timeout values',
                'Add performance monitoring',
                'Investigate performance bottlenecks'
            ])
        elif pattern_type == 'Infrastructure Issue':
            fixes.extend([
                'Check service health',
                'Verify environment configuration',
                'Implement retry mechanisms'
            ])
        elif pattern_type == 'Code Structure Change':
            fixes.extend([
                'Update test code to match API changes',
                'Review recent code changes',
                'Add missing dependencies'
            ])
            
        return fixes
        
    def _calculate_pattern_confidence(self, frequency: int, 
                                    failed_tests: List[TestExecution]) -> float:
        """Calculate confidence in pattern identification."""
        
        base_confidence = 0.5
        
        # Higher frequency increases confidence
        if frequency > 5:
            base_confidence += 0.3
        elif frequency > 2:
            base_confidence += 0.1
            
        # Consistent error messages increase confidence
        error_messages = [test.error_message for test in failed_tests if test.error_message]
        if error_messages:
            unique_errors = len(set(error_messages))
            consistency = 1.0 - (unique_errors / len(error_messages))
            base_confidence += consistency * 0.2
            
        return min(base_confidence, 0.95)


class QualityGateEngine:
    """Automated quality gates for test results."""
    
    def __init__(self):
        self.gates = {
            QualityGate.COVERAGE_THRESHOLD: {
                'threshold': 80.0,
                'enabled': True,
                'severity': 'high'
            },
            QualityGate.FAILURE_RATE: {
                'threshold': 5.0,
                'enabled': True,
                'severity': 'critical'
            },
            QualityGate.PERFORMANCE_REGRESSION: {
                'threshold': 20.0,  # 20% slowdown
                'enabled': True,
                'severity': 'medium'
            }
        }
        
    def evaluate_gates(self, suite_metrics: TestSuiteMetrics,
                      historical_data: List[TestSuiteMetrics] = None) -> List[QualityGateResult]:
        """Evaluate all quality gates."""
        
        results = []
        
        # Coverage gate
        if self.gates[QualityGate.COVERAGE_THRESHOLD]['enabled']:
            coverage_result = self._evaluate_coverage_gate(suite_metrics)
            results.append(coverage_result)
            
        # Failure rate gate
        if self.gates[QualityGate.FAILURE_RATE]['enabled']:
            failure_result = self._evaluate_failure_rate_gate(suite_metrics)
            results.append(failure_result)
            
        # Performance regression gate (needs historical data)
        if (self.gates[QualityGate.PERFORMANCE_REGRESSION]['enabled'] and 
            historical_data):
            performance_result = self._evaluate_performance_gate(
                suite_metrics, historical_data
            )
            results.append(performance_result)
            
        return results
        
    def _evaluate_coverage_gate(self, metrics: TestSuiteMetrics) -> QualityGateResult:
        """Evaluate coverage threshold gate."""
        
        gate_config = self.gates[QualityGate.COVERAGE_THRESHOLD]
        threshold = gate_config['threshold']
        current_coverage = metrics.coverage_percentage
        
        passed = current_coverage >= threshold
        
        return QualityGateResult(
            gate_name="Coverage Threshold",
            gate_type=QualityGate.COVERAGE_THRESHOLD,
            passed=passed,
            current_value=current_coverage,
            threshold=threshold,
            severity=gate_config['severity'],
            message=f"Code coverage is {current_coverage:.1f}% (threshold: {threshold}%)",
            recommendations=[
                "Add tests for uncovered code paths",
                "Review coverage report for gaps",
                "Consider integration tests"
            ] if not passed else []
        )
        
    def _evaluate_failure_rate_gate(self, metrics: TestSuiteMetrics) -> QualityGateResult:
        """Evaluate failure rate gate."""
        
        gate_config = self.gates[QualityGate.FAILURE_RATE]
        threshold = gate_config['threshold']
        
        total_tests = metrics.total_tests
        failed_tests = metrics.failed_tests + metrics.error_tests
        failure_rate = (failed_tests / max(total_tests, 1)) * 100
        
        passed = failure_rate <= threshold
        
        return QualityGateResult(
            gate_name="Failure Rate",
            gate_type=QualityGate.FAILURE_RATE,
            passed=passed,
            current_value=failure_rate,
            threshold=threshold,
            severity=gate_config['severity'],
            message=f"Test failure rate is {failure_rate:.1f}% (threshold: {threshold}%)",
            recommendations=[
                "Investigate failing tests",
                "Fix flaky tests",
                "Review test stability"
            ] if not passed else []
        )
        
    def _evaluate_performance_gate(self, current_metrics: TestSuiteMetrics,
                                 historical_data: List[TestSuiteMetrics]) -> QualityGateResult:
        """Evaluate performance regression gate."""
        
        gate_config = self.gates[QualityGate.PERFORMANCE_REGRESSION]
        threshold = gate_config['threshold']
        
        # Calculate baseline performance
        if len(historical_data) < 5:
            return QualityGateResult(
                gate_name="Performance Regression",
                gate_type=QualityGate.PERFORMANCE_REGRESSION,
                passed=True,
                current_value=0.0,
                threshold=threshold,
                severity=gate_config['severity'],
                message="Insufficient historical data for performance comparison",
                recommendations=[]
            )
            
        # Use recent history as baseline
        recent_durations = [m.total_duration for m in historical_data[-5:]]
        baseline_duration = np.mean(recent_durations)
        
        current_duration = current_metrics.total_duration
        regression_percentage = ((current_duration - baseline_duration) / baseline_duration) * 100
        
        passed = regression_percentage <= threshold
        
        return QualityGateResult(
            gate_name="Performance Regression",
            gate_type=QualityGate.PERFORMANCE_REGRESSION,
            passed=passed,
            current_value=regression_percentage,
            threshold=threshold,
            severity=gate_config['severity'],
            message=f"Test execution {regression_percentage:+.1f}% vs baseline (threshold: +{threshold}%)",
            recommendations=[
                "Investigate performance bottlenecks",
                "Profile test execution",
                "Check for infrastructure issues"
            ] if not passed else []
        )


class TestAnalyticsDashboard:
    """Generates comprehensive test analytics dashboard."""
    
    def __init__(self, data_collector: TestDataCollector):
        self.data_collector = data_collector
        
    def generate_dashboard(self, output_dir: str = "test_analytics_dashboard/"):
        """Generate comprehensive analytics dashboard."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get recent test data
        recent_executions = self.data_collector.get_test_history(days=30)
        
        if not recent_executions:
            logging.warning("No test data available for dashboard generation")
            return
            
        # Generate different dashboard sections
        self._generate_overview_charts(recent_executions, output_path)
        self._generate_trend_analysis(recent_executions, output_path)
        self._generate_failure_analysis_charts(recent_executions, output_path)
        self._generate_html_dashboard(output_path)
        
        logging.info(f"Test analytics dashboard generated in {output_path}")
        
    def _generate_overview_charts(self, executions: List[TestExecution], 
                                output_path: Path):
        """Generate overview charts."""
        
        # Test outcome distribution
        outcomes = [exec.outcome.value for exec in executions]
        outcome_counts = Counter(outcomes)
        
        fig = px.pie(
            values=list(outcome_counts.values()),
            names=list(outcome_counts.keys()),
            title="Test Outcome Distribution (Last 30 Days)"
        )
        fig.write_html(str(output_path / "test_outcomes.html"))
        
        # Test duration distribution
        durations = [exec.duration for exec in executions if exec.duration > 0]
        
        fig = px.histogram(
            x=durations,
            nbins=30,
            title="Test Duration Distribution",
            labels={'x': 'Duration (seconds)', 'y': 'Count'}
        )
        fig.write_html(str(output_path / "duration_distribution.html"))
        
    def _generate_trend_analysis(self, executions: List[TestExecution],
                               output_path: Path):
        """Generate trend analysis charts."""
        
        # Group by date
        daily_data = defaultdict(lambda: {
            'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0, 'total_duration': 0
        })
        
        for exec in executions:
            date = exec.timestamp.date()
            daily_data[date][exec.outcome.value] += 1
            daily_data[date]['total_duration'] += exec.duration
            
        # Convert to DataFrame for easier plotting
        dates = sorted(daily_data.keys())
        trend_data = {
            'date': dates,
            'passed': [daily_data[date]['passed'] for date in dates],
            'failed': [daily_data[date]['failed'] for date in dates],
            'skipped': [daily_data[date]['skipped'] for date in dates],
            'error': [daily_data[date]['error'] for date in dates],
            'avg_duration': [
                daily_data[date]['total_duration'] / max(
                    sum(daily_data[date][k] for k in ['passed', 'failed', 'skipped', 'error']), 1
                ) for date in dates
            ]
        }
        
        # Test results trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['passed'], 
                                mode='lines+markers', name='Passed', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['failed'], 
                                mode='lines+markers', name='Failed', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['error'], 
                                mode='lines+markers', name='Error', line=dict(color='orange')))
        
        fig.update_layout(title='Test Results Trend', xaxis_title='Date', yaxis_title='Number of Tests')
        fig.write_html(str(output_path / "test_trends.html"))
        
        # Performance trend
        fig = px.line(
            x=trend_data['date'],
            y=trend_data['avg_duration'],
            title="Average Test Duration Trend",
            labels={'x': 'Date', 'y': 'Average Duration (seconds)'}
        )
        fig.write_html(str(output_path / "performance_trend.html"))
        
    def _generate_failure_analysis_charts(self, executions: List[TestExecution],
                                        output_path: Path):
        """Generate failure analysis charts."""
        
        failed_tests = [e for e in executions if e.outcome == TestOutcome.FAILED]
        
        if not failed_tests:
            return
            
        # Most failing tests
        test_failures = Counter([test.test_name for test in failed_tests])
        top_failing = dict(test_failures.most_common(10))
        
        fig = px.bar(
            x=list(top_failing.values()),
            y=list(top_failing.keys()),
            orientation='h',
            title="Top 10 Most Failing Tests",
            labels={'x': 'Failure Count', 'y': 'Test Name'}
        )
        fig.write_html(str(output_path / "top_failing_tests.html"))
        
        # Failure heatmap by hour and day
        failure_times = [(test.timestamp.weekday(), test.timestamp.hour) 
                        for test in failed_tests]
        
        # Create heatmap data
        heatmap_data = np.zeros((7, 24))  # 7 days, 24 hours
        for day, hour in failure_times:
            heatmap_data[day, hour] += 1
            
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Failure Count"),
            x=list(range(24)),
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            title="Test Failure Heatmap"
        )
        fig.write_html(str(output_path / "failure_heatmap.html"))
        
    def _generate_html_dashboard(self, output_path: Path):
        """Generate main HTML dashboard."""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Analytics Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                .chart-container { width: 100%; height: 500px; margin: 20px 0; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                iframe { width: 100%; height: 500px; border: none; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Analytics Dashboard</h1>
                <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="section">
                <h2>Overview</h2>
                <div class="grid">
                    <div class="chart-container">
                        <iframe src="test_outcomes.html"></iframe>
                    </div>
                    <div class="chart-container">
                        <iframe src="duration_distribution.html"></iframe>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Trends</h2>
                <div class="grid">
                    <div class="chart-container">
                        <iframe src="test_trends.html"></iframe>
                    </div>
                    <div class="chart-container">
                        <iframe src="performance_trend.html"></iframe>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Failure Analysis</h2>
                <div class="grid">
                    <div class="chart-container">
                        <iframe src="top_failing_tests.html"></iframe>
                    </div>
                    <div class="chart-container">
                        <iframe src="failure_heatmap.html"></iframe>
                    </div>
                </div>
            </div>
            
        </body>
        </html>
        """
        
        with open(output_path / "dashboard.html", 'w') as f:
            f.write(html_content)


# Main analytics system
class TestAnalyticsSystem:
    """Main test analytics system."""
    
    def __init__(self, db_path: str = "test_analytics.db"):
        self.data_collector = TestDataCollector(db_path)
        self.coverage_predictor = CoveragePredictor()
        self.failure_analyzer = FailureAnalyzer()
        self.quality_gate_engine = QualityGateEngine()
        self.dashboard = TestAnalyticsDashboard(self.data_collector)
        
    def analyze_test_results(self, junit_xml_path: str = None,
                           coverage_xml_path: str = None,
                           execution_id: str = None) -> Dict[str, Any]:
        """Comprehensive analysis of test results."""
        
        if execution_id is None:
            execution_id = f"exec_{int(datetime.now().timestamp())}"
            
        # Import test results
        if junit_xml_path:
            self.data_collector.import_junit_results(junit_xml_path, execution_id)
            
        if coverage_xml_path:
            self.data_collector.import_coverage_report(coverage_xml_path, execution_id)
            
        # Get recent test data
        recent_executions = self.data_collector.get_test_history(days=30)
        
        # Analyze failures
        failure_patterns = self.failure_analyzer.analyze_failures(recent_executions)
        
        # Calculate metrics
        total_tests = len(recent_executions)
        if total_tests > 0:
            passed_tests = len([e for e in recent_executions if e.outcome == TestOutcome.PASSED])
            failed_tests = len([e for e in recent_executions if e.outcome == TestOutcome.FAILED])
            error_tests = len([e for e in recent_executions if e.outcome == TestOutcome.ERROR])
            skipped_tests = len([e for e in recent_executions if e.outcome == TestOutcome.SKIPPED])
            
            avg_duration = np.mean([e.duration for e in recent_executions])
            
            # Create suite metrics
            suite_metrics = TestSuiteMetrics(
                suite_name="main_suite",
                execution_id=execution_id,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                total_duration=sum(e.duration for e in recent_executions),
                coverage_percentage=75.0,  # Would come from actual coverage data
                timestamp=datetime.now()
            )
            
            # Evaluate quality gates
            quality_gate_results = self.quality_gate_engine.evaluate_gates(suite_metrics)
            
        else:
            suite_metrics = None
            quality_gate_results = []
            
        # Generate dashboard
        self.dashboard.generate_dashboard()
        
        return {
            'execution_id': execution_id,
            'suite_metrics': suite_metrics.__dict__ if suite_metrics else None,
            'failure_patterns': [
                {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'description': pattern.description,
                    'frequency': pattern.frequency,
                    'confidence': pattern.confidence,
                    'suggested_fixes': pattern.suggested_fixes
                }
                for pattern in failure_patterns
            ],
            'quality_gates': [
                {
                    'gate_name': gate.gate_name,
                    'passed': gate.passed,
                    'current_value': gate.current_value,
                    'threshold': gate.threshold,
                    'message': gate.message,
                    'recommendations': gate.recommendations
                }
                for gate in quality_gate_results
            ],
            'recommendations': self._generate_overall_recommendations(
                failure_patterns, quality_gate_results
            )
        }
        
    def _generate_overall_recommendations(self, 
                                       failure_patterns: List[FailurePattern],
                                       quality_gates: List[QualityGateResult]) -> List[str]:
        """Generate overall recommendations."""
        
        recommendations = []
        
        # Based on failure patterns
        if len(failure_patterns) > 3:
            recommendations.append("High number of failure patterns detected - prioritize test stability")
            
        # Based on quality gates
        failed_gates = [gate for gate in quality_gates if not gate.passed]
        if failed_gates:
            recommendations.append(f"{len(failed_gates)} quality gate(s) failed - address before deployment")
            
        # General recommendations
        if not recommendations:
            recommendations.append("Test suite appears healthy - continue monitoring")
            
        return recommendations


if __name__ == "__main__":
    # Example usage
    analytics_system = TestAnalyticsSystem()
    
    # Analyze test results
    results = analytics_system.analyze_test_results(
        # junit_xml_path="path/to/junit-results.xml",
        # coverage_xml_path="path/to/coverage.xml"
    )
    
    print("Test Analytics Results:")
    print(f"Failure patterns found: {len(results['failure_patterns'])}")
    print(f"Quality gates: {len(results['quality_gates'])}")
    
    for gate in results['quality_gates']:
        status = "PASSED" if gate['passed'] else "FAILED"
        print(f"  {gate['gate_name']}: {status} ({gate['message']})")
        
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")