"""
AI-Powered Bug Prediction System

This module provides intelligent bug prediction using machine learning models
that analyze code complexity, historical patterns, and developer activity.
"""

import ast
import os
import json
import logging
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import subprocess
import git


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeMetrics:
    """Represents code complexity and quality metrics."""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int
    return_statement_count: int
    branch_count: int
    loop_count: int
    exception_handling_count: int
    code_duplication_ratio: float
    test_coverage: float
    documentation_ratio: float


@dataclass
class HistoricalMetrics:
    """Represents historical bug and change patterns."""
    bug_count_last_30_days: int
    bug_count_last_90_days: int
    bug_count_total: int
    change_frequency: float
    lines_changed_ratio: float
    author_count: int
    commit_count_last_30_days: int
    hotspot_score: float
    fix_time_average: float
    reopened_bug_ratio: float


@dataclass
class DeveloperMetrics:
    """Represents developer activity and experience metrics."""
    experience_months: int
    lines_of_code_contributed: int
    bug_introduction_rate: float
    code_review_participation: float
    knowledge_area_count: int
    team_collaboration_score: float
    code_quality_score: float


@dataclass
class BugPrediction:
    """Represents a bug prediction result."""
    file_path: str
    function_name: Optional[str]
    risk_level: RiskLevel
    probability: float
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    priority_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplexityAnalyzer:
    """Analyzes code complexity metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_file(self, file_path: str) -> Dict[str, CodeMetrics]:
        """Analyze complexity metrics for all functions in a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return {}
            
        metrics = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_metrics = self._analyze_function(node, content)
                metrics[node.name] = function_metrics
                
        # Also analyze file-level metrics
        file_metrics = self._analyze_file_level(tree, content)
        metrics['__file__'] = file_metrics
        
        return metrics
        
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> CodeMetrics:
        """Analyze metrics for a specific function."""
        # Calculate cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic_complexity(node)
        
        # Calculate cognitive complexity (more human-friendly)
        cognitive = self._calculate_cognitive_complexity(node)
        
        # Calculate nesting depth
        nesting_depth = self._calculate_nesting_depth(node)
        
        # Function length (lines)
        function_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        
        # Parameter count
        param_count = len(node.args.args)
        
        # Return statement count
        return_count = len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
        
        # Branch count (if/elif/else, try/except, for/while)
        branch_count = len([n for n in ast.walk(node) 
                          if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))])
        
        # Loop count
        loop_count = len([n for n in ast.walk(node) 
                        if isinstance(n, (ast.For, ast.While))])
        
        # Exception handling count
        exception_count = len([n for n in ast.walk(node) 
                             if isinstance(n, (ast.Try, ast.Raise, ast.ExceptHandler))])
        
        # Code duplication ratio (simplified)
        duplication_ratio = self._calculate_duplication_ratio(node)
        
        # Documentation ratio
        doc_ratio = self._calculate_documentation_ratio(node, content)
        
        return CodeMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            nesting_depth=nesting_depth,
            function_length=function_lines,
            parameter_count=param_count,
            return_statement_count=return_count,
            branch_count=branch_count,
            loop_count=loop_count,
            exception_handling_count=exception_count,
            code_duplication_ratio=duplication_ratio,
            test_coverage=0.0,  # Will be filled by coverage analysis
            documentation_ratio=doc_ratio
        )
        
    def _analyze_file_level(self, tree: ast.AST, content: str) -> CodeMetrics:
        """Analyze file-level metrics."""
        total_lines = len(content.split('\n'))
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        
        # Average complexity across all functions
        function_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        avg_cyclomatic = np.mean([self._calculate_cyclomatic_complexity(f) 
                                for f in function_nodes]) if function_nodes else 0
        
        return CodeMetrics(
            cyclomatic_complexity=int(avg_cyclomatic),
            cognitive_complexity=int(avg_cyclomatic * 1.2),  # Approximation
            nesting_depth=max([self._calculate_nesting_depth(f) for f in function_nodes] or [0]),
            function_length=total_lines // max(function_count, 1),
            parameter_count=0,
            return_statement_count=0,
            branch_count=len([n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While))]),
            loop_count=len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]),
            exception_handling_count=len([n for n in ast.walk(tree) if isinstance(n, ast.Try)]),
            code_duplication_ratio=0.0,
            test_coverage=0.0,
            documentation_ratio=self._calculate_file_documentation_ratio(tree, content)
        )
        
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                
        return complexity
        
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (more human-friendly than cyclomatic)."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(n, level=0):
            nonlocal complexity, nesting_level
            
            if isinstance(n, (ast.If, ast.While, ast.For)):
                complexity += 1 + level
                nesting_level = max(nesting_level, level + 1)
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level + 1)
            elif isinstance(n, ast.Try):
                complexity += 1
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, level)
                    
        visit_node(node)
        return complexity
        
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(n, current_depth=0):
            max_depth = current_depth
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.Try, ast.With, 
                             ast.FunctionDef, ast.ClassDef, ast.AsyncWith, ast.AsyncFor)):
                current_depth += 1
                
            for child in ast.iter_child_nodes(n):
                child_depth = get_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
                
            return max_depth
            
        return get_depth(node)
        
    def _calculate_duplication_ratio(self, node: ast.AST) -> float:
        """Calculate code duplication ratio (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated algorithms
        
        # Count similar subtrees
        subtree_hashes = []
        
        def hash_node(n):
            if isinstance(n, ast.AST):
                node_type = type(n).__name__
                children = [hash_node(child) for child in ast.iter_child_nodes(n)]
                return hash((node_type, tuple(sorted(children))))
            return hash(str(n))
            
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.FunctionDef)):
                subtree_hashes.append(hash_node(child))
                
        if not subtree_hashes:
            return 0.0
            
        unique_hashes = set(subtree_hashes)
        return 1.0 - (len(unique_hashes) / len(subtree_hashes))
        
    def _calculate_documentation_ratio(self, node: ast.FunctionDef, content: str) -> float:
        """Calculate documentation ratio for a function."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return 0.0
            
        # Count lines of documentation vs code
        doc_lines = len(docstring.split('\n'))
        total_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
        
        return min(doc_lines / max(total_lines - doc_lines, 1), 1.0)
        
    def _calculate_file_documentation_ratio(self, tree: ast.AST, content: str) -> float:
        """Calculate overall file documentation ratio."""
        total_lines = len(content.split('\n'))
        
        # Count docstrings and comments
        doc_lines = 0
        
        # Count docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    doc_lines += len(docstring.split('\n'))
                    
        # Count comment lines (simplified)
        comment_lines = len([line for line in content.split('\n') 
                           if line.strip().startswith('#')])
        
        total_doc_lines = doc_lines + comment_lines
        return min(total_doc_lines / max(total_lines, 1), 1.0)


class GitHistoryAnalyzer:
    """Analyzes Git history for bug patterns and developer metrics."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = None
            logging.warning(f"Invalid Git repository: {repo_path}")
            
    def analyze_file_history(self, file_path: str) -> HistoricalMetrics:
        """Analyze historical metrics for a specific file."""
        if not self.repo:
            return self._default_historical_metrics()
            
        try:
            # Get commits that modified this file
            commits = list(self.repo.iter_commits(paths=file_path, max_count=1000))
            
            if not commits:
                return self._default_historical_metrics()
                
            # Analyze time periods
            now = datetime.now()
            recent_commits = [c for c in commits 
                            if (now - datetime.fromtimestamp(c.committed_date)).days <= 30]
            three_month_commits = [c for c in commits 
                                 if (now - datetime.fromtimestamp(c.committed_date)).days <= 90]
            
            # Analyze bug patterns from commit messages
            bug_keywords = ['fix', 'bug', 'issue', 'error', 'patch', 'hotfix', 'critical']
            bug_commits = [c for c in commits 
                         if any(keyword in c.message.lower() for keyword in bug_keywords)]
            
            recent_bugs = [c for c in recent_commits 
                         if any(keyword in c.message.lower() for keyword in bug_keywords)]
            three_month_bugs = [c for c in three_month_commits 
                              if any(keyword in c.message.lower() for keyword in bug_keywords)]
            
            # Calculate metrics
            change_frequency = len(recent_commits) / max(30, 1)  # Changes per day
            
            # Lines changed analysis
            total_lines_changed = 0
            for commit in recent_commits[:10]:  # Sample recent commits
                try:
                    stats = commit.stats.files.get(file_path, {'insertions': 0, 'deletions': 0})
                    total_lines_changed += stats['insertions'] + stats['deletions']
                except:
                    continue
                    
            # Get current file size
            try:
                with open(os.path.join(self.repo_path, file_path), 'r') as f:
                    current_lines = len(f.readlines())
                lines_changed_ratio = total_lines_changed / max(current_lines, 1)
            except:
                lines_changed_ratio = 0.0
                
            # Author diversity
            authors = set(c.author.name for c in commits[:50])  # Sample commits
            
            # Hotspot score (files changed frequently with bugs)
            hotspot_score = (len(recent_bugs) / max(len(recent_commits), 1)) * len(recent_commits)
            
            # Average fix time (simplified approximation)
            fix_times = []
            for i in range(len(bug_commits) - 1):
                bug_commit = bug_commits[i]
                prev_commit = bug_commits[i + 1]
                fix_time_hours = (bug_commit.committed_date - prev_commit.committed_date) / 3600
                if 0 < fix_time_hours < 24 * 30:  # Reasonable fix time
                    fix_times.append(fix_time_hours)
                    
            avg_fix_time = np.mean(fix_times) if fix_times else 24.0
            
            # Reopened bug ratio (simplified)
            reopened_ratio = 0.0
            if len(bug_commits) > 1:
                # Look for patterns of multiple bug fixes close together
                close_fixes = 0
                for i in range(len(bug_commits) - 1):
                    time_diff = bug_commits[i].committed_date - bug_commits[i + 1].committed_date
                    if time_diff < 24 * 3600 * 7:  # Within a week
                        close_fixes += 1
                reopened_ratio = close_fixes / len(bug_commits)
            
            return HistoricalMetrics(
                bug_count_last_30_days=len(recent_bugs),
                bug_count_last_90_days=len(three_month_bugs),
                bug_count_total=len(bug_commits),
                change_frequency=change_frequency,
                lines_changed_ratio=min(lines_changed_ratio, 2.0),  # Cap at reasonable value
                author_count=len(authors),
                commit_count_last_30_days=len(recent_commits),
                hotspot_score=hotspot_score,
                fix_time_average=avg_fix_time,
                reopened_bug_ratio=reopened_ratio
            )
            
        except Exception as e:
            logging.warning(f"Error analyzing Git history for {file_path}: {e}")
            return self._default_historical_metrics()
            
    def analyze_developer_metrics(self, author_email: str) -> DeveloperMetrics:
        """Analyze developer-specific metrics."""
        if not self.repo:
            return self._default_developer_metrics()
            
        try:
            # Get all commits by this author
            commits = list(self.repo.iter_commits(author=author_email, max_count=1000))
            
            if not commits:
                return self._default_developer_metrics()
                
            # Calculate experience (months since first commit)
            first_commit_date = datetime.fromtimestamp(commits[-1].committed_date)
            experience_months = max(1, (datetime.now() - first_commit_date).days // 30)
            
            # Lines of code contributed (approximation)
            total_insertions = sum(c.stats.total['insertions'] for c in commits if c.stats.total)
            
            # Bug introduction rate
            bug_keywords = ['fix', 'bug', 'issue', 'error', 'patch']
            bug_fix_commits = [c for c in commits 
                             if any(keyword in c.message.lower() for keyword in bug_keywords)]
            bug_introduction_rate = len(bug_fix_commits) / max(len(commits), 1)
            
            # Simplified metrics for demo
            return DeveloperMetrics(
                experience_months=experience_months,
                lines_of_code_contributed=total_insertions,
                bug_introduction_rate=bug_introduction_rate,
                code_review_participation=0.7,  # Would need PR data
                knowledge_area_count=min(10, len(set(c.message.split()[0] for c in commits[:50]))),
                team_collaboration_score=0.8,  # Would need collaboration data
                code_quality_score=max(0.1, 1.0 - bug_introduction_rate)
            )
            
        except Exception as e:
            logging.warning(f"Error analyzing developer metrics for {author_email}: {e}")
            return self._default_developer_metrics()
            
    def _default_historical_metrics(self) -> HistoricalMetrics:
        """Return default historical metrics when Git analysis fails."""
        return HistoricalMetrics(
            bug_count_last_30_days=0,
            bug_count_last_90_days=0,
            bug_count_total=0,
            change_frequency=0.0,
            lines_changed_ratio=0.0,
            author_count=1,
            commit_count_last_30_days=0,
            hotspot_score=0.0,
            fix_time_average=24.0,
            reopened_bug_ratio=0.0
        )
        
    def _default_developer_metrics(self) -> DeveloperMetrics:
        """Return default developer metrics when analysis fails."""
        return DeveloperMetrics(
            experience_months=6,
            lines_of_code_contributed=1000,
            bug_introduction_rate=0.1,
            code_review_participation=0.5,
            knowledge_area_count=3,
            team_collaboration_score=0.6,
            code_quality_score=0.7
        )


class BugPredictor:
    """Main bug prediction engine using machine learning."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.complexity_analyzer = ComplexityAnalyzer()
        self.git_analyzer = GitHistoryAnalyzer(repo_path)
        
        # ML models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def extract_features(self, file_path: str, function_name: Optional[str] = None) -> np.ndarray:
        """Extract features for bug prediction."""
        # Get complexity metrics
        complexity_metrics = self.complexity_analyzer.analyze_file(file_path)
        
        if function_name and function_name in complexity_metrics:
            code_metrics = complexity_metrics[function_name]
        else:
            code_metrics = complexity_metrics.get('__file__', CodeMetrics(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0
            ))
            
        # Get historical metrics
        historical_metrics = self.git_analyzer.analyze_file_history(file_path)
        
        # Get developer metrics (use most recent author)
        developer_metrics = DeveloperMetrics(6, 1000, 0.1, 0.5, 3, 0.6, 0.7)
        
        # Combine all features
        features = [
            # Code complexity features
            code_metrics.cyclomatic_complexity,
            code_metrics.cognitive_complexity,
            code_metrics.nesting_depth,
            code_metrics.function_length,
            code_metrics.parameter_count,
            code_metrics.return_statement_count,
            code_metrics.branch_count,
            code_metrics.loop_count,
            code_metrics.exception_handling_count,
            code_metrics.code_duplication_ratio,
            code_metrics.test_coverage,
            code_metrics.documentation_ratio,
            
            # Historical features
            historical_metrics.bug_count_last_30_days,
            historical_metrics.bug_count_last_90_days,
            historical_metrics.bug_count_total,
            historical_metrics.change_frequency,
            historical_metrics.lines_changed_ratio,
            historical_metrics.author_count,
            historical_metrics.commit_count_last_30_days,
            historical_metrics.hotspot_score,
            historical_metrics.fix_time_average,
            historical_metrics.reopened_bug_ratio,
            
            # Developer features
            developer_metrics.experience_months,
            np.log1p(developer_metrics.lines_of_code_contributed),
            developer_metrics.bug_introduction_rate,
            developer_metrics.code_review_participation,
            developer_metrics.knowledge_area_count,
            developer_metrics.team_collaboration_score,
            developer_metrics.code_quality_score
        ]
        
        return np.array(features).reshape(1, -1)
        
    def train(self, training_data: List[Tuple[str, Optional[str], bool]]):
        """Train the bug prediction models."""
        if not training_data:
            logging.warning("No training data provided")
            return
            
        X = []
        y = []
        
        for file_path, function_name, has_bug in training_data:
            try:
                features = self.extract_features(file_path, function_name)
                X.append(features.flatten())
                y.append(1 if has_bug else 0)
            except Exception as e:
                logging.warning(f"Error extracting features for {file_path}: {e}")
                continue
                
        if not X:
            logging.error("No valid training data extracted")
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        
        # Train models
        model_scores = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_scores[name] = cv_scores.mean()
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            if len(set(y_test)) > 1:
                auc_score = roc_auc_score(y_test, y_pred_proba)
                logging.info(f"{name} - CV Score: {cv_scores.mean():.3f}, Test AUC: {auc_score:.3f}")
            
        # Select best model
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[self.best_model_name]
        
        # Store feature names for interpretation
        self.feature_names = [
            'cyclomatic_complexity', 'cognitive_complexity', 'nesting_depth',
            'function_length', 'parameter_count', 'return_statement_count',
            'branch_count', 'loop_count', 'exception_handling_count',
            'code_duplication_ratio', 'test_coverage', 'documentation_ratio',
            'bug_count_last_30_days', 'bug_count_last_90_days', 'bug_count_total',
            'change_frequency', 'lines_changed_ratio', 'author_count',
            'commit_count_last_30_days', 'hotspot_score', 'fix_time_average',
            'reopened_bug_ratio', 'experience_months', 'lines_contributed',
            'bug_introduction_rate', 'code_review_participation',
            'knowledge_area_count', 'team_collaboration_score', 'code_quality_score'
        ]
        
        self.is_trained = True
        logging.info(f"Model training completed. Best model: {self.best_model_name}")
        
    def predict(self, file_path: str, function_name: Optional[str] = None) -> BugPrediction:
        """Predict bug risk for a file or function."""
        if not self.is_trained:
            # Use rule-based prediction if no trained model
            return self._rule_based_prediction(file_path, function_name)
            
        try:
            # Extract features
            features = self.extract_features(file_path, function_name)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction
            probability = self.best_model.predict_proba(features_scaled)[0, 1]
            
            # Determine risk level
            if probability >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif probability >= 0.6:
                risk_level = RiskLevel.HIGH
            elif probability >= 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
                
            # Get feature importance for explanation
            contributing_factors = self._get_contributing_factors(features.flatten())
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, contributing_factors)
            
            # Calculate confidence based on model certainty
            confidence = max(probability, 1 - probability)
            
            return BugPrediction(
                file_path=file_path,
                function_name=function_name,
                risk_level=risk_level,
                probability=probability,
                confidence=confidence,
                contributing_factors=contributing_factors,
                recommended_actions=recommendations,
                priority_score=probability * confidence,
                metadata={
                    'model_used': self.best_model_name,
                    'feature_count': len(self.feature_names)
                }
            )
            
        except Exception as e:
            logging.error(f"Error predicting bug risk for {file_path}: {e}")
            return self._rule_based_prediction(file_path, function_name)
            
    def _rule_based_prediction(self, file_path: str, function_name: Optional[str]) -> BugPrediction:
        """Fallback rule-based prediction when ML model is not available."""
        try:
            complexity_metrics = self.complexity_analyzer.analyze_file(file_path)
            
            if function_name and function_name in complexity_metrics:
                metrics = complexity_metrics[function_name]
            else:
                metrics = complexity_metrics.get('__file__', CodeMetrics(
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0
                ))
                
            # Simple rule-based risk assessment
            risk_score = 0
            factors = []
            
            if metrics.cyclomatic_complexity > 15:
                risk_score += 0.3
                factors.append("High cyclomatic complexity")
                
            if metrics.nesting_depth > 5:
                risk_score += 0.2
                factors.append("Deep nesting")
                
            if metrics.function_length > 100:
                risk_score += 0.2
                factors.append("Long function")
                
            if metrics.parameter_count > 7:
                risk_score += 0.1
                factors.append("Many parameters")
                
            if metrics.documentation_ratio < 0.1:
                risk_score += 0.2
                factors.append("Poor documentation")
                
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
                
            recommendations = self._generate_recommendations(risk_level, factors)
            
            return BugPrediction(
                file_path=file_path,
                function_name=function_name,
                risk_level=risk_level,
                probability=risk_score,
                confidence=0.6,  # Lower confidence for rule-based
                contributing_factors=factors,
                recommended_actions=recommendations,
                priority_score=risk_score * 0.6,
                metadata={'model_used': 'rule_based'}
            )
            
        except Exception as e:
            logging.error(f"Error in rule-based prediction for {file_path}: {e}")
            return BugPrediction(
                file_path=file_path,
                function_name=function_name,
                risk_level=RiskLevel.LOW,
                probability=0.1,
                confidence=0.3,
                contributing_factors=["Analysis failed"],
                recommended_actions=["Manual code review recommended"],
                priority_score=0.03,
                metadata={'model_used': 'fallback'}
            )
            
    def _get_contributing_factors(self, features: np.ndarray) -> List[str]:
        """Get the most important contributing factors."""
        if not hasattr(self.best_model, 'feature_importances_'):
            return ["Model-based factors"]
            
        importances = self.best_model.feature_importances_
        
        # Get top factors
        top_indices = np.argsort(importances)[-5:][::-1]
        
        factors = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                feature_value = features[idx]
                factors.append(f"{feature_name}: {feature_value:.2f}")
                
        return factors
        
    def _generate_recommendations(self, risk_level: RiskLevel, factors: List[str]) -> List[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Prioritize comprehensive testing",
                "Conduct thorough code review",
                "Consider refactoring to reduce complexity"
            ])
            
        if "complexity" in ' '.join(factors).lower():
            recommendations.append("Break down complex functions into smaller units")
            
        if "nesting" in ' '.join(factors).lower():
            recommendations.append("Reduce nesting depth using guard clauses")
            
        if "documentation" in ' '.join(factors).lower():
            recommendations.append("Add comprehensive documentation and comments")
            
        if "bug_count" in ' '.join(factors).lower():
            recommendations.append("Investigate historical bug patterns")
            
        if not recommendations:
            recommendations.append("Continue monitoring and maintain good practices")
            
        return recommendations
        
    def analyze_codebase(self, extensions: List[str] = ['.py']) -> List[BugPrediction]:
        """Analyze entire codebase for bug risks."""
        predictions = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    
                    try:
                        prediction = self.predict(file_path)
                        predictions.append(prediction)
                    except Exception as e:
                        logging.warning(f"Error analyzing {relative_path}: {e}")
                        continue
                        
        # Sort by priority score
        predictions.sort(key=lambda p: p.priority_score, reverse=True)
        return predictions
        
    def save_model(self, model_path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            logging.warning("No trained model to save")
            return
            
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logging.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.models = model_data['models']
            self.best_model_name = model_data['best_model_name']
            self.best_model = self.models[self.best_model_name]
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logging.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")


if __name__ == "__main__":
    # Example usage
    predictor = BugPredictor("/path/to/repo")
    
    # Analyze a specific file
    prediction = predictor.predict("app/models/user.py", "validate_email")
    print(f"Risk Level: {prediction.risk_level.value}")
    print(f"Probability: {prediction.probability:.3f}")
    print(f"Contributing Factors: {prediction.contributing_factors}")
    print(f"Recommendations: {prediction.recommended_actions}")
    
    # Analyze entire codebase
    all_predictions = predictor.analyze_codebase()
    print(f"\nTop 5 highest risk files:")
    for pred in all_predictions[:5]:
        print(f"{pred.file_path}: {pred.risk_level.value} ({pred.probability:.3f})")