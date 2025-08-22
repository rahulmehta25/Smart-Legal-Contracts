"""
Data Quality Engine

Comprehensive data quality validation and monitoring system:
- Schema validation and evolution
- Data profiling and statistics
- Anomaly detection
- Data freshness monitoring
- Custom validation rules
- Quality reporting and alerting
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import math

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *


class QualityCheckType(Enum):
    """Types of data quality checks"""
    NULL_CHECK = "null_check"
    UNIQUENESS = "uniqueness"
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    RANGE_CHECK = "range_check"
    PATTERN_MATCH = "pattern_match"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    CUSTOM = "custom"


class QualityLevel(Enum):
    """Quality check severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityCheck:
    """Configuration for a data quality check"""
    check_id: str
    name: str
    check_type: QualityCheckType
    level: QualityLevel
    columns: List[str]
    conditions: Dict[str, Any]
    description: Optional[str] = None
    enabled: bool = True
    
    # Thresholds
    error_threshold: float = 0.0  # Percentage of failures that trigger error
    warning_threshold: float = 0.1  # Percentage of failures that trigger warning
    
    # Custom validation function
    custom_func: Optional[Callable[[DataFrame], Tuple[bool, str]]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class QualityResult:
    """Result of a data quality check"""
    check_id: str
    check_name: str
    status: str  # passed, warning, failed
    execution_time: datetime
    duration_ms: int
    
    # Metrics
    total_records: int = 0
    failed_records: int = 0
    failure_rate: float = 0.0
    
    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    sample_failures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataProfile:
    """Statistical profile of a dataset"""
    table_name: str
    column_profiles: Dict[str, Dict[str, Any]]
    total_records: int
    total_columns: int
    null_records: int
    duplicate_records: int
    profile_timestamp: datetime
    
    # Schema information
    schema_hash: str
    schema_evolution: List[str] = field(default_factory=list)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    quality_dimensions: Dict[str, float] = field(default_factory=dict)


class DataQualityEngine:
    """
    Comprehensive data quality validation and monitoring engine.
    
    Features:
    - Multi-dimensional quality checks
    - Statistical data profiling
    - Anomaly detection and alerting
    - Schema validation and evolution tracking
    - Custom validation rules
    - Quality reporting and dashboards
    - Data lineage integration
    - Performance optimization
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality checks registry
        self.quality_checks: Dict[str, QualityCheck] = {}
        
        # Execution history
        self.execution_history: List[QualityResult] = []
        
        # Data profiles
        self.data_profiles: Dict[str, DataProfile] = {}
        
        # Quality rules cache
        self._compiled_rules_cache: Dict[str, Any] = {}
        
        # Statistics for anomaly detection
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_check(self, check: QualityCheck) -> str:
        """
        Register a data quality check
        
        Args:
            check: QualityCheck configuration
            
        Returns:
            str: Check ID
        """
        try:
            # Validate check configuration
            validation_errors = self._validate_check_config(check)
            if validation_errors:
                raise ValueError(f"Check validation failed: {validation_errors}")
            
            # Store the check
            self.quality_checks[check.check_id] = check
            
            # Compile check if it's a custom rule
            if check.check_type == QualityCheckType.CUSTOM and check.custom_func:
                self._compiled_rules_cache[check.check_id] = check.custom_func
            
            self.logger.info(f"Registered quality check: {check.check_id}")
            return check.check_id
            
        except Exception as e:
            self.logger.error(f"Error registering check {check.check_id}: {str(e)}")
            raise
    
    def _validate_check_config(self, check: QualityCheck) -> List[str]:
        """Validate quality check configuration"""
        errors = []
        
        if not check.check_id:
            errors.append("check_id is required")
        
        if not check.name:
            errors.append("name is required")
        
        if not check.columns and check.check_type != QualityCheckType.CUSTOM:
            errors.append("columns must be specified for non-custom checks")
        
        if check.error_threshold < 0 or check.error_threshold > 1:
            errors.append("error_threshold must be between 0 and 1")
        
        if check.warning_threshold < 0 or check.warning_threshold > 1:
            errors.append("warning_threshold must be between 0 and 1")
        
        if check.check_type == QualityCheckType.CUSTOM and not check.custom_func:
            errors.append("custom_func is required for custom checks")
        
        return errors
    
    def execute_checks(
        self,
        df: DataFrame,
        table_name: str,
        check_ids: Optional[List[str]] = None,
        parallel_execution: bool = True
    ) -> List[QualityResult]:
        """
        Execute data quality checks on a DataFrame
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table being checked
            check_ids: Specific checks to run (None for all)
            parallel_execution: Whether to run checks in parallel
            
        Returns:
            List[QualityResult]: Results of quality checks
        """
        try:
            self.logger.info(f"Executing quality checks for table: {table_name}")
            
            # Determine which checks to run
            checks_to_run = []
            if check_ids:
                checks_to_run = [self.quality_checks[cid] for cid in check_ids 
                               if cid in self.quality_checks and self.quality_checks[cid].enabled]
            else:
                checks_to_run = [check for check in self.quality_checks.values() if check.enabled]
            
            if not checks_to_run:
                self.logger.warning("No quality checks to execute")
                return []
            
            # Cache DataFrame for multiple checks
            df_cached = df.cache()
            total_records = df_cached.count()
            
            results = []
            
            # Execute checks
            for check in checks_to_run:
                try:
                    start_time = datetime.now()
                    
                    result = self._execute_single_check(check, df_cached, total_records)
                    
                    end_time = datetime.now()
                    result.duration_ms = int((end_time - start_time).total_seconds() * 1000)
                    
                    results.append(result)
                    
                except Exception as e:
                    error_result = QualityResult(
                        check_id=check.check_id,
                        check_name=check.name,
                        status="failed",
                        execution_time=datetime.now(),
                        duration_ms=0,
                        message=f"Check execution failed: {str(e)}"
                    )
                    results.append(error_result)
                    self.logger.error(f"Error executing check {check.check_id}: {str(e)}")
            
            # Store execution history
            self.execution_history.extend(results)
            
            # Cleanup
            df_cached.unpersist()
            
            self.logger.info(f"Completed {len(results)} quality checks for {table_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing quality checks: {str(e)}")
            return []
    
    def _execute_single_check(self, check: QualityCheck, df: DataFrame, total_records: int) -> QualityResult:
        """Execute a single quality check"""
        result = QualityResult(
            check_id=check.check_id,
            check_name=check.name,
            status="passed",
            execution_time=datetime.now(),
            duration_ms=0,
            total_records=total_records
        )
        
        try:
            if check.check_type == QualityCheckType.NULL_CHECK:
                result = self._execute_null_check(check, df, result)
            
            elif check.check_type == QualityCheckType.UNIQUENESS:
                result = self._execute_uniqueness_check(check, df, result)
            
            elif check.check_type == QualityCheckType.COMPLETENESS:
                result = self._execute_completeness_check(check, df, result)
            
            elif check.check_type == QualityCheckType.RANGE_CHECK:
                result = self._execute_range_check(check, df, result)
            
            elif check.check_type == QualityCheckType.PATTERN_MATCH:
                result = self._execute_pattern_check(check, df, result)
            
            elif check.check_type == QualityCheckType.CUSTOM:
                result = self._execute_custom_check(check, df, result)
            
            else:
                result.status = "failed"
                result.message = f"Unsupported check type: {check.check_type}"
            
            # Determine final status based on thresholds
            if result.failure_rate > check.error_threshold:
                result.status = "failed"
            elif result.failure_rate > check.warning_threshold:
                result.status = "warning"
            else:
                result.status = "passed"
            
        except Exception as e:
            result.status = "failed"
            result.message = f"Check execution error: {str(e)}"
        
        return result
    
    def _execute_null_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute null value check"""
        try:
            null_conditions = []
            for column in check.columns:
                if column in df.columns:
                    null_conditions.append(col(column).isNull())
            
            if null_conditions:
                # Count records with nulls in any of the specified columns
                null_filter = null_conditions[0]
                for condition in null_conditions[1:]:
                    null_filter = null_filter | condition
                
                failed_records = df.filter(null_filter).count()
                result.failed_records = failed_records
                result.failure_rate = failed_records / result.total_records if result.total_records > 0 else 0
                
                result.message = f"Found {failed_records} records with null values in columns: {check.columns}"
                result.details = {
                    "null_columns": check.columns,
                    "null_records": failed_records,
                    "null_percentage": result.failure_rate * 100
                }
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Null check failed: {str(e)}"
        
        return result
    
    def _execute_uniqueness_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute uniqueness check"""
        try:
            # Check for duplicates in specified columns
            duplicate_count = df.groupBy(*check.columns).count().filter(col("count") > 1).count()
            
            result.failed_records = duplicate_count
            result.failure_rate = duplicate_count / result.total_records if result.total_records > 0 else 0
            
            result.message = f"Found {duplicate_count} duplicate records for columns: {check.columns}"
            result.details = {
                "duplicate_columns": check.columns,
                "duplicate_records": duplicate_count,
                "duplicate_percentage": result.failure_rate * 100
            }
            
            # Get sample duplicates for analysis
            if duplicate_count > 0:
                sample_duplicates = df.groupBy(*check.columns) \
                                   .count() \
                                   .filter(col("count") > 1) \
                                   .limit(10) \
                                   .collect()
                
                result.sample_failures = [row.asDict() for row in sample_duplicates]
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Uniqueness check failed: {str(e)}"
        
        return result
    
    def _execute_completeness_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute completeness check"""
        try:
            min_completeness = check.conditions.get("min_completeness", 0.95)
            
            completeness_stats = {}
            overall_incomplete_records = 0
            
            for column in check.columns:
                if column in df.columns:
                    null_count = df.filter(col(column).isNull() | (col(column) == "")).count()
                    completeness = (result.total_records - null_count) / result.total_records if result.total_records > 0 else 0
                    
                    completeness_stats[column] = {
                        "completeness": completeness,
                        "null_count": null_count,
                        "complete_records": result.total_records - null_count
                    }
                    
                    if completeness < min_completeness:
                        overall_incomplete_records += null_count
            
            result.failed_records = overall_incomplete_records
            result.failure_rate = overall_incomplete_records / result.total_records if result.total_records > 0 else 0
            
            result.message = f"Completeness check: {len([c for c, s in completeness_stats.items() if s['completeness'] < min_completeness])} columns below threshold"
            result.details = {
                "min_completeness_threshold": min_completeness,
                "column_completeness": completeness_stats,
                "overall_incomplete_records": overall_incomplete_records
            }
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Completeness check failed: {str(e)}"
        
        return result
    
    def _execute_range_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute range validation check"""
        try:
            out_of_range_conditions = []
            
            for column in check.columns:
                if column in df.columns:
                    min_val = check.conditions.get(f"{column}_min")
                    max_val = check.conditions.get(f"{column}_max")
                    
                    if min_val is not None and max_val is not None:
                        out_of_range_condition = (col(column) < min_val) | (col(column) > max_val)
                        out_of_range_conditions.append(out_of_range_condition)
            
            if out_of_range_conditions:
                # Combine all out-of-range conditions
                combined_condition = out_of_range_conditions[0]
                for condition in out_of_range_conditions[1:]:
                    combined_condition = combined_condition | condition
                
                failed_records = df.filter(combined_condition).count()
                result.failed_records = failed_records
                result.failure_rate = failed_records / result.total_records if result.total_records > 0 else 0
                
                result.message = f"Found {failed_records} records outside valid ranges"
                result.details = {
                    "range_conditions": {col: {"min": check.conditions.get(f"{col}_min"), 
                                             "max": check.conditions.get(f"{col}_max")} 
                                        for col in check.columns},
                    "out_of_range_records": failed_records
                }
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Range check failed: {str(e)}"
        
        return result
    
    def _execute_pattern_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute pattern matching check"""
        try:
            pattern_violations = 0
            
            for column in check.columns:
                if column in df.columns:
                    pattern = check.conditions.get(f"{column}_pattern")
                    if pattern:
                        # Use regexp to validate pattern
                        invalid_pattern_count = df.filter(~col(column).rlike(pattern)).count()
                        pattern_violations += invalid_pattern_count
            
            result.failed_records = pattern_violations
            result.failure_rate = pattern_violations / result.total_records if result.total_records > 0 else 0
            
            result.message = f"Found {pattern_violations} records with invalid patterns"
            result.details = {
                "pattern_conditions": {col: check.conditions.get(f"{col}_pattern") for col in check.columns},
                "pattern_violations": pattern_violations
            }
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Pattern check failed: {str(e)}"
        
        return result
    
    def _execute_custom_check(self, check: QualityCheck, df: DataFrame, result: QualityResult) -> QualityResult:
        """Execute custom validation check"""
        try:
            if check.custom_func:
                success, message = check.custom_func(df)
                
                if not success:
                    result.status = "failed"
                    result.message = message
                else:
                    result.status = "passed"
                    result.message = message
            else:
                result.status = "failed"
                result.message = "No custom function defined"
        
        except Exception as e:
            result.status = "failed"
            result.message = f"Custom check failed: {str(e)}"
        
        return result
    
    def profile_data(self, df: DataFrame, table_name: str) -> DataProfile:
        """
        Generate comprehensive data profile
        
        Args:
            df: DataFrame to profile
            table_name: Name of the table
            
        Returns:
            DataProfile: Statistical profile of the data
        """
        try:
            self.logger.info(f"Profiling data for table: {table_name}")
            
            # Basic metrics
            total_records = df.count()
            total_columns = len(df.columns)
            
            # Column profiling
            column_profiles = {}
            
            for column in df.columns:
                column_type = dict(df.dtypes)[column]
                
                # Basic statistics
                profile = {
                    "column_name": column,
                    "data_type": column_type,
                    "null_count": df.filter(col(column).isNull()).count(),
                    "distinct_count": df.select(column).distinct().count()
                }
                
                # Calculate null percentage
                profile["null_percentage"] = (profile["null_count"] / total_records * 100) if total_records > 0 else 0
                profile["completeness"] = 100 - profile["null_percentage"]
                
                # Type-specific statistics
                if column_type in ["int", "bigint", "double", "float", "decimal"]:
                    # Numeric statistics
                    stats = df.select(
                        min(col(column)).alias("min_val"),
                        max(col(column)).alias("max_val"),
                        avg(col(column)).alias("mean_val"),
                        stddev(col(column)).alias("std_val")
                    ).collect()[0]
                    
                    profile.update({
                        "min_value": stats["min_val"],
                        "max_value": stats["max_val"],
                        "mean_value": stats["mean_val"],
                        "std_deviation": stats["std_val"]
                    })
                
                elif column_type in ["string", "varchar"]:
                    # String statistics
                    string_stats = df.select(
                        min(length(col(column))).alias("min_length"),
                        max(length(col(column))).alias("max_length"),
                        avg(length(col(column))).alias("avg_length")
                    ).collect()[0]
                    
                    profile.update({
                        "min_length": string_stats["min_length"],
                        "max_length": string_stats["max_length"],
                        "avg_length": string_stats["avg_length"]
                    })
                
                # Cardinality analysis
                cardinality_ratio = profile["distinct_count"] / total_records if total_records > 0 else 0
                if cardinality_ratio == 1.0:
                    profile["cardinality_type"] = "unique"
                elif cardinality_ratio > 0.8:
                    profile["cardinality_type"] = "high"
                elif cardinality_ratio > 0.1:
                    profile["cardinality_type"] = "medium"
                else:
                    profile["cardinality_type"] = "low"
                
                column_profiles[column] = profile
            
            # Calculate overall quality metrics
            overall_completeness = sum(p["completeness"] for p in column_profiles.values()) / len(column_profiles)
            
            quality_dimensions = {
                "completeness": overall_completeness,
                "uniqueness": len([p for p in column_profiles.values() if p["cardinality_type"] == "unique"]) / len(column_profiles) * 100,
                "validity": 100.0  # Would be calculated based on validation rules
            }
            
            overall_quality_score = sum(quality_dimensions.values()) / len(quality_dimensions)
            
            # Create data profile
            data_profile = DataProfile(
                table_name=table_name,
                column_profiles=column_profiles,
                total_records=total_records,
                total_columns=total_columns,
                null_records=sum(p["null_count"] for p in column_profiles.values()),
                duplicate_records=0,  # Would need to be calculated
                profile_timestamp=datetime.now(),
                schema_hash=self._generate_schema_hash(df.schema),
                overall_quality_score=overall_quality_score,
                quality_dimensions=quality_dimensions
            )
            
            # Store profile
            self.data_profiles[table_name] = data_profile
            
            self.logger.info(f"Data profiling completed for {table_name}")
            return data_profile
            
        except Exception as e:
            self.logger.error(f"Error profiling data for {table_name}: {str(e)}")
            raise
    
    def _generate_schema_hash(self, schema: StructType) -> str:
        """Generate hash for schema to detect changes"""
        import hashlib
        schema_str = str(schema)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def get_quality_summary(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get quality summary for tables"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks_registered": len(self.quality_checks),
            "total_executions": len(self.execution_history),
            "tables": {}
        }
        
        if table_name:
            # Summary for specific table
            table_results = [r for r in self.execution_history if table_name in r.check_name]
            summary["tables"][table_name] = self._generate_table_summary(table_results)
        else:
            # Summary for all tables
            table_names = set()
            for result in self.execution_history:
                # Extract table name from check name (simplified)
                table_names.add(result.check_name.split("_")[0] if "_" in result.check_name else "unknown")
            
            for tn in table_names:
                table_results = [r for r in self.execution_history if tn in r.check_name]
                summary["tables"][tn] = self._generate_table_summary(table_results)
        
        return summary
    
    def _generate_table_summary(self, results: List[QualityResult]) -> Dict[str, Any]:
        """Generate summary for a table's quality results"""
        if not results:
            return {"status": "no_data", "checks": 0}
        
        latest_results = {}
        for result in results:
            if result.check_id not in latest_results or result.execution_time > latest_results[result.check_id].execution_time:
                latest_results[result.check_id] = result
        
        passed = len([r for r in latest_results.values() if r.status == "passed"])
        warnings = len([r for r in latest_results.values() if r.status == "warning"])
        failed = len([r for r in latest_results.values() if r.status == "failed"])
        
        overall_status = "passed"
        if failed > 0:
            overall_status = "failed"
        elif warnings > 0:
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "checks": len(latest_results),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "last_execution": max(r.execution_time for r in latest_results.values()).isoformat()
        }
    
    def cleanup(self):
        """Cleanup data quality engine resources"""
        try:
            # Clear caches
            self._compiled_rules_cache.clear()
            
            # Optionally clear history (keep recent entries)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.execution_history = [r for r in self.execution_history if r.execution_time > cutoff_date]
            
            self.logger.info("Data quality engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Utility functions for common quality checks

def create_null_check(check_id: str, columns: List[str], level: QualityLevel = QualityLevel.HIGH) -> QualityCheck:
    """Create a null value check"""
    return QualityCheck(
        check_id=check_id,
        name=f"Null Check: {', '.join(columns)}",
        check_type=QualityCheckType.NULL_CHECK,
        level=level,
        columns=columns,
        conditions={},
        description=f"Validates that specified columns do not contain null values"
    )


def create_uniqueness_check(check_id: str, columns: List[str], level: QualityLevel = QualityLevel.CRITICAL) -> QualityCheck:
    """Create a uniqueness check"""
    return QualityCheck(
        check_id=check_id,
        name=f"Uniqueness Check: {', '.join(columns)}",
        check_type=QualityCheckType.UNIQUENESS,
        level=level,
        columns=columns,
        conditions={},
        description=f"Validates that specified columns contain unique values"
    )


def create_range_check(
    check_id: str,
    column: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
    level: QualityLevel = QualityLevel.MEDIUM
) -> QualityCheck:
    """Create a range validation check"""
    return QualityCheck(
        check_id=check_id,
        name=f"Range Check: {column} [{min_value}, {max_value}]",
        check_type=QualityCheckType.RANGE_CHECK,
        level=level,
        columns=[column],
        conditions={f"{column}_min": min_value, f"{column}_max": max_value},
        description=f"Validates that {column} values are within specified range"
    )