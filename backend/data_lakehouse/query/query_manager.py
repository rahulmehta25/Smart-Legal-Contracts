"""
Query Engine Manager

Unified query management system supporting multiple engines:
- Intelligent query routing and optimization
- Cross-engine federation and data virtualization
- Query result caching and materialization
- Performance monitoring and optimization
- Cost-based query execution planning
- Multi-engine workload management
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


class QueryEngine(Enum):
    """Supported query engines"""
    SPARK_SQL = "spark_sql"
    PRESTO = "presto"
    TRINO = "trino"
    DREMIO = "dremio"
    DRILL = "drill"
    AUTO = "auto"  # Automatic engine selection


class QueryType(Enum):
    """Types of queries"""
    ANALYTICAL = "analytical"
    TRANSACTIONAL = "transactional"
    REPORTING = "reporting"
    ETL = "etl"
    STREAMING = "streaming"
    FEDERATED = "federated"
    EXPLORATORY = "exploratory"


class QueryStatus(Enum):
    """Query execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CACHED = "cached"


@dataclass
class QueryRequest:
    """Query execution request"""
    query_id: str
    sql: str
    engine: QueryEngine = QueryEngine.AUTO
    
    # Context and configuration
    user: Optional[str] = None
    session_id: Optional[str] = None
    database: Optional[str] = None
    
    # Execution settings
    max_execution_time_seconds: int = 3600
    memory_limit_gb: int = 8
    parallelism: int = 4
    
    # Optimization hints
    query_type: Optional[QueryType] = None
    cache_results: bool = True
    use_materialized_views: bool = True
    
    # Priority and scheduling
    priority: int = 5  # 1-10, higher is more priority
    schedule_time: Optional[datetime] = None
    
    # Custom properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)


@dataclass
class QueryResult:
    """Query execution result"""
    query_id: str
    status: QueryStatus
    engine_used: QueryEngine
    
    # Result data
    data: Optional[DataFrame] = None
    row_count: int = 0
    column_count: int = 0
    schema: Optional[StructType] = None
    
    # Execution metrics
    execution_time_ms: int = 0
    planning_time_ms: int = 0
    cpu_time_ms: int = 0
    memory_used_mb: int = 0
    bytes_scanned: int = 0
    bytes_processed: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Caching information
    cache_hit: bool = False
    cache_key: Optional[str] = None
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Query execution plan"""
    query_id: str
    original_sql: str
    optimized_sql: str
    selected_engine: QueryEngine
    
    # Plan details
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    estimated_bytes: int = 0
    estimated_time_ms: int = 0
    
    # Optimization information
    optimizations_applied: List[str] = field(default_factory=list)
    materialized_views_used: List[str] = field(default_factory=list)
    
    # Resource requirements
    required_memory_mb: int = 0
    required_cpu_cores: int = 0
    
    # Plan tree (simplified)
    plan_tree: Dict[str, Any] = field(default_factory=dict)


class QueryEngineManager:
    """
    Comprehensive query engine manager for the lakehouse.
    
    Features:
    - Multi-engine support with intelligent routing
    - Query optimization and rewriting
    - Result caching and materialization
    - Performance monitoring and analytics
    - Cost-based optimization
    - Workload management and queuing
    - Federation across multiple data sources
    - Security and access control integration
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Engine registry
        self.engines: Dict[QueryEngine, Any] = {}
        
        # Query tracking
        self.active_queries: Dict[str, QueryRequest] = {}
        self.query_history: List[QueryResult] = []
        self.query_plans: Dict[str, QueryPlan] = {}
        
        # Performance metrics
        self.engine_metrics: Dict[QueryEngine, Dict[str, Any]] = {}
        
        # Caching layer
        self.cache_layer = None
        
        # Query queue for workload management
        self.query_queue: List[QueryRequest] = []
        
        # Initialize engines
        self._initialize_engines()
        
        # Initialize caching
        self._initialize_caching()
    
    def _initialize_engines(self):
        """Initialize available query engines"""
        try:
            # Spark SQL (always available)
            from .spark_sql_engine import SparkSQLEngine
            self.engines[QueryEngine.SPARK_SQL] = SparkSQLEngine(self.spark, self.config.get("spark_sql", {}))
            
            # Presto/Trino (if configured)
            presto_config = self.config.get("presto")
            if presto_config:
                try:
                    from .presto_engine import PrestoEngine
                    self.engines[QueryEngine.PRESTO] = PrestoEngine(presto_config)
                except ImportError:
                    self.logger.warning("Presto engine not available")
            
            # Dremio (if configured)
            dremio_config = self.config.get("dremio")
            if dremio_config:
                try:
                    from .dremio_engine import DremioEngine
                    self.engines[QueryEngine.DREMIO] = DremioEngine(dremio_config)
                except ImportError:
                    self.logger.warning("Dremio engine not available")
            
            # Initialize engine metrics
            for engine in self.engines.keys():
                self.engine_metrics[engine] = {
                    "queries_executed": 0,
                    "total_execution_time_ms": 0,
                    "success_rate": 0.0,
                    "avg_response_time_ms": 0.0,
                    "last_activity": None
                }
            
            self.logger.info(f"Initialized {len(self.engines)} query engines")
            
        except Exception as e:
            self.logger.error(f"Error initializing query engines: {str(e)}")
    
    def _initialize_caching(self):
        """Initialize query result caching"""
        try:
            from .caching_layer import QueryCachingLayer
            cache_config = self.config.get("caching", {})
            self.cache_layer = QueryCachingLayer(self.spark, cache_config)
            
            self.logger.info("Query caching layer initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize caching layer: {str(e)}")
    
    def execute_query(
        self,
        sql: str,
        engine: QueryEngine = QueryEngine.AUTO,
        user: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Execute a SQL query with intelligent engine selection
        
        Args:
            sql: SQL query to execute
            engine: Preferred engine (AUTO for automatic selection)
            user: User executing the query
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult: Query execution result
        """
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Create query request
            request = QueryRequest(
                query_id=query_id,
                sql=sql,
                engine=engine,
                user=user,
                **kwargs
            )
            
            # Store active query
            self.active_queries[query_id] = request
            
            self.logger.info(f"Executing query {query_id} for user {user}")
            
            # Check cache first
            if request.cache_results and self.cache_layer:
                cached_result = self.cache_layer.get_cached_result(sql, user)
                if cached_result:
                    result = QueryResult(
                        query_id=query_id,
                        status=QueryStatus.CACHED,
                        engine_used=QueryEngine.SPARK_SQL,  # Doesn't matter for cached results
                        data=cached_result["data"],
                        row_count=cached_result["row_count"],
                        cache_hit=True,
                        cache_key=cached_result["cache_key"],
                        execution_time_ms=0
                    )
                    self._complete_query(query_id, result)
                    return result
            
            # Generate query plan
            plan = self._create_query_plan(request)
            self.query_plans[query_id] = plan
            
            # Select engine
            selected_engine = self._select_engine(request, plan)
            
            # Execute query on selected engine
            engine_instance = self.engines[selected_engine]
            result = self._execute_on_engine(request, engine_instance, selected_engine)
            
            # Cache results if enabled
            if request.cache_results and self.cache_layer and result.status == QueryStatus.COMPLETED:
                self.cache_layer.cache_result(sql, result.data, user)
            
            # Update metrics
            self._update_engine_metrics(selected_engine, result)
            
            # Complete query
            self._complete_query(query_id, result)
            
            return result
            
        except Exception as e:
            # Handle query failure
            error_result = QueryResult(
                query_id=query_id,
                status=QueryStatus.FAILED,
                engine_used=engine,
                error_message=str(e),
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            self._complete_query(query_id, error_result)
            self.logger.error(f"Query {query_id} failed: {str(e)}")
            
            return error_result
    
    def _create_query_plan(self, request: QueryRequest) -> QueryPlan:
        """Create query execution plan"""
        try:
            from .query_optimizer import QueryOptimizer
            optimizer = QueryOptimizer(self.spark, self.config.get("optimizer", {}))
            
            # Analyze query
            analysis = optimizer.analyze_query(request.sql)
            
            # Estimate costs for different engines
            engine_costs = {}
            for engine in self.engines.keys():
                cost = self._estimate_engine_cost(request.sql, engine, analysis)
                engine_costs[engine] = cost
            
            # Select best engine based on cost
            best_engine = min(engine_costs.keys(), key=lambda e: engine_costs[e])
            
            # Optimize query
            optimized_sql = optimizer.optimize_query(request.sql, best_engine)
            
            plan = QueryPlan(
                query_id=request.query_id,
                original_sql=request.sql,
                optimized_sql=optimized_sql,
                selected_engine=best_engine,
                estimated_cost=engine_costs[best_engine],
                optimizations_applied=optimizer.get_applied_optimizations()
            )
            
            return plan
            
        except Exception as e:
            self.logger.warning(f"Error creating query plan: {str(e)}")
            # Return basic plan
            return QueryPlan(
                query_id=request.query_id,
                original_sql=request.sql,
                optimized_sql=request.sql,
                selected_engine=QueryEngine.SPARK_SQL  # Default fallback
            )
    
    def _select_engine(self, request: QueryRequest, plan: QueryPlan) -> QueryEngine:
        """Select optimal query engine"""
        try:
            if request.engine != QueryEngine.AUTO:
                # User specified engine
                if request.engine in self.engines:
                    return request.engine
                else:
                    self.logger.warning(f"Requested engine {request.engine} not available, using auto-selection")
            
            # Use plan recommendation
            if plan.selected_engine in self.engines:
                return plan.selected_engine
            
            # Fallback logic based on query characteristics
            sql_upper = request.sql.upper()
            
            # Large aggregations or complex analytics -> Spark
            if any(keyword in sql_upper for keyword in ["GROUP BY", "WINDOW", "WITH", "CTE"]):
                if QueryEngine.SPARK_SQL in self.engines:
                    return QueryEngine.SPARK_SQL
            
            # Fast interactive queries -> Presto/Trino
            if "LIMIT" in sql_upper and any(engine in self.engines for engine in [QueryEngine.PRESTO, QueryEngine.TRINO]):
                for engine in [QueryEngine.PRESTO, QueryEngine.TRINO]:
                    if engine in self.engines:
                        return engine
            
            # Default to Spark SQL
            return QueryEngine.SPARK_SQL
            
        except Exception as e:
            self.logger.error(f"Error selecting engine: {str(e)}")
            return QueryEngine.SPARK_SQL
    
    def _estimate_engine_cost(self, sql: str, engine: QueryEngine, analysis: Dict[str, Any]) -> float:
        """Estimate cost of executing query on specific engine"""
        try:
            base_cost = 100.0  # Base cost
            
            # Get engine performance characteristics
            metrics = self.engine_metrics.get(engine, {})
            avg_response_time = metrics.get("avg_response_time_ms", 1000)
            success_rate = metrics.get("success_rate", 1.0)
            
            # Adjust cost based on query characteristics
            if analysis.get("complexity_score", 0) > 5:
                base_cost *= 1.5  # Complex queries cost more
            
            if analysis.get("estimated_rows", 0) > 1000000:
                # Large result sets
                if engine == QueryEngine.SPARK_SQL:
                    base_cost *= 0.8  # Spark is better for large data
                else:
                    base_cost *= 1.2
            
            # Factor in engine performance
            performance_factor = avg_response_time / 1000.0  # Normalize to seconds
            reliability_factor = 2.0 - success_rate  # Lower success rate increases cost
            
            final_cost = base_cost * performance_factor * reliability_factor
            
            return final_cost
            
        except Exception as e:
            self.logger.warning(f"Error estimating cost for {engine}: {str(e)}")
            return 1000.0  # High default cost
    
    def _execute_on_engine(
        self,
        request: QueryRequest,
        engine_instance: Any,
        engine_type: QueryEngine
    ) -> QueryResult:
        """Execute query on specific engine"""
        start_time = datetime.now()
        
        try:
            # Get optimized SQL from plan
            plan = self.query_plans.get(request.query_id)
            sql_to_execute = plan.optimized_sql if plan else request.sql
            
            # Execute query
            if hasattr(engine_instance, 'execute_query'):
                result_df = engine_instance.execute_query(sql_to_execute, request.properties)
            else:
                # Fallback to Spark SQL
                result_df = self.spark.sql(sql_to_execute)
            
            # Create result
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            result = QueryResult(
                query_id=request.query_id,
                status=QueryStatus.COMPLETED,
                engine_used=engine_type,
                data=result_df,
                row_count=result_df.count() if result_df else 0,
                column_count=len(result_df.columns) if result_df else 0,
                schema=result_df.schema if result_df else None,
                execution_time_ms=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return QueryResult(
                query_id=request.query_id,
                status=QueryStatus.FAILED,
                engine_used=engine_type,
                error_message=str(e),
                execution_time_ms=execution_time,
                start_time=start_time,
                end_time=end_time
            )
    
    def _update_engine_metrics(self, engine: QueryEngine, result: QueryResult):
        """Update performance metrics for an engine"""
        try:
            metrics = self.engine_metrics[engine]
            
            metrics["queries_executed"] += 1
            metrics["total_execution_time_ms"] += result.execution_time_ms
            metrics["last_activity"] = datetime.now()
            
            # Calculate success rate
            total_queries = metrics["queries_executed"]
            if result.status == QueryStatus.COMPLETED:
                success_count = getattr(metrics, "_success_count", 0) + 1
            else:
                success_count = getattr(metrics, "_success_count", 0)
            
            metrics["_success_count"] = success_count
            metrics["success_rate"] = success_count / total_queries if total_queries > 0 else 0.0
            
            # Calculate average response time
            metrics["avg_response_time_ms"] = metrics["total_execution_time_ms"] / total_queries
            
        except Exception as e:
            self.logger.error(f"Error updating engine metrics: {str(e)}")
    
    def _complete_query(self, query_id: str, result: QueryResult):
        """Complete query execution and cleanup"""
        try:
            # Move from active to history
            if query_id in self.active_queries:
                del self.active_queries[query_id]
            
            self.query_history.append(result)
            
            # Cleanup old history (keep last 1000 queries)
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error completing query {query_id}: {str(e)}")
    
    def get_query_status(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a query"""
        # Check active queries
        if query_id in self.active_queries:
            request = self.active_queries[query_id]
            return {
                "query_id": query_id,
                "status": "running",
                "sql": request.sql,
                "engine": request.engine.value,
                "user": request.user,
                "start_time": request.created_time.isoformat()
            }
        
        # Check history
        for result in self.query_history:
            if result.query_id == query_id:
                return {
                    "query_id": query_id,
                    "status": result.status.value,
                    "engine_used": result.engine_used.value,
                    "execution_time_ms": result.execution_time_ms,
                    "row_count": result.row_count,
                    "cache_hit": result.cache_hit,
                    "error_message": result.error_message,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None
                }
        
        return None
    
    def cancel_query(self, query_id: str) -> bool:
        """Cancel a running query"""
        try:
            if query_id in self.active_queries:
                # This would need engine-specific cancellation logic
                request = self.active_queries[query_id]
                
                # Create cancelled result
                result = QueryResult(
                    query_id=query_id,
                    status=QueryStatus.CANCELLED,
                    engine_used=request.engine,
                    error_message="Query cancelled by user"
                )
                
                self._complete_query(query_id, result)
                
                self.logger.info(f"Cancelled query: {query_id}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling query {query_id}: {str(e)}")
            return False
    
    def get_engine_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all engines"""
        return {
            "engines": {
                engine.value: metrics
                for engine, metrics in self.engine_metrics.items()
            },
            "total_queries": sum(m["queries_executed"] for m in self.engine_metrics.values()),
            "active_queries": len(self.active_queries),
            "avg_success_rate": sum(m["success_rate"] for m in self.engine_metrics.values()) / len(self.engine_metrics) if self.engine_metrics else 0.0
        }
    
    def create_materialized_view(
        self,
        view_name: str,
        sql: str,
        refresh_interval: Optional[timedelta] = None
    ) -> bool:
        """Create a materialized view for query acceleration"""
        try:
            # This would integrate with the storage layer to create Delta tables
            # For now, we'll create a cached result
            
            if self.cache_layer:
                # Execute query to materialize
                result = self.execute_query(sql, user="system")
                
                if result.status == QueryStatus.COMPLETED:
                    # Store as materialized view
                    self.cache_layer.create_materialized_view(
                        view_name, 
                        result.data,
                        refresh_interval
                    )
                    
                    self.logger.info(f"Created materialized view: {view_name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error creating materialized view {view_name}: {str(e)}")
            return False
    
    def get_query_recommendations(self, sql: str, user: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get query optimization recommendations"""
        try:
            recommendations = []
            
            sql_upper = sql.upper()
            
            # Basic recommendations
            if "SELECT *" in sql_upper:
                recommendations.append({
                    "type": "performance",
                    "message": "Avoid SELECT * - specify only needed columns",
                    "impact": "medium"
                })
            
            if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
                recommendations.append({
                    "type": "performance",
                    "message": "Consider adding LIMIT when using ORDER BY",
                    "impact": "high"
                })
            
            if sql.count("JOIN") > 3:
                recommendations.append({
                    "type": "complexity",
                    "message": "Complex query with multiple JOINs - consider using Spark SQL",
                    "impact": "medium"
                })
            
            # Check for common anti-patterns
            if "WHERE" not in sql_upper and ("SELECT" in sql_upper and "FROM" in sql_upper):
                recommendations.append({
                    "type": "performance",
                    "message": "Add WHERE clause to filter data early",
                    "impact": "high"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def cleanup(self):
        """Cleanup query manager resources"""
        try:
            # Cancel active queries
            for query_id in list(self.active_queries.keys()):
                self.cancel_query(query_id)
            
            # Cleanup engines
            for engine in self.engines.values():
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
            
            # Cleanup cache layer
            if self.cache_layer:
                self.cache_layer.cleanup()
            
            self.logger.info("Query engine manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Utility functions

def create_query_request(
    sql: str,
    user: str,
    engine: QueryEngine = QueryEngine.AUTO,
    **kwargs
) -> QueryRequest:
    """Create a query request with defaults"""
    return QueryRequest(
        query_id=str(uuid.uuid4()),
        sql=sql,
        engine=engine,
        user=user,
        **kwargs
    )


def analyze_query_complexity(sql: str) -> Dict[str, Any]:
    """Analyze query complexity for engine selection"""
    sql_upper = sql.upper()
    
    complexity_score = 0
    
    # Count different SQL features
    features = {
        "joins": sql_upper.count("JOIN"),
        "subqueries": sql_upper.count("SELECT") - 1,  # Subtract main SELECT
        "aggregations": sum(sql_upper.count(agg) for agg in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]),
        "window_functions": sum(sql_upper.count(win) for win in ["OVER(", "ROW_NUMBER", "RANK"]),
        "ctes": sql_upper.count("WITH"),
        "unions": sql_upper.count("UNION")
    }
    
    # Calculate complexity score
    complexity_score = (
        features["joins"] * 2 +
        features["subqueries"] * 1.5 +
        features["aggregations"] * 1 +
        features["window_functions"] * 3 +
        features["ctes"] * 2 +
        features["unions"] * 1.5
    )
    
    return {
        "complexity_score": complexity_score,
        "features": features,
        "estimated_rows": 10000,  # Would need actual estimation
        "recommended_engine": QueryEngine.SPARK_SQL if complexity_score > 5 else QueryEngine.PRESTO
    }