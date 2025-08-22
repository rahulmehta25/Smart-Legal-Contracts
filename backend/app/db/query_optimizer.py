"""
Enterprise Query Optimizer Module

Implements intelligent query optimization with execution plan analysis,
automatic index recommendations, and slow query detection.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

from sqlalchemy import create_engine, text, inspect, MetaData, Index
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT" 
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    

class OptimizationLevel(Enum):
    """Query optimization levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"
    

@dataclass
class QueryExecution:
    """Query execution statistics"""
    query_hash: str
    original_query: str
    optimized_query: Optional[str]
    execution_time: float
    rows_examined: int
    rows_returned: int
    index_usage: List[str]
    execution_plan: Dict[str, Any]
    timestamp: datetime
    database_name: str
    table_names: List[str]
    

@dataclass
class IndexRecommendation:
    """Index recommendation with cost/benefit analysis"""
    table_name: str
    columns: List[str]
    index_type: str  # btree, hash, gin, gist
    priority: int  # 1-10, higher is more important
    estimated_benefit: float  # Performance improvement estimate
    storage_cost: int  # Estimated storage overhead in MB
    maintenance_cost: float  # Estimated maintenance overhead
    usage_frequency: int  # How often this index would be used
    recommendation_reason: str
    sql_create_statement: str
    

@dataclass
class SlowQueryAlert:
    """Alert for slow query detection"""
    query_hash: str
    query_text: str
    avg_execution_time: float
    max_execution_time: float
    execution_count: int
    first_seen: datetime
    last_seen: datetime
    affected_tables: List[str]
    optimization_suggestions: List[str]
    alert_level: str  # info, warning, critical
    

class QueryOptimizer:
    """
    Advanced query optimizer for enterprise-scale database performance
    
    Features:
    - Automatic query plan analysis and optimization
    - Smart index recommendations based on query patterns
    - Slow query detection and alerting
    - Query rewriting for performance improvements
    - Cost-based optimization decisions
    - Real-time performance monitoring
    """
    
    def __init__(self, engine, metadata: MetaData):
        self.engine = engine
        self.metadata = metadata
        self.query_history: Dict[str, List[QueryExecution]] = defaultdict(list)
        self.index_recommendations: List[IndexRecommendation] = []
        self.slow_queries: Dict[str, SlowQueryAlert] = {}
        self.optimization_rules: List[Dict] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.very_slow_query_threshold = 5.0  # seconds
        self.high_rows_examined_threshold = 10000
        
        # Initialize optimization rules
        self._initialize_optimization_rules()
        
    def _initialize_optimization_rules(self):
        """Initialize query optimization rules"""
        
        self.optimization_rules = [
            {
                "name": "eliminate_subqueries",
                "pattern": r"WHERE\s+\w+\s+IN\s*\(\s*SELECT\s+.*?\)",
                "replacement": self._optimize_subquery_to_join,
                "priority": 8
            },
            {
                "name": "add_limit_to_unbounded_queries", 
                "pattern": r"SELECT\s+.*\s+FROM\s+\w+(?!\s+.*LIMIT)",
                "replacement": self._add_safety_limit,
                "priority": 6
            },
            {
                "name": "optimize_wildcard_selects",
                "pattern": r"SELECT\s+\*\s+FROM",
                "replacement": self._optimize_wildcard_select,
                "priority": 5
            },
            {
                "name": "push_down_predicates",
                "pattern": r"SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*",
                "replacement": self._push_down_predicates,
                "priority": 7
            },
            {
                "name": "optimize_order_by",
                "pattern": r"ORDER\s+BY\s+.*",
                "replacement": self._optimize_order_by,
                "priority": 4
            }
        ]
    
    async def analyze_query(self, query: str, session: Session = None) -> Dict[str, Any]:
        """
        Comprehensive query analysis with execution plan and optimization suggestions
        
        Args:
            query: SQL query to analyze
            session: Database session
            
        Returns:
            Dict: Complete analysis results
        """
        start_time = time.time()
        query_hash = self._hash_query(query)
        
        analysis_result = {
            "query_hash": query_hash,
            "original_query": query,
            "query_type": self._detect_query_type(query),
            "table_names": self._extract_table_names(query),
            "execution_plan": {},
            "optimization_suggestions": [],
            "index_recommendations": [],
            "estimated_cost": 0,
            "complexity_score": 0,
            "execution_stats": None
        }
        
        try:
            # Get execution plan
            if session:
                execution_plan = await self._get_execution_plan(query, session)
                analysis_result["execution_plan"] = execution_plan
                analysis_result["estimated_cost"] = self._extract_cost_from_plan(execution_plan)
            
            # Analyze query complexity
            analysis_result["complexity_score"] = self._calculate_complexity_score(query)
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(query, analysis_result)
            analysis_result["optimization_suggestions"] = optimization_suggestions
            
            # Generate index recommendations
            index_recommendations = await self._generate_index_recommendations(query, analysis_result)
            analysis_result["index_recommendations"] = index_recommendations
            
            # Check for common performance anti-patterns
            anti_patterns = self._detect_anti_patterns(query)
            analysis_result["anti_patterns"] = anti_patterns
            
            analysis_time = time.time() - start_time
            logger.debug(f"Query analysis completed in {analysis_time:.3f}s for query hash {query_hash}")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    async def _get_execution_plan(self, query: str, session: Session) -> Dict[str, Any]:
        """Get database execution plan"""
        
        try:
            if self.engine.dialect.name == 'postgresql':
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                result = session.execute(text(explain_query)).fetchone()
                return json.loads(result[0])[0] if result else {}
                
            elif self.engine.dialect.name == 'mysql':
                explain_query = f"EXPLAIN FORMAT=JSON {query}"
                result = session.execute(text(explain_query)).fetchone()
                return json.loads(result[0]) if result else {}
                
            elif self.engine.dialect.name == 'sqlite':
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                results = session.execute(text(explain_query)).fetchall()
                return {
                    "plan": [
                        {"id": row[0], "parent": row[1], "detail": row[3]}
                        for row in results
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get execution plan: {e}")
            return {"error": str(e)}
    
    def _extract_cost_from_plan(self, execution_plan: Dict) -> float:
        """Extract cost estimate from execution plan"""
        
        if not execution_plan or "error" in execution_plan:
            return 0.0
            
        try:
            if self.engine.dialect.name == 'postgresql':
                return execution_plan.get("Plan", {}).get("Total Cost", 0.0)
            elif self.engine.dialect.name == 'mysql':
                return execution_plan.get("query_block", {}).get("cost_info", {}).get("query_cost", 0.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_complexity_score(self, query: str) -> int:
        """Calculate query complexity score (1-100)"""
        
        score = 0
        query_upper = query.upper()
        
        # Base complexity factors
        score += len(query.split()) // 10  # Length factor
        score += query_upper.count("JOIN") * 5  # Join complexity
        score += query_upper.count("SUBQUERY") * 8  # Subquery complexity
        score += query_upper.count("UNION") * 6  # Union complexity
        score += query_upper.count("GROUP BY") * 3  # Aggregation complexity
        score += query_upper.count("ORDER BY") * 2  # Sorting complexity
        score += query_upper.count("HAVING") * 4  # Having clause complexity
        score += query_upper.count("CASE WHEN") * 3  # Conditional logic
        
        # Window functions add significant complexity
        window_functions = ["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "NTILE"]
        for func in window_functions:
            score += query_upper.count(func) * 6
            
        # Nested queries
        nested_depth = query_upper.count("(SELECT")
        score += nested_depth * 10
        
        return min(score, 100)  # Cap at 100
    
    async def _generate_optimization_suggestions(self, query: str, analysis: Dict) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        
        suggestions = []
        query_upper = query.upper()
        
        # Check for SELECT *
        if "SELECT *" in query_upper:
            suggestions.append("Avoid SELECT * - specify only needed columns to reduce I/O")
        
        # Check for missing WHERE clause on large tables
        table_names = analysis.get("table_names", [])
        if not re.search(r"\bWHERE\b", query_upper) and table_names:
            suggestions.append("Consider adding WHERE clause to limit rows scanned")
        
        # Check for ORDER BY without LIMIT
        if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            suggestions.append("ORDER BY without LIMIT may sort entire result set - consider adding LIMIT")
        
        # Check for inefficient JOINs
        if "JOIN" in query_upper:
            if "ON" not in query_upper:
                suggestions.append("Use explicit JOIN conditions instead of WHERE clause joins")
            
            join_count = query_upper.count("JOIN")
            if join_count > 3:
                suggestions.append(f"Query has {join_count} JOINs - consider denormalization or materialized views")
        
        # Check for subqueries that could be JOINs
        if "IN (SELECT" in query_upper:
            suggestions.append("Consider rewriting IN subquery as JOIN for better performance")
        
        # Check for DISTINCT usage
        if "DISTINCT" in query_upper:
            suggestions.append("DISTINCT can be expensive - ensure it's necessary and consider GROUP BY alternative")
        
        # Check for OR conditions
        if query_upper.count(" OR ") > 2:
            suggestions.append("Multiple OR conditions can prevent index usage - consider UNION ALL")
        
        # Check for function calls on columns in WHERE
        function_pattern = r"WHERE\s+\w+\([^)]*\w+\.[^)]*\)"
        if re.search(function_pattern, query_upper):
            suggestions.append("Avoid functions on columns in WHERE clause - consider functional indexes")
        
        # Check for LIKE with leading wildcards
        if re.search(r"LIKE\s+['\"]%", query_upper):
            suggestions.append("LIKE patterns starting with % cannot use indexes - consider full-text search")
        
        # Check query complexity
        complexity = analysis.get("complexity_score", 0)
        if complexity > 70:
            suggestions.append("High query complexity - consider breaking into smaller queries or using views")
        
        return suggestions
    
    async def _generate_index_recommendations(self, query: str, analysis: Dict) -> List[IndexRecommendation]:
        """Generate index recommendations based on query analysis"""
        
        recommendations = []
        table_names = analysis.get("table_names", [])
        
        # Extract WHERE clause columns
        where_columns = self._extract_where_columns(query)
        
        # Extract JOIN columns
        join_columns = self._extract_join_columns(query)
        
        # Extract ORDER BY columns
        order_columns = self._extract_order_by_columns(query)
        
        # Generate recommendations for WHERE clause columns
        for table_name, columns in where_columns.items():
            if table_name in table_names:
                for column in columns:
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=[column],
                        index_type="btree",
                        priority=8,
                        estimated_benefit=0.7,
                        storage_cost=5,
                        maintenance_cost=0.1,
                        usage_frequency=10,
                        recommendation_reason=f"WHERE clause filtering on {column}",
                        sql_create_statement=f"CREATE INDEX idx_{table_name}_{column} ON {table_name} ({column})"
                    )
                    recommendations.append(recommendation)
        
        # Generate recommendations for JOIN columns
        for table_name, columns in join_columns.items():
            if table_name in table_names:
                for column in columns:
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=[column],
                        index_type="btree",
                        priority=9,
                        estimated_benefit=0.8,
                        storage_cost=7,
                        maintenance_cost=0.15,
                        usage_frequency=15,
                        recommendation_reason=f"JOIN condition on {column}",
                        sql_create_statement=f"CREATE INDEX idx_{table_name}_{column}_join ON {table_name} ({column})"
                    )
                    recommendations.append(recommendation)
        
        # Generate composite index recommendations
        for table_name in table_names:
            table_where_cols = where_columns.get(table_name, [])
            table_order_cols = order_columns.get(table_name, [])
            
            if len(table_where_cols) > 1:
                # Multi-column index for WHERE conditions
                columns = list(set(table_where_cols))[:3]  # Limit to 3 columns
                if len(columns) > 1:
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=columns,
                        index_type="btree",
                        priority=7,
                        estimated_benefit=0.6,
                        storage_cost=10,
                        maintenance_cost=0.2,
                        usage_frequency=8,
                        recommendation_reason=f"Composite index for multiple WHERE conditions",
                        sql_create_statement=f"CREATE INDEX idx_{table_name}_{'_'.join(columns)} ON {table_name} ({', '.join(columns)})"
                    )
                    recommendations.append(recommendation)
            
            # Covering index for SELECT + WHERE + ORDER BY
            if table_where_cols and table_order_cols:
                covering_columns = list(set(table_where_cols + table_order_cols))[:4]
                if len(covering_columns) > 1:
                    recommendation = IndexRecommendation(
                        table_name=table_name,
                        columns=covering_columns,
                        index_type="btree",
                        priority=6,
                        estimated_benefit=0.5,
                        storage_cost=15,
                        maintenance_cost=0.25,
                        usage_frequency=5,
                        recommendation_reason=f"Covering index for query optimization",
                        sql_create_statement=f"CREATE INDEX idx_{table_name}_covering ON {table_name} ({', '.join(covering_columns)})"
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _extract_where_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in WHERE clauses by table"""
        
        where_columns = defaultdict(list)
        
        # Find WHERE clause
        where_match = re.search(r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)", 
                               query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract table.column or column references
            column_patterns = [
                r"(\w+)\.(\w+)\s*[=<>!]",  # table.column
                r"\b(\w+)\s*[=<>!]",       # column only
            ]
            
            for pattern in column_patterns:
                matches = re.findall(pattern, where_clause, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        table_name, column_name = match
                        where_columns[table_name].append(column_name)
                    else:
                        # For column-only matches, we'd need table context
                        pass
        
        return dict(where_columns)
    
    def _extract_join_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in JOIN conditions by table"""
        
        join_columns = defaultdict(list)
        
        # Find JOIN clauses
        join_pattern = r"\bJOIN\s+(\w+).*?\bON\s+([^)]+?)(?:\bWHERE\b|\bJOIN\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|$)"
        matches = re.findall(join_pattern, query, re.IGNORECASE | re.DOTALL)
        
        for table_name, condition in matches:
            # Extract columns from join condition
            column_matches = re.findall(r"(\w+)\.(\w+)", condition)
            for table, column in column_matches:
                join_columns[table].append(column)
        
        return dict(join_columns)
    
    def _extract_order_by_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in ORDER BY clauses by table"""
        
        order_columns = defaultdict(list)
        
        # Find ORDER BY clause
        order_match = re.search(r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)", query, re.IGNORECASE | re.DOTALL)
        
        if order_match:
            order_clause = order_match.group(1)
            
            # Extract table.column references
            column_matches = re.findall(r"(\w+)\.(\w+)", order_clause)
            for table_name, column_name in column_matches:
                order_columns[table_name].append(column_name)
        
        return dict(order_columns)
    
    def _detect_anti_patterns(self, query: str) -> List[Dict[str, str]]:
        """Detect common SQL anti-patterns"""
        
        anti_patterns = []
        query_upper = query.upper()
        
        # N+1 Query Pattern (multiple similar queries)
        if self._is_potential_n_plus_one(query):
            anti_patterns.append({
                "pattern": "N+1 Query",
                "description": "Potential N+1 query pattern detected",
                "severity": "high",
                "suggestion": "Consider using JOINs or batch loading"
            })
        
        # Cartesian Product
        join_count = query_upper.count("JOIN")
        on_count = query_upper.count(" ON ")
        if join_count > 0 and on_count < join_count:
            anti_patterns.append({
                "pattern": "Cartesian Product",
                "description": "Missing JOIN conditions may cause cartesian product",
                "severity": "critical",
                "suggestion": "Ensure all JOINs have proper ON conditions"
            })
        
        # Inefficient DISTINCT
        if "DISTINCT" in query_upper and "ORDER BY" in query_upper:
            anti_patterns.append({
                "pattern": "DISTINCT with ORDER BY",
                "description": "DISTINCT with ORDER BY can be inefficient",
                "severity": "medium",
                "suggestion": "Consider GROUP BY or ensure proper indexing"
            })
        
        # Implicit Type Conversion
        if re.search(r"=\s*['\"][0-9]+['\"]", query):
            anti_patterns.append({
                "pattern": "Implicit Type Conversion",
                "description": "Comparing numeric columns with string literals",
                "severity": "medium",
                "suggestion": "Use proper data types in comparisons"
            })
        
        # Unnecessary Subqueries
        if query_upper.count("SELECT") > 1 and "JOIN" not in query_upper:
            anti_patterns.append({
                "pattern": "Unnecessary Subquery",
                "description": "Subquery might be replaceable with JOIN",
                "severity": "medium",
                "suggestion": "Consider rewriting subquery as JOIN"
            })
        
        return anti_patterns
    
    def _is_potential_n_plus_one(self, query: str) -> bool:
        """Check if query is potentially part of N+1 pattern"""
        
        # This is a simplified check - in practice, you'd track query patterns over time
        query_hash = self._hash_query(query)
        
        # Check if similar queries have been executed recently
        recent_queries = [
            exec_record for exec_records in self.query_history.values() 
            for exec_record in exec_records 
            if exec_record.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        similar_count = sum(1 for q in recent_queries if q.query_hash.startswith(query_hash[:8]))
        return similar_count > 10
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of SQL query"""
        
        query_trimmed = query.strip().upper()
        
        if query_trimmed.startswith("SELECT"):
            return QueryType.SELECT
        elif query_trimmed.startswith("INSERT"):
            return QueryType.INSERT
        elif query_trimmed.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_trimmed.startswith("DELETE"):
            return QueryType.DELETE
        elif query_trimmed.startswith("CREATE"):
            return QueryType.CREATE
        elif query_trimmed.startswith("ALTER"):
            return QueryType.ALTER
        else:
            return QueryType.SELECT  # Default
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query"""
        
        table_names = []
        
        # FROM clause tables
        from_pattern = r"\bFROM\s+(\w+)"
        from_matches = re.findall(from_pattern, query, re.IGNORECASE)
        table_names.extend(from_matches)
        
        # JOIN clause tables
        join_pattern = r"\bJOIN\s+(\w+)"
        join_matches = re.findall(join_pattern, query, re.IGNORECASE)
        table_names.extend(join_matches)
        
        # UPDATE clause tables
        update_pattern = r"\bUPDATE\s+(\w+)"
        update_matches = re.findall(update_pattern, query, re.IGNORECASE)
        table_names.extend(update_matches)
        
        # INSERT INTO clause tables
        insert_pattern = r"\bINSERT\s+INTO\s+(\w+)"
        insert_matches = re.findall(insert_pattern, query, re.IGNORECASE)
        table_names.extend(insert_matches)
        
        # DELETE FROM clause tables
        delete_pattern = r"\bDELETE\s+FROM\s+(\w+)"
        delete_matches = re.findall(delete_pattern, query, re.IGNORECASE)
        table_names.extend(delete_matches)
        
        return list(set(table_names))  # Remove duplicates
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query (normalizing for similar queries)"""
        
        # Normalize query for hashing
        normalized = re.sub(r"\s+", " ", query.strip().upper())
        normalized = re.sub(r"['\"][^'\"]*['\"]", "?", normalized)  # Replace literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)  # Replace numbers
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    async def optimize_query(self, query: str, optimization_level: OptimizationLevel = OptimizationLevel.INTERMEDIATE) -> str:
        """
        Optimize a query based on the specified optimization level
        
        Args:
            query: Original SQL query
            optimization_level: Level of optimization to apply
            
        Returns:
            str: Optimized query
        """
        
        optimized_query = query
        
        # Apply optimization rules based on level
        applicable_rules = [
            rule for rule in self.optimization_rules 
            if self._should_apply_rule(rule, optimization_level)
        ]
        
        # Sort rules by priority (higher first)
        applicable_rules.sort(key=lambda x: x["priority"], reverse=True)
        
        for rule in applicable_rules:
            try:
                if isinstance(rule["replacement"], str):
                    # Simple regex replacement
                    optimized_query = re.sub(
                        rule["pattern"], 
                        rule["replacement"], 
                        optimized_query, 
                        flags=re.IGNORECASE
                    )
                else:
                    # Function-based replacement
                    optimized_query = rule["replacement"](optimized_query)
                    
            except Exception as e:
                logger.warning(f"Failed to apply optimization rule {rule['name']}: {e}")
        
        return optimized_query
    
    def _should_apply_rule(self, rule: Dict, optimization_level: OptimizationLevel) -> bool:
        """Determine if an optimization rule should be applied"""
        
        priority = rule["priority"]
        
        if optimization_level == OptimizationLevel.BASIC:
            return priority >= 8
        elif optimization_level == OptimizationLevel.INTERMEDIATE:
            return priority >= 6
        elif optimization_level == OptimizationLevel.ADVANCED:
            return priority >= 4
        elif optimization_level == OptimizationLevel.AGGRESSIVE:
            return priority >= 1
        
        return False
    
    def _optimize_subquery_to_join(self, query: str) -> str:
        """Convert IN subqueries to JOINs when possible"""
        
        # This is a simplified implementation
        # In practice, this would need sophisticated parsing
        
        subquery_pattern = r"WHERE\s+(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)(?:\s+WHERE\s+([^)]+))?\s*\)"
        
        def replace_subquery(match):
            col1, col2, table2, where_clause = match.groups()
            
            join_clause = f"JOIN {table2} ON {col1} = {table2}.{col2}"
            if where_clause:
                join_clause += f" AND {where_clause}"
            
            return join_clause
        
        return re.sub(subquery_pattern, replace_subquery, query, flags=re.IGNORECASE)
    
    def _add_safety_limit(self, query: str) -> str:
        """Add LIMIT to potentially unbounded queries"""
        
        if "LIMIT" not in query.upper() and "SELECT" in query.upper():
            # Add a reasonable default limit
            return f"{query.rstrip()} LIMIT 10000"
        
        return query
    
    def _optimize_wildcard_select(self, query: str) -> str:
        """Suggest specific columns instead of SELECT *"""
        
        # This would require table schema knowledge to be fully implemented
        # For now, just add a comment
        if "SELECT *" in query.upper():
            return f"/* Consider specifying columns instead of * */ {query}"
        
        return query
    
    def _push_down_predicates(self, query: str) -> str:
        """Push WHERE conditions closer to data sources"""
        
        # This is a complex optimization that would require full query parsing
        # For now, return the original query
        return query
    
    def _optimize_order_by(self, query: str) -> str:
        """Optimize ORDER BY clauses"""
        
        # Remove unnecessary ORDER BY in subqueries
        subquery_order_pattern = r"\(\s*SELECT\s+[^)]+ORDER\s+BY\s+[^)]+\)\s*(?!LIMIT)"
        
        def remove_subquery_order(match):
            subquery = match.group(0)
            return re.sub(r"ORDER\s+BY\s+[^)]+", "", subquery, flags=re.IGNORECASE)
        
        return re.sub(subquery_order_pattern, remove_subquery_order, query, flags=re.IGNORECASE | re.DOTALL)
    
    async def track_query_execution(self, query: str, execution_time: float, 
                                  rows_examined: int = 0, rows_returned: int = 0,
                                  execution_plan: Dict = None):
        """
        Track query execution for performance monitoring
        
        Args:
            query: Executed SQL query
            execution_time: Time taken to execute
            rows_examined: Number of rows examined
            rows_returned: Number of rows returned
            execution_plan: Database execution plan
        """
        
        query_hash = self._hash_query(query)
        
        execution_record = QueryExecution(
            query_hash=query_hash,
            original_query=query,
            optimized_query=None,
            execution_time=execution_time,
            rows_examined=rows_examined,
            rows_returned=rows_returned,
            index_usage=[],
            execution_plan=execution_plan or {},
            timestamp=datetime.now(),
            database_name=self.engine.url.database or "unknown",
            table_names=self._extract_table_names(query)
        )
        
        # Store execution history
        self.query_history[query_hash].append(execution_record)
        
        # Keep only last 100 executions per query
        if len(self.query_history[query_hash]) > 100:
            self.query_history[query_hash] = self.query_history[query_hash][-100:]
        
        # Check for slow queries
        await self._check_slow_query(query_hash, execution_record)
    
    async def _check_slow_query(self, query_hash: str, execution_record: QueryExecution):
        """Check if query should be flagged as slow"""
        
        if execution_record.execution_time >= self.slow_query_threshold:
            
            if query_hash in self.slow_queries:
                # Update existing slow query alert
                alert = self.slow_queries[query_hash]
                alert.execution_count += 1
                alert.last_seen = execution_record.timestamp
                
                if execution_record.execution_time > alert.max_execution_time:
                    alert.max_execution_time = execution_record.execution_time
                
                # Recalculate average
                executions = self.query_history[query_hash]
                avg_time = sum(e.execution_time for e in executions) / len(executions)
                alert.avg_execution_time = avg_time
                
            else:
                # Create new slow query alert
                alert_level = "critical" if execution_record.execution_time >= self.very_slow_query_threshold else "warning"
                
                analysis = await self.analyze_query(execution_record.original_query)
                optimization_suggestions = analysis.get("optimization_suggestions", [])
                
                self.slow_queries[query_hash] = SlowQueryAlert(
                    query_hash=query_hash,
                    query_text=execution_record.original_query,
                    avg_execution_time=execution_record.execution_time,
                    max_execution_time=execution_record.execution_time,
                    execution_count=1,
                    first_seen=execution_record.timestamp,
                    last_seen=execution_record.timestamp,
                    affected_tables=execution_record.table_names,
                    optimization_suggestions=optimization_suggestions,
                    alert_level=alert_level
                )
                
                logger.warning(f"Slow query detected: {query_hash[:8]} ({execution_record.execution_time:.2f}s)")
    
    async def get_slow_queries(self, limit: int = 50) -> List[SlowQueryAlert]:
        """Get list of slow queries sorted by severity"""
        
        slow_queries = list(self.slow_queries.values())
        
        # Sort by severity and execution time
        slow_queries.sort(key=lambda x: (
            {"critical": 3, "warning": 2, "info": 1}.get(x.alert_level, 0),
            x.avg_execution_time
        ), reverse=True)
        
        return slow_queries[:limit]
    
    async def get_index_recommendations(self, limit: int = 20) -> List[IndexRecommendation]:
        """Get prioritized index recommendations"""
        
        # Analyze all recent queries to generate comprehensive recommendations
        all_recommendations = []
        
        # Get recent unique queries
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_queries = set()
        
        for executions in self.query_history.values():
            for execution in executions:
                if execution.timestamp >= recent_cutoff:
                    recent_queries.add(execution.original_query)
        
        # Generate recommendations for recent queries
        for query in recent_queries:
            analysis = await self.analyze_query(query)
            recommendations = analysis.get("index_recommendations", [])
            all_recommendations.extend(recommendations)
        
        # Deduplicate and prioritize
        unique_recommendations = {}
        for rec in all_recommendations:
            key = (rec.table_name, tuple(rec.columns))
            if key not in unique_recommendations or rec.priority > unique_recommendations[key].priority:
                unique_recommendations[key] = rec
        
        # Sort by priority and estimated benefit
        sorted_recommendations = sorted(
            unique_recommendations.values(),
            key=lambda x: (x.priority, x.estimated_benefit),
            reverse=True
        )
        
        return sorted_recommendations[:limit]
    
    async def apply_index_recommendation(self, recommendation: IndexRecommendation, 
                                       session: Session) -> bool:
        """
        Apply an index recommendation
        
        Args:
            recommendation: Index recommendation to apply
            session: Database session
            
        Returns:
            bool: Success status
        """
        
        try:
            session.execute(text(recommendation.sql_create_statement))
            session.commit()
            
            logger.info(f"Created index: {recommendation.sql_create_statement}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            session.rollback()
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        total_queries = sum(len(executions) for executions in self.query_history.values())
        
        if total_queries == 0:
            return {"message": "No query history available"}
        
        all_executions = [
            execution for executions in self.query_history.values() 
            for execution in executions
        ]
        
        execution_times = [e.execution_time for e in all_executions]
        
        stats = {
            "total_queries_tracked": total_queries,
            "unique_queries": len(self.query_history),
            "slow_queries_count": len(self.slow_queries),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "p95_execution_time": sorted(execution_times)[int(0.95 * len(execution_times))],
            "queries_by_type": self._get_query_type_distribution(all_executions),
            "most_frequent_tables": self._get_table_usage_stats(all_executions),
            "recent_trends": self._get_recent_performance_trends()
        }
        
        return stats
    
    def _get_query_type_distribution(self, executions: List[QueryExecution]) -> Dict[str, int]:
        """Get distribution of query types"""
        
        type_counter = Counter()
        for execution in executions:
            query_type = self._detect_query_type(execution.original_query)
            type_counter[query_type.value] += 1
        
        return dict(type_counter)
    
    def _get_table_usage_stats(self, executions: List[QueryExecution]) -> Dict[str, int]:
        """Get table usage statistics"""
        
        table_counter = Counter()
        for execution in executions:
            for table in execution.table_names:
                table_counter[table] += 1
        
        return dict(table_counter.most_common(10))
    
    def _get_recent_performance_trends(self) -> Dict[str, Any]:
        """Get recent performance trends"""
        
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_executions = [
            execution for executions in self.query_history.values()
            for execution in executions
            if execution.timestamp >= recent_cutoff
        ]
        
        if not recent_executions:
            return {"message": "No recent executions"}
        
        recent_times = [e.execution_time for e in recent_executions]
        
        return {
            "recent_query_count": len(recent_executions),
            "recent_avg_time": sum(recent_times) / len(recent_times),
            "recent_max_time": max(recent_times),
            "queries_per_minute": len(recent_executions) / 60
        }


# Background task for automatic optimization
async def query_optimizer_maintenance(optimizer: QueryOptimizer):
    """
    Background task for automatic query optimization maintenance
    """
    
    while True:
        try:
            # Clean up old query history (keep last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            
            for query_hash in list(optimizer.query_history.keys()):
                executions = optimizer.query_history[query_hash]
                recent_executions = [
                    e for e in executions if e.timestamp >= cutoff
                ]
                
                if recent_executions:
                    optimizer.query_history[query_hash] = recent_executions
                else:
                    del optimizer.query_history[query_hash]
            
            # Update index recommendations based on recent query patterns
            # This would involve analyzing query patterns and updating recommendations
            
            # Clean up resolved slow queries
            resolved_queries = []
            for query_hash, alert in optimizer.slow_queries.items():
                if alert.last_seen < datetime.now() - timedelta(hours=24):
                    # Check if recent executions are now fast
                    recent_executions = [
                        e for e in optimizer.query_history.get(query_hash, [])
                        if e.timestamp >= datetime.now() - timedelta(hours=1)
                    ]
                    
                    if recent_executions:
                        avg_recent_time = sum(e.execution_time for e in recent_executions) / len(recent_executions)
                        if avg_recent_time < optimizer.slow_query_threshold:
                            resolved_queries.append(query_hash)
            
            for query_hash in resolved_queries:
                del optimizer.slow_queries[query_hash]
                logger.info(f"Resolved slow query: {query_hash[:8]}")
            
            # Sleep for 10 minutes before next maintenance cycle
            await asyncio.sleep(600)
            
        except Exception as e:
            logger.error(f"Query optimizer maintenance failed: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error


async def setup_enterprise_query_optimizer(engine, metadata: MetaData) -> QueryOptimizer:
    """
    Set up enterprise-level query optimization
    
    Args:
        engine: SQLAlchemy engine
        metadata: SQLAlchemy metadata
        
    Returns:
        QueryOptimizer: Configured query optimizer
    """
    
    optimizer = QueryOptimizer(engine, metadata)
    
    logger.info("Enterprise query optimizer setup completed")
    return optimizer