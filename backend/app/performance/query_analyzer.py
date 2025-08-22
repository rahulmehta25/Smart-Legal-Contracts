"""
Database Query Analyzer and Optimizer
Analyzes query performance, identifies slow queries, and provides optimization recommendations
"""

import time
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import sqlparse
from sqlalchemy import event, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool
import threading
import queue

@dataclass
class QueryProfile:
    """Profile data for a database query"""
    query: str
    normalized_query: str
    query_hash: str
    execution_time: float
    rows_affected: int
    timestamp: datetime
    parameters: Dict
    explain_plan: Optional[Dict] = None
    index_usage: List[str] = field(default_factory=list)
    table_scans: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

@dataclass 
class QueryPattern:
    """Pattern analysis for similar queries"""
    pattern: str
    count: int
    total_time: float
    avg_time: float
    max_time: float
    min_time: float
    variations: List[str]

class QueryAnalyzer:
    """Advanced database query analyzer with optimization recommendations"""
    
    def __init__(self, 
                 slow_query_threshold: float = 0.1,
                 enable_explain: bool = True,
                 cache_size: int = 1000):
        self.slow_query_threshold = slow_query_threshold
        self.enable_explain = enable_explain
        self.query_cache: Dict[str, QueryProfile] = {}
        self.query_history: List[QueryProfile] = []
        self.slow_queries: List[QueryProfile] = []
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.table_stats: Dict[str, Dict] = defaultdict(lambda: {
            'selects': 0,
            'inserts': 0,
            'updates': 0,
            'deletes': 0,
            'total_time': 0.0
        })
        self.index_recommendations: List[Dict] = []
        self._lock = threading.Lock()
        self._analysis_queue = queue.Queue()
        self._start_analysis_worker()
    
    def _start_analysis_worker(self):
        """Start background worker for query analysis"""
        def worker():
            while True:
                try:
                    profile = self._analysis_queue.get(timeout=1)
                    if profile:
                        self._analyze_query_async(profile)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Analysis worker error: {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def setup_monitoring(self, engine: Engine):
        """Setup query monitoring for SQLAlchemy engine"""
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.perf_counter())
            conn.info.setdefault('query_statement', []).append(statement)
            conn.info.setdefault('query_parameters', []).append(parameters)
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.perf_counter() - conn.info['query_start_time'].pop()
            
            # Create query profile
            profile = self._create_profile(
                statement=conn.info['query_statement'].pop(),
                parameters=conn.info['query_parameters'].pop(),
                execution_time=total_time,
                rows_affected=cursor.rowcount if hasattr(cursor, 'rowcount') else 0
            )
            
            # Analyze query
            self.analyze_query(profile, conn)
        
        @event.listens_for(Pool, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Enable query statistics when connection is created"""
            if hasattr(dbapi_conn, 'execute'):
                # Enable statistics for PostgreSQL
                try:
                    dbapi_conn.execute("SET track_io_timing = ON")
                    dbapi_conn.execute("SET log_statement_stats = ON")
                except Exception:
                    pass
    
    def _create_profile(self, 
                       statement: str, 
                       parameters: Any,
                       execution_time: float,
                       rows_affected: int) -> QueryProfile:
        """Create a query profile"""
        normalized = self._normalize_query(statement)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        return QueryProfile(
            query=statement,
            normalized_query=normalized,
            query_hash=query_hash,
            execution_time=execution_time,
            rows_affected=rows_affected,
            timestamp=datetime.now(),
            parameters=parameters if isinstance(parameters, dict) else {}
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern detection"""
        # Parse and format SQL
        formatted = sqlparse.format(
            query,
            reindent=True,
            keyword_case='upper'
        )
        
        # Remove specific values to find patterns
        # Replace numbers with ?
        normalized = re.sub(r'\b\d+\b', '?', formatted)
        # Replace quoted strings with ?
        normalized = re.sub(r"'[^']*'", '?', normalized)
        normalized = re.sub(r'"[^"]*"', '?', normalized)
        
        return normalized.strip()
    
    def analyze_query(self, profile: QueryProfile, connection=None):
        """Analyze a query and provide optimization suggestions"""
        with self._lock:
            # Add to history
            self.query_history.append(profile)
            
            # Check if slow query
            if profile.execution_time > self.slow_query_threshold:
                self.slow_queries.append(profile)
                profile.suggestions.append(
                    f"Query exceeded threshold ({profile.execution_time:.3f}s > {self.slow_query_threshold}s)"
                )
            
            # Update pattern tracking
            self._track_pattern(profile)
            
            # Update table statistics
            self._update_table_stats(profile)
            
            # Get explain plan if available
            if self.enable_explain and connection:
                self._get_explain_plan(profile, connection)
            
            # Analyze for optimization opportunities
            self._analyze_for_optimizations(profile)
            
            # Queue for async analysis
            self._analysis_queue.put(profile)
        
        return profile
    
    def _track_pattern(self, profile: QueryProfile):
        """Track query patterns"""
        pattern_key = profile.normalized_query
        
        if pattern_key not in self.query_patterns:
            self.query_patterns[pattern_key] = QueryPattern(
                pattern=pattern_key,
                count=0,
                total_time=0,
                avg_time=0,
                max_time=0,
                min_time=float('inf'),
                variations=[]
            )
        
        pattern = self.query_patterns[pattern_key]
        pattern.count += 1
        pattern.total_time += profile.execution_time
        pattern.avg_time = pattern.total_time / pattern.count
        pattern.max_time = max(pattern.max_time, profile.execution_time)
        pattern.min_time = min(pattern.min_time, profile.execution_time)
        
        # Track variations
        if profile.query not in pattern.variations:
            pattern.variations.append(profile.query[:200])  # Keep first 200 chars
    
    def _update_table_stats(self, profile: QueryProfile):
        """Update table-level statistics"""
        # Extract table names from query
        tables = self._extract_table_names(profile.query)
        
        # Determine operation type
        operation = self._get_operation_type(profile.query)
        
        for table in tables:
            self.table_stats[table][operation] += 1
            self.table_stats[table]['total_time'] += profile.execution_time
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        
        # Parse SQL
        parsed = sqlparse.parse(query)[0]
        
        # Find table names
        from_seen = False
        for token in parsed.tokens:
            if from_seen:
                if token.ttype is None:
                    tables.append(str(token).strip())
                from_seen = False
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ['JOIN', 'INTO', 'UPDATE']:
                # Next non-keyword token is likely a table
                from_seen = True
        
        return tables
    
    def _get_operation_type(self, query: str) -> str:
        """Determine the operation type of a query"""
        query_upper = query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            return 'selects'
        elif query_upper.startswith('INSERT'):
            return 'inserts'
        elif query_upper.startswith('UPDATE'):
            return 'updates'
        elif query_upper.startswith('DELETE'):
            return 'deletes'
        else:
            return 'other'
    
    def _get_explain_plan(self, profile: QueryProfile, connection):
        """Get query execution plan"""
        try:
            # PostgreSQL EXPLAIN
            if 'postgresql' in str(connection.engine.url):
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {profile.query}"
                result = connection.execute(text(explain_query))
                profile.explain_plan = result.fetchone()[0]
                self._analyze_explain_plan(profile)
            
            # MySQL EXPLAIN
            elif 'mysql' in str(connection.engine.url):
                explain_query = f"EXPLAIN {profile.query}"
                result = connection.execute(text(explain_query))
                profile.explain_plan = [dict(row) for row in result]
                self._analyze_mysql_explain(profile)
                
        except Exception as e:
            # Explain might fail for some queries
            pass
    
    def _analyze_explain_plan(self, profile: QueryProfile):
        """Analyze PostgreSQL explain plan"""
        if not profile.explain_plan:
            return
        
        plan = profile.explain_plan[0]['Plan'] if isinstance(profile.explain_plan, list) else profile.explain_plan
        
        # Check for sequential scans
        if 'Seq Scan' in str(plan):
            profile.table_scans.append('Sequential scan detected')
            profile.suggestions.append('Consider adding an index to avoid sequential scan')
        
        # Check for missing indexes
        if 'Index Scan' not in str(plan) and 'Index Only Scan' not in str(plan):
            profile.suggestions.append('Query may benefit from indexes')
        
        # Check cost
        if 'Total Cost' in plan and plan['Total Cost'] > 1000:
            profile.suggestions.append(f"High query cost: {plan['Total Cost']}")
    
    def _analyze_mysql_explain(self, profile: QueryProfile):
        """Analyze MySQL explain plan"""
        if not profile.explain_plan:
            return
        
        for row in profile.explain_plan:
            # Check for full table scans
            if row.get('type') == 'ALL':
                profile.table_scans.append(f"Full table scan on {row.get('table', 'unknown')}")
                profile.suggestions.append(f"Add index to table {row.get('table', 'unknown')}")
            
            # Check for missing keys
            if not row.get('key'):
                profile.suggestions.append(f"No index used for table {row.get('table', 'unknown')}")
            
            # Check rows examined
            if row.get('rows', 0) > 10000:
                profile.suggestions.append(f"Large number of rows examined: {row.get('rows')}")
    
    def _analyze_for_optimizations(self, profile: QueryProfile):
        """Analyze query for optimization opportunities"""
        query_upper = profile.query.upper()
        
        # Check for SELECT *
        if 'SELECT *' in query_upper:
            profile.suggestions.append('Avoid SELECT *, specify needed columns')
        
        # Check for missing WHERE clause
        if 'WHERE' not in query_upper and 'SELECT' in query_upper:
            profile.suggestions.append('Query has no WHERE clause, may return too many rows')
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+'%", query_upper):
            profile.suggestions.append('LIKE with leading wildcard prevents index usage')
        
        # Check for OR conditions
        if ' OR ' in query_upper:
            profile.suggestions.append('OR conditions may prevent index usage, consider UNION')
        
        # Check for NOT IN
        if 'NOT IN' in query_upper:
            profile.suggestions.append('NOT IN can be slow, consider NOT EXISTS or LEFT JOIN')
        
        # Check for functions in WHERE
        if re.search(r'WHERE.*\(.*\)', query_upper):
            profile.suggestions.append('Functions in WHERE clause may prevent index usage')
        
        # Check for implicit conversions
        if re.search(r"WHERE\s+\w+\s*=\s*'?\d+'?", profile.query):
            profile.suggestions.append('Check for implicit type conversions in WHERE clause')
        
        # Check for missing JOIN conditions
        if 'JOIN' in query_upper and 'ON' not in query_upper:
            profile.suggestions.append('JOIN without ON condition creates cartesian product')
    
    def _analyze_query_async(self, profile: QueryProfile):
        """Perform async deep analysis of query"""
        # Additional complex analysis that doesn't block
        
        # Check for N+1 query patterns
        self._detect_n_plus_one(profile)
        
        # Generate index recommendations
        self._generate_index_recommendations(profile)
        
        # Check for query similarity
        self._check_query_similarity(profile)
    
    def _detect_n_plus_one(self, profile: QueryProfile):
        """Detect N+1 query patterns"""
        # Look for repeated similar queries in recent history
        recent_queries = self.query_history[-100:]
        similar_count = 0
        
        for recent in recent_queries:
            if recent.normalized_query == profile.normalized_query:
                similar_count += 1
        
        if similar_count > 10:
            profile.suggestions.append(f'Possible N+1 pattern detected ({similar_count} similar queries)')
    
    def _generate_index_recommendations(self, profile: QueryProfile):
        """Generate index recommendations based on query patterns"""
        # Extract WHERE clause columns
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', profile.query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column names
            columns = re.findall(r'(\w+)\s*[=<>]', where_clause)
            
            if columns:
                tables = self._extract_table_names(profile.query)
                for table in tables:
                    for column in columns:
                        recommendation = {
                            'table': table,
                            'column': column,
                            'type': 'btree',
                            'reason': f'Column {column} used in WHERE clause'
                        }
                        
                        if recommendation not in self.index_recommendations:
                            self.index_recommendations.append(recommendation)
    
    def _check_query_similarity(self, profile: QueryProfile):
        """Check for similar queries that could be combined"""
        similar_queries = []
        
        for pattern_key, pattern in self.query_patterns.items():
            if pattern.count > 5 and pattern_key != profile.normalized_query:
                # Calculate similarity
                similarity = self._calculate_similarity(
                    profile.normalized_query, 
                    pattern_key
                )
                
                if similarity > 0.8:
                    similar_queries.append(pattern)
        
        if similar_queries:
            profile.suggestions.append(
                f'Found {len(similar_queries)} similar query patterns that might be combined'
            )
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple similarity based on common tokens
        tokens1 = set(query1.split())
        tokens2 = set(query2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryProfile]:
        """Get slowest queries"""
        return sorted(
            self.slow_queries, 
            key=lambda x: x.execution_time, 
            reverse=True
        )[:limit]
    
    def get_most_frequent_queries(self, limit: int = 10) -> List[QueryPattern]:
        """Get most frequently executed query patterns"""
        return sorted(
            self.query_patterns.values(),
            key=lambda x: x.count,
            reverse=True
        )[:limit]
    
    def get_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        report = {
            'summary': {
                'total_queries': len(self.query_history),
                'slow_queries': len(self.slow_queries),
                'unique_patterns': len(self.query_patterns),
                'tables_analyzed': len(self.table_stats)
            },
            'slow_queries': [
                {
                    'query': q.query[:200],
                    'execution_time': q.execution_time,
                    'timestamp': q.timestamp.isoformat(),
                    'suggestions': q.suggestions
                }
                for q in self.get_slow_queries()
            ],
            'frequent_patterns': [
                {
                    'pattern': p.pattern[:200],
                    'count': p.count,
                    'avg_time': p.avg_time,
                    'total_time': p.total_time
                }
                for p in self.get_most_frequent_queries()
            ],
            'table_statistics': dict(self.table_stats),
            'index_recommendations': self.index_recommendations[:20],
            'optimization_suggestions': self._generate_optimization_suggestions()
        }
        
        return report
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate general optimization suggestions"""
        suggestions = []
        
        # Check for tables with many sequential scans
        for table, stats in self.table_stats.items():
            if stats['selects'] > 100 and stats['total_time'] > 10:
                suggestions.append(f"Table '{table}' has high SELECT activity, review indexes")
        
        # Check for patterns with high variation
        for pattern in self.query_patterns.values():
            if pattern.max_time / pattern.avg_time > 10:
                suggestions.append(
                    f"Query pattern has high time variance (max: {pattern.max_time:.3f}s, "
                    f"avg: {pattern.avg_time:.3f}s), investigate data distribution"
                )
        
        # General recommendations
        if len(self.slow_queries) > 50:
            suggestions.append("High number of slow queries detected, consider query optimization sprint")
        
        if len(self.index_recommendations) > 10:
            suggestions.append(f"Consider adding {len(self.index_recommendations)} recommended indexes")
        
        # Check for N+1 patterns
        n_plus_one_patterns = sum(
            1 for p in self.query_patterns.values() 
            if p.count > 100 and p.avg_time < 0.01
        )
        if n_plus_one_patterns > 0:
            suggestions.append(f"Detected {n_plus_one_patterns} possible N+1 query patterns")
        
        return suggestions
    
    def export_recommendations_sql(self) -> str:
        """Export index recommendations as SQL"""
        sql_statements = []
        
        for rec in self.index_recommendations:
            index_name = f"idx_{rec['table']}_{rec['column']}"
            sql = f"CREATE INDEX {index_name} ON {rec['table']} ({rec['column']}) USING {rec['type']};"
            sql_statements.append(sql)
        
        return "\n".join(sql_statements)

# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = QueryAnalyzer(slow_query_threshold=0.1)
    
    # Setup with SQLAlchemy engine
    # engine = create_engine('postgresql://user:pass@localhost/db')
    # analyzer.setup_monitoring(engine)
    
    # Simulate some queries
    sample_queries = [
        "SELECT * FROM users WHERE id = 123",
        "SELECT name, email FROM users WHERE status = 'active'",
        "SELECT u.*, p.* FROM users u JOIN posts p ON u.id = p.user_id",
        "SELECT COUNT(*) FROM orders WHERE created_at > '2024-01-01'",
        "UPDATE users SET last_login = NOW() WHERE id = 456"
    ]
    
    # Analyze queries
    for query in sample_queries:
        profile = analyzer._create_profile(
            statement=query,
            parameters={},
            execution_time=0.05 + (len(query) * 0.001),
            rows_affected=10
        )
        analyzer.analyze_query(profile)
    
    # Get report
    report = analyzer.get_optimization_report()
    print(json.dumps(report, indent=2))