"""
Query Complexity Analysis for GraphQL
"""

import time
from typing import Dict, Any, List, Optional
from graphql import DocumentNode, FieldNode, OperationDefinitionNode, validate, ValidationRule
from graphql.validation import ValidationContext
from graphql.error import GraphQLError
from graphql.language.ast import Node


class QueryComplexityError(Exception):
    """Exception raised when query complexity exceeds limits"""
    
    def __init__(self, message: str, complexity: int, limit: int):
        super().__init__(message)
        self.complexity = complexity
        self.limit = limit


class ComplexityAnalyzer:
    """Analyzes GraphQL query complexity to prevent resource exhaustion"""
    
    def __init__(
        self,
        max_complexity: int = 1000,
        max_depth: int = 10,
        scaler_map: Optional[Dict[str, int]] = None,
        introspection_complexity: int = 1000
    ):
        self.max_complexity = max_complexity
        self.max_depth = max_depth
        self.scaler_map = scaler_map or {}
        self.introspection_complexity = introspection_complexity
        
        # Default complexity values for field types
        self.default_complexities = {
            "String": 1,
            "Int": 1,
            "Float": 1,
            "Boolean": 1,
            "ID": 1,
            "DateTime": 1,
            "JSON": 2,
            "List": 2,  # Base complexity for lists
            "Connection": 10,  # Base complexity for connections
            "Node": 5,  # Base complexity for node interfaces
        }
        
        # Field-specific complexity rules
        self.field_complexities = {
            # Document operations
            "documents": 20,
            "document": 10,
            "searchDocuments": 50,
            
            # Analysis operations
            "analyses": 30,
            "analysis": 15,
            "quickAnalysis": 100,  # High complexity for AI analysis
            
            # Detection operations
            "detections": 25,
            "detection": 12,
            
            # Pattern operations
            "patterns": 15,
            "pattern": 8,
            
            # User operations
            "users": 10,
            "user": 5,
            
            # Statistics (expensive aggregations)
            "systemStats": 200,
            "documentStats": 100,
            "detectionStats": 150,
            
            # Relationships (potentially expensive)
            "chunks": 15,
            "clauses": 20,
            "comments": 10,
            "annotations": 8,
            
            # Mutations (generally more expensive)
            "uploadDocument": 50,
            "requestAnalysis": 200,
            "quickAnalysis": 300,
            "createPattern": 25,
            "updatePattern": 20,
            "deletePattern": 15,
            "registerUser": 30,
            "loginUser": 20,
        }
        
        # Multiplier fields that affect complexity calculation
        self.multiplier_fields = {
            "first": lambda value: max(1, min(value or 20, 100)),  # Pagination limit
            "limit": lambda value: max(1, min(value or 20, 100)),
            "batchSize": lambda value: max(1, min(value or 10, 50)),
        }
    
    def analyze_query(self, document: DocumentNode, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze query complexity and depth"""
        variables = variables or {}
        
        try:
            analysis_result = {
                "complexity": 0,
                "depth": 0,
                "field_count": 0,
                "operation_type": "query",
                "is_introspection": False,
                "warnings": [],
                "breakdown": {}
            }
            
            for definition in document.definitions:
                if isinstance(definition, OperationDefinitionNode):
                    analysis_result["operation_type"] = definition.operation.value
                    
                    # Check for introspection queries
                    if self._is_introspection_query(definition):
                        analysis_result["is_introspection"] = True
                        analysis_result["complexity"] = self.introspection_complexity
                        analysis_result["warnings"].append("Introspection query detected")
                        continue
                    
                    # Calculate complexity and depth
                    field_analysis = self._analyze_selection_set(
                        definition.selection_set,
                        variables,
                        depth=1,
                        parent_type="Query" if definition.operation.value == "query" else definition.operation.value.title()
                    )
                    
                    analysis_result["complexity"] += field_analysis["complexity"]
                    analysis_result["depth"] = max(analysis_result["depth"], field_analysis["depth"])
                    analysis_result["field_count"] += field_analysis["field_count"]
                    analysis_result["breakdown"].update(field_analysis["breakdown"])
            
            # Check limits
            if analysis_result["complexity"] > self.max_complexity:
                raise QueryComplexityError(
                    f"Query complexity {analysis_result['complexity']} exceeds limit {self.max_complexity}",
                    analysis_result["complexity"],
                    self.max_complexity
                )
            
            if analysis_result["depth"] > self.max_depth:
                raise QueryComplexityError(
                    f"Query depth {analysis_result['depth']} exceeds limit {self.max_depth}",
                    analysis_result["depth"],
                    self.max_depth
                )
            
            return analysis_result
            
        except GraphQLError as e:
            raise QueryComplexityError(f"GraphQL validation error: {str(e)}", 0, 0)
        except Exception as e:
            raise QueryComplexityError(f"Complexity analysis error: {str(e)}", 0, 0)
    
    def _analyze_selection_set(
        self,
        selection_set,
        variables: Dict[str, Any],
        depth: int = 1,
        parent_type: str = "Query"
    ) -> Dict[str, Any]:
        """Analyze a selection set (fields, fragments, etc.)"""
        total_complexity = 0
        max_depth = depth
        field_count = 0
        breakdown = {}
        
        if not selection_set or not selection_set.selections:
            return {
                "complexity": 0,
                "depth": depth,
                "field_count": 0,
                "breakdown": {}
            }
        
        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                field_name = selection.name.value
                field_complexity = self._calculate_field_complexity(
                    field_name,
                    selection,
                    variables,
                    parent_type
                )
                
                # Handle nested selections
                nested_analysis = {"complexity": 0, "depth": depth, "field_count": 0, "breakdown": {}}
                if selection.selection_set:
                    nested_analysis = self._analyze_selection_set(
                        selection.selection_set,
                        variables,
                        depth + 1,
                        self._get_field_type(field_name, parent_type)
                    )
                
                # Calculate total field complexity
                total_field_complexity = field_complexity + nested_analysis["complexity"]
                total_complexity += total_field_complexity
                max_depth = max(max_depth, nested_analysis["depth"])
                field_count += 1 + nested_analysis["field_count"]
                
                # Track complexity breakdown
                breakdown[f"{parent_type}.{field_name}"] = {
                    "base_complexity": field_complexity,
                    "nested_complexity": nested_analysis["complexity"],
                    "total_complexity": total_field_complexity,
                    "depth": nested_analysis["depth"] - depth + 1,
                    "multipliers": self._get_field_multipliers(selection, variables)
                }
                
                # Merge nested breakdowns
                breakdown.update(nested_analysis["breakdown"])
        
        return {
            "complexity": total_complexity,
            "depth": max_depth,
            "field_count": field_count,
            "breakdown": breakdown
        }
    
    def _calculate_field_complexity(
        self,
        field_name: str,
        field_node: FieldNode,
        variables: Dict[str, Any],
        parent_type: str
    ) -> int:
        """Calculate complexity for a specific field"""
        # Get base complexity for the field
        base_complexity = self.field_complexities.get(
            field_name,
            self.default_complexities.get(parent_type, 5)
        )
        
        # Apply multipliers from arguments
        multiplier = 1
        if field_node.arguments:
            for arg in field_node.arguments:
                arg_name = arg.name.value
                if arg_name in self.multiplier_fields:
                    arg_value = self._resolve_argument_value(arg.value, variables)
                    multiplier *= self.multiplier_fields[arg_name](arg_value)
        
        # Apply scaler map if available
        scaler_key = f"{parent_type}.{field_name}"
        if scaler_key in self.scaler_map:
            multiplier *= self.scaler_map[scaler_key]
        
        return int(base_complexity * multiplier)
    
    def _get_field_multipliers(self, field_node: FieldNode, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Get multipliers applied to a field"""
        multipliers = {}
        
        if field_node.arguments:
            for arg in field_node.arguments:
                arg_name = arg.name.value
                if arg_name in self.multiplier_fields:
                    arg_value = self._resolve_argument_value(arg.value, variables)
                    multipliers[arg_name] = arg_value
        
        return multipliers
    
    def _resolve_argument_value(self, value_node, variables: Dict[str, Any]) -> Any:
        """Resolve argument value from AST node"""
        if hasattr(value_node, 'value'):
            return value_node.value
        elif hasattr(value_node, 'name') and value_node.name.value in variables:
            return variables[value_node.name.value]
        elif hasattr(value_node, 'values'):
            return [self._resolve_argument_value(v, variables) for v in value_node.values]
        else:
            return None
    
    def _get_field_type(self, field_name: str, parent_type: str) -> str:
        """Get the return type of a field"""
        # This would typically use schema introspection
        # For now, use simple heuristics
        
        if field_name.endswith("s") and field_name != "systemStats":
            return "List"
        elif field_name.endswith("Connection"):
            return "Connection"
        elif field_name in ["document", "user", "analysis", "detection", "pattern"]:
            return field_name.title()
        else:
            return "Object"
    
    def _is_introspection_query(self, operation: OperationDefinitionNode) -> bool:
        """Check if the query is an introspection query"""
        if not operation.selection_set:
            return False
        
        introspection_fields = {"__schema", "__type", "__typename"}
        
        for selection in operation.selection_set.selections:
            if isinstance(selection, FieldNode):
                if selection.name.value in introspection_fields:
                    return True
        
        return False
    
    def get_complexity_breakdown(self, analysis_result: Dict[str, Any]) -> str:
        """Get human-readable complexity breakdown"""
        lines = [
            f"Query Complexity Analysis:",
            f"  Total Complexity: {analysis_result['complexity']}/{self.max_complexity}",
            f"  Max Depth: {analysis_result['depth']}/{self.max_depth}",
            f"  Field Count: {analysis_result['field_count']}",
            f"  Operation Type: {analysis_result['operation_type']}",
        ]
        
        if analysis_result.get("is_introspection"):
            lines.append("  Type: Introspection Query")
        
        if analysis_result.get("warnings"):
            lines.append("  Warnings:")
            for warning in analysis_result["warnings"]:
                lines.append(f"    - {warning}")
        
        if analysis_result.get("breakdown"):
            lines.append("  Field Breakdown:")
            for field_path, details in analysis_result["breakdown"].items():
                lines.append(
                    f"    {field_path}: {details['total_complexity']} "
                    f"(base: {details['base_complexity']}, nested: {details['nested_complexity']})"
                )
                if details.get("multipliers"):
                    multiplier_str = ", ".join(f"{k}={v}" for k, v in details["multipliers"].items())
                    lines.append(f"      multipliers: {multiplier_str}")
        
        return "\n".join(lines)


# Query complexity validation rule
class QueryComplexityValidationRule(ValidationRule):
    """GraphQL validation rule for query complexity"""
    
    def __init__(self, analyzer: ComplexityAnalyzer):
        super().__init__()
        self.analyzer = analyzer
    
    def enter_document(self, node: DocumentNode, *_):
        """Validate query complexity when entering document"""
        try:
            analysis_result = self.analyzer.analyze_query(node)
            
            # The validation will be handled by the analyzer
            # If no exception is raised, the query passes validation
            
        except QueryComplexityError as e:
            self.report_error(
                GraphQLError(
                    message=str(e),
                    nodes=[node],
                    extensions={
                        "code": "QUERY_COMPLEXITY_LIMIT_EXCEEDED",
                        "complexity": e.complexity,
                        "limit": e.limit
                    }
                )
            )


# Complexity middleware
class ComplexityMiddleware:
    """Middleware for query complexity analysis"""
    
    def __init__(self, analyzer: ComplexityAnalyzer, log_analysis: bool = True):
        self.analyzer = analyzer
        self.log_analysis = log_analysis
    
    async def process_request(self, info, **kwargs):
        """Process request and analyze complexity"""
        start_time = time.time()
        
        try:
            # Analyze query complexity
            document = info.context.get("document")
            variables = info.context.get("variables", {})
            
            if document:
                analysis_result = self.analyzer.analyze_query(document, variables)
                
                # Add analysis to context
                info.context["complexity_analysis"] = analysis_result
                
                # Log analysis if enabled
                if self.log_analysis:
                    analysis_time = (time.time() - start_time) * 1000
                    print(f"Query complexity analysis completed in {analysis_time:.2f}ms")
                    if analysis_result["complexity"] > self.analyzer.max_complexity * 0.8:
                        print(f"High complexity query detected: {analysis_result['complexity']}")
            
        except QueryComplexityError as e:
            # Add error to context for proper error handling
            info.context["complexity_error"] = e
            raise e


# Query timeout based on complexity
class ComplexityBasedTimeout:
    """Calculate query timeout based on complexity"""
    
    def __init__(
        self,
        base_timeout: int = 30,  # Base timeout in seconds
        complexity_factor: float = 0.01,  # Additional seconds per complexity point
        max_timeout: int = 300  # Maximum timeout in seconds
    ):
        self.base_timeout = base_timeout
        self.complexity_factor = complexity_factor
        self.max_timeout = max_timeout
    
    def calculate_timeout(self, complexity: int) -> int:
        """Calculate timeout based on query complexity"""
        timeout = self.base_timeout + (complexity * self.complexity_factor)
        return min(int(timeout), self.max_timeout)
    
    def get_timeout_from_context(self, context: Dict[str, Any]) -> int:
        """Get timeout from request context"""
        analysis = context.get("complexity_analysis", {})
        complexity = analysis.get("complexity", 100)
        return self.calculate_timeout(complexity)


# Pre-computed complexity cache
class ComplexityCache:
    """Cache for pre-computed query complexities"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_complexity(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached complexity analysis"""
        if query_hash in self.cache:
            analysis, timestamp = self.cache[query_hash]
            if time.time() - timestamp < self.ttl:
                return analysis
            else:
                del self.cache[query_hash]
        return None
    
    def set_complexity(self, query_hash: str, analysis: Dict[str, Any]):
        """Cache complexity analysis"""
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[query_hash] = (analysis, time.time())
    
    def clear_cache(self):
        """Clear complexity cache"""
        self.cache.clear()


# Global instances
default_complexity_analyzer = ComplexityAnalyzer()
complexity_cache = ComplexityCache()
complexity_based_timeout = ComplexityBasedTimeout()