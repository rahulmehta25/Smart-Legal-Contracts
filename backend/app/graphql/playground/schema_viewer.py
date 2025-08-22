"""
Schema Viewer and Documentation Generator
"""

import json
from typing import Dict, Any, List, Optional
from graphql import build_schema, get_introspection_query, graphql_sync
from graphql.type import GraphQLSchema


class SchemaViewer:
    """View and document GraphQL schema"""
    
    def __init__(self, schema: GraphQLSchema):
        self.schema = schema
        self.introspection_result = None
    
    def get_introspection(self) -> Dict[str, Any]:
        """Get schema introspection result"""
        if self.introspection_result is None:
            introspection_query = get_introspection_query()
            result = graphql_sync(self.schema, introspection_query)
            self.introspection_result = result.data
        
        return self.introspection_result
    
    def get_types(self) -> List[Dict[str, Any]]:
        """Get all types from schema"""
        introspection = self.get_introspection()
        return introspection.get("__schema", {}).get("types", [])
    
    def get_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific type by name"""
        types = self.get_types()
        for type_def in types:
            if type_def.get("name") == name:
                return type_def
        return None
    
    def get_root_types(self) -> Dict[str, str]:
        """Get root operation types"""
        introspection = self.get_introspection()
        schema_def = introspection.get("__schema", {})
        
        return {
            "query": schema_def.get("queryType", {}).get("name"),
            "mutation": schema_def.get("mutationType", {}).get("name"),
            "subscription": schema_def.get("subscriptionType", {}).get("name")
        }
    
    def get_directives(self) -> List[Dict[str, Any]]:
        """Get schema directives"""
        introspection = self.get_introspection()
        return introspection.get("__schema", {}).get("directives", [])
    
    def generate_schema_doc(self) -> str:
        """Generate human-readable schema documentation"""
        introspection = self.get_introspection()
        schema_def = introspection.get("__schema", {})
        
        doc = []
        doc.append("# GraphQL Schema Documentation")
        doc.append("")
        
        # Root types
        root_types = self.get_root_types()
        doc.append("## Root Types")
        for op_type, type_name in root_types.items():
            if type_name:
                doc.append(f"- **{op_type.title()}**: `{type_name}`")
        doc.append("")
        
        # Object types
        object_types = [t for t in self.get_types() if t.get("kind") == "OBJECT" and not t.get("name", "").startswith("__")]
        if object_types:
            doc.append("## Object Types")
            for type_def in sorted(object_types, key=lambda t: t.get("name", "")):
                doc.extend(self._format_type_doc(type_def))
            doc.append("")
        
        # Input types
        input_types = [t for t in self.get_types() if t.get("kind") == "INPUT_OBJECT"]
        if input_types:
            doc.append("## Input Types")
            for type_def in sorted(input_types, key=lambda t: t.get("name", "")):
                doc.extend(self._format_type_doc(type_def))
            doc.append("")
        
        # Enums
        enum_types = [t for t in self.get_types() if t.get("kind") == "ENUM"]
        if enum_types:
            doc.append("## Enums")
            for type_def in sorted(enum_types, key=lambda t: t.get("name", "")):
                doc.extend(self._format_enum_doc(type_def))
            doc.append("")
        
        # Interfaces
        interface_types = [t for t in self.get_types() if t.get("kind") == "INTERFACE"]
        if interface_types:
            doc.append("## Interfaces")
            for type_def in sorted(interface_types, key=lambda t: t.get("name", "")):
                doc.extend(self._format_type_doc(type_def))
            doc.append("")
        
        # Directives
        directives = self.get_directives()
        if directives:
            doc.append("## Directives")
            for directive in sorted(directives, key=lambda d: d.get("name", "")):
                doc.extend(self._format_directive_doc(directive))
            doc.append("")
        
        return "\\n".join(doc)
    
    def _format_type_doc(self, type_def: Dict[str, Any]) -> List[str]:
        """Format type documentation"""
        lines = []
        name = type_def.get("name", "")
        description = type_def.get("description", "")
        kind = type_def.get("kind", "")
        
        # Type header
        lines.append(f"### {name}")
        if description:
            lines.append(f"*{description}*")
        lines.append(f"**Kind**: {kind}")
        lines.append("")
        
        # Fields
        fields = type_def.get("fields", [])
        if fields:
            lines.append("**Fields**:")
            for field in fields:
                field_name = field.get("name", "")
                field_type = self._format_type_ref(field.get("type", {}))
                field_desc = field.get("description", "")
                
                field_line = f"- `{field_name}`: {field_type}"
                if field_desc:
                    field_line += f" - {field_desc}"
                lines.append(field_line)
                
                # Arguments
                args = field.get("args", [])
                if args:
                    for arg in args:
                        arg_name = arg.get("name", "")
                        arg_type = self._format_type_ref(arg.get("type", {}))
                        arg_desc = arg.get("description", "")
                        
                        arg_line = f"  - `{arg_name}`: {arg_type}"
                        if arg_desc:
                            arg_line += f" - {arg_desc}"
                        lines.append(arg_line)
            lines.append("")
        
        # Input fields (for input types)
        input_fields = type_def.get("inputFields", [])
        if input_fields:
            lines.append("**Input Fields**:")
            for field in input_fields:
                field_name = field.get("name", "")
                field_type = self._format_type_ref(field.get("type", {}))
                field_desc = field.get("description", "")
                
                field_line = f"- `{field_name}`: {field_type}"
                if field_desc:
                    field_line += f" - {field_desc}"
                lines.append(field_line)
            lines.append("")
        
        return lines
    
    def _format_enum_doc(self, type_def: Dict[str, Any]) -> List[str]:
        """Format enum documentation"""
        lines = []
        name = type_def.get("name", "")
        description = type_def.get("description", "")
        
        lines.append(f"### {name}")
        if description:
            lines.append(f"*{description}*")
        lines.append("")
        
        enum_values = type_def.get("enumValues", [])
        if enum_values:
            lines.append("**Values**:")
            for value in enum_values:
                value_name = value.get("name", "")
                value_desc = value.get("description", "")
                
                value_line = f"- `{value_name}`"
                if value_desc:
                    value_line += f" - {value_desc}"
                lines.append(value_line)
            lines.append("")
        
        return lines
    
    def _format_directive_doc(self, directive: Dict[str, Any]) -> List[str]:
        """Format directive documentation"""
        lines = []
        name = directive.get("name", "")
        description = directive.get("description", "")
        locations = directive.get("locations", [])
        
        lines.append(f"### @{name}")
        if description:
            lines.append(f"*{description}*")
        
        if locations:
            lines.append(f"**Locations**: {', '.join(locations)}")
        
        args = directive.get("args", [])
        if args:
            lines.append("**Arguments**:")
            for arg in args:
                arg_name = arg.get("name", "")
                arg_type = self._format_type_ref(arg.get("type", {}))
                arg_desc = arg.get("description", "")
                
                arg_line = f"- `{arg_name}`: {arg_type}"
                if arg_desc:
                    arg_line += f" - {arg_desc}"
                lines.append(arg_line)
        
        lines.append("")
        return lines
    
    def _format_type_ref(self, type_ref: Dict[str, Any]) -> str:
        """Format type reference"""
        if not type_ref:
            return "Unknown"
        
        kind = type_ref.get("kind", "")
        
        if kind == "NON_NULL":
            inner_type = self._format_type_ref(type_ref.get("ofType", {}))
            return f"{inner_type}!"
        elif kind == "LIST":
            inner_type = self._format_type_ref(type_ref.get("ofType", {}))
            return f"[{inner_type}]"
        elif kind in ["SCALAR", "OBJECT", "INTERFACE", "UNION", "ENUM", "INPUT_OBJECT"]:
            return type_ref.get("name", "Unknown")
        else:
            return "Unknown"
    
    def get_field_usage_stats(self) -> Dict[str, Any]:
        """Get field usage statistics"""
        types = self.get_types()
        stats = {
            "total_types": len(types),
            "object_types": len([t for t in types if t.get("kind") == "OBJECT"]),
            "input_types": len([t for t in types if t.get("kind") == "INPUT_OBJECT"]),
            "enum_types": len([t for t in types if t.get("kind") == "ENUM"]),
            "interface_types": len([t for t in types if t.get("kind") == "INTERFACE"]),
            "scalar_types": len([t for t in types if t.get("kind") == "SCALAR"]),
            "union_types": len([t for t in types if t.get("kind") == "UNION"]),
        }
        
        # Count fields
        total_fields = 0
        for type_def in types:
            fields = type_def.get("fields", [])
            total_fields += len(fields)
        
        stats["total_fields"] = total_fields
        
        return stats
    
    def export_schema_json(self) -> str:
        """Export schema as JSON"""
        return json.dumps(self.get_introspection(), indent=2)
    
    def export_schema_sdl(self) -> str:
        """Export schema as SDL (Schema Definition Language)"""
        from graphql import print_schema
        return print_schema(self.schema)
    
    def validate_schema(self) -> List[str]:
        """Validate schema and return any errors"""
        from graphql import validate_schema
        
        errors = validate_schema(self.schema)
        return [str(error) for error in errors]
    
    def get_deprecated_fields(self) -> List[Dict[str, Any]]:
        """Get list of deprecated fields"""
        deprecated_fields = []
        
        for type_def in self.get_types():
            if type_def.get("kind") != "OBJECT":
                continue
            
            type_name = type_def.get("name", "")
            fields = type_def.get("fields", [])
            
            for field in fields:
                if field.get("isDeprecated"):
                    deprecated_fields.append({
                        "type": type_name,
                        "field": field.get("name", ""),
                        "reason": field.get("deprecationReason", "")
                    })
        
        return deprecated_fields
    
    def get_complex_fields(self) -> List[Dict[str, Any]]:
        """Get fields with high complexity (many arguments or nested types)"""
        complex_fields = []
        
        for type_def in self.get_types():
            if type_def.get("kind") != "OBJECT":
                continue
            
            type_name = type_def.get("name", "")
            fields = type_def.get("fields", [])
            
            for field in fields:
                args = field.get("args", [])
                if len(args) > 5:  # Fields with many arguments
                    complex_fields.append({
                        "type": type_name,
                        "field": field.get("name", ""),
                        "argument_count": len(args),
                        "complexity_reason": "Many arguments"
                    })
        
        return complex_fields