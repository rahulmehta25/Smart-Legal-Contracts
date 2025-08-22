"""
Federation Directives for GraphQL Schema
"""

import strawberry
from typing import List, Optional, Any
from strawberry.directive import DirectiveLocation


# Federation directives
@strawberry.directive(
    locations=[DirectiveLocation.OBJECT, DirectiveLocation.INTERFACE],
    description="Indicates that an object type is an entity in the federated graph"
)
class Key:
    """@key directive for federation entities"""
    fields: str  # FieldSet scalar representing the key fields
    resolvable: bool = True  # Whether this entity can be resolved by this service


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that a field is owned by another service"
)
class External:
    """@external directive for fields owned by other services"""
    pass


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that this field requires data from another service"
)
class Requires:
    """@requires directive for fields that need external data"""
    fields: str  # FieldSet scalar representing required fields


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that this field provides data to other services"
)
class Provides:
    """@provides directive for fields that provide data to other services"""
    fields: str  # FieldSet scalar representing provided fields


@strawberry.directive(
    locations=[DirectiveLocation.OBJECT, DirectiveLocation.INTERFACE],
    description="Indicates that an object type should be extended by other services"
)
class Extends:
    """@extends directive for extending types from other services"""
    pass


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that a field should be shareable across services"
)
class Shareable:
    """@shareable directive for fields that can be shared"""
    pass


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that a field should not be overridden"
)
class Override:
    """@override directive for field overrides"""
    from_: str = strawberry.field(name="from")  # Service name to override from


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates that a field is for internal use only"
)
class Inaccessible:
    """@inaccessible directive for internal-only fields"""
    pass


@strawberry.directive(
    locations=[DirectiveLocation.SCHEMA],
    description="Links to another federated schema"
)
class Link:
    """@link directive for schema linking"""
    url: str
    as_: Optional[str] = strawberry.field(name="as", default=None)
    for_: Optional[str] = strawberry.field(name="for", default=None)
    import_: Optional[List[str]] = strawberry.field(name="import", default=None)


# Composition directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates field composition rules"
)
class ComposeDirective:
    """@composeDirective for field composition"""
    name: str


@strawberry.directive(
    locations=[DirectiveLocation.OBJECT],
    description="Indicates interface object implementation"
)
class InterfaceObject:
    """@interfaceObject directive"""
    pass


# Policy directives for security
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Authentication policy for field access"
)
class Authenticated:
    """@authenticated directive for auth requirements"""
    pass


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Authorization policy for field access"
)
class RequiresScopes:
    """@requiresScopes directive for authorization"""
    scopes: List[List[str]]  # List of scope lists (OR of ANDs)


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Policy directive for access control"
)
class Policy:
    """@policy directive for custom policies"""
    policies: List[List[str]]


# Cost analysis directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Query cost analysis"
)
class Cost:
    """@cost directive for query cost analysis"""
    complexity: int = 1
    multipliers: Optional[List[str]] = None
    use_multipliers: bool = True


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Query complexity analysis"
)
class Complexity:
    """@complexity directive for query complexity"""
    value: int
    introspection: bool = True


# Rate limiting directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Rate limiting for field access"
)
class RateLimit:
    """@rateLimit directive for rate limiting"""
    max: int
    window: int  # Time window in seconds
    message: Optional[str] = None


# Caching directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION, DirectiveLocation.OBJECT],
    description="Caching policy"
)
class CacheControl:
    """@cacheControl directive for caching"""
    max_age: Optional[int] = None
    scope: Optional[str] = None  # PUBLIC or PRIVATE
    inherit_max_age: bool = False


# Deprecation directive
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION, DirectiveLocation.ENUM_VALUE],
    description="Deprecation information"
)
class Deprecated:
    """@deprecated directive"""
    reason: str = "No longer supported"


# Validation directives
@strawberry.directive(
    locations=[DirectiveLocation.INPUT_FIELD_DEFINITION, DirectiveLocation.ARGUMENT_DEFINITION],
    description="Input validation constraints"
)
class Constraint:
    """@constraint directive for input validation"""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    format: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


# Documentation directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION, DirectiveLocation.OBJECT],
    description="Specifies that a field or object is an example"
)
class Example:
    """@example directive for documentation"""
    value: str


@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Indicates a field is for internal use"
)
class Internal:
    """@internal directive for internal fields"""
    pass


# Transform directives
@strawberry.directive(
    locations=[DirectiveLocation.FIELD_DEFINITION],
    description="Transform field values"
)
class Transform:
    """@transform directive for field transformation"""
    operation: str  # UPPERCASE, LOWERCASE, etc.


# Federation directives collection
federation_directives = [
    Key,
    External,
    Requires,
    Provides,
    Extends,
    Shareable,
    Override,
    Inaccessible,
    Link,
    ComposeDirective,
    InterfaceObject,
    Authenticated,
    RequiresScopes,
    Policy,
    Cost,
    Complexity,
    RateLimit,
    CacheControl,
    Deprecated,
    Constraint,
    Example,
    Internal,
    Transform,
]


# Directive processors
class DirectiveProcessor:
    """Process federation directives"""
    
    def __init__(self):
        self.processors = {
            "key": self._process_key,
            "external": self._process_external,
            "requires": self._process_requires,
            "provides": self._process_provides,
            "extends": self._process_extends,
            "shareable": self._process_shareable,
            "override": self._process_override,
            "authenticated": self._process_authenticated,
            "requiresScopes": self._process_requires_scopes,
            "cost": self._process_cost,
            "rateLimit": self._process_rate_limit,
            "cacheControl": self._process_cache_control,
        }
    
    def process_directive(self, directive_name: str, directive_args: dict, context: dict):
        """Process a directive with its arguments"""
        if directive_name in self.processors:
            return self.processors[directive_name](directive_args, context)
        return context
    
    def _process_key(self, args: dict, context: dict) -> dict:
        """Process @key directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["key_fields"] = args.get("fields", "")
        context["federation"]["resolvable"] = args.get("resolvable", True)
        return context
    
    def _process_external(self, args: dict, context: dict) -> dict:
        """Process @external directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["external"] = True
        return context
    
    def _process_requires(self, args: dict, context: dict) -> dict:
        """Process @requires directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["requires"] = args.get("fields", "")
        return context
    
    def _process_provides(self, args: dict, context: dict) -> dict:
        """Process @provides directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["provides"] = args.get("fields", "")
        return context
    
    def _process_extends(self, args: dict, context: dict) -> dict:
        """Process @extends directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["extends"] = True
        return context
    
    def _process_shareable(self, args: dict, context: dict) -> dict:
        """Process @shareable directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["shareable"] = True
        return context
    
    def _process_override(self, args: dict, context: dict) -> dict:
        """Process @override directive"""
        context["federation"] = context.get("federation", {})
        context["federation"]["override_from"] = args.get("from", "")
        return context
    
    def _process_authenticated(self, args: dict, context: dict) -> dict:
        """Process @authenticated directive"""
        context["auth"] = context.get("auth", {})
        context["auth"]["authenticated"] = True
        return context
    
    def _process_requires_scopes(self, args: dict, context: dict) -> dict:
        """Process @requiresScopes directive"""
        context["auth"] = context.get("auth", {})
        context["auth"]["required_scopes"] = args.get("scopes", [])
        return context
    
    def _process_cost(self, args: dict, context: dict) -> dict:
        """Process @cost directive"""
        context["cost"] = context.get("cost", {})
        context["cost"]["complexity"] = args.get("complexity", 1)
        context["cost"]["multipliers"] = args.get("multipliers", [])
        return context
    
    def _process_rate_limit(self, args: dict, context: dict) -> dict:
        """Process @rateLimit directive"""
        context["rate_limit"] = context.get("rate_limit", {})
        context["rate_limit"]["max"] = args.get("max", 100)
        context["rate_limit"]["window"] = args.get("window", 60)
        context["rate_limit"]["message"] = args.get("message")
        return context
    
    def _process_cache_control(self, args: dict, context: dict) -> dict:
        """Process @cacheControl directive"""
        context["cache"] = context.get("cache", {})
        context["cache"]["max_age"] = args.get("max_age")
        context["cache"]["scope"] = args.get("scope")
        context["cache"]["inherit_max_age"] = args.get("inherit_max_age", False)
        return context


# Global directive processor
directive_processor = DirectiveProcessor()


# Directive utilities
def extract_directives_from_schema(schema_sdl: str) -> dict:
    """Extract directives from schema SDL"""
    directives = {}
    
    # Parse SDL and extract directive usage
    # This would be more complex in a real implementation
    
    return directives


def validate_directive_usage(directive_name: str, location: str, args: dict) -> List[str]:
    """Validate directive usage"""
    errors = []
    
    # Validate directive exists
    # Validate location is allowed
    # Validate arguments
    
    return errors


def apply_directive_transformations(schema, directives: dict):
    """Apply directive transformations to schema"""
    # Transform schema based on directive usage
    # Add federation metadata
    # Add authorization rules
    # Add caching policies
    
    return schema