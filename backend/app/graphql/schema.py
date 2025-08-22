"""
Main GraphQL Schema Definition
"""

import strawberry
from strawberry.extensions import AddValidationRules
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
from typing import Dict, Any, Optional

from .resolvers import Query, Mutation, Subscription
from .utils.complexity import ComplexityAnalyzer, QueryComplexityValidationRule
from .utils.rate_limiting import RateLimitMiddleware, default_rate_limiter
from .utils.errors import error_handler, custom_format_error
from .federation.schema import create_federated_schema


# Create complexity analyzer
complexity_analyzer = ComplexityAnalyzer(
    max_complexity=1000,
    max_depth=10,
    introspection_complexity=1000
)

# Create validation rules
validation_rules = [
    QueryComplexityValidationRule(complexity_analyzer)
]

# Create extensions
extensions = [
    AddValidationRules(validation_rules)
]


def create_schema(enable_federation: bool = False):
    """Create GraphQL schema with all configurations"""
    
    if enable_federation:
        # Create federated schema
        return create_federated_schema()
    
    # Create standard schema
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
        extensions=extensions,
        execution_context_class=None,  # Can add custom execution context
    )
    
    return schema


def create_graphql_context(request, background_tasks=None) -> Dict[str, Any]:
    """Create GraphQL context from request"""
    from ..db.database import get_session
    from .dataloaders import (
        create_document_loaders,
        create_analysis_loaders,
        create_user_loaders,
        create_pattern_loaders
    )
    
    # Create database session
    session = next(get_session())
    
    # Create DataLoaders
    loaders = {}
    loaders.update(create_document_loaders(session))
    loaders.update(create_analysis_loaders(session))
    loaders.update(create_user_loaders(session))
    loaders.update(create_pattern_loaders(session))
    
    context = {
        "request": request,
        "session": session,
        "loaders": loaders,
        "rate_limiter": default_rate_limiter,
        "complexity_analyzer": complexity_analyzer,
        "background_tasks": background_tasks,
    }
    
    return context


async def process_request_middleware(request, call_next):
    """Middleware to process GraphQL requests"""
    import time
    import uuid
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timing
    start_time = time.time()
    
    try:
        # Apply rate limiting
        rate_limit_middleware = RateLimitMiddleware(default_rate_limiter)
        
        # Process request
        response = await call_next(request)
        
        # Add timing header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Handle middleware errors
        error_handler.logger.log_error(e, {"request_id": request_id})
        raise


# Schema instance
schema = create_schema()
federated_schema = create_schema(enable_federation=True)


# Schema configuration for different environments
def get_schema_config(environment: str = "production") -> Dict[str, Any]:
    """Get schema configuration for environment"""
    
    base_config = {
        "schema": schema,
        "context_getter": create_graphql_context,
        "error_formatter": custom_format_error,
        "subscription_protocols": [
            GRAPHQL_TRANSPORT_WS_PROTOCOL,
            GRAPHQL_WS_PROTOCOL,
        ],
    }
    
    if environment == "development":
        base_config.update({
            "introspection": True,
            "graphiql": True,
            "debug": True,
        })
    elif environment == "production":
        base_config.update({
            "introspection": False,
            "graphiql": False,
            "debug": False,
        })
    elif environment == "federation":
        base_config.update({
            "schema": federated_schema,
            "introspection": True,
            "graphiql": False,
        })
    
    return base_config