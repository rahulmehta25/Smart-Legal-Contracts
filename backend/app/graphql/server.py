"""
GraphQL Server Integration with FastAPI
"""

from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL
import os
from typing import Dict, Any

from .schema import get_schema_config, create_graphql_context, process_request_middleware
from .playground import create_playground_app
from ..core.config import settings


def create_graphql_app(environment: str = None) -> FastAPI:
    """Create FastAPI app with GraphQL integration"""
    
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    # Create FastAPI app
    app = FastAPI(
        title="Arbitration Detection GraphQL API",
        description="Advanced GraphQL API for arbitration clause detection and analysis",
        version="1.0.0",
        docs_url="/docs" if environment == "development" else None,
        redoc_url="/redoc" if environment == "development" else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, 'ALLOWED_ORIGINS', ["*"]),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Add request processing middleware
    app.middleware("http")(process_request_middleware)
    
    # Get schema configuration
    schema_config = get_schema_config(environment)
    
    # Create GraphQL router
    graphql_router = GraphQLRouter(
        schema=schema_config["schema"],
        context_getter=schema_config["context_getter"],
        introspection=schema_config.get("introspection", False),
        graphiql=schema_config.get("graphiql", False),
        subscription_protocols=[
            GRAPHQL_TRANSPORT_WS_PROTOCOL,
            GRAPHQL_WS_PROTOCOL,
        ],
    )
    
    # Include GraphQL router
    app.include_router(graphql_router, prefix="/graphql")
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "arbitration-detection-graphql",
            "version": "1.0.0",
            "environment": environment
        }
    
    # Add schema introspection endpoint (for tooling)
    if environment in ["development", "federation"]:
        @app.get("/graphql/schema")
        async def get_schema():
            """Get GraphQL schema SDL"""
            from graphql import print_schema
            return {
                "schema": print_schema(schema_config["schema"].graphql_schema)
            }
    
    # Add GraphQL Playground in development
    if environment == "development":
        playground_app = create_playground_app(
            endpoint="/graphql",
            subscriptions_endpoint="/graphql/ws"
        )
        app.mount("/playground", playground_app)
    
    return app


def integrate_with_existing_fastapi(app: FastAPI, prefix: str = "/graphql") -> None:
    """Integrate GraphQL with existing FastAPI application"""
    
    environment = os.getenv("ENVIRONMENT", "production")
    schema_config = get_schema_config(environment)
    
    # Create GraphQL router
    graphql_router = GraphQLRouter(
        schema=schema_config["schema"],
        context_getter=schema_config["context_getter"],
        introspection=schema_config.get("introspection", False),
        graphiql=schema_config.get("graphiql", False),
    )
    
    # Include GraphQL router
    app.include_router(graphql_router, prefix=prefix, tags=["GraphQL"])
    
    # Add GraphQL-specific health check
    @app.get(f"{prefix}/health", tags=["GraphQL"])
    async def graphql_health():
        """GraphQL service health check"""
        return {
            "status": "healthy",
            "service": "graphql",
            "schema_types": len(schema_config["schema"].graphql_schema.type_map),
            "environment": environment
        }


# Main application factory
def create_app() -> FastAPI:
    """Create main application with GraphQL"""
    return create_graphql_app()


# Development server setup
def setup_development_server():
    """Setup development server with additional features"""
    import uvicorn
    from .utils.complexity import complexity_cache
    from .utils.rate_limiting import default_rate_limiter
    
    app = create_graphql_app("development")
    
    # Add development-specific endpoints
    @app.get("/dev/clear-cache")
    async def clear_cache():
        """Clear all caches (development only)"""
        complexity_cache.clear_cache()
        # Clear rate limiter cache if using in-memory storage
        return {"message": "Caches cleared"}
    
    @app.get("/dev/rate-limits")
    async def get_rate_limits():
        """Get current rate limiting status (development only)"""
        return {
            "rules": {name: {
                "max_requests": rule.max_requests,
                "window_seconds": rule.window_seconds,
                "strategy": rule.strategy.value
            } for name, rule in default_rate_limiter.rules.items()}
        }
    
    return app


# Production server setup
def setup_production_server():
    """Setup production server with optimizations"""
    app = create_graphql_app("production")
    
    # Add production monitoring endpoints
    @app.get("/metrics")
    async def get_metrics():
        """Get server metrics (production monitoring)"""
        from .utils.complexity import complexity_cache
        
        return {
            "cache_size": len(complexity_cache.cache),
            "schema_version": "1.0.0",
            "status": "operational"
        }
    
    return app


# Federation gateway setup
def setup_federation_gateway():
    """Setup Apollo Federation gateway"""
    app = create_graphql_app("federation")
    
    @app.get("/federation/services")
    async def get_services():
        """Get federation services configuration"""
        from .federation.schema import setup_federation_gateway
        
        gateway_config = setup_federation_gateway()
        return gateway_config.get_composition_config()
    
    @app.get("/federation/health")
    async def federation_health():
        """Federation health check"""
        from .federation.schema import federation_health_check
        
        return await federation_health_check()
    
    return app


# CLI commands for server management
def run_development_server(host: str = "0.0.0.0", port: int = 8000):
    """Run development server"""
    import uvicorn
    
    app = setup_development_server()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,
        log_level="info",
        access_log=True
    )


def run_production_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
    """Run production server"""
    import uvicorn
    
    app = setup_production_server()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="warning",
        access_log=False
    )


def run_federation_gateway(host: str = "0.0.0.0", port: int = 4000):
    """Run federation gateway"""
    import uvicorn
    
    app = setup_federation_gateway()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


# WebSocket connection management
class GraphQLWebSocketManager:
    """Manage WebSocket connections for GraphQL subscriptions"""
    
    def __init__(self):
        self.connections = {}
    
    async def connect(self, websocket, user_id: str = None):
        """Accept WebSocket connection"""
        await websocket.accept()
        connection_id = id(websocket)
        
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "subscriptions": set()
        }
    
    async def disconnect(self, websocket):
        """Handle WebSocket disconnection"""
        connection_id = id(websocket)
        if connection_id in self.connections:
            del self.connections[connection_id]
    
    async def broadcast(self, message: dict, user_filter: callable = None):
        """Broadcast message to connections"""
        for connection in self.connections.values():
            if user_filter and not user_filter(connection["user_id"]):
                continue
            
            try:
                await connection["websocket"].send_json(message)
            except:
                # Connection is dead, will be cleaned up later
                pass


# Global WebSocket manager
websocket_manager = GraphQLWebSocketManager()


# Server configuration validation
def validate_server_config():
    """Validate server configuration"""
    errors = []
    
    # Check required environment variables
    required_vars = ["DATABASE_URL", "SECRET_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Check GraphQL schema
    try:
        schema_config = get_schema_config()
        schema = schema_config["schema"]
        
        # Validate schema
        from .playground.schema_viewer import SchemaViewer
        viewer = SchemaViewer(schema.graphql_schema)
        schema_errors = viewer.validate_schema()
        errors.extend(schema_errors)
        
    except Exception as e:
        errors.append(f"Schema validation error: {str(e)}")
    
    return errors


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "dev":
            run_development_server()
        elif command == "prod":
            run_production_server()
        elif command == "federation":
            run_federation_gateway()
        elif command == "validate":
            errors = validate_server_config()
            if errors:
                print("Configuration errors:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("Configuration is valid")
        else:
            print("Usage: python server.py [dev|prod|federation|validate]")
    else:
        run_development_server()