"""
Apollo Federation Schema Implementation
"""

import strawberry
from typing import List, Union, Optional, Any
from strawberry.federation import Schema
from strawberry.types import Info

from ..types import (
    Document, Chunk, Detection, Pattern, 
    ArbitrationAnalysis, ArbitrationClause,
    User, Organization, Comment, Annotation
)
from ..resolvers import Query, Mutation, Subscription
from .entities import EntityResolver
from .directives import federation_directives


# Federation entity types with @key directives
@strawberry.federation.type(keys=["id"])
class DocumentEntity:
    """Document entity for federation"""
    id: strawberry.ID = strawberry.federation.field(external=True)
    filename: str = strawberry.federation.field(external=True)
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info: Info):
        """Resolve document reference from other services"""
        try:
            loaders = info.context["loaders"]
            return loaders["document"].load(int(id))
        except Exception:
            return None


@strawberry.federation.type(keys=["id"])
class UserEntity:
    """User entity for federation"""
    id: strawberry.ID = strawberry.federation.field(external=True)
    username: str = strawberry.federation.field(external=True)
    email: str = strawberry.federation.field(external=True)
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info: Info):
        """Resolve user reference from other services"""
        try:
            loaders = info.context["loaders"]
            return loaders["user"].load(int(id))
        except Exception:
            return None


@strawberry.federation.type(keys=["id"])
class AnalysisEntity:
    """Analysis entity for federation"""
    id: strawberry.ID = strawberry.federation.field(external=True)
    document_id: strawberry.ID = strawberry.federation.field(external=True)
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info: Info):
        """Resolve analysis reference from other services"""
        try:
            loaders = info.context["loaders"]
            return loaders["analysis"].load(int(id))
        except Exception:
            return None


@strawberry.federation.type(keys=["id"])
class DetectionEntity:
    """Detection entity for federation"""
    id: strawberry.ID = strawberry.federation.field(external=True)
    document_id: strawberry.ID = strawberry.federation.field(external=True)
    detection_type: str = strawberry.federation.field(external=True)
    
    @classmethod
    def resolve_reference(cls, id: strawberry.ID, info: Info):
        """Resolve detection reference from other services"""
        try:
            loaders = info.context["loaders"]
            return loaders["detection"].load(int(id))
        except Exception:
            return None


# Extended Query for federation
@strawberry.federation.type(extend=True)
class FederatedQuery(Query):
    """Extended Query with federation support"""
    
    @strawberry.field
    def _entities(
        self, 
        info: Info, 
        representations: List[Any]
    ) -> List[Union[DocumentEntity, UserEntity, AnalysisEntity, DetectionEntity]]:
        """Resolve entities from other services"""
        entities = []
        
        for representation in representations:
            entity_type = representation.get("__typename")
            entity_id = representation.get("id")
            
            if entity_type == "Document":
                entity = DocumentEntity.resolve_reference(entity_id, info)
            elif entity_type == "User":
                entity = UserEntity.resolve_reference(entity_id, info)
            elif entity_type == "ArbitrationAnalysis":
                entity = AnalysisEntity.resolve_reference(entity_id, info)
            elif entity_type == "Detection":
                entity = DetectionEntity.resolve_reference(entity_id, info)
            else:
                entity = None
            
            if entity:
                entities.append(entity)
        
        return entities
    
    @strawberry.field
    def _service(self, info: Info) -> "ServiceDefinition":
        """Return service definition for federation"""
        return ServiceDefinition(
            sdl=get_federated_schema_sdl()
        )


@strawberry.type
class ServiceDefinition:
    """Service definition for Apollo Federation"""
    sdl: str


def get_federated_schema_sdl() -> str:
    """Get the SDL representation of the federated schema"""
    return """
    type Document @key(fields: "id") {
        id: ID!
        filename: String!
    }
    
    type User @key(fields: "id") {
        id: ID!
        username: String!
        email: String!
    }
    
    type ArbitrationAnalysis @key(fields: "id") {
        id: ID!
        document_id: ID!
    }
    
    type Detection @key(fields: "id") {
        id: ID!
        document_id: ID!
        detection_type: String!
    }
    
    extend type Query {
        _entities(representations: [_Any!]!): [_Entity]!
        _service: _Service!
    }
    
    scalar _Any
    scalar _FieldSet
    
    type _Service {
        sdl: String
    }
    
    union _Entity = Document | User | ArbitrationAnalysis | Detection
    """


# Subgraph configuration
class SubgraphConfig:
    """Configuration for Apollo Federation subgraph"""
    
    def __init__(self, name: str, url: str, version: str = "1.0.0"):
        self.name = name
        self.url = url
        self.version = version
    
    def to_dict(self) -> dict:
        """Convert to dictionary for gateway configuration"""
        return {
            "name": self.name,
            "url": self.url,
            "version": self.version
        }


# Gateway composition configuration
class GatewayConfig:
    """Configuration for Apollo Federation Gateway"""
    
    def __init__(self):
        self.subgraphs = []
    
    def add_subgraph(self, config: SubgraphConfig):
        """Add a subgraph to the gateway"""
        self.subgraphs.append(config)
    
    def get_composition_config(self) -> dict:
        """Get gateway composition configuration"""
        return {
            "subgraphs": [subgraph.to_dict() for subgraph in self.subgraphs],
            "introspection": True,
            "playground": True
        }


def create_federated_schema():
    """Create federated GraphQL schema"""
    try:
        # Create the base schema with federation support
        schema = Schema(
            query=FederatedQuery,
            mutation=Mutation,
            subscription=Subscription,
            # Enable federation
            enable_federation_2=True,
            # Add federation directives
            directives=federation_directives
        )
        
        return schema
        
    except Exception as e:
        # Fallback to non-federated schema
        return strawberry.Schema(
            query=Query,
            mutation=Mutation,
            subscription=Subscription
        )


def setup_federation_gateway():
    """Setup Apollo Federation Gateway configuration"""
    gateway_config = GatewayConfig()
    
    # Add arbitration detection subgraph
    arbitration_subgraph = SubgraphConfig(
        name="arbitration-detection",
        url="http://localhost:8000/graphql",
        version="1.0.0"
    )
    gateway_config.add_subgraph(arbitration_subgraph)
    
    # Add user management subgraph (if separate)
    # user_subgraph = SubgraphConfig(
    #     name="user-management",
    #     url="http://localhost:8001/graphql",
    #     version="1.0.0"
    # )
    # gateway_config.add_subgraph(user_subgraph)
    
    # Add document processing subgraph (if separate)
    # document_subgraph = SubgraphConfig(
    #     name="document-processing",
    #     url="http://localhost:8002/graphql",
    #     version="1.0.0"
    # )
    # gateway_config.add_subgraph(document_subgraph)
    
    return gateway_config


# Schema composition utilities
def compose_schemas(*schemas):
    """Compose multiple schemas for federation"""
    composed_types = {}
    composed_fields = {}
    
    for schema in schemas:
        # Extract types and fields from each schema
        # This would be more complex in a real implementation
        pass
    
    return composed_types, composed_fields


def validate_federation_schema(schema):
    """Validate federated schema for composition"""
    errors = []
    
    # Check for required federation directives
    # Check for entity key fields
    # Check for field conflicts
    # Validate entity resolution
    
    return errors


# Federation middleware
class FederationMiddleware:
    """Middleware for handling federation requests"""
    
    def __init__(self, schema):
        self.schema = schema
    
    async def process_request(self, request, context):
        """Process federation-specific requests"""
        # Handle _entities queries
        # Handle _service queries
        # Add federation context
        
        return context
    
    async def process_response(self, response, context):
        """Process federation-specific responses"""
        # Add federation metadata
        # Handle entity references
        
        return response


# Entity caching for federation
class EntityCache:
    """Cache for federated entities"""
    
    def __init__(self, ttl: int = 300):  # 5 minutes default
        self.cache = {}
        self.ttl = ttl
    
    async def get_entity(self, entity_type: str, entity_id: str):
        """Get entity from cache"""
        cache_key = f"{entity_type}:{entity_id}"
        
        if cache_key in self.cache:
            entity, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return entity
            else:
                del self.cache[cache_key]
        
        return None
    
    async def set_entity(self, entity_type: str, entity_id: str, entity):
        """Set entity in cache"""
        cache_key = f"{entity_type}:{entity_id}"
        self.cache[cache_key] = (entity, time.time())
    
    async def invalidate_entity(self, entity_type: str, entity_id: str):
        """Invalidate entity in cache"""
        cache_key = f"{entity_type}:{entity_id}"
        if cache_key in self.cache:
            del self.cache[cache_key]
    
    async def clear_cache(self):
        """Clear entire cache"""
        self.cache.clear()


# Global entity cache instance
entity_cache = EntityCache()


# Federation health check
async def federation_health_check():
    """Check federation service health"""
    health_status = {
        "status": "healthy",
        "subgraphs": [],
        "timestamp": time.time()
    }
    
    # Check each subgraph health
    gateway_config = setup_federation_gateway()
    
    for subgraph in gateway_config.subgraphs:
        try:
            # Make health check request to subgraph
            subgraph_health = await check_subgraph_health(subgraph.url)
            health_status["subgraphs"].append({
                "name": subgraph.name,
                "url": subgraph.url,
                "status": "healthy" if subgraph_health else "unhealthy"
            })
        except Exception as e:
            health_status["subgraphs"].append({
                "name": subgraph.name,
                "url": subgraph.url,
                "status": "unhealthy",
                "error": str(e)
            })
    
    # Set overall status based on subgraph health
    unhealthy_count = sum(1 for sg in health_status["subgraphs"] if sg["status"] == "unhealthy")
    if unhealthy_count > 0:
        health_status["status"] = "degraded" if unhealthy_count < len(health_status["subgraphs"]) else "unhealthy"
    
    return health_status


async def check_subgraph_health(url: str) -> bool:
    """Check individual subgraph health"""
    try:
        # Would make actual health check request
        return True
    except Exception:
        return False