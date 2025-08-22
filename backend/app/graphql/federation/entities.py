"""
Federation Entity Resolvers
"""

import strawberry
from typing import List, Optional, Union, Any
from strawberry.types import Info

from ..types import Document, User, ArbitrationAnalysis, Detection


class EntityResolver:
    """Resolves federated entities from other services"""
    
    def __init__(self):
        self.entity_resolvers = {
            "Document": self._resolve_document,
            "User": self._resolve_user,
            "ArbitrationAnalysis": self._resolve_analysis,
            "Detection": self._resolve_detection,
        }
    
    async def resolve_entities(
        self, 
        info: Info, 
        representations: List[Any]
    ) -> List[Union[Document, User, ArbitrationAnalysis, Detection]]:
        """Resolve multiple entities from representations"""
        entities = []
        
        for representation in representations:
            entity = await self._resolve_single_entity(info, representation)
            if entity:
                entities.append(entity)
        
        return entities
    
    async def _resolve_single_entity(self, info: Info, representation: dict):
        """Resolve a single entity from its representation"""
        entity_type = representation.get("__typename")
        
        if entity_type in self.entity_resolvers:
            resolver = self.entity_resolvers[entity_type]
            return await resolver(info, representation)
        
        return None
    
    async def _resolve_document(self, info: Info, representation: dict) -> Optional[Document]:
        """Resolve Document entity"""
        try:
            entity_id = representation.get("id")
            if not entity_id:
                return None
            
            loaders = info.context["loaders"]
            return await loaders["document"].load(int(entity_id))
            
        except Exception as e:
            return None
    
    async def _resolve_user(self, info: Info, representation: dict) -> Optional[User]:
        """Resolve User entity"""
        try:
            entity_id = representation.get("id")
            if not entity_id:
                return None
            
            loaders = info.context["loaders"]
            return await loaders["user"].load(int(entity_id))
            
        except Exception as e:
            return None
    
    async def _resolve_analysis(self, info: Info, representation: dict) -> Optional[ArbitrationAnalysis]:
        """Resolve ArbitrationAnalysis entity"""
        try:
            entity_id = representation.get("id")
            if not entity_id:
                return None
            
            loaders = info.context["loaders"]
            return await loaders["analysis"].load(int(entity_id))
            
        except Exception as e:
            return None
    
    async def _resolve_detection(self, info: Info, representation: dict) -> Optional[Detection]:
        """Resolve Detection entity"""
        try:
            entity_id = representation.get("id")
            if not entity_id:
                return None
            
            loaders = info.context["loaders"]
            return await loaders["detection"].load(int(entity_id))
            
        except Exception as e:
            return None


# Entity reference builders
class EntityReferenceBuilder:
    """Builds entity references for federation"""
    
    @staticmethod
    def build_document_reference(document_id: str) -> dict:
        """Build Document entity reference"""
        return {
            "__typename": "Document",
            "id": document_id
        }
    
    @staticmethod
    def build_user_reference(user_id: str) -> dict:
        """Build User entity reference"""
        return {
            "__typename": "User",
            "id": user_id
        }
    
    @staticmethod
    def build_analysis_reference(analysis_id: str) -> dict:
        """Build ArbitrationAnalysis entity reference"""
        return {
            "__typename": "ArbitrationAnalysis",
            "id": analysis_id
        }
    
    @staticmethod
    def build_detection_reference(detection_id: str) -> dict:
        """Build Detection entity reference"""
        return {
            "__typename": "Detection",
            "id": detection_id
        }


# Entity key extractors
class EntityKeyExtractor:
    """Extracts entity keys for federation"""
    
    @staticmethod
    def extract_document_key(document: Document) -> dict:
        """Extract key fields from Document"""
        return {
            "id": document.id
        }
    
    @staticmethod
    def extract_user_key(user: User) -> dict:
        """Extract key fields from User"""
        return {
            "id": user.id
        }
    
    @staticmethod
    def extract_analysis_key(analysis: ArbitrationAnalysis) -> dict:
        """Extract key fields from ArbitrationAnalysis"""
        return {
            "id": analysis.id,
            "document_id": analysis.document_id
        }
    
    @staticmethod
    def extract_detection_key(detection: Detection) -> dict:
        """Extract key fields from Detection"""
        return {
            "id": detection.id,
            "document_id": detection.document_id
        }


# Federation entity registry
class EntityRegistry:
    """Registry for federated entities"""
    
    def __init__(self):
        self.entities = {}
        self.key_extractors = {
            "Document": EntityKeyExtractor.extract_document_key,
            "User": EntityKeyExtractor.extract_user_key,
            "ArbitrationAnalysis": EntityKeyExtractor.extract_analysis_key,
            "Detection": EntityKeyExtractor.extract_detection_key,
        }
    
    def register_entity(self, entity_type: str, entity_class, key_fields: List[str]):
        """Register an entity type for federation"""
        self.entities[entity_type] = {
            "class": entity_class,
            "key_fields": key_fields
        }
    
    def get_entity_info(self, entity_type: str) -> Optional[dict]:
        """Get entity information"""
        return self.entities.get(entity_type)
    
    def extract_entity_key(self, entity_type: str, entity) -> dict:
        """Extract entity key"""
        if entity_type in self.key_extractors:
            return self.key_extractors[entity_type](entity)
        return {}
    
    def get_registered_entities(self) -> List[str]:
        """Get list of registered entity types"""
        return list(self.entities.keys())


# Global entity registry
entity_registry = EntityRegistry()

# Register default entities
entity_registry.register_entity("Document", Document, ["id"])
entity_registry.register_entity("User", User, ["id"])
entity_registry.register_entity("ArbitrationAnalysis", ArbitrationAnalysis, ["id", "document_id"])
entity_registry.register_entity("Detection", Detection, ["id", "document_id"])


# Cross-service entity relationships
class CrossServiceRelationshipResolver:
    """Resolves relationships across federated services"""
    
    def __init__(self):
        self.service_mappings = {
            "documents": "arbitration-detection",
            "users": "user-management",
            "analyses": "arbitration-detection",
            "detections": "arbitration-detection",
        }
    
    async def resolve_document_owner(self, document_id: str, info: Info) -> Optional[User]:
        """Resolve document owner from user service"""
        try:
            # In a federated setup, this would make a request to the user service
            # For now, use local loaders
            loaders = info.context["loaders"]
            
            # Would need document-user relationship
            # For now, return None
            return None
            
        except Exception:
            return None
    
    async def resolve_user_documents(self, user_id: str, info: Info) -> List[Document]:
        """Resolve user's documents"""
        try:
            loaders = info.context["loaders"]
            
            # Would need to query documents by user_id
            # For now, return empty list
            return []
            
        except Exception:
            return []
    
    async def resolve_document_analyses(self, document_id: str, info: Info) -> List[ArbitrationAnalysis]:
        """Resolve document's analyses"""
        try:
            loaders = info.context["loaders"]
            return await loaders["analyses_by_document"].load(int(document_id))
            
        except Exception:
            return []
    
    async def resolve_analysis_detections(self, analysis_id: str, info: Info) -> List[Detection]:
        """Resolve analysis detections"""
        try:
            loaders = info.context["loaders"]
            
            # Would need to get detections by analysis_id
            # For now, return empty list
            return []
            
        except Exception:
            return []


# Global cross-service resolver
cross_service_resolver = CrossServiceRelationshipResolver()


# Federation schema composition utilities
def compose_entity_schemas(*entity_schemas):
    """Compose multiple entity schemas"""
    composed_entities = {}
    
    for schema in entity_schemas:
        for entity_type, entity_info in schema.items():
            if entity_type in composed_entities:
                # Merge entity definitions
                composed_entities[entity_type] = merge_entity_definitions(
                    composed_entities[entity_type],
                    entity_info
                )
            else:
                composed_entities[entity_type] = entity_info
    
    return composed_entities


def merge_entity_definitions(entity1: dict, entity2: dict) -> dict:
    """Merge two entity definitions"""
    merged = entity1.copy()
    
    # Merge key fields
    if "key_fields" in entity2:
        merged_keys = set(merged.get("key_fields", []))
        merged_keys.update(entity2["key_fields"])
        merged["key_fields"] = list(merged_keys)
    
    # Merge fields
    if "fields" in entity2:
        merged_fields = merged.get("fields", {})
        merged_fields.update(entity2["fields"])
        merged["fields"] = merged_fields
    
    return merged


# Entity validation for federation
def validate_entity_keys(entity_type: str, entity_data: dict) -> List[str]:
    """Validate entity keys for federation"""
    errors = []
    
    entity_info = entity_registry.get_entity_info(entity_type)
    if not entity_info:
        errors.append(f"Unknown entity type: {entity_type}")
        return errors
    
    required_keys = entity_info["key_fields"]
    
    for key_field in required_keys:
        if key_field not in entity_data:
            errors.append(f"Missing required key field '{key_field}' for entity type '{entity_type}'")
    
    return errors


def validate_entity_reference(reference: dict) -> List[str]:
    """Validate entity reference format"""
    errors = []
    
    if "__typename" not in reference:
        errors.append("Missing __typename in entity reference")
        return errors
    
    entity_type = reference["__typename"]
    entity_errors = validate_entity_keys(entity_type, reference)
    errors.extend(entity_errors)
    
    return errors