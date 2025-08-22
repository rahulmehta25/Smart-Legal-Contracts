"""
Data Catalog Manager

Central data catalog for metadata management and data governance:
- Unified metadata repository
- Schema registry and evolution
- Data discovery and search
- Lineage tracking and impact analysis
- Data classification and compliance
- Access control and security
"""

import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField


class DataAssetType(Enum):
    """Types of data assets in the catalog"""
    TABLE = "table"
    VIEW = "view"
    DATASET = "dataset"
    STREAM = "stream"
    FILE = "file"
    API = "api"
    MODEL = "model"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_CONFIDENTIAL = "highly_confidential"


class AccessLevel(Enum):
    """Access control levels"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


@dataclass
class DataAsset:
    """Represents a data asset in the catalog"""
    asset_id: str
    name: str
    asset_type: DataAssetType
    location: str
    
    # Metadata
    description: Optional[str] = None
    owner: Optional[str] = None
    steward: Optional[str] = None
    
    # Schema and structure
    schema: Optional[StructType] = None
    schema_version: int = 1
    schema_evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Classification and governance
    classification: DataClassification = DataClassification.INTERNAL
    tags: Set[str] = field(default_factory=set)
    business_glossary_terms: Set[str] = field(default_factory=set)
    
    # Quality and usage
    quality_score: Optional[float] = None
    usage_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    upstream_assets: Set[str] = field(default_factory=set)
    downstream_assets: Set[str] = field(default_factory=set)
    
    # Technical metadata
    format: Optional[str] = None
    compression: Optional[str] = None
    partitioning: List[str] = field(default_factory=list)
    
    # Temporal metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    
    # Custom properties
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """Access control policy for data assets"""
    policy_id: str
    asset_id: str
    principal: str  # User, group, or role
    access_level: AccessLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    expiry_date: Optional[datetime] = None
    created_by: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessGlossaryTerm:
    """Business glossary term definition"""
    term_id: str
    name: str
    definition: str
    category: Optional[str] = None
    synonyms: Set[str] = field(default_factory=set)
    related_terms: Set[str] = field(default_factory=set)
    steward: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class LineageRelation:
    """Represents a lineage relationship between assets"""
    relation_id: str
    source_asset_id: str
    target_asset_id: str
    relation_type: str  # "reads_from", "writes_to", "transforms", "derives_from"
    transformation_logic: Optional[str] = None
    confidence_score: float = 1.0
    created_date: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None


class DataCatalogManager:
    """
    Comprehensive data catalog manager for metadata management and governance.
    
    Features:
    - Unified metadata repository
    - Schema registry and evolution tracking
    - Data discovery and search capabilities
    - Lineage tracking and impact analysis
    - Data classification and compliance management
    - Access control and security policies
    - Business glossary integration
    - Usage analytics and insights
    - API for external integrations
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core repositories
        self.assets: Dict[str, DataAsset] = {}
        self.access_policies: Dict[str, List[AccessPolicy]] = {}  # asset_id -> policies
        self.glossary_terms: Dict[str, BusinessGlossaryTerm] = {}
        self.lineage_relations: Dict[str, List[LineageRelation]] = {}  # asset_id -> relations
        
        # Indexes for fast search
        self.name_index: Dict[str, Set[str]] = {}  # name -> asset_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> asset_ids
        self.owner_index: Dict[str, Set[str]] = {}  # owner -> asset_ids
        self.classification_index: Dict[DataClassification, Set[str]] = {}
        
        # Usage tracking
        self.usage_tracker: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize indexes
        self._initialize_indexes()
    
    def _initialize_indexes(self):
        """Initialize search indexes"""
        for classification in DataClassification:
            self.classification_index[classification] = set()
    
    def register_asset(
        self,
        name: str,
        asset_type: DataAssetType,
        location: str,
        schema: Optional[StructType] = None,
        **kwargs
    ) -> str:
        """
        Register a new data asset in the catalog
        
        Args:
            name: Name of the asset
            asset_type: Type of the asset
            location: Physical location/path
            schema: Schema definition
            **kwargs: Additional asset properties
            
        Returns:
            str: Asset ID
        """
        try:
            asset_id = kwargs.get("asset_id") or str(uuid.uuid4())
            
            # Create data asset
            asset = DataAsset(
                asset_id=asset_id,
                name=name,
                asset_type=asset_type,
                location=location,
                schema=schema,
                description=kwargs.get("description"),
                owner=kwargs.get("owner"),
                steward=kwargs.get("steward"),
                classification=kwargs.get("classification", DataClassification.INTERNAL),
                tags=set(kwargs.get("tags", [])),
                business_glossary_terms=set(kwargs.get("business_terms", [])),
                format=kwargs.get("format"),
                compression=kwargs.get("compression"),
                partitioning=kwargs.get("partitioning", []),
                properties=kwargs.get("properties", {})
            )
            
            # Store asset
            self.assets[asset_id] = asset
            
            # Update indexes
            self._update_indexes(asset)
            
            self.logger.info(f"Registered asset: {name} (ID: {asset_id})")
            return asset_id
            
        except Exception as e:
            self.logger.error(f"Error registering asset {name}: {str(e)}")
            raise
    
    def _update_indexes(self, asset: DataAsset):
        """Update search indexes for an asset"""
        asset_id = asset.asset_id
        
        # Name index
        if asset.name not in self.name_index:
            self.name_index[asset.name] = set()
        self.name_index[asset.name].add(asset_id)
        
        # Tag index
        for tag in asset.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(asset_id)
        
        # Owner index
        if asset.owner:
            if asset.owner not in self.owner_index:
                self.owner_index[asset.owner] = set()
            self.owner_index[asset.owner].add(asset_id)
        
        # Classification index
        self.classification_index[asset.classification].add(asset_id)
    
    def update_asset(self, asset_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing data asset
        
        Args:
            asset_id: ID of the asset to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: Success status
        """
        try:
            if asset_id not in self.assets:
                raise ValueError(f"Asset {asset_id} not found")
            
            asset = self.assets[asset_id]
            old_asset = asset  # For index cleanup
            
            # Track schema evolution
            if "schema" in updates and updates["schema"] != asset.schema:
                schema_change = {
                    "version": asset.schema_version + 1,
                    "previous_schema": str(asset.schema) if asset.schema else None,
                    "new_schema": str(updates["schema"]),
                    "change_date": datetime.now(),
                    "change_type": "evolution"
                }
                asset.schema_evolution_history.append(schema_change)
                asset.schema_version += 1
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(asset, field):
                    setattr(asset, field, value)
            
            asset.last_modified = datetime.now()
            
            # Update indexes
            self._cleanup_indexes(old_asset)
            self._update_indexes(asset)
            
            self.logger.info(f"Updated asset: {asset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating asset {asset_id}: {str(e)}")
            return False
    
    def _cleanup_indexes(self, asset: DataAsset):
        """Remove asset from indexes before update"""
        asset_id = asset.asset_id
        
        # Clean name index
        if asset.name in self.name_index:
            self.name_index[asset.name].discard(asset_id)
            if not self.name_index[asset.name]:
                del self.name_index[asset.name]
        
        # Clean tag index
        for tag in asset.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(asset_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Clean owner index
        if asset.owner and asset.owner in self.owner_index:
            self.owner_index[asset.owner].discard(asset_id)
            if not self.owner_index[asset.owner]:
                del self.owner_index[asset.owner]
        
        # Clean classification index
        self.classification_index[asset.classification].discard(asset_id)
    
    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """Get a data asset by ID"""
        return self.assets.get(asset_id)
    
    def search_assets(
        self,
        query: Optional[str] = None,
        asset_type: Optional[DataAssetType] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        classification: Optional[DataClassification] = None,
        limit: int = 100
    ) -> List[DataAsset]:
        """
        Search for data assets based on various criteria
        
        Args:
            query: Text search query (searches names and descriptions)
            asset_type: Filter by asset type
            tags: Filter by tags (any of the tags)
            owner: Filter by owner
            classification: Filter by classification level
            limit: Maximum number of results
            
        Returns:
            List[DataAsset]: Matching assets
        """
        try:
            candidate_asset_ids = set(self.assets.keys())
            
            # Apply filters
            if query:
                query_matches = set()
                query_lower = query.lower()
                
                # Search in names
                for name, asset_ids in self.name_index.items():
                    if query_lower in name.lower():
                        query_matches.update(asset_ids)
                
                # Search in descriptions
                for asset_id, asset in self.assets.items():
                    if asset.description and query_lower in asset.description.lower():
                        query_matches.add(asset_id)
                
                candidate_asset_ids &= query_matches
            
            if asset_type:
                type_matches = {aid for aid, asset in self.assets.items() if asset.asset_type == asset_type}
                candidate_asset_ids &= type_matches
            
            if tags:
                tag_matches = set()
                for tag in tags:
                    if tag in self.tag_index:
                        tag_matches.update(self.tag_index[tag])
                candidate_asset_ids &= tag_matches
            
            if owner:
                owner_matches = self.owner_index.get(owner, set())
                candidate_asset_ids &= owner_matches
            
            if classification:
                classification_matches = self.classification_index.get(classification, set())
                candidate_asset_ids &= classification_matches
            
            # Get matching assets
            matching_assets = [self.assets[aid] for aid in candidate_asset_ids if aid in self.assets]
            
            # Sort by relevance (simplified - by last_modified)
            matching_assets.sort(key=lambda a: a.last_modified, reverse=True)
            
            return matching_assets[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching assets: {str(e)}")
            return []
    
    def add_lineage_relation(
        self,
        source_asset_id: str,
        target_asset_id: str,
        relation_type: str,
        transformation_logic: Optional[str] = None,
        confidence_score: float = 1.0
    ) -> str:
        """
        Add a lineage relationship between assets
        
        Args:
            source_asset_id: Source asset ID
            target_asset_id: Target asset ID
            relation_type: Type of relationship
            transformation_logic: Description of transformation
            confidence_score: Confidence in the relationship
            
        Returns:
            str: Relation ID
        """
        try:
            relation_id = str(uuid.uuid4())
            
            relation = LineageRelation(
                relation_id=relation_id,
                source_asset_id=source_asset_id,
                target_asset_id=target_asset_id,
                relation_type=relation_type,
                transformation_logic=transformation_logic,
                confidence_score=confidence_score
            )
            
            # Store in lineage relations
            if source_asset_id not in self.lineage_relations:
                self.lineage_relations[source_asset_id] = []
            self.lineage_relations[source_asset_id].append(relation)
            
            # Update asset lineage references
            if source_asset_id in self.assets:
                self.assets[source_asset_id].downstream_assets.add(target_asset_id)
            
            if target_asset_id in self.assets:
                self.assets[target_asset_id].upstream_assets.add(source_asset_id)
            
            self.logger.info(f"Added lineage relation: {source_asset_id} -> {target_asset_id}")
            return relation_id
            
        except Exception as e:
            self.logger.error(f"Error adding lineage relation: {str(e)}")
            raise
    
    def get_lineage(self, asset_id: str, direction: str = "both", depth: int = 3) -> Dict[str, Any]:
        """
        Get lineage information for an asset
        
        Args:
            asset_id: Asset ID to get lineage for
            direction: "upstream", "downstream", or "both"
            depth: Maximum depth to traverse
            
        Returns:
            Dict[str, Any]: Lineage graph
        """
        try:
            lineage_graph = {
                "asset_id": asset_id,
                "direction": direction,
                "depth": depth,
                "nodes": {},
                "edges": []
            }
            
            visited = set()
            
            def traverse_lineage(current_id: str, current_depth: int, is_upstream: bool):
                if current_depth > depth or current_id in visited:
                    return
                
                visited.add(current_id)
                
                # Add current asset to nodes
                if current_id in self.assets:
                    asset = self.assets[current_id]
                    lineage_graph["nodes"][current_id] = {
                        "name": asset.name,
                        "type": asset.asset_type.value,
                        "classification": asset.classification.value
                    }
                
                # Get relations for current asset
                relations = self.lineage_relations.get(current_id, [])
                
                for relation in relations:
                    if is_upstream:
                        # For upstream, we want sources of current asset
                        next_id = relation.source_asset_id if relation.target_asset_id == current_id else None
                    else:
                        # For downstream, we want targets of current asset
                        next_id = relation.target_asset_id if relation.source_asset_id == current_id else None
                    
                    if next_id:
                        # Add edge
                        lineage_graph["edges"].append({
                            "source": relation.source_asset_id,
                            "target": relation.target_asset_id,
                            "relation_type": relation.relation_type,
                            "confidence": relation.confidence_score
                        })
                        
                        # Recurse
                        traverse_lineage(next_id, current_depth + 1, is_upstream)
            
            # Traverse based on direction
            if direction in ["upstream", "both"]:
                traverse_lineage(asset_id, 0, True)
            
            if direction in ["downstream", "both"]:
                traverse_lineage(asset_id, 0, False)
            
            return lineage_graph
            
        except Exception as e:
            self.logger.error(f"Error getting lineage for {asset_id}: {str(e)}")
            return {"error": str(e)}
    
    def add_access_policy(
        self,
        asset_id: str,
        principal: str,
        access_level: AccessLevel,
        conditions: Optional[Dict[str, Any]] = None,
        expiry_date: Optional[datetime] = None
    ) -> str:
        """
        Add an access control policy for an asset
        
        Args:
            asset_id: Asset ID
            principal: User, group, or role
            access_level: Level of access
            conditions: Additional access conditions
            expiry_date: Policy expiry date
            
        Returns:
            str: Policy ID
        """
        try:
            policy_id = str(uuid.uuid4())
            
            policy = AccessPolicy(
                policy_id=policy_id,
                asset_id=asset_id,
                principal=principal,
                access_level=access_level,
                conditions=conditions or {},
                expiry_date=expiry_date
            )
            
            # Store policy
            if asset_id not in self.access_policies:
                self.access_policies[asset_id] = []
            self.access_policies[asset_id].append(policy)
            
            self.logger.info(f"Added access policy for {asset_id}: {principal} -> {access_level.value}")
            return policy_id
            
        except Exception as e:
            self.logger.error(f"Error adding access policy: {str(e)}")
            raise
    
    def check_access(self, asset_id: str, principal: str, requested_access: AccessLevel) -> bool:
        """
        Check if a principal has access to an asset
        
        Args:
            asset_id: Asset ID
            principal: User, group, or role
            requested_access: Requested access level
            
        Returns:
            bool: Whether access is granted
        """
        try:
            policies = self.access_policies.get(asset_id, [])
            
            for policy in policies:
                # Check if policy applies to principal
                if policy.principal != principal:
                    continue
                
                # Check if policy has expired
                if policy.expiry_date and datetime.now() > policy.expiry_date:
                    continue
                
                # Check access level hierarchy
                access_hierarchy = {
                    AccessLevel.READ: 1,
                    AccessLevel.WRITE: 2,
                    AccessLevel.ADMIN: 3,
                    AccessLevel.OWNER: 4
                }
                
                if access_hierarchy.get(policy.access_level, 0) >= access_hierarchy.get(requested_access, 0):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking access: {str(e)}")
            return False
    
    def track_usage(self, asset_id: str, user: str, operation: str, **metadata):
        """
        Track usage of a data asset
        
        Args:
            asset_id: Asset ID
            user: User who accessed the asset
            operation: Type of operation (read, write, etc.)
            **metadata: Additional metadata about the usage
        """
        try:
            usage_event = {
                "timestamp": datetime.now(),
                "user": user,
                "operation": operation,
                "metadata": metadata
            }
            
            if asset_id not in self.usage_tracker:
                self.usage_tracker[asset_id] = []
            
            self.usage_tracker[asset_id].append(usage_event)
            
            # Update last accessed time
            if asset_id in self.assets:
                self.assets[asset_id].last_accessed = datetime.now()
            
            # Keep only recent usage events (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.usage_tracker[asset_id] = [
                event for event in self.usage_tracker[asset_id]
                if event["timestamp"] > cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error tracking usage for {asset_id}: {str(e)}")
    
    def get_usage_analytics(self, asset_id: str) -> Dict[str, Any]:
        """
        Get usage analytics for an asset
        
        Args:
            asset_id: Asset ID
            
        Returns:
            Dict[str, Any]: Usage analytics
        """
        try:
            usage_events = self.usage_tracker.get(asset_id, [])
            
            if not usage_events:
                return {"asset_id": asset_id, "total_accesses": 0}
            
            # Calculate analytics
            total_accesses = len(usage_events)
            unique_users = len(set(event["user"] for event in usage_events))
            
            # Recent activity (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_events = [e for e in usage_events if e["timestamp"] > recent_cutoff]
            recent_accesses = len(recent_events)
            
            # Operation breakdown
            operations = {}
            for event in usage_events:
                op = event["operation"]
                operations[op] = operations.get(op, 0) + 1
            
            # User activity
            user_activity = {}
            for event in usage_events:
                user = event["user"]
                user_activity[user] = user_activity.get(user, 0) + 1
            
            return {
                "asset_id": asset_id,
                "total_accesses": total_accesses,
                "unique_users": unique_users,
                "recent_accesses_7d": recent_accesses,
                "operations": operations,
                "top_users": sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10],
                "last_accessed": max(e["timestamp"] for e in usage_events) if usage_events else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting usage analytics for {asset_id}: {str(e)}")
            return {"error": str(e)}
    
    def add_glossary_term(
        self,
        name: str,
        definition: str,
        category: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        steward: Optional[str] = None
    ) -> str:
        """Add a business glossary term"""
        try:
            term_id = str(uuid.uuid4())
            
            term = BusinessGlossaryTerm(
                term_id=term_id,
                name=name,
                definition=definition,
                category=category,
                synonyms=set(synonyms or []),
                steward=steward
            )
            
            self.glossary_terms[term_id] = term
            
            self.logger.info(f"Added glossary term: {name}")
            return term_id
            
        except Exception as e:
            self.logger.error(f"Error adding glossary term {name}: {str(e)}")
            raise
    
    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get overall catalog statistics"""
        try:
            # Asset statistics by type
            asset_types = {}
            classifications = {}
            
            for asset in self.assets.values():
                asset_type = asset.asset_type.value
                asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
                
                classification = asset.classification.value
                classifications[classification] = classifications.get(classification, 0) + 1
            
            # Recent activity
            recent_cutoff = datetime.now() - timedelta(days=7)
            recently_created = len([a for a in self.assets.values() if a.created_date > recent_cutoff])
            recently_modified = len([a for a in self.assets.values() if a.last_modified > recent_cutoff])
            
            return {
                "total_assets": len(self.assets),
                "asset_types": asset_types,
                "data_classifications": classifications,
                "access_policies": sum(len(policies) for policies in self.access_policies.values()),
                "glossary_terms": len(self.glossary_terms),
                "lineage_relations": sum(len(relations) for relations in self.lineage_relations.values()),
                "recently_created_7d": recently_created,
                "recently_modified_7d": recently_modified
            }
            
        except Exception as e:
            self.logger.error(f"Error getting catalog stats: {str(e)}")
            return {"error": str(e)}
    
    def export_catalog(self, format: str = "json") -> str:
        """Export catalog metadata"""
        try:
            catalog_export = {
                "export_timestamp": datetime.now().isoformat(),
                "assets": {},
                "glossary_terms": {},
                "access_policies": {},
                "lineage_relations": {}
            }
            
            # Export assets (simplified serialization)
            for asset_id, asset in self.assets.items():
                catalog_export["assets"][asset_id] = {
                    "name": asset.name,
                    "type": asset.asset_type.value,
                    "location": asset.location,
                    "description": asset.description,
                    "owner": asset.owner,
                    "classification": asset.classification.value,
                    "tags": list(asset.tags),
                    "created_date": asset.created_date.isoformat(),
                    "last_modified": asset.last_modified.isoformat()
                }
            
            # Export glossary terms
            for term_id, term in self.glossary_terms.items():
                catalog_export["glossary_terms"][term_id] = {
                    "name": term.name,
                    "definition": term.definition,
                    "category": term.category,
                    "synonyms": list(term.synonyms),
                    "steward": term.steward
                }
            
            if format == "json":
                return json.dumps(catalog_export, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting catalog: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup catalog manager resources"""
        try:
            # Clean up expired policies
            for asset_id, policies in self.access_policies.items():
                active_policies = [
                    p for p in policies
                    if not p.expiry_date or datetime.now() <= p.expiry_date
                ]
                self.access_policies[asset_id] = active_policies
            
            # Clean up old usage events (already done in track_usage)
            
            self.logger.info("Data catalog cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during catalog cleanup: {str(e)}")


# Utility functions

def create_table_asset(
    name: str,
    location: str,
    schema: StructType,
    owner: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a table asset configuration"""
    return {
        "name": name,
        "asset_type": DataAssetType.TABLE,
        "location": location,
        "schema": schema,
        "owner": owner,
        "description": description,
        "tags": tags or [],
        "classification": DataClassification.INTERNAL
    }