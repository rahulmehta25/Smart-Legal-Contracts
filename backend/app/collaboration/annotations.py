"""
Collaborative annotations system for document markup and highlighting.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import json

import redis.asyncio as redis
from pydantic import BaseModel, Field


class AnnotationType(Enum):
    HIGHLIGHT = "highlight"
    NOTE = "note"
    ARROW = "arrow"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    FREEHAND = "freehand"
    TEXT_BOX = "text_box"
    STAMP = "stamp"
    STRIKETHROUGH = "strikethrough"
    UNDERLINE = "underline"


class AnnotationPermission(Enum):
    PRIVATE = "private"  # Only creator can see
    SHARED = "shared"    # All room members can see
    PUBLIC = "public"    # Anyone with document access can see


@dataclass
class AnnotationStyle:
    """Styling information for annotations."""
    color: str = "#FFD700"  # Default yellow
    opacity: float = 0.7
    stroke_width: int = 2
    font_size: int = 12
    font_family: str = "Arial"
    fill_color: Optional[str] = None
    border_color: Optional[str] = None
    dash_pattern: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationStyle':
        return cls(**data)


@dataclass
class AnnotationPosition:
    """Position and geometry information for annotations."""
    page: int
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    z_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationPosition':
        return cls(**data)


@dataclass
class TextRange:
    """Text selection range for text-based annotations."""
    start_offset: int
    end_offset: int
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    selected_text: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextRange':
        return cls(**data)


@dataclass
class Annotation:
    """Complete annotation object."""
    id: str
    document_id: str
    room_id: str
    user_id: str
    username: str
    annotation_type: AnnotationType
    content: str  # Text content, path data for drawings, etc.
    position: AnnotationPosition
    style: AnnotationStyle
    permission: AnnotationPermission
    created_at: datetime
    updated_at: datetime
    text_range: Optional[TextRange] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_resolved: bool = False
    parent_id: Optional[str] = None  # For threaded annotations
    replies: List[str] = field(default_factory=list)  # Child annotation IDs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_id': self.document_id,
            'room_id': self.room_id,
            'user_id': self.user_id,
            'username': self.username,
            'annotation_type': self.annotation_type.value,
            'content': self.content,
            'position': self.position.to_dict(),
            'style': self.style.to_dict(),
            'permission': self.permission.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'text_range': self.text_range.to_dict() if self.text_range else None,
            'tags': self.tags,
            'metadata': self.metadata,
            'is_resolved': self.is_resolved,
            'parent_id': self.parent_id,
            'replies': self.replies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Annotation':
        text_range = None
        if data.get('text_range'):
            text_range = TextRange.from_dict(data['text_range'])
        
        return cls(
            id=data['id'],
            document_id=data['document_id'],
            room_id=data['room_id'],
            user_id=data['user_id'],
            username=data['username'],
            annotation_type=AnnotationType(data['annotation_type']),
            content=data['content'],
            position=AnnotationPosition.from_dict(data['position']),
            style=AnnotationStyle.from_dict(data['style']),
            permission=AnnotationPermission(data['permission']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            text_range=text_range,
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            is_resolved=data.get('is_resolved', False),
            parent_id=data.get('parent_id'),
            replies=data.get('replies', [])
        )


@dataclass
class AnnotationEvent:
    """Event for annotation changes."""
    event_id: str
    event_type: str  # created, updated, deleted, resolved
    annotation: Annotation
    timestamp: datetime
    changes: Optional[Dict[str, Any]] = None  # What was changed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'annotation': self.annotation.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'changes': self.changes
        }


class AnnotationManager:
    """Manages collaborative annotations and document markup."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.document_annotations: Dict[str, List[Annotation]] = {}  # document_id -> annotations
        self.room_annotations: Dict[str, List[Annotation]] = {}     # room_id -> annotations
        self.user_annotations: Dict[str, List[Annotation]] = {}     # user_id -> annotations
        self.annotation_layers: Dict[str, Dict[str, List[Annotation]]] = {}  # document_id -> layer_name -> annotations
        self.cleanup_task = None
        
    async def start(self):
        """Start the annotation manager."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("Annotation manager started")
    
    async def stop(self):
        """Stop the annotation manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        await self.redis.close()
        logging.info("Annotation manager stopped")
    
    async def create_annotation(self, document_id: str, room_id: str, user_id: str, username: str,
                              annotation_type: AnnotationType, content: str,
                              position: AnnotationPosition, style: Optional[AnnotationStyle] = None,
                              permission: AnnotationPermission = AnnotationPermission.SHARED,
                              text_range: Optional[TextRange] = None,
                              tags: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None,
                              parent_id: Optional[str] = None) -> Annotation:
        """Create a new annotation."""
        annotation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        if style is None:
            style = AnnotationStyle()
        
        annotation = Annotation(
            id=annotation_id,
            document_id=document_id,
            room_id=room_id,
            user_id=user_id,
            username=username,
            annotation_type=annotation_type,
            content=content,
            position=position,
            style=style,
            permission=permission,
            created_at=now,
            updated_at=now,
            text_range=text_range,
            tags=tags or [],
            metadata=metadata or {},
            parent_id=parent_id
        )
        
        # Store annotation
        await self._store_annotation(annotation)
        
        # Update indexes
        await self._index_annotation(annotation)
        
        # Handle threading if this is a reply
        if parent_id:
            await self._add_reply_to_parent(parent_id, annotation_id)
        
        # Create event
        event = AnnotationEvent(
            event_id=str(uuid.uuid4()),
            event_type="created",
            annotation=annotation,
            timestamp=now
        )
        
        # Store event for audit trail
        await self._store_event(event)
        
        logging.info(f"Created annotation {annotation_id} by {username} in document {document_id}")
        return annotation
    
    async def update_annotation(self, annotation_id: str, user_id: str,
                              content: Optional[str] = None,
                              position: Optional[AnnotationPosition] = None,
                              style: Optional[AnnotationStyle] = None,
                              tags: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[Annotation]:
        """Update an existing annotation."""
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None
        
        # Check permissions
        if not await self._can_edit_annotation(annotation, user_id):
            return None
        
        changes = {}
        
        # Update fields
        if content is not None and content != annotation.content:
            changes['content'] = {'old': annotation.content, 'new': content}
            annotation.content = content
        
        if position is not None:
            changes['position'] = {'old': annotation.position.to_dict(), 'new': position.to_dict()}
            annotation.position = position
        
        if style is not None:
            changes['style'] = {'old': annotation.style.to_dict(), 'new': style.to_dict()}
            annotation.style = style
        
        if tags is not None:
            changes['tags'] = {'old': annotation.tags, 'new': tags}
            annotation.tags = tags
        
        if metadata is not None:
            changes['metadata'] = {'old': annotation.metadata, 'new': metadata}
            annotation.metadata.update(metadata)
        
        if not changes:
            return annotation  # No changes made
        
        annotation.updated_at = datetime.utcnow()
        
        # Store updated annotation
        await self._store_annotation(annotation)
        
        # Create event
        event = AnnotationEvent(
            event_id=str(uuid.uuid4()),
            event_type="updated",
            annotation=annotation,
            timestamp=annotation.updated_at,
            changes=changes
        )
        
        await self._store_event(event)
        
        logging.info(f"Updated annotation {annotation_id} by {user_id}")
        return annotation
    
    async def delete_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Delete an annotation."""
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return False
        
        # Check permissions
        if not await self._can_delete_annotation(annotation, user_id):
            return False
        
        # Remove from all indexes
        await self._remove_annotation_from_indexes(annotation)
        
        # Delete from Redis
        await self.redis.delete(f"annotation:{annotation_id}")
        
        # Delete replies if any
        for reply_id in annotation.replies:
            await self.delete_annotation(reply_id, user_id)
        
        # Remove from parent's replies if this was a reply
        if annotation.parent_id:
            await self._remove_reply_from_parent(annotation.parent_id, annotation_id)
        
        # Create event
        event = AnnotationEvent(
            event_id=str(uuid.uuid4()),
            event_type="deleted",
            annotation=annotation,
            timestamp=datetime.utcnow()
        )
        
        await self._store_event(event)
        
        logging.info(f"Deleted annotation {annotation_id} by {user_id}")
        return True
    
    async def resolve_annotation(self, annotation_id: str, user_id: str) -> Optional[Annotation]:
        """Mark an annotation as resolved."""
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None
        
        # Check permissions (room members can resolve)
        # This could be made more restrictive if needed
        
        annotation.is_resolved = True
        annotation.updated_at = datetime.utcnow()
        
        # Store updated annotation
        await self._store_annotation(annotation)
        
        # Create event
        event = AnnotationEvent(
            event_id=str(uuid.uuid4()),
            event_type="resolved",
            annotation=annotation,
            timestamp=annotation.updated_at
        )
        
        await self._store_event(event)
        
        logging.info(f"Resolved annotation {annotation_id} by {user_id}")
        return annotation
    
    async def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Get a specific annotation by ID."""
        data = await self.redis.get(f"annotation:{annotation_id}")
        if data:
            try:
                annotation_data = json.loads(data)
                return Annotation.from_dict(annotation_data)
            except json.JSONDecodeError:
                logging.error(f"Invalid annotation data for ID {annotation_id}")
        
        return None
    
    async def get_document_annotations(self, document_id: str, user_id: Optional[str] = None,
                                     include_resolved: bool = True,
                                     annotation_types: Optional[List[AnnotationType]] = None) -> List[Annotation]:
        """Get all annotations for a document."""
        # First check cache
        if document_id in self.document_annotations:
            annotations = self.document_annotations[document_id][:]
        else:
            # Load from Redis
            annotation_ids = await self.redis.smembers(f"document_annotations:{document_id}")
            annotations = []
            
            for annotation_id in annotation_ids:
                annotation = await self.get_annotation(annotation_id)
                if annotation:
                    annotations.append(annotation)
            
            # Cache the results
            self.document_annotations[document_id] = annotations
        
        # Filter annotations
        filtered = []
        for annotation in annotations:
            # Permission check
            if annotation.permission == AnnotationPermission.PRIVATE and user_id != annotation.user_id:
                continue
            
            # Resolved filter
            if not include_resolved and annotation.is_resolved:
                continue
            
            # Type filter
            if annotation_types and annotation.annotation_type not in annotation_types:
                continue
            
            filtered.append(annotation)
        
        return filtered
    
    async def get_room_annotations(self, room_id: str, user_id: Optional[str] = None) -> List[Annotation]:
        """Get all annotations for a room."""
        annotation_ids = await self.redis.smembers(f"room_annotations:{room_id}")
        annotations = []
        
        for annotation_id in annotation_ids:
            annotation = await self.get_annotation(annotation_id)
            if annotation:
                # Permission check
                if annotation.permission == AnnotationPermission.PRIVATE and user_id != annotation.user_id:
                    continue
                annotations.append(annotation)
        
        return annotations
    
    async def get_user_annotations(self, user_id: str, document_id: Optional[str] = None) -> List[Annotation]:
        """Get all annotations by a specific user."""
        annotation_ids = await self.redis.smembers(f"user_annotations:{user_id}")
        annotations = []
        
        for annotation_id in annotation_ids:
            annotation = await self.get_annotation(annotation_id)
            if annotation:
                if document_id is None or annotation.document_id == document_id:
                    annotations.append(annotation)
        
        return annotations
    
    async def get_annotations_in_area(self, document_id: str, page: int, x: float, y: float,
                                    width: float, height: float, user_id: Optional[str] = None) -> List[Annotation]:
        """Get annotations that intersect with a specific area."""
        annotations = await self.get_document_annotations(document_id, user_id)
        intersecting = []
        
        for annotation in annotations:
            if annotation.position.page == page:
                # Check if annotation intersects with the area
                if self._rectangles_intersect(
                    annotation.position.x, annotation.position.y,
                    annotation.position.width, annotation.position.height,
                    x, y, width, height
                ):
                    intersecting.append(annotation)
        
        return intersecting
    
    async def get_annotations_by_tags(self, document_id: str, tags: List[str],
                                    user_id: Optional[str] = None) -> List[Annotation]:
        """Get annotations that have any of the specified tags."""
        annotations = await self.get_document_annotations(document_id, user_id)
        tagged = []
        
        for annotation in annotations:
            if any(tag in annotation.tags for tag in tags):
                tagged.append(annotation)
        
        return tagged
    
    async def search_annotations(self, document_id: str, query: str,
                               user_id: Optional[str] = None) -> List[Annotation]:
        """Search annotations by content."""
        annotations = await self.get_document_annotations(document_id, user_id)
        results = []
        
        query_lower = query.lower()
        for annotation in annotations:
            if (query_lower in annotation.content.lower() or
                any(query_lower in tag.lower() for tag in annotation.tags)):
                results.append(annotation)
        
        return results
    
    def _rectangles_intersect(self, x1: float, y1: float, w1: float, h1: float,
                            x2: float, y2: float, w2: float, h2: float) -> bool:
        """Check if two rectangles intersect."""
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    async def _store_annotation(self, annotation: Annotation):
        """Store annotation in Redis."""
        try:
            data = json.dumps(annotation.to_dict())
            await self.redis.set(f"annotation:{annotation.id}", data)
        except Exception as e:
            logging.error(f"Error storing annotation: {e}")
    
    async def _index_annotation(self, annotation: Annotation):
        """Add annotation to various indexes."""
        try:
            # Document index
            await self.redis.sadd(f"document_annotations:{annotation.document_id}", annotation.id)
            
            # Room index
            await self.redis.sadd(f"room_annotations:{annotation.room_id}", annotation.id)
            
            # User index
            await self.redis.sadd(f"user_annotations:{annotation.user_id}", annotation.id)
            
            # Update local cache
            if annotation.document_id not in self.document_annotations:
                self.document_annotations[annotation.document_id] = []
            self.document_annotations[annotation.document_id].append(annotation)
            
        except Exception as e:
            logging.error(f"Error indexing annotation: {e}")
    
    async def _remove_annotation_from_indexes(self, annotation: Annotation):
        """Remove annotation from all indexes."""
        try:
            # Remove from Redis indexes
            await self.redis.srem(f"document_annotations:{annotation.document_id}", annotation.id)
            await self.redis.srem(f"room_annotations:{annotation.room_id}", annotation.id)
            await self.redis.srem(f"user_annotations:{annotation.user_id}", annotation.id)
            
            # Update local cache
            if annotation.document_id in self.document_annotations:
                self.document_annotations[annotation.document_id] = [
                    a for a in self.document_annotations[annotation.document_id]
                    if a.id != annotation.id
                ]
            
        except Exception as e:
            logging.error(f"Error removing annotation from indexes: {e}")
    
    async def _add_reply_to_parent(self, parent_id: str, reply_id: str):
        """Add a reply to parent annotation."""
        parent = await self.get_annotation(parent_id)
        if parent:
            parent.replies.append(reply_id)
            await self._store_annotation(parent)
    
    async def _remove_reply_from_parent(self, parent_id: str, reply_id: str):
        """Remove a reply from parent annotation."""
        parent = await self.get_annotation(parent_id)
        if parent and reply_id in parent.replies:
            parent.replies.remove(reply_id)
            await self._store_annotation(parent)
    
    async def _can_edit_annotation(self, annotation: Annotation, user_id: str) -> bool:
        """Check if user can edit annotation."""
        # Only creator can edit their annotations
        return annotation.user_id == user_id
    
    async def _can_delete_annotation(self, annotation: Annotation, user_id: str) -> bool:
        """Check if user can delete annotation."""
        # Creator or room owner can delete
        return annotation.user_id == user_id
    
    async def _store_event(self, event: AnnotationEvent):
        """Store annotation event for audit trail."""
        try:
            data = json.dumps(event.to_dict())
            await self.redis.lpush(f"annotation_events:{event.annotation.document_id}", data)
            await self.redis.ltrim(f"annotation_events:{event.annotation.document_id}", 0, 999)
        except Exception as e:
            logging.error(f"Error storing annotation event: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of annotation caches."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_caches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in annotation cleanup: {e}")
    
    async def _cleanup_caches(self):
        """Clean up local annotation caches."""
        # Simple cleanup - clear caches to force reload from Redis
        # In production, you might want more sophisticated cache management
        self.document_annotations.clear()
        self.room_annotations.clear()
        self.user_annotations.clear()
    
    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get annotation statistics."""
        total_annotations = sum(len(annotations) for annotations in self.document_annotations.values())
        documents_with_annotations = len(self.document_annotations)
        
        return {
            'total_annotations': total_annotations,
            'documents_with_annotations': documents_with_annotations,
            'annotation_types': len(AnnotationType),
            'cached_documents': len(self.document_annotations)
        }


# Global instance
annotation_manager = AnnotationManager()