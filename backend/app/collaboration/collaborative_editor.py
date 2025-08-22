"""
Collaborative editor that integrates all collaboration features.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import json

from .websocket_server import CollaborationWebSocketServer, collaboration_server
from .presence import PresenceManager, presence_manager, PresenceStatus, UserPresence
from .cursors import CursorManager, cursor_manager, CursorPosition, TextSelection, CursorState
from .annotations import AnnotationManager, annotation_manager, Annotation, AnnotationType
from .comments import CommentManager, comment_manager, Comment, CommentType
from .conflict_resolution import CRDTManager, crdt_manager, Operation, OperationType


class DocumentEventType(Enum):
    DOCUMENT_OPENED = "document_opened"
    DOCUMENT_CLOSED = "document_closed"
    CONTENT_CHANGED = "content_changed"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    ANNOTATION_ADDED = "annotation_added"
    COMMENT_ADDED = "comment_added"
    CONFLICT_RESOLVED = "conflict_resolved"


@dataclass
class DocumentSession:
    """Active document editing session."""
    document_id: str
    room_id: str
    title: str
    content: str
    version: int
    created_at: datetime
    last_modified: datetime
    active_users: Set[str] = field(default_factory=set)
    locked_by: Optional[str] = None
    lock_expires: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'room_id': self.room_id,
            'title': self.title,
            'content': self.content,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'active_users': list(self.active_users),
            'locked_by': self.locked_by,
            'lock_expires': self.lock_expires.isoformat() if self.lock_expires else None,
            'metadata': self.metadata
        }


@dataclass
class CollaborativeChange:
    """A change made to a document during collaboration."""
    change_id: str
    document_id: str
    user_id: str
    operation_type: str
    position: int
    content: Any
    timestamp: datetime
    applied: bool = False
    conflicts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'change_id': self.change_id,
            'document_id': self.document_id,
            'user_id': self.user_id,
            'operation_type': self.operation_type,
            'position': self.position,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'applied': self.applied,
            'conflicts': self.conflicts
        }


class CollaborativeEditor:
    """Main collaborative editor coordinator."""
    
    def __init__(self):
        self.websocket_server = collaboration_server
        self.presence_manager = presence_manager
        self.cursor_manager = cursor_manager
        self.annotation_manager = annotation_manager
        self.comment_manager = comment_manager
        self.crdt_manager = crdt_manager
        
        self.active_sessions: Dict[str, DocumentSession] = {}
        self.event_callbacks: Dict[DocumentEventType, List[Callable]] = {}
        self.change_history: Dict[str, List[CollaborativeChange]] = {}
        self.auto_save_intervals: Dict[str, float] = {}
        
        self.is_running = False
        self.save_task = None
        
    async def start(self):
        """Start the collaborative editor."""
        if self.is_running:
            return
        
        # Start all managers
        await self.presence_manager.start()
        await self.cursor_manager.start()
        await self.annotation_manager.start()
        await self.comment_manager.start()
        await self.crdt_manager.start()
        
        # Start auto-save task
        self.save_task = asyncio.create_task(self._auto_save_loop())
        
        # Setup event handlers
        self._setup_event_handlers()
        
        self.is_running = True
        logging.info("Collaborative editor started")
    
    async def stop(self):
        """Stop the collaborative editor."""
        if not self.is_running:
            return
        
        # Stop auto-save task
        if self.save_task:
            self.save_task.cancel()
            try:
                await self.save_task
            except asyncio.CancelledError:
                pass
        
        # Stop all managers
        await self.presence_manager.stop()
        await self.cursor_manager.stop()
        await self.annotation_manager.stop()
        await self.comment_manager.stop()
        await self.crdt_manager.stop()
        
        self.is_running = False
        logging.info("Collaborative editor stopped")
    
    def _setup_event_handlers(self):
        """Setup event handlers for various components."""
        # Add notification callback for comments
        self.comment_manager.add_notification_callback(self._handle_comment_notification)
    
    async def open_document(self, document_id: str, user_id: str, username: str,
                          room_id: Optional[str] = None, initial_content: str = "",
                          title: str = "Untitled Document") -> DocumentSession:
        """Open a document for collaborative editing."""
        if room_id is None:
            room_id = f"doc_{document_id}"
        
        # Get or create session
        if document_id in self.active_sessions:
            session = self.active_sessions[document_id]
        else:
            # Create new session
            session = DocumentSession(
                document_id=document_id,
                room_id=room_id,
                title=title,
                content=initial_content,
                version=1,
                created_at=datetime.utcnow(),
                last_modified=datetime.utcnow()
            )
            
            self.active_sessions[document_id] = session
            self.change_history[document_id] = []
            
            # Initialize CRDT document
            await self.crdt_manager.create_document(document_id, initial_content)
        
        # Add user to session
        session.active_users.add(user_id)
        
        # Set user presence
        await self.presence_manager.set_user_online(user_id, username, f"session_{document_id}")
        await self.presence_manager.join_room(user_id, room_id)
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.DOCUMENT_OPENED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'session': session.to_dict()
        })
        
        logging.info(f"User {username} opened document {document_id}")
        return session
    
    async def close_document(self, document_id: str, user_id: str) -> bool:
        """Close a document for a user."""
        if document_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[document_id]
        session.active_users.discard(user_id)
        
        # Update presence
        await self.presence_manager.leave_room(user_id, session.room_id)
        await self.cursor_manager.remove_cursor(user_id, session.room_id)
        
        # If no active users, clean up session
        if not session.active_users:
            del self.active_sessions[document_id]
            if document_id in self.change_history:
                del self.change_history[document_id]
        
        # Emit event
        await self._emit_event(DocumentEventType.DOCUMENT_CLOSED, {
            'document_id': document_id,
            'user_id': user_id,
            'remaining_users': len(session.active_users) if session.active_users else 0
        })
        
        logging.info(f"User {user_id} closed document {document_id}")
        return True
    
    async def apply_text_change(self, document_id: str, user_id: str, username: str,
                              operation_type: str, position: int, content: str,
                              length: Optional[int] = None) -> Optional[CollaborativeChange]:
        """Apply a text change to a document."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Create collaborative change
        change = CollaborativeChange(
            change_id=str(uuid.uuid4()),
            document_id=document_id,
            user_id=user_id,
            operation_type=operation_type,
            position=position,
            content=content,
            timestamp=datetime.utcnow()
        )
        
        # Apply change through CRDT
        operation = None
        if operation_type == "insert":
            operation = await self.crdt_manager.insert_text(document_id, position, content, user_id)
        elif operation_type == "delete":
            delete_length = length or len(content)
            operation = await self.crdt_manager.delete_text(document_id, position, delete_length, user_id)
        
        if operation:
            # Get updated content
            crdt_state = await self.crdt_manager.get_document_state(document_id)
            if crdt_state:
                session.content = crdt_state.content
                session.version = crdt_state.version
                session.last_modified = datetime.utcnow()
                
                change.applied = True
        
        # Add to change history
        self.change_history[document_id].append(change)
        
        # Keep only last 100 changes
        if len(self.change_history[document_id]) > 100:
            self.change_history[document_id] = self.change_history[document_id][-100:]
        
        # Update user activity
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.CONTENT_CHANGED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'change': change.to_dict(),
            'session': session.to_dict()
        })
        
        return change
    
    async def update_cursor_position(self, document_id: str, user_id: str, username: str,
                                   position: CursorPosition, metadata: Optional[Dict[str, Any]] = None) -> Optional[CursorState]:
        """Update a user's cursor position."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Update cursor through cursor manager
        cursor_state = await self.cursor_manager.update_cursor_position(
            user_id, username, session.room_id, position, metadata
        )
        
        # Update user activity
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.CURSOR_MOVED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'cursor_state': cursor_state.to_dict() if cursor_state else None
        })
        
        return cursor_state
    
    async def update_text_selection(self, document_id: str, user_id: str, username: str,
                                  selection: TextSelection, metadata: Optional[Dict[str, Any]] = None) -> Optional[CursorState]:
        """Update a user's text selection."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Update selection through cursor manager
        cursor_state = await self.cursor_manager.update_text_selection(
            user_id, username, session.room_id, selection, metadata
        )
        
        # Update user activity
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.SELECTION_CHANGED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'cursor_state': cursor_state.to_dict() if cursor_state else None
        })
        
        return cursor_state
    
    async def add_annotation(self, document_id: str, user_id: str, username: str,
                           annotation_type: AnnotationType, content: str,
                           position: Dict[str, Any], style: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Optional[Annotation]:
        """Add an annotation to a document."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Create annotation through annotation manager
        from .annotations import AnnotationPosition, AnnotationStyle
        
        annotation_position = AnnotationPosition.from_dict(position)
        annotation_style = AnnotationStyle.from_dict(style) if style else None
        
        annotation = await self.annotation_manager.create_annotation(
            document_id=document_id,
            room_id=session.room_id,
            user_id=user_id,
            username=username,
            annotation_type=annotation_type,
            content=content,
            position=annotation_position,
            style=annotation_style,
            **kwargs
        )
        
        # Update user activity
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.ANNOTATION_ADDED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'annotation': annotation.to_dict() if annotation else None
        })
        
        return annotation
    
    async def add_comment(self, document_id: str, user_id: str, username: str,
                        content: str, comment_type: CommentType = CommentType.TEXT,
                        position: Optional[Dict[str, Any]] = None,
                        parent_id: Optional[str] = None,
                        **kwargs) -> Optional[Comment]:
        """Add a comment to a document."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Create comment through comment manager
        comment = await self.comment_manager.create_comment(
            document_id=document_id,
            room_id=session.room_id,
            user_id=user_id,
            username=username,
            content=content,
            comment_type=comment_type,
            position=position,
            parent_id=parent_id,
            **kwargs
        )
        
        # Update user activity
        await self.presence_manager.update_user_activity(user_id, document_id)
        
        # Emit event
        await self._emit_event(DocumentEventType.COMMENT_ADDED, {
            'document_id': document_id,
            'user_id': user_id,
            'username': username,
            'comment': comment.to_dict() if comment else None
        })
        
        return comment
    
    async def get_document_state(self, document_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get complete document state for a user."""
        if document_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[document_id]
        
        # Get all data
        cursors = await self.cursor_manager.get_room_cursors(session.room_id)
        annotations = await self.annotation_manager.get_document_annotations(document_id, user_id)
        comments = await self.comment_manager.get_document_comments(document_id)
        presence = await self.presence_manager.get_room_members(session.room_id)
        crdt_state = await self.crdt_manager.get_document_state(document_id)
        
        return {
            'session': session.to_dict(),
            'cursors': [cursor.to_dict() for cursor in cursors],
            'annotations': [annotation.to_dict() for annotation in annotations],
            'comments': [comment.to_dict() for comment in comments],
            'presence': [user.to_dict() for user in presence],
            'crdt_state': crdt_state.to_dict() if crdt_state else None,
            'change_history': [change.to_dict() for change in self.change_history.get(document_id, [])]
        }
    
    async def sync_document(self, document_id: str, remote_operations: List[Dict[str, Any]]) -> bool:
        """Sync document with remote operations."""
        if document_id not in self.active_sessions:
            return False
        
        # Convert and apply operations
        operations = []
        for op_data in remote_operations:
            try:
                operation = Operation.from_dict(op_data)
                operations.append(operation)
            except Exception as e:
                logging.error(f"Error parsing operation: {e}")
                continue
        
        # Resolve conflicts
        resolved_operations = await self.crdt_manager.resolve_conflicts(document_id, operations)
        
        if resolved_operations:
            # Update session content
            session = self.active_sessions[document_id]
            crdt_state = await self.crdt_manager.get_document_state(document_id)
            if crdt_state:
                session.content = crdt_state.content
                session.version = crdt_state.version
                session.last_modified = datetime.utcnow()
            
            # Emit conflict resolution event
            await self._emit_event(DocumentEventType.CONFLICT_RESOLVED, {
                'document_id': document_id,
                'resolved_operations': [op.to_dict() for op in resolved_operations],
                'session': session.to_dict()
            })
        
        return True
    
    async def lock_document(self, document_id: str, user_id: str, duration_minutes: int = 30) -> bool:
        """Lock a document for exclusive editing."""
        if document_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[document_id]
        
        # Check if already locked
        if session.locked_by and session.lock_expires and session.lock_expires > datetime.utcnow():
            return False
        
        # Lock document
        session.locked_by = user_id
        session.lock_expires = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        logging.info(f"Document {document_id} locked by {user_id}")
        return True
    
    async def unlock_document(self, document_id: str, user_id: str) -> bool:
        """Unlock a document."""
        if document_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[document_id]
        
        # Check if user can unlock
        if session.locked_by != user_id:
            return False
        
        # Unlock document
        session.locked_by = None
        session.lock_expires = None
        
        logging.info(f"Document {document_id} unlocked by {user_id}")
        return True
    
    def add_event_callback(self, event_type: DocumentEventType, callback: Callable):
        """Add a callback for document events."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event_type: DocumentEventType, data: Dict[str, Any]):
        """Emit a document event."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    await callback(event_type, data)
                except Exception as e:
                    logging.error(f"Error in event callback: {e}")
    
    async def _handle_comment_notification(self, notification_type: str, data: Dict[str, Any]):
        """Handle comment notifications."""
        if notification_type == 'mention':
            # Handle user mentions in comments
            comment_data = data.get('comment', {})
            mentioned_user = data.get('mentioned_user', {})
            
            logging.info(f"User {mentioned_user.get('user_id')} mentioned in comment {comment_data.get('id')}")
            
            # Additional notification logic can be added here
    
    async def _auto_save_loop(self):
        """Periodic auto-save of documents."""
        while True:
            try:
                await asyncio.sleep(30)  # Auto-save every 30 seconds
                await self._auto_save_all_documents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in auto-save loop: {e}")
    
    async def _auto_save_all_documents(self):
        """Auto-save all active documents."""
        for document_id, session in list(self.active_sessions.items()):
            try:
                # Save CRDT state
                crdt_state = await self.crdt_manager.get_document_state(document_id)
                if crdt_state:
                    # Here you would typically save to your database
                    # For now, we just update the session
                    session.content = crdt_state.content
                    session.version = crdt_state.version
                    
                logging.debug(f"Auto-saved document {document_id}")
                
            except Exception as e:
                logging.error(f"Error auto-saving document {document_id}: {e}")
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        active_documents = len(self.active_sessions)
        total_users = sum(len(session.active_users) for session in self.active_sessions.values())
        
        presence_stats = self.presence_manager.get_presence_stats()
        cursor_stats = self.cursor_manager.get_cursor_stats()
        annotation_stats = self.annotation_manager.get_annotation_stats()
        comment_stats = self.comment_manager.get_comment_stats()
        crdt_stats = self.crdt_manager.get_crdt_stats()
        
        return {
            'active_documents': active_documents,
            'total_active_users': total_users,
            'is_running': self.is_running,
            'presence': presence_stats,
            'cursors': cursor_stats,
            'annotations': annotation_stats,
            'comments': comment_stats,
            'crdt': crdt_stats
        }


# Global instance
collaborative_editor = CollaborativeEditor()