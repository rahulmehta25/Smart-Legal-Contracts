"""
Room Management for WebSocket real-time features.
"""

import asyncio
import logging
from typing import Dict, Set, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from .events import WebSocketEvent, EventType, create_presence_event

logger = logging.getLogger(__name__)


class RoomType(Enum):
    """Types of WebSocket rooms."""
    DOCUMENT = "document"          # Document collaboration
    ANALYSIS = "analysis"          # Real-time analysis
    NOTIFICATION = "notification"  # User notifications
    WORKSPACE = "workspace"        # Shared workspace
    VIDEO_CALL = "video_call"      # Video/audio collaboration
    CHAT = "chat"                  # Text chat
    SYSTEM = "system"              # System-wide broadcasts
    CUSTOM = "custom"              # Custom room type


class RoomPermission(Enum):
    """Room permissions."""
    READ = "read"
    WRITE = "write"
    COMMENT = "comment"
    MODERATE = "moderate"
    ADMIN = "admin"


@dataclass
class RoomMember:
    """Information about a room member."""
    user_id: str
    username: str
    connection_ids: Set[str] = field(default_factory=set)
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    permissions: Set[RoomPermission] = field(default_factory=lambda: {RoomPermission.READ, RoomPermission.WRITE, RoomPermission.COMMENT})
    status: str = "online"
    avatar_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def has_permission(self, permission: RoomPermission) -> bool:
        """Check if member has specific permission."""
        return permission in self.permissions or RoomPermission.ADMIN in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "joined_at": self.joined_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "permissions": [p.value for p in self.permissions],
            "status": self.status,
            "avatar_url": self.avatar_url,
            "metadata": self.metadata,
            "connection_count": len(self.connection_ids)
        }


@dataclass
class RoomSettings:
    """Room configuration settings."""
    max_members: Optional[int] = None
    require_approval: bool = False
    allow_anonymous: bool = False
    auto_cleanup: bool = True
    cleanup_delay_minutes: int = 60
    persistent: bool = False
    enable_cursor_tracking: bool = True
    enable_presence: bool = True
    enable_voice: bool = False
    enable_video: bool = False
    message_history_limit: int = 1000
    file_sharing_enabled: bool = True
    annotation_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_members": self.max_members,
            "require_approval": self.require_approval,
            "allow_anonymous": self.allow_anonymous,
            "auto_cleanup": self.auto_cleanup,
            "cleanup_delay_minutes": self.cleanup_delay_minutes,
            "persistent": self.persistent,
            "enable_cursor_tracking": self.enable_cursor_tracking,
            "enable_presence": self.enable_presence,
            "enable_voice": self.enable_voice,
            "enable_video": self.enable_video,
            "message_history_limit": self.message_history_limit,
            "file_sharing_enabled": self.file_sharing_enabled,
            "annotation_enabled": self.annotation_enabled
        }


class Room:
    """Represents a WebSocket room for real-time collaboration."""
    
    def __init__(
        self,
        room_id: str,
        room_type: RoomType = RoomType.DOCUMENT,
        name: Optional[str] = None,
        description: Optional[str] = None,
        owner_id: Optional[str] = None,
        settings: Optional[RoomSettings] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.room_id = room_id
        self.room_type = room_type
        self.name = name or f"Room {room_id[:8]}"
        self.description = description
        self.owner_id = owner_id
        self.settings = settings or RoomSettings()
        self.metadata = metadata or {}
        
        # State management
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.members: Dict[str, RoomMember] = {}
        self.pending_members: Dict[str, Dict[str, Any]] = {}
        
        # Content state
        self.document_state: Dict[str, Any] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.annotations: Dict[str, Any] = {}
        self.shared_cursors: Dict[str, Dict[str, Any]] = {}
        
        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_members_joined": 0,
            "total_messages": 0,
            "total_annotations": 0,
            "total_edits": 0,
            "created_at": self.created_at.isoformat()
        }
    
    def add_member(
        self,
        user_id: str,
        username: str,
        connection_id: str,
        permissions: Optional[Set[RoomPermission]] = None,
        avatar_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_approve: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Add a member to the room.
        
        Returns:
            tuple: (success, error_message)
        """
        # Check room capacity
        if (self.settings.max_members and 
            len(self.members) >= self.settings.max_members):
            return False, "Room is at maximum capacity"
        
        # Check if approval is required
        if self.settings.require_approval and not auto_approve:
            # Add to pending members
            self.pending_members[user_id] = {
                "username": username,
                "connection_id": connection_id,
                "requested_at": datetime.utcnow().isoformat(),
                "permissions": [p.value for p in permissions] if permissions else ["read", "write", "comment"],
                "avatar_url": avatar_url,
                "metadata": metadata or {}
            }
            return True, "Approval required"
        
        # Set default permissions if not provided
        if permissions is None:
            if user_id == self.owner_id:
                permissions = {RoomPermission.ADMIN}
            else:
                permissions = {RoomPermission.READ, RoomPermission.WRITE, RoomPermission.COMMENT}
        
        # Create or update member
        if user_id in self.members:
            member = self.members[user_id]
            member.connection_ids.add(connection_id)
            member.update_activity()
        else:
            member = RoomMember(
                user_id=user_id,
                username=username,
                connection_ids={connection_id},
                permissions=permissions,
                avatar_url=avatar_url,
                metadata=metadata or {}
            )
            self.members[user_id] = member
            self.stats["total_members_joined"] += 1
        
        # Remove from pending if it was there
        self.pending_members.pop(user_id, None)
        
        self.last_activity = datetime.utcnow()
        return True, None
    
    def remove_member(self, user_id: str, connection_id: str) -> bool:
        """
        Remove a member's connection from the room.
        
        Returns:
            bool: True if member was completely removed from room
        """
        if user_id not in self.members:
            return False
        
        member = self.members[user_id]
        member.connection_ids.discard(connection_id)
        
        # Remove member completely if no more connections
        if not member.connection_ids:
            del self.members[user_id]
            # Clean up member's cursors
            self.shared_cursors.pop(user_id, None)
            self.last_activity = datetime.utcnow()
            return True
        
        return False
    
    def update_member_activity(self, user_id: str):
        """Update member's last activity."""
        if user_id in self.members:
            self.members[user_id].update_activity()
            self.last_activity = datetime.utcnow()
    
    def update_member_status(self, user_id: str, status: str):
        """Update member's status."""
        if user_id in self.members:
            self.members[user_id].status = status
            self.last_activity = datetime.utcnow()
    
    def update_member_permissions(self, user_id: str, permissions: Set[RoomPermission]) -> bool:
        """Update member's permissions."""
        if user_id in self.members:
            self.members[user_id].permissions = permissions
            return True
        return False
    
    def approve_member(self, user_id: str) -> bool:
        """Approve a pending member."""
        if user_id not in self.pending_members:
            return False
        
        pending = self.pending_members[user_id]
        permissions = {RoomPermission(p) for p in pending["permissions"]}
        
        success, error = self.add_member(
            user_id=user_id,
            username=pending["username"],
            connection_id=pending["connection_id"],
            permissions=permissions,
            avatar_url=pending["avatar_url"],
            metadata=pending["metadata"],
            auto_approve=True
        )
        
        return success
    
    def reject_member(self, user_id: str) -> bool:
        """Reject a pending member."""
        if user_id in self.pending_members:
            del self.pending_members[user_id]
            return True
        return False
    
    def has_permission(self, user_id: str, permission: RoomPermission) -> bool:
        """Check if user has specific permission."""
        if user_id == self.owner_id:
            return True
        if user_id in self.members:
            return self.members[user_id].has_permission(permission)
        return False
    
    def get_active_members(self) -> List[RoomMember]:
        """Get list of active members."""
        return [member for member in self.members.values() if member.connection_ids]
    
    def get_member_count(self) -> int:
        """Get current number of active members."""
        return len([member for member in self.members.values() if member.connection_ids])
    
    def update_cursor(self, user_id: str, cursor_data: Dict[str, Any]):
        """Update user's cursor position."""
        if not self.settings.enable_cursor_tracking:
            return
        
        if user_id in self.members:
            self.shared_cursors[user_id] = {
                **cursor_data,
                "user_id": user_id,
                "username": self.members[user_id].username,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.update_member_activity(user_id)
    
    def add_message(self, user_id: str, message: Dict[str, Any]):
        """Add a message to room history."""
        if user_id in self.members:
            message_data = {
                **message,
                "user_id": user_id,
                "username": self.members[user_id].username,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": str(uuid.uuid4())
            }
            
            self.message_history.append(message_data)
            
            # Trim history if needed
            if len(self.message_history) > self.settings.message_history_limit:
                self.message_history = self.message_history[-self.settings.message_history_limit:]
            
            self.stats["total_messages"] += 1
            self.update_member_activity(user_id)
    
    def add_annotation(self, user_id: str, annotation: Dict[str, Any]) -> str:
        """Add an annotation to the room."""
        if not self.settings.annotation_enabled:
            return ""
        
        if user_id in self.members:
            annotation_id = str(uuid.uuid4())
            annotation_data = {
                **annotation,
                "id": annotation_id,
                "user_id": user_id,
                "username": self.members[user_id].username,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.annotations[annotation_id] = annotation_data
            self.stats["total_annotations"] += 1
            self.update_member_activity(user_id)
            
            return annotation_id
        
        return ""
    
    def update_annotation(self, annotation_id: str, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing annotation."""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        
        # Check if user can edit this annotation
        if annotation["user_id"] != user_id and not self.has_permission(user_id, RoomPermission.MODERATE):
            return False
        
        annotation.update(updates)
        annotation["updated_at"] = datetime.utcnow().isoformat()
        
        self.update_member_activity(user_id)
        return True
    
    def delete_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Delete an annotation."""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        
        # Check if user can delete this annotation
        if annotation["user_id"] != user_id and not self.has_permission(user_id, RoomPermission.MODERATE):
            return False
        
        del self.annotations[annotation_id]
        self.update_member_activity(user_id)
        return True
    
    def update_document_state(self, user_id: str, operation: Dict[str, Any]):
        """Update document state with an operation."""
        if user_id in self.members and self.has_permission(user_id, RoomPermission.WRITE):
            # Add operation metadata
            operation_data = {
                **operation,
                "user_id": user_id,
                "username": self.members[user_id].username,
                "timestamp": datetime.utcnow().isoformat(),
                "operation_id": str(uuid.uuid4())
            }
            
            # Apply operation to document state (simplified)
            # In a real implementation, this would use CRDT or OT
            self.document_state.setdefault("operations", []).append(operation_data)
            
            # Keep only recent operations
            if len(self.document_state["operations"]) > 1000:
                self.document_state["operations"] = self.document_state["operations"][-1000:]
            
            self.stats["total_edits"] += 1
            self.update_member_activity(user_id)
    
    def is_empty(self) -> bool:
        """Check if room has no active members."""
        return self.get_member_count() == 0
    
    def should_cleanup(self) -> bool:
        """Check if room should be cleaned up."""
        if not self.settings.auto_cleanup or self.settings.persistent:
            return False
        
        if not self.is_empty():
            return False
        
        cleanup_threshold = datetime.utcnow() - timedelta(minutes=self.settings.cleanup_delay_minutes)
        return self.last_activity < cleanup_threshold
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert room to dictionary."""
        result = {
            "room_id": self.room_id,
            "room_type": self.room_type.value,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "settings": self.settings.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "member_count": self.get_member_count(),
            "stats": self.stats.copy()
        }
        
        if include_sensitive:
            result.update({
                "members": [member.to_dict() for member in self.get_active_members()],
                "pending_members": list(self.pending_members.keys()),
                "cursors": self.shared_cursors,
                "recent_messages": self.message_history[-10:],  # Last 10 messages
                "annotation_count": len(self.annotations)
            })
        
        return result


class RoomManager:
    """Manages WebSocket rooms and their lifecycle."""
    
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.user_rooms: Dict[str, Set[str]] = {}  # user_id -> set of room_ids
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            "rooms_created": 0,
            "rooms_deleted": 0,
            "total_member_joins": 0,
            "total_member_leaves": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_empty_rooms())
    
    async def _cleanup_empty_rooms(self):
        """Periodically clean up empty rooms."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_rooms()
            except Exception as e:
                logger.error(f"Error in room cleanup task: {e}")
    
    def create_room(
        self,
        room_id: Optional[str] = None,
        room_type: RoomType = RoomType.DOCUMENT,
        name: Optional[str] = None,
        description: Optional[str] = None,
        owner_id: Optional[str] = None,
        settings: Optional[RoomSettings] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Room:
        """
        Create a new room.
        
        Returns:
            Room: The created room
        """
        if room_id is None:
            room_id = str(uuid.uuid4())
        
        if room_id in self.rooms:
            raise ValueError(f"Room {room_id} already exists")
        
        room = Room(
            room_id=room_id,
            room_type=room_type,
            name=name,
            description=description,
            owner_id=owner_id,
            settings=settings,
            metadata=metadata
        )
        
        self.rooms[room_id] = room
        self.stats["rooms_created"] += 1
        
        logger.info(f"Created room {room_id} of type {room_type.value}")
        return room
    
    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self.rooms.get(room_id)
    
    def delete_room(self, room_id: str) -> bool:
        """Delete a room."""
        if room_id not in self.rooms:
            return False
        
        room = self.rooms[room_id]
        
        # Remove from user room mappings
        for user_id in list(room.members.keys()):
            self._remove_user_from_room_mapping(user_id, room_id)
        
        # Cancel cleanup task if exists
        if room.cleanup_task:
            room.cleanup_task.cancel()
        
        del self.rooms[room_id]
        self.stats["rooms_deleted"] += 1
        
        logger.info(f"Deleted room {room_id}")
        return True
    
    def join_room(
        self,
        room_id: str,
        user_id: str,
        username: str,
        connection_id: str,
        permissions: Optional[Set[RoomPermission]] = None,
        avatar_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_create: bool = True
    ) -> tuple[bool, Optional[str], Optional[Room]]:
        """
        Join a user to a room.
        
        Returns:
            tuple: (success, error_message, room)
        """
        # Create room if it doesn't exist and auto_create is True
        if room_id not in self.rooms:
            if auto_create:
                room = self.create_room(
                    room_id=room_id,
                    owner_id=user_id,
                    name=f"Room {room_id[:8]}"
                )
            else:
                return False, "Room does not exist", None
        else:
            room = self.rooms[room_id]
        
        # Add member to room
        success, error = room.add_member(
            user_id=user_id,
            username=username,
            connection_id=connection_id,
            permissions=permissions,
            avatar_url=avatar_url,
            metadata=metadata
        )
        
        if success and error != "Approval required":
            # Update user room mapping
            if user_id not in self.user_rooms:
                self.user_rooms[user_id] = set()
            self.user_rooms[user_id].add(room_id)
            self.stats["total_member_joins"] += 1
        
        return success, error, room
    
    def leave_room(self, room_id: str, user_id: str, connection_id: str) -> bool:
        """
        Remove a user from a room.
        
        Returns:
            bool: True if user was completely removed from room
        """
        if room_id not in self.rooms:
            return False
        
        room = self.rooms[room_id]
        completely_removed = room.remove_member(user_id, connection_id)
        
        if completely_removed:
            self._remove_user_from_room_mapping(user_id, room_id)
            self.stats["total_member_leaves"] += 1
        
        return completely_removed
    
    def _remove_user_from_room_mapping(self, user_id: str, room_id: str):
        """Remove user from room mapping."""
        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room_id)
            if not self.user_rooms[user_id]:
                del self.user_rooms[user_id]
    
    def get_user_rooms(self, user_id: str) -> List[Room]:
        """Get all rooms a user is in."""
        if user_id not in self.user_rooms:
            return []
        
        return [
            self.rooms[room_id] 
            for room_id in self.user_rooms[user_id]
            if room_id in self.rooms
        ]
    
    def get_rooms_by_type(self, room_type: RoomType) -> List[Room]:
        """Get all rooms of a specific type."""
        return [
            room for room in self.rooms.values()
            if room.room_type == room_type
        ]
    
    def get_public_rooms(self) -> List[Dict[str, Any]]:
        """Get list of public rooms (non-sensitive info)."""
        return [
            room.to_dict(include_sensitive=False)
            for room in self.rooms.values()
            if not room.settings.require_approval
        ]
    
    async def cleanup_rooms(self):
        """Clean up empty rooms that should be removed."""
        rooms_to_delete = []
        
        for room_id, room in self.rooms.items():
            if room.should_cleanup():
                rooms_to_delete.append(room_id)
        
        for room_id in rooms_to_delete:
            self.delete_room(room_id)
        
        if rooms_to_delete:
            logger.info(f"Cleaned up {len(rooms_to_delete)} empty rooms")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get room manager statistics."""
        return {
            **self.stats,
            "active_rooms": len(self.rooms),
            "active_users": len(self.user_rooms),
            "rooms_by_type": {
                room_type.value: len(self.get_rooms_by_type(room_type))
                for room_type in RoomType
            },
            "total_active_members": sum(
                room.get_member_count() for room in self.rooms.values()
            )
        }
    
    async def broadcast_to_room_type(
        self,
        room_type: RoomType,
        event: WebSocketEvent,
        exclude_rooms: Optional[Set[str]] = None
    ):
        """Broadcast event to all rooms of a specific type."""
        # This would be implemented with connection manager
        # Left as placeholder for integration
        pass
    
    async def shutdown(self):
        """Shutdown room manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all room cleanup tasks
        for room in self.rooms.values():
            if room.cleanup_task:
                room.cleanup_task.cancel()
        
        # Clear all data
        self.rooms.clear()
        self.user_rooms.clear()
        
        logger.info("RoomManager shutdown complete")