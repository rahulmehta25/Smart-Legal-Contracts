"""
WebSocket Event Definitions for real-time features.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from pydantic import BaseModel, Field


class EventType(Enum):
    """WebSocket event types for real-time features."""
    
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"
    HEARTBEAT = "heartbeat"
    
    # User presence
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_STATUS_UPDATE = "user_status_update"
    PRESENCE_UPDATE = "presence_update"
    
    # Real-time analysis
    ANALYSIS_START = "analysis_start"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    ANALYSIS_CANCELLED = "analysis_cancelled"
    
    # Document sharing and collaboration
    DOCUMENT_SHARED = "document_shared"
    DOCUMENT_UNSHARED = "document_unshared"
    DOCUMENT_EDIT = "document_edit"
    DOCUMENT_LOCK = "document_lock"
    DOCUMENT_UNLOCK = "document_unlock"
    
    # Collaborative annotations
    ANNOTATION_ADDED = "annotation_added"
    ANNOTATION_UPDATED = "annotation_updated"
    ANNOTATION_DELETED = "annotation_deleted"
    ANNOTATION_HIGHLIGHTED = "annotation_highlighted"
    
    # Comments and discussions
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    COMMENT_REPLY = "comment_reply"
    
    # Real-time notifications
    NOTIFICATION_NEW = "notification_new"
    NOTIFICATION_READ = "notification_read"
    NOTIFICATION_DISMISSED = "notification_dismissed"
    NOTIFICATION_BULK_UPDATE = "notification_bulk_update"
    
    # Cursor and selection tracking
    CURSOR_MOVE = "cursor_move"
    SELECTION_UPDATE = "selection_update"
    VIEWPORT_CHANGE = "viewport_change"
    
    # Room management
    ROOM_CREATED = "room_created"
    ROOM_JOINED = "room_joined"
    ROOM_LEFT = "room_left"
    ROOM_UPDATED = "room_updated"
    ROOM_DELETED = "room_deleted"
    
    # System events
    SYSTEM_MAINTENANCE = "system_maintenance"
    SYSTEM_UPDATE = "system_update"
    SYSTEM_ERROR = "system_error"
    
    # Video/Audio collaboration
    VIDEO_CALL_START = "video_call_start"
    VIDEO_CALL_END = "video_call_end"
    AUDIO_TOGGLE = "audio_toggle"
    VIDEO_TOGGLE = "video_toggle"
    SCREEN_SHARE_START = "screen_share_start"
    SCREEN_SHARE_END = "screen_share_end"
    
    # File operations
    FILE_UPLOAD_START = "file_upload_start"
    FILE_UPLOAD_PROGRESS = "file_upload_progress"
    FILE_UPLOAD_COMPLETE = "file_upload_complete"
    FILE_UPLOAD_ERROR = "file_upload_error"
    
    # Custom events
    CUSTOM_EVENT = "custom_event"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WebSocketEvent:
    """Standardized WebSocket event structure."""
    
    event_type: EventType
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context information
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    priority: EventPriority = EventPriority.NORMAL
    retryable: bool = True
    ttl: Optional[int] = None  # Time to live in seconds
    requires_ack: bool = False  # Whether this event requires acknowledgment
    
    # Routing information
    target_users: Optional[List[str]] = None
    target_rooms: Optional[List[str]] = None
    exclude_users: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "priority": self.priority.value,
            "retryable": self.retryable,
            "requires_ack": self.requires_ack
        }
        
        # Add optional fields if present
        if self.user_id:
            result["user_id"] = self.user_id
        if self.room_id:
            result["room_id"] = self.room_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.ttl:
            result["ttl"] = self.ttl
        if self.target_users:
            result["target_users"] = self.target_users
        if self.target_rooms:
            result["target_rooms"] = self.target_rooms
        if self.exclude_users:
            result["exclude_users"] = self.exclude_users
            
        return result
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketEvent':
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            user_id=data.get("user_id"),
            room_id=data.get("room_id"),
            session_id=data.get("session_id"),
            data=data.get("data", {}),
            priority=EventPriority(data.get("priority", "normal")),
            retryable=data.get("retryable", True),
            ttl=data.get("ttl"),
            requires_ack=data.get("requires_ack", False),
            target_users=data.get("target_users"),
            target_rooms=data.get("target_rooms"),
            exclude_users=data.get("exclude_users")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketEvent':
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Event factory functions for common event types

def create_analysis_progress_event(
    analysis_id: str,
    user_id: str,
    progress: float,
    stage: str,
    details: Optional[Dict[str, Any]] = None
) -> WebSocketEvent:
    """Create analysis progress event."""
    return WebSocketEvent(
        event_type=EventType.ANALYSIS_PROGRESS,
        user_id=user_id,
        data={
            "analysis_id": analysis_id,
            "progress": progress,
            "stage": stage,
            "details": details or {}
        },
        priority=EventPriority.HIGH
    )


def create_notification_event(
    user_id: str,
    title: str,
    message: str,
    notification_type: str = "info",
    action_url: Optional[str] = None,
    expires_at: Optional[datetime] = None
) -> WebSocketEvent:
    """Create notification event."""
    data = {
        "title": title,
        "message": message,
        "type": notification_type,
        "created_at": datetime.utcnow().isoformat()
    }
    
    if action_url:
        data["action_url"] = action_url
    if expires_at:
        data["expires_at"] = expires_at.isoformat()
    
    return WebSocketEvent(
        event_type=EventType.NOTIFICATION_NEW,
        user_id=user_id,
        data=data,
        priority=EventPriority.NORMAL,
        requires_ack=True
    )


def create_presence_event(
    user_id: str,
    username: str,
    status: str,
    room_id: Optional[str] = None,
    avatar_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WebSocketEvent:
    """Create user presence event."""
    data = {
        "user_id": user_id,
        "username": username,
        "status": status,
        "last_seen": datetime.utcnow().isoformat()
    }
    
    if avatar_url:
        data["avatar_url"] = avatar_url
    if metadata:
        data["metadata"] = metadata
    
    return WebSocketEvent(
        event_type=EventType.PRESENCE_UPDATE,
        user_id=user_id,
        room_id=room_id,
        data=data,
        priority=EventPriority.NORMAL
    )


def create_annotation_event(
    event_type: EventType,
    user_id: str,
    room_id: str,
    annotation_id: str,
    annotation_data: Dict[str, Any],
    username: Optional[str] = None
) -> WebSocketEvent:
    """Create annotation-related event."""
    data = {
        "annotation_id": annotation_id,
        "annotation": annotation_data,
        "user_id": user_id
    }
    
    if username:
        data["username"] = username
    
    return WebSocketEvent(
        event_type=event_type,
        user_id=user_id,
        room_id=room_id,
        data=data,
        priority=EventPriority.NORMAL
    )


def create_document_event(
    event_type: EventType,
    user_id: str,
    room_id: str,
    document_id: str,
    operation: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WebSocketEvent:
    """Create document-related event."""
    data = {
        "document_id": document_id,
        "user_id": user_id
    }
    
    if operation:
        data["operation"] = operation
    if metadata:
        data["metadata"] = metadata
    
    return WebSocketEvent(
        event_type=event_type,
        user_id=user_id,
        room_id=room_id,
        data=data,
        priority=EventPriority.HIGH
    )


def create_cursor_event(
    user_id: str,
    username: str,
    room_id: str,
    cursor_position: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None
) -> WebSocketEvent:
    """Create cursor movement event."""
    data = {
        "user_id": user_id,
        "username": username,
        "cursor": cursor_position
    }
    
    if selection:
        data["selection"] = selection
    
    return WebSocketEvent(
        event_type=EventType.CURSOR_MOVE,
        user_id=user_id,
        room_id=room_id,
        data=data,
        priority=EventPriority.LOW
    )


def create_system_event(
    event_type: EventType,
    message: str,
    severity: str = "info",
    affected_users: Optional[List[str]] = None,
    maintenance_window: Optional[Dict[str, str]] = None
) -> WebSocketEvent:
    """Create system-wide event."""
    data = {
        "message": message,
        "severity": severity
    }
    
    if maintenance_window:
        data["maintenance_window"] = maintenance_window
    
    return WebSocketEvent(
        event_type=event_type,
        data=data,
        priority=EventPriority.HIGH if severity == "critical" else EventPriority.NORMAL,
        target_users=affected_users,
        requires_ack=True
    )


def create_file_upload_event(
    event_type: EventType,
    user_id: str,
    file_id: str,
    filename: str,
    progress: Optional[float] = None,
    error: Optional[str] = None,
    file_url: Optional[str] = None
) -> WebSocketEvent:
    """Create file upload event."""
    data = {
        "file_id": file_id,
        "filename": filename
    }
    
    if progress is not None:
        data["progress"] = progress
    if error:
        data["error"] = error
    if file_url:
        data["file_url"] = file_url
    
    return WebSocketEvent(
        event_type=event_type,
        user_id=user_id,
        data=data,
        priority=EventPriority.NORMAL
    )


# Event validation functions

def validate_event(event: WebSocketEvent) -> tuple[bool, Optional[str]]:
    """
    Validate a WebSocket event.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check required fields
    if not event.event_type:
        return False, "Event type is required"
    
    if not event.event_id:
        return False, "Event ID is required"
    
    # Validate timestamp
    if not isinstance(event.timestamp, datetime):
        return False, "Timestamp must be a datetime object"
    
    # Validate data is serializable
    try:
        json.dumps(event.data, default=str)
    except (TypeError, ValueError) as e:
        return False, f"Event data is not JSON serializable: {e}"
    
    # Event-specific validations
    if event.event_type in [EventType.ANALYSIS_PROGRESS, EventType.ANALYSIS_COMPLETE]:
        if "analysis_id" not in event.data:
            return False, "Analysis events require analysis_id in data"
    
    if event.event_type in [EventType.ANNOTATION_ADDED, EventType.ANNOTATION_UPDATED]:
        if "annotation_id" not in event.data:
            return False, "Annotation events require annotation_id in data"
    
    if event.event_type == EventType.NOTIFICATION_NEW:
        required_fields = ["title", "message"]
        for field in required_fields:
            if field not in event.data:
                return False, f"Notification events require {field} in data"
    
    return True, None


# Event serialization utilities

class EventEncoder(json.JSONEncoder):
    """Custom JSON encoder for WebSocket events."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def serialize_event(event: WebSocketEvent) -> str:
    """Serialize event to JSON string."""
    return json.dumps(event.to_dict(), cls=EventEncoder)


def deserialize_event(json_str: str) -> WebSocketEvent:
    """Deserialize event from JSON string."""
    return WebSocketEvent.from_json(json_str)