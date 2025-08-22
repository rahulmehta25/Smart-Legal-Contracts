"""
WebSocket server for real-time collaboration with room management.
"""
import json
import asyncio
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import uuid

import socketio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.security import verify_token
from ..models.user import User


# Event types for collaboration
class CollaborationEventType(Enum):
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    USER_PRESENCE = "user_presence"
    CURSOR_MOVE = "cursor_move"
    DOCUMENT_EDIT = "document_edit"
    ANNOTATION_ADD = "annotation_add"
    ANNOTATION_UPDATE = "annotation_update"
    ANNOTATION_DELETE = "annotation_delete"
    COMMENT_ADD = "comment_add"
    COMMENT_UPDATE = "comment_update"
    COMMENT_DELETE = "comment_delete"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class CollaborationEvent:
    """Standardized collaboration event structure."""
    event_type: CollaborationEventType
    room_id: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class UserStatus(Enum):
    ONLINE = "online"
    AWAY = "away" 
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class RoomMember:
    """Information about a room member."""
    user_id: str
    username: str
    avatar_url: Optional[str]
    status: UserStatus
    join_time: datetime
    last_activity: datetime
    cursor_position: Optional[Dict[str, Any]] = None
    permissions: Set[str] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {"read", "write", "comment"}


class CollaborationRoom:
    """Manages a collaboration room with its members and state."""
    
    def __init__(self, room_id: str, document_id: str, owner_id: str):
        self.room_id = room_id
        self.document_id = document_id
        self.owner_id = owner_id
        self.created_at = datetime.utcnow()
        self.members: Dict[str, RoomMember] = {}
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.document_state = {}
        self.operation_queue = []
        self.last_sync_timestamp = datetime.utcnow()
        
    def add_member(self, user_id: str, username: str, websocket: WebSocket, 
                   avatar_url: Optional[str] = None) -> RoomMember:
        """Add a member to the room."""
        member = RoomMember(
            user_id=user_id,
            username=username,
            avatar_url=avatar_url,
            status=UserStatus.ONLINE,
            join_time=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        
        self.members[user_id] = member
        
        if user_id not in self.connections:
            self.connections[user_id] = set()
        self.connections[user_id].add(websocket)
        
        return member
    
    def remove_member(self, user_id: str, websocket: WebSocket):
        """Remove a member from the room."""
        if user_id in self.connections:
            self.connections[user_id].discard(websocket)
            if not self.connections[user_id]:
                del self.connections[user_id]
                if user_id in self.members:
                    del self.members[user_id]
    
    def update_member_activity(self, user_id: str):
        """Update member's last activity timestamp."""
        if user_id in self.members:
            self.members[user_id].last_activity = datetime.utcnow()
    
    def get_active_members(self) -> Dict[str, RoomMember]:
        """Get currently active members."""
        return {uid: member for uid, member in self.members.items() 
                if uid in self.connections}
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        if user_id == self.owner_id:
            return True
        if user_id in self.members:
            return permission in self.members[user_id].permissions
        return False


class CollaborationWebSocketServer:
    """WebSocket server managing real-time collaboration."""
    
    def __init__(self):
        self.rooms: Dict[str, CollaborationRoom] = {}
        self.user_rooms: Dict[str, Set[str]] = {}  # user_id -> set of room_ids
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )
        self.setup_handlers()
        
    def setup_handlers(self):
        """Set up WebSocket event handlers."""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection."""
            try:
                # Extract token from auth or query params
                token = auth.get('token') if auth else None
                if not token:
                    raise HTTPException(status_code=401, detail="No token provided")
                
                # Verify token and get user info
                user = await verify_token(token)
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                # Store user info in session
                await self.sio.save_session(sid, {
                    'user_id': user.id,
                    'username': user.username,
                    'avatar_url': getattr(user, 'avatar_url', None)
                })
                
                logging.info(f"User {user.username} connected with session {sid}")
                
            except Exception as e:
                logging.error(f"Connection failed: {e}")
                return False
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            try:
                session = await self.sio.get_session(sid)
                user_id = session.get('user_id')
                
                if user_id and user_id in self.user_rooms:
                    # Leave all rooms
                    for room_id in list(self.user_rooms[user_id]):
                        await self.leave_room(sid, {'room_id': room_id})
                
                logging.info(f"User {user_id} disconnected")
                
            except Exception as e:
                logging.error(f"Disconnect error: {e}")
        
        @self.sio.event
        async def join_room(sid, data):
            """Handle joining a collaboration room."""
            return await self.join_room(sid, data)
        
        @self.sio.event
        async def leave_room(sid, data):
            """Handle leaving a collaboration room."""
            return await self.leave_room(sid, data)
        
        @self.sio.event
        async def cursor_move(sid, data):
            """Handle cursor movement."""
            return await self.handle_cursor_move(sid, data)
        
        @self.sio.event
        async def document_edit(sid, data):
            """Handle document editing operations."""
            return await self.handle_document_edit(sid, data)
        
        @self.sio.event
        async def add_annotation(sid, data):
            """Handle adding annotations."""
            return await self.handle_add_annotation(sid, data)
        
        @self.sio.event
        async def add_comment(sid, data):
            """Handle adding comments."""
            return await self.handle_add_comment(sid, data)
        
        @self.sio.event
        async def sync_request(sid, data):
            """Handle synchronization requests."""
            return await self.handle_sync_request(sid, data)
    
    async def join_room(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user joining a room."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            username = session['username']
            avatar_url = session.get('avatar_url')
            
            room_id = data['room_id']
            document_id = data.get('document_id')
            
            # Create room if it doesn't exist
            if room_id not in self.rooms:
                self.rooms[room_id] = CollaborationRoom(
                    room_id=room_id,
                    document_id=document_id,
                    owner_id=user_id
                )
            
            room = self.rooms[room_id]
            
            # Add user to room
            # Note: We need a WebSocket object here, but socket.io abstracts this
            # We'll use sid as a pseudo-websocket for now
            member = room.add_member(user_id, username, sid, avatar_url)
            
            # Track user's rooms
            if user_id not in self.user_rooms:
                self.user_rooms[user_id] = set()
            self.user_rooms[user_id].add(room_id)
            
            # Join socket.io room
            await self.sio.enter_room(sid, room_id)
            
            # Notify other members
            event = CollaborationEvent(
                event_type=CollaborationEventType.USER_PRESENCE,
                room_id=room_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                data={
                    'action': 'joined',
                    'user': asdict(member),
                    'active_members': [asdict(m) for m in room.get_active_members().values()]
                }
            )
            
            await self.sio.emit('user_presence', asdict(event), room=room_id, skip_sid=sid)
            
            return {
                'success': True,
                'room_state': {
                    'room_id': room_id,
                    'document_id': room.document_id,
                    'members': [asdict(m) for m in room.get_active_members().values()],
                    'document_state': room.document_state,
                    'last_sync': room.last_sync_timestamp.isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"Error joining room: {e}")
            return {'success': False, 'error': str(e)}
    
    async def leave_room(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user leaving a room."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                room.remove_member(user_id, sid)
                
                # Leave socket.io room
                await self.sio.leave_room(sid, room_id)
                
                # Update user's rooms
                if user_id in self.user_rooms:
                    self.user_rooms[user_id].discard(room_id)
                
                # Notify other members
                event = CollaborationEvent(
                    event_type=CollaborationEventType.USER_PRESENCE,
                    room_id=room_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    data={
                        'action': 'left',
                        'user_id': user_id,
                        'active_members': [asdict(m) for m in room.get_active_members().values()]
                    }
                )
                
                await self.sio.emit('user_presence', asdict(event), room=room_id)
                
                # Clean up empty rooms
                if not room.get_active_members():
                    del self.rooms[room_id]
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error leaving room: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_cursor_move(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cursor movement events."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            cursor_data = data['cursor']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                # Update member's cursor position
                if user_id in room.members:
                    room.members[user_id].cursor_position = cursor_data
                    room.update_member_activity(user_id)
                
                # Broadcast to other members
                event = CollaborationEvent(
                    event_type=CollaborationEventType.CURSOR_MOVE,
                    room_id=room_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    data={'cursor': cursor_data, 'username': session['username']}
                )
                
                await self.sio.emit('cursor_move', asdict(event), room=room_id, skip_sid=sid)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling cursor move: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_document_edit(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document editing operations."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            operation = data['operation']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                # Check permissions
                if not room.has_permission(user_id, 'write'):
                    return {'success': False, 'error': 'Insufficient permissions'}
                
                # Add operation to queue for CRDT processing
                room.operation_queue.append({
                    'operation': operation,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow(),
                    'operation_id': str(uuid.uuid4())
                })
                
                # Update member activity
                room.update_member_activity(user_id)
                
                # Broadcast to other members
                event = CollaborationEvent(
                    event_type=CollaborationEventType.DOCUMENT_EDIT,
                    room_id=room_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    data={
                        'operation': operation,
                        'username': session['username']
                    }
                )
                
                await self.sio.emit('document_edit', asdict(event), room=room_id, skip_sid=sid)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling document edit: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_add_annotation(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adding annotations."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            annotation = data['annotation']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                # Check permissions
                if not room.has_permission(user_id, 'comment'):
                    return {'success': False, 'error': 'Insufficient permissions'}
                
                # Add metadata to annotation
                annotation.update({
                    'id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'username': session['username'],
                    'created_at': datetime.utcnow().isoformat()
                })
                
                # Update member activity
                room.update_member_activity(user_id)
                
                # Broadcast to all members
                event = CollaborationEvent(
                    event_type=CollaborationEventType.ANNOTATION_ADD,
                    room_id=room_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    data={'annotation': annotation}
                )
                
                await self.sio.emit('annotation_add', asdict(event), room=room_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error adding annotation: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_add_comment(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adding comments."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            comment = data['comment']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                # Check permissions
                if not room.has_permission(user_id, 'comment'):
                    return {'success': False, 'error': 'Insufficient permissions'}
                
                # Add metadata to comment
                comment.update({
                    'id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'username': session['username'],
                    'created_at': datetime.utcnow().isoformat(),
                    'replies': []
                })
                
                # Update member activity
                room.update_member_activity(user_id)
                
                # Broadcast to all members
                event = CollaborationEvent(
                    event_type=CollaborationEventType.COMMENT_ADD,
                    room_id=room_id,
                    user_id=user_id,
                    timestamp=datetime.utcnow(),
                    data={'comment': comment}
                )
                
                await self.sio.emit('comment_add', asdict(event), room=room_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error adding comment: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_sync_request(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronization requests."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                # Send current state to requesting user
                sync_data = {
                    'document_state': room.document_state,
                    'operation_queue': room.operation_queue[-50:],  # Last 50 operations
                    'members': [asdict(m) for m in room.get_active_members().values()],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.sio.emit('sync_response', {
                    'room_id': room_id,
                    'sync_data': sync_data
                }, to=sid)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling sync request: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_room_stats(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a room."""
        if room_id not in self.rooms:
            return None
        
        room = self.rooms[room_id]
        return {
            'room_id': room_id,
            'document_id': room.document_id,
            'owner_id': room.owner_id,
            'created_at': room.created_at.isoformat(),
            'active_members': len(room.get_active_members()),
            'total_operations': len(room.operation_queue),
            'last_sync': room.last_sync_timestamp.isoformat()
        }
    
    def get_all_room_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rooms."""
        return {room_id: self.get_room_stats(room_id) 
                for room_id in self.rooms.keys()}


# Global instance
collaboration_server = CollaborationWebSocketServer()


def get_collaboration_app():
    """Create Socket.IO ASGI application."""
    return socketio.ASGIApp(collaboration_server.sio)