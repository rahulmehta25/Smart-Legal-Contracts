"""
WebSocket Event Handlers for real-time features.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from fastapi import WebSocket, WebSocketDisconnect, HTTPException

from .connection_manager import ConnectionManager, UserStatus
from .events import (
    WebSocketEvent, EventType, EventPriority,
    create_analysis_progress_event, create_notification_event,
    create_presence_event, create_annotation_event, create_document_event,
    create_cursor_event, create_system_event, create_file_upload_event,
    validate_event
)
from .rooms import RoomManager, Room
from ..core.security import verify_token
from ..models.user import User

logger = logging.getLogger(__name__)


class WebSocketHandlers:
    """WebSocket event handlers for real-time features."""
    
    def __init__(self, connection_manager: ConnectionManager, room_manager: RoomManager):
        self.connection_manager = connection_manager
        self.room_manager = room_manager
        
        # Event handler registry
        self.handlers: Dict[EventType, callable] = {
            # Connection management
            EventType.CONNECT: self.handle_connect,
            EventType.DISCONNECT: self.handle_disconnect,
            EventType.HEARTBEAT: self.handle_heartbeat,
            
            # User presence
            EventType.USER_STATUS_UPDATE: self.handle_user_status_update,
            EventType.PRESENCE_UPDATE: self.handle_presence_update,
            
            # Real-time analysis
            EventType.ANALYSIS_START: self.handle_analysis_start,
            EventType.ANALYSIS_PROGRESS: self.handle_analysis_progress,
            EventType.ANALYSIS_COMPLETE: self.handle_analysis_complete,
            EventType.ANALYSIS_ERROR: self.handle_analysis_error,
            EventType.ANALYSIS_CANCELLED: self.handle_analysis_cancelled,
            
            # Document collaboration
            EventType.DOCUMENT_SHARED: self.handle_document_shared,
            EventType.DOCUMENT_EDIT: self.handle_document_edit,
            EventType.DOCUMENT_LOCK: self.handle_document_lock,
            EventType.DOCUMENT_UNLOCK: self.handle_document_unlock,
            
            # Annotations
            EventType.ANNOTATION_ADDED: self.handle_annotation_added,
            EventType.ANNOTATION_UPDATED: self.handle_annotation_updated,
            EventType.ANNOTATION_DELETED: self.handle_annotation_deleted,
            EventType.ANNOTATION_HIGHLIGHTED: self.handle_annotation_highlighted,
            
            # Comments
            EventType.COMMENT_ADDED: self.handle_comment_added,
            EventType.COMMENT_UPDATED: self.handle_comment_updated,
            EventType.COMMENT_DELETED: self.handle_comment_deleted,
            EventType.COMMENT_REPLY: self.handle_comment_reply,
            
            # Notifications
            EventType.NOTIFICATION_NEW: self.handle_notification_new,
            EventType.NOTIFICATION_READ: self.handle_notification_read,
            EventType.NOTIFICATION_DISMISSED: self.handle_notification_dismissed,
            
            # Cursor tracking
            EventType.CURSOR_MOVE: self.handle_cursor_move,
            EventType.SELECTION_UPDATE: self.handle_selection_update,
            EventType.VIEWPORT_CHANGE: self.handle_viewport_change,
            
            # Room management
            EventType.ROOM_JOINED: self.handle_room_joined,
            EventType.ROOM_LEFT: self.handle_room_left,
            EventType.ROOM_UPDATED: self.handle_room_updated,
            
            # File operations
            EventType.FILE_UPLOAD_START: self.handle_file_upload_start,
            EventType.FILE_UPLOAD_PROGRESS: self.handle_file_upload_progress,
            EventType.FILE_UPLOAD_COMPLETE: self.handle_file_upload_complete,
            EventType.FILE_UPLOAD_ERROR: self.handle_file_upload_error,
            
            # Video/Audio
            EventType.VIDEO_CALL_START: self.handle_video_call_start,
            EventType.VIDEO_CALL_END: self.handle_video_call_end,
            EventType.AUDIO_TOGGLE: self.handle_audio_toggle,
            EventType.VIDEO_TOGGLE: self.handle_video_toggle,
            
            # Custom events
            EventType.CUSTOM_EVENT: self.handle_custom_event,
        }
        
        # Active analysis tracking
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        
        # File upload tracking
        self.active_uploads: Dict[str, Dict[str, Any]] = {}
    
    async def handle_websocket_connection(self, websocket: WebSocket, token: str):
        """Handle new WebSocket connection."""
        try:
            # Verify authentication
            user = await verify_token(token)
            if not user:
                await websocket.close(code=4001, reason="Invalid token")
                return
            
            await websocket.accept()
            
            # Register connection
            connection_id = await self.connection_manager.connect(
                websocket=websocket,
                user_id=str(user.id),
                username=user.username,
                avatar_url=getattr(user, 'avatar_url', None)
            )
            
            # Send welcome message
            welcome_event = WebSocketEvent(
                event_type=EventType.CONNECT,
                user_id=str(user.id),
                data={
                    "connection_id": connection_id,
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "avatar_url": getattr(user, 'avatar_url', None)
                    },
                    "server_time": datetime.utcnow().isoformat(),
                    "features": [
                        "real_time_analysis",
                        "collaborative_annotations",
                        "live_presence",
                        "notifications",
                        "document_sharing"
                    ]
                }
            )
            
            await self.connection_manager.send_to_connection(
                connection_id, welcome_event.to_dict()
            )
            
            # Listen for messages
            await self._listen_for_messages(connection_id, websocket)
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            try:
                await websocket.close(code=4000, reason="Connection error")
            except:
                pass
    
    async def _listen_for_messages(self, connection_id: str, websocket: WebSocket):
        """Listen for incoming WebSocket messages."""
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                
                try:
                    # Parse event
                    event_data = json.loads(data)
                    event = WebSocketEvent.from_dict(event_data)
                    
                    # Validate event
                    is_valid, error_msg = validate_event(event)
                    if not is_valid:
                        await self._send_error(connection_id, f"Invalid event: {error_msg}")
                        continue
                    
                    # Update connection activity
                    connection = self.connection_manager.get_connection(connection_id)
                    if connection:
                        connection.last_activity = datetime.utcnow()
                    
                    # Handle event
                    await self.handle_event(connection_id, event)
                    
                except json.JSONDecodeError:
                    await self._send_error(connection_id, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error processing message from {connection_id}: {e}")
                    await self._send_error(connection_id, "Error processing message")
                    
        except WebSocketDisconnect:
            await self.connection_manager.disconnect(connection_id, "client_disconnect")
        except Exception as e:
            logger.error(f"WebSocket listen error for {connection_id}: {e}")
            await self.connection_manager.disconnect(connection_id, "listen_error")
    
    async def handle_event(self, connection_id: str, event: WebSocketEvent):
        """Route event to appropriate handler."""
        handler = self.handlers.get(event.event_type)
        if handler:
            try:
                await handler(connection_id, event)
            except Exception as e:
                logger.error(f"Error handling event {event.event_type}: {e}")
                await self._send_error(connection_id, f"Error handling event: {e}")
        else:
            logger.warning(f"No handler for event type: {event.event_type}")
            await self._send_error(connection_id, f"Unknown event type: {event.event_type}")
    
    async def _send_error(self, connection_id: str, message: str):
        """Send error message to connection."""
        error_event = WebSocketEvent(
            event_type=EventType.SYSTEM_ERROR,
            data={"error": message, "timestamp": datetime.utcnow().isoformat()}
        )
        await self.connection_manager.send_to_connection(
            connection_id, error_event.to_dict()
        )
    
    # Connection management handlers
    
    async def handle_connect(self, connection_id: str, event: WebSocketEvent):
        """Handle connection event."""
        logger.info(f"Connection {connection_id} established")
    
    async def handle_disconnect(self, connection_id: str, event: WebSocketEvent):
        """Handle disconnect event."""
        await self.connection_manager.disconnect(connection_id, "client_request")
    
    async def handle_heartbeat(self, connection_id: str, event: WebSocketEvent):
        """Handle heartbeat/ping event."""
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            connection.last_ping = datetime.utcnow()
            # Send pong response
            pong_event = WebSocketEvent(
                event_type=EventType.HEARTBEAT,
                data={"pong": True, "timestamp": datetime.utcnow().isoformat()}
            )
            await self.connection_manager.send_to_connection(
                connection_id, pong_event.to_dict()
            )
    
    # User presence handlers
    
    async def handle_user_status_update(self, connection_id: str, event: WebSocketEvent):
        """Handle user status update."""
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        status_str = event.data.get("status", "online")
        try:
            status = UserStatus(status_str)
            await self.connection_manager.update_user_status(connection.user_id, status)
        except ValueError:
            await self._send_error(connection_id, f"Invalid status: {status_str}")
    
    async def handle_presence_update(self, connection_id: str, event: WebSocketEvent):
        """Handle presence update."""
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        # Broadcast presence update to relevant rooms
        for room_id in connection.rooms:
            presence_event = create_presence_event(
                user_id=connection.user_id,
                username=event.data.get("username", "Unknown"),
                status=event.data.get("status", "online"),
                room_id=room_id,
                metadata=event.data.get("metadata")
            )
            
            await self.connection_manager.send_to_room(
                room_id, presence_event.to_dict(), exclude_connection=connection_id
            )
    
    # Analysis handlers
    
    async def handle_analysis_start(self, connection_id: str, event: WebSocketEvent):
        """Handle analysis start event."""
        analysis_id = event.data.get("analysis_id")
        if not analysis_id:
            await self._send_error(connection_id, "analysis_id required")
            return
        
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        # Track active analysis
        self.active_analyses[analysis_id] = {
            "user_id": connection.user_id,
            "connection_id": connection_id,
            "started_at": datetime.utcnow(),
            "progress": 0.0,
            "stage": "initializing"
        }
        
        # Broadcast start event if room is specified
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
    
    async def handle_analysis_progress(self, connection_id: str, event: WebSocketEvent):
        """Handle analysis progress update."""
        analysis_id = event.data.get("analysis_id")
        progress = event.data.get("progress", 0.0)
        stage = event.data.get("stage", "processing")
        
        if analysis_id in self.active_analyses:
            self.active_analyses[analysis_id].update({
                "progress": progress,
                "stage": stage,
                "last_update": datetime.utcnow()
            })
        
        # Broadcast to room if specified
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict()
            )
        else:
            # Send to user's other connections
            connection = self.connection_manager.get_connection(connection_id)
            if connection:
                await self.connection_manager.send_to_user(
                    connection.user_id, event.to_dict()
                )
    
    async def handle_analysis_complete(self, connection_id: str, event: WebSocketEvent):
        """Handle analysis completion."""
        analysis_id = event.data.get("analysis_id")
        
        if analysis_id in self.active_analyses:
            analysis_info = self.active_analyses.pop(analysis_id)
            
            # Add completion metadata
            event.data.update({
                "completed_at": datetime.utcnow().isoformat(),
                "duration": (datetime.utcnow() - analysis_info["started_at"]).total_seconds()
            })
        
        # Broadcast completion
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict()
            )
        else:
            connection = self.connection_manager.get_connection(connection_id)
            if connection:
                await self.connection_manager.send_to_user(
                    connection.user_id, event.to_dict()
                )
    
    async def handle_analysis_error(self, connection_id: str, event: WebSocketEvent):
        """Handle analysis error."""
        analysis_id = event.data.get("analysis_id")
        
        if analysis_id in self.active_analyses:
            del self.active_analyses[analysis_id]
        
        # Broadcast error
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict()
            )
        else:
            connection = self.connection_manager.get_connection(connection_id)
            if connection:
                await self.connection_manager.send_to_user(
                    connection.user_id, event.to_dict()
                )
    
    async def handle_analysis_cancelled(self, connection_id: str, event: WebSocketEvent):
        """Handle analysis cancellation."""
        analysis_id = event.data.get("analysis_id")
        
        if analysis_id in self.active_analyses:
            del self.active_analyses[analysis_id]
        
        # Broadcast cancellation
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict()
            )
    
    # Document collaboration handlers
    
    async def handle_document_shared(self, connection_id: str, event: WebSocketEvent):
        """Handle document sharing."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required for document sharing")
            return
        
        # Add user to room if not already there
        await self.connection_manager.join_room(connection_id, event.room_id)
        
        # Broadcast document shared event
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    async def handle_document_edit(self, connection_id: str, event: WebSocketEvent):
        """Handle document edit."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required for document edit")
            return
        
        # Validate edit operation
        operation = event.data.get("operation")
        if not operation:
            await self._send_error(connection_id, "operation required for document edit")
            return
        
        # Add operation metadata
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            event.data.update({
                "user_id": connection.user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "operation_id": event.event_id
            })
        
        # Broadcast to room
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    async def handle_document_lock(self, connection_id: str, event: WebSocketEvent):
        """Handle document lock."""
        # Broadcast lock event
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
    
    async def handle_document_unlock(self, connection_id: str, event: WebSocketEvent):
        """Handle document unlock."""
        # Broadcast unlock event
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
    
    # Annotation handlers
    
    async def handle_annotation_added(self, connection_id: str, event: WebSocketEvent):
        """Handle annotation addition."""
        await self._handle_annotation_event(connection_id, event)
    
    async def handle_annotation_updated(self, connection_id: str, event: WebSocketEvent):
        """Handle annotation update."""
        await self._handle_annotation_event(connection_id, event)
    
    async def handle_annotation_deleted(self, connection_id: str, event: WebSocketEvent):
        """Handle annotation deletion."""
        await self._handle_annotation_event(connection_id, event)
    
    async def handle_annotation_highlighted(self, connection_id: str, event: WebSocketEvent):
        """Handle annotation highlighting."""
        await self._handle_annotation_event(connection_id, event)
    
    async def _handle_annotation_event(self, connection_id: str, event: WebSocketEvent):
        """Common handler for annotation events."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required for annotation events")
            return
        
        # Add user info
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            event.data.update({
                "user_id": connection.user_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Broadcast to room
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    # Comment handlers
    
    async def handle_comment_added(self, connection_id: str, event: WebSocketEvent):
        """Handle comment addition."""
        await self._handle_comment_event(connection_id, event)
    
    async def handle_comment_updated(self, connection_id: str, event: WebSocketEvent):
        """Handle comment update."""
        await self._handle_comment_event(connection_id, event)
    
    async def handle_comment_deleted(self, connection_id: str, event: WebSocketEvent):
        """Handle comment deletion."""
        await self._handle_comment_event(connection_id, event)
    
    async def handle_comment_reply(self, connection_id: str, event: WebSocketEvent):
        """Handle comment reply."""
        await self._handle_comment_event(connection_id, event)
    
    async def _handle_comment_event(self, connection_id: str, event: WebSocketEvent):
        """Common handler for comment events."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required for comment events")
            return
        
        # Add user info
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            event.data.update({
                "user_id": connection.user_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Broadcast to room
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    # Notification handlers
    
    async def handle_notification_new(self, connection_id: str, event: WebSocketEvent):
        """Handle new notification."""
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        # Send to user's all connections
        await self.connection_manager.send_to_user(
            connection.user_id, event.to_dict()
        )
    
    async def handle_notification_read(self, connection_id: str, event: WebSocketEvent):
        """Handle notification read."""
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        # Broadcast to user's other connections
        await self.connection_manager.send_to_user(
            connection.user_id, event.to_dict()
        )
    
    async def handle_notification_dismissed(self, connection_id: str, event: WebSocketEvent):
        """Handle notification dismissal."""
        connection = self.connection_manager.get_connection(connection_id)
        if not connection:
            return
        
        # Broadcast to user's other connections
        await self.connection_manager.send_to_user(
            connection.user_id, event.to_dict()
        )
    
    # Cursor tracking handlers
    
    async def handle_cursor_move(self, connection_id: str, event: WebSocketEvent):
        """Handle cursor movement."""
        if not event.room_id:
            return
        
        # Add user info
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            event.data.update({
                "user_id": connection.user_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Broadcast to room (low priority, no ack needed)
        event.priority = EventPriority.LOW
        event.requires_ack = False
        
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    async def handle_selection_update(self, connection_id: str, event: WebSocketEvent):
        """Handle text selection update."""
        await self.handle_cursor_move(connection_id, event)
    
    async def handle_viewport_change(self, connection_id: str, event: WebSocketEvent):
        """Handle viewport change."""
        await self.handle_cursor_move(connection_id, event)
    
    # Room management handlers
    
    async def handle_room_joined(self, connection_id: str, event: WebSocketEvent):
        """Handle room join."""
        room_id = event.room_id or event.data.get("room_id")
        if not room_id:
            await self._send_error(connection_id, "room_id required")
            return
        
        # Join room
        success = await self.connection_manager.join_room(connection_id, room_id)
        if success:
            # Get room users and send room state
            room_users = self.connection_manager.get_room_users(room_id)
            
            room_state_event = WebSocketEvent(
                event_type=EventType.ROOM_JOINED,
                room_id=room_id,
                data={
                    "room_id": room_id,
                    "users": room_users,
                    "joined_at": datetime.utcnow().isoformat()
                }
            )
            
            # Send to joining user
            await self.connection_manager.send_to_connection(
                connection_id, room_state_event.to_dict()
            )
            
            # Notify other room members
            connection = self.connection_manager.get_connection(connection_id)
            if connection:
                user_conn = self.connection_manager.get_user_connection(connection.user_id)
                if user_conn:
                    user_joined_event = WebSocketEvent(
                        event_type=EventType.USER_JOINED,
                        room_id=room_id,
                        data={
                            "user": {
                                "user_id": connection.user_id,
                                "username": user_conn.username,
                                "avatar_url": user_conn.avatar_url,
                                "status": user_conn.status.value
                            },
                            "joined_at": datetime.utcnow().isoformat()
                        }
                    )
                    
                    await self.connection_manager.send_to_room(
                        room_id, user_joined_event.to_dict(), exclude_connection=connection_id
                    )
        else:
            await self._send_error(connection_id, "Failed to join room")
    
    async def handle_room_left(self, connection_id: str, event: WebSocketEvent):
        """Handle room leave."""
        room_id = event.room_id or event.data.get("room_id")
        if not room_id:
            await self._send_error(connection_id, "room_id required")
            return
        
        # Get user info before leaving
        connection = self.connection_manager.get_connection(connection_id)
        user_info = None
        if connection:
            user_conn = self.connection_manager.get_user_connection(connection.user_id)
            if user_conn:
                user_info = {
                    "user_id": connection.user_id,
                    "username": user_conn.username
                }
        
        # Leave room
        success = await self.connection_manager.leave_room(connection_id, room_id)
        if success and user_info:
            # Notify other room members
            user_left_event = WebSocketEvent(
                event_type=EventType.USER_LEFT,
                room_id=room_id,
                data={
                    "user": user_info,
                    "left_at": datetime.utcnow().isoformat()
                }
            )
            
            await self.connection_manager.send_to_room(
                room_id, user_left_event.to_dict()
            )
    
    async def handle_room_updated(self, connection_id: str, event: WebSocketEvent):
        """Handle room update."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required")
            return
        
        # Broadcast room update
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    # File operation handlers
    
    async def handle_file_upload_start(self, connection_id: str, event: WebSocketEvent):
        """Handle file upload start."""
        file_id = event.data.get("file_id")
        if file_id:
            self.active_uploads[file_id] = {
                "connection_id": connection_id,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }
    
    async def handle_file_upload_progress(self, connection_id: str, event: WebSocketEvent):
        """Handle file upload progress."""
        file_id = event.data.get("file_id")
        progress = event.data.get("progress", 0.0)
        
        if file_id in self.active_uploads:
            self.active_uploads[file_id]["progress"] = progress
        
        # Send progress to user
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            await self.connection_manager.send_to_user(
                connection.user_id, event.to_dict()
            )
    
    async def handle_file_upload_complete(self, connection_id: str, event: WebSocketEvent):
        """Handle file upload completion."""
        file_id = event.data.get("file_id")
        if file_id in self.active_uploads:
            del self.active_uploads[file_id]
        
        # Broadcast completion if room specified
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict()
            )
        else:
            connection = self.connection_manager.get_connection(connection_id)
            if connection:
                await self.connection_manager.send_to_user(
                    connection.user_id, event.to_dict()
                )
    
    async def handle_file_upload_error(self, connection_id: str, event: WebSocketEvent):
        """Handle file upload error."""
        file_id = event.data.get("file_id")
        if file_id in self.active_uploads:
            del self.active_uploads[file_id]
        
        # Send error to user
        connection = self.connection_manager.get_connection(connection_id)
        if connection:
            await self.connection_manager.send_to_user(
                connection.user_id, event.to_dict()
            )
    
    # Video/Audio handlers
    
    async def handle_video_call_start(self, connection_id: str, event: WebSocketEvent):
        """Handle video call start."""
        if not event.room_id:
            await self._send_error(connection_id, "room_id required for video call")
            return
        
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    async def handle_video_call_end(self, connection_id: str, event: WebSocketEvent):
        """Handle video call end."""
        if not event.room_id:
            return
        
        await self.connection_manager.send_to_room(
            event.room_id, event.to_dict(), exclude_connection=connection_id
        )
    
    async def handle_audio_toggle(self, connection_id: str, event: WebSocketEvent):
        """Handle audio toggle."""
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
    
    async def handle_video_toggle(self, connection_id: str, event: WebSocketEvent):
        """Handle video toggle."""
        if event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
    
    # Custom event handler
    
    async def handle_custom_event(self, connection_id: str, event: WebSocketEvent):
        """Handle custom events."""
        # Validate custom event has required fields
        if "custom_type" not in event.data:
            await self._send_error(connection_id, "custom_type required for custom events")
            return
        
        # Route based on target
        if event.target_users:
            for user_id in event.target_users:
                await self.connection_manager.send_to_user(user_id, event.to_dict())
        elif event.target_rooms:
            for room_id in event.target_rooms:
                await self.connection_manager.send_to_room(
                    room_id, event.to_dict(), exclude_connection=connection_id
                )
        elif event.room_id:
            await self.connection_manager.send_to_room(
                event.room_id, event.to_dict(), exclude_connection=connection_id
            )
        else:
            # Send back to sender if no target specified
            await self.connection_manager.send_to_connection(
                connection_id, event.to_dict()
            )
    
    # Utility methods
    
    def get_active_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active analyses."""
        return self.active_analyses.copy()
    
    def get_active_uploads(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active file uploads."""
        return self.active_uploads.copy()
    
    async def cleanup_stale_operations(self, max_age_hours: int = 24):
        """Clean up stale operations."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Clean up stale analyses
        stale_analyses = [
            analysis_id for analysis_id, info in self.active_analyses.items()
            if info["started_at"] < cutoff_time
        ]
        for analysis_id in stale_analyses:
            del self.active_analyses[analysis_id]
        
        # Clean up stale uploads
        stale_uploads = [
            file_id for file_id, info in self.active_uploads.items()
            if info["started_at"] < cutoff_time
        ]
        for file_id in stale_uploads:
            del self.active_uploads[file_id]
        
        if stale_analyses or stale_uploads:
            logger.info(f"Cleaned up {len(stale_analyses)} stale analyses and {len(stale_uploads)} stale uploads")