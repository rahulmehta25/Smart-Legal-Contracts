"""
WebSocket Connection Manager for handling real-time connections.
"""

import json
import asyncio
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


class UserStatus(Enum):
    """User presence status."""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class Connection:
    """Represents a WebSocket connection."""
    connection_id: str
    user_id: str
    websocket: WebSocket
    status: ConnectionStatus
    connected_at: datetime
    last_ping: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    rooms: Set[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.rooms is None:
            self.rooms = set()
        if self.last_activity is None:
            self.last_activity = self.connected_at


@dataclass
class UserConnection:
    """Aggregated user connection information."""
    user_id: str
    username: str
    status: UserStatus
    connections: List[Connection]
    primary_connection_id: Optional[str] = None
    avatar_url: Optional[str] = None
    permissions: Set[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {"read", "write", "comment"}
        if self.primary_connection_id is None and self.connections:
            self.primary_connection_id = self.connections[0].connection_id


class ConnectionManager:
    """Manages WebSocket connections and user presence."""
    
    def __init__(self):
        # Core connection tracking
        self.connections: Dict[str, Connection] = {}  # connection_id -> Connection
        self.user_connections: Dict[str, UserConnection] = {}  # user_id -> UserConnection
        self.connection_to_user: Dict[str, str] = {}  # connection_id -> user_id
        
        # Room and namespace management
        self.room_connections: Dict[str, Set[str]] = {}  # room_id -> set of connection_ids
        self.namespace_connections: Dict[str, Set[str]] = {}  # namespace -> set of connection_ids
        
        # Heartbeat and cleanup
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "disconnections": 0,
            "reconnections": 0,
            "message_count": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
    
    async def _cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._check_connection_health()
                await self._cleanup_inactive_connections()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _check_connection_health(self):
        """Check health of all connections via ping."""
        current_time = datetime.utcnow()
        stale_connections = []
        
        for connection_id, connection in self.connections.items():
            if connection.status == ConnectionStatus.CONNECTED:
                # Check if we need to ping
                if (connection.last_ping is None or 
                    current_time - connection.last_ping > timedelta(seconds=self.heartbeat_interval)):
                    try:
                        await connection.websocket.ping()
                        connection.last_ping = current_time
                    except Exception:
                        stale_connections.append(connection_id)
        
        # Remove stale connections
        for connection_id in stale_connections:
            await self.disconnect(connection_id, reason="ping_failed")
    
    async def _cleanup_inactive_connections(self):
        """Remove connections that have been inactive too long."""
        current_time = datetime.utcnow()
        inactive_connections = []
        
        for connection_id, connection in self.connections.items():
            if (current_time - connection.last_activity > 
                timedelta(seconds=self.connection_timeout)):
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            await self.disconnect(connection_id, reason="timeout")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        username: str,
        avatar_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            user_id: User identifier
            username: User display name
            avatar_url: User avatar URL
            metadata: Additional connection metadata
            
        Returns:
            connection_id: Unique connection identifier
        """
        connection_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        
        # Create connection object
        connection = Connection(
            connection_id=connection_id,
            user_id=user_id,
            websocket=websocket,
            status=ConnectionStatus.CONNECTED,
            connected_at=current_time,
            last_activity=current_time,
            metadata=metadata or {}
        )
        
        # Store connection
        self.connections[connection_id] = connection
        self.connection_to_user[connection_id] = user_id
        
        # Update user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = UserConnection(
                user_id=user_id,
                username=username,
                status=UserStatus.ONLINE,
                connections=[],
                avatar_url=avatar_url
            )
        
        user_conn = self.user_connections[user_id]
        user_conn.connections.append(connection)
        user_conn.status = UserStatus.ONLINE
        
        # Set as primary if it's the first connection
        if not user_conn.primary_connection_id:
            user_conn.primary_connection_id = connection_id
        
        # Update statistics
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] = len(self.connections)
        
        logger.info(f"User {username} connected with connection {connection_id}")
        
        return connection_id
    
    async def disconnect(self, connection_id: str, reason: str = "client_disconnect"):
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection to disconnect
            reason: Reason for disconnection
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user_id
        
        # Update connection status
        connection.status = ConnectionStatus.DISCONNECTED
        
        # Leave all rooms
        for room_id in list(connection.rooms):
            await self.leave_room(connection_id, room_id)
        
        # Clean up connection tracking
        del self.connections[connection_id]
        del self.connection_to_user[connection_id]
        
        # Update user connections
        if user_id in self.user_connections:
            user_conn = self.user_connections[user_id]
            user_conn.connections = [
                conn for conn in user_conn.connections 
                if conn.connection_id != connection_id
            ]
            
            # Update primary connection
            if user_conn.primary_connection_id == connection_id:
                user_conn.primary_connection_id = (
                    user_conn.connections[0].connection_id 
                    if user_conn.connections else None
                )
            
            # Update user status
            if not user_conn.connections:
                user_conn.status = UserStatus.OFFLINE
                # Keep user connection info for a short time for reconnection
                asyncio.create_task(self._cleanup_user_after_delay(user_id))
        
        # Update statistics
        self.connection_stats["disconnections"] += 1
        self.connection_stats["active_connections"] = len(self.connections)
        
        logger.info(f"Connection {connection_id} disconnected: {reason}")
    
    async def _cleanup_user_after_delay(self, user_id: str, delay: int = 300):
        """Remove user connection info after delay if still offline."""
        await asyncio.sleep(delay)  # 5 minutes
        if (user_id in self.user_connections and 
            not self.user_connections[user_id].connections):
            del self.user_connections[user_id]
    
    async def join_room(self, connection_id: str, room_id: str):
        """Add connection to a room."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.rooms.add(room_id)
        
        if room_id not in self.room_connections:
            self.room_connections[room_id] = set()
        self.room_connections[room_id].add(connection_id)
        
        logger.debug(f"Connection {connection_id} joined room {room_id}")
        return True
    
    async def leave_room(self, connection_id: str, room_id: str):
        """Remove connection from a room."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.rooms.discard(room_id)
        
        if room_id in self.room_connections:
            self.room_connections[room_id].discard(connection_id)
            # Clean up empty rooms
            if not self.room_connections[room_id]:
                del self.room_connections[room_id]
        
        logger.debug(f"Connection {connection_id} left room {room_id}")
        return True
    
    async def send_to_connection(
        self, 
        connection_id: str, 
        message: Dict[str, Any]
    ) -> bool:
        """Send message to a specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        try:
            await connection.websocket.send_text(json.dumps(message))
            connection.last_activity = datetime.utcnow()
            self.connection_stats["message_count"] += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id, reason="send_error")
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Send message to all connections of a user."""
        if user_id not in self.user_connections:
            return 0
        
        user_conn = self.user_connections[user_id]
        sent_count = 0
        
        for connection in user_conn.connections:
            if await self.send_to_connection(connection.connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def send_to_room(
        self, 
        room_id: str, 
        message: Dict[str, Any],
        exclude_connection: Optional[str] = None
    ) -> int:
        """Send message to all connections in a room."""
        if room_id not in self.room_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.room_connections[room_id]:
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast(
        self, 
        message: Dict[str, Any],
        exclude_connection: Optional[str] = None
    ) -> int:
        """Broadcast message to all connections."""
        sent_count = 0
        for connection_id in self.connections:
            if connection_id != exclude_connection:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    def get_connection(self, connection_id: str) -> Optional[Connection]:
        """Get connection by ID."""
        return self.connections.get(connection_id)
    
    def get_user_connection(self, user_id: str) -> Optional[UserConnection]:
        """Get user connection info."""
        return self.user_connections.get(user_id)
    
    def get_room_connections(self, room_id: str) -> Set[str]:
        """Get all connection IDs in a room."""
        return self.room_connections.get(room_id, set()).copy()
    
    def get_room_users(self, room_id: str) -> List[Dict[str, Any]]:
        """Get user information for all users in a room."""
        if room_id not in self.room_connections:
            return []
        
        users = {}
        for connection_id in self.room_connections[room_id]:
            if connection_id in self.connection_to_user:
                user_id = self.connection_to_user[connection_id]
                if user_id in self.user_connections:
                    user_conn = self.user_connections[user_id]
                    users[user_id] = {
                        "user_id": user_id,
                        "username": user_conn.username,
                        "status": user_conn.status.value,
                        "avatar_url": user_conn.avatar_url,
                        "permissions": list(user_conn.permissions),
                        "connection_count": len(user_conn.connections)
                    }
        
        return list(users.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.connection_stats,
            "active_users": len(self.user_connections),
            "active_rooms": len(self.room_connections),
            "rooms_with_users": len([r for r in self.room_connections.values() if r])
        }
    
    async def update_user_status(self, user_id: str, status: UserStatus):
        """Update user's presence status."""
        if user_id in self.user_connections:
            self.user_connections[user_id].status = status
            
            # Notify all rooms the user is in
            user_rooms = set()
            for connection in self.user_connections[user_id].connections:
                user_rooms.update(connection.rooms)
            
            status_message = {
                "type": "user_status_update",
                "user_id": user_id,
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for room_id in user_rooms:
                await self.send_to_room(room_id, status_message)
    
    async def shutdown(self):
        """Shutdown connection manager and clean up resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection in list(self.connections.values()):
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        # Clear all data
        self.connections.clear()
        self.user_connections.clear()
        self.connection_to_user.clear()
        self.room_connections.clear()
        self.namespace_connections.clear()
        
        logger.info("ConnectionManager shutdown complete")