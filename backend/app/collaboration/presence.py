"""
User presence tracking system for real-time collaboration.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

import redis.asyncio as redis
from pydantic import BaseModel


class PresenceStatus(Enum):
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    IDLE = "idle"
    OFFLINE = "offline"


@dataclass
class UserPresence:
    """User presence information."""
    user_id: str
    username: str
    status: PresenceStatus
    last_seen: datetime
    current_document: Optional[str] = None
    active_rooms: Set[str] = None
    avatar_url: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.active_rooms is None:
            self.active_rooms = set()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'status': self.status.value,
            'last_seen': self.last_seen.isoformat(),
            'current_document': self.current_document,
            'active_rooms': list(self.active_rooms),
            'avatar_url': self.avatar_url,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPresence':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            status=PresenceStatus(data['status']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            current_document=data.get('current_document'),
            active_rooms=set(data.get('active_rooms', [])),
            avatar_url=data.get('avatar_url'),
            metadata=data.get('metadata', {})
        )


class PresenceManager:
    """Manages user presence across the collaboration platform."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.local_presence: Dict[str, UserPresence] = {}
        self.room_members: Dict[str, Set[str]] = {}  # room_id -> set of user_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.idle_threshold = timedelta(minutes=5)
        self.offline_threshold = timedelta(minutes=15)
        self.cleanup_task = None
        
    async def start(self):
        """Start the presence manager."""
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("Presence manager started")
    
    async def stop(self):
        """Stop the presence manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        await self.redis.close()
        logging.info("Presence manager stopped")
    
    async def set_user_online(self, user_id: str, username: str, 
                             connection_id: str, avatar_url: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> UserPresence:
        """Set user as online."""
        now = datetime.utcnow()
        
        # Get existing presence or create new
        presence = await self.get_user_presence(user_id)
        if presence:
            presence.status = PresenceStatus.ONLINE
            presence.last_seen = now
            if avatar_url:
                presence.avatar_url = avatar_url
            if metadata:
                presence.metadata.update(metadata)
        else:
            presence = UserPresence(
                user_id=user_id,
                username=username,
                status=PresenceStatus.ONLINE,
                last_seen=now,
                avatar_url=avatar_url,
                metadata=metadata or {}
            )
        
        # Track connection
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Store locally and in Redis
        self.local_presence[user_id] = presence
        await self._store_presence_redis(presence)
        
        logging.info(f"User {username} ({user_id}) is now online")
        return presence
    
    async def set_user_offline(self, user_id: str, connection_id: Optional[str] = None) -> Optional[UserPresence]:
        """Set user as offline."""
        if user_id not in self.local_presence:
            return None
        
        # Remove connection
        if connection_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            
            # Don't set offline if user has other active connections
            if self.user_connections[user_id]:
                return self.local_presence[user_id]
        
        presence = self.local_presence[user_id]
        presence.status = PresenceStatus.OFFLINE
        presence.last_seen = datetime.utcnow()
        presence.active_rooms.clear()
        presence.current_document = None
        
        # Clean up connections for this user
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        
        # Remove from all rooms
        for room_id in list(presence.active_rooms):
            await self.leave_room(user_id, room_id)
        
        # Update storage
        await self._store_presence_redis(presence)
        
        logging.info(f"User {presence.username} ({user_id}) is now offline")
        return presence
    
    async def update_user_status(self, user_id: str, status: PresenceStatus,
                                metadata: Optional[Dict[str, Any]] = None) -> Optional[UserPresence]:
        """Update user's presence status."""
        if user_id not in self.local_presence:
            return None
        
        presence = self.local_presence[user_id]
        presence.status = status
        presence.last_seen = datetime.utcnow()
        
        if metadata:
            presence.metadata.update(metadata)
        
        await self._store_presence_redis(presence)
        
        logging.debug(f"User {presence.username} status updated to {status.value}")
        return presence
    
    async def update_user_activity(self, user_id: str, current_document: Optional[str] = None) -> Optional[UserPresence]:
        """Update user's activity timestamp and current document."""
        if user_id not in self.local_presence:
            return None
        
        presence = self.local_presence[user_id]
        presence.last_seen = datetime.utcnow()
        
        # Update status from idle/away to online if user is active
        if presence.status in [PresenceStatus.IDLE, PresenceStatus.AWAY]:
            presence.status = PresenceStatus.ONLINE
        
        if current_document is not None:
            presence.current_document = current_document
        
        await self._store_presence_redis(presence)
        return presence
    
    async def join_room(self, user_id: str, room_id: str) -> bool:
        """Add user to a room."""
        if user_id not in self.local_presence:
            return False
        
        presence = self.local_presence[user_id]
        presence.active_rooms.add(room_id)
        
        # Track room membership
        if room_id not in self.room_members:
            self.room_members[room_id] = set()
        self.room_members[room_id].add(user_id)
        
        await self._store_presence_redis(presence)
        
        logging.debug(f"User {presence.username} joined room {room_id}")
        return True
    
    async def leave_room(self, user_id: str, room_id: str) -> bool:
        """Remove user from a room."""
        if user_id not in self.local_presence:
            return False
        
        presence = self.local_presence[user_id]
        presence.active_rooms.discard(room_id)
        
        # Update room membership
        if room_id in self.room_members:
            self.room_members[room_id].discard(user_id)
            if not self.room_members[room_id]:
                del self.room_members[room_id]
        
        await self._store_presence_redis(presence)
        
        logging.debug(f"User {presence.username} left room {room_id}")
        return True
    
    async def get_user_presence(self, user_id: str) -> Optional[UserPresence]:
        """Get user's current presence."""
        # Check local cache first
        if user_id in self.local_presence:
            return self.local_presence[user_id]
        
        # Check Redis
        presence_data = await self.redis.hget(f"presence:{user_id}", "data")
        if presence_data:
            try:
                import json
                data = json.loads(presence_data)
                presence = UserPresence.from_dict(data)
                self.local_presence[user_id] = presence
                return presence
            except Exception as e:
                logging.error(f"Error loading presence from Redis: {e}")
        
        return None
    
    async def get_room_members(self, room_id: str) -> List[UserPresence]:
        """Get all members currently in a room."""
        if room_id not in self.room_members:
            return []
        
        members = []
        for user_id in self.room_members[room_id]:
            presence = await self.get_user_presence(user_id)
            if presence and presence.status != PresenceStatus.OFFLINE:
                members.append(presence)
        
        return members
    
    async def get_online_users(self) -> List[UserPresence]:
        """Get all currently online users."""
        online_users = []
        
        # Check local presence
        for presence in self.local_presence.values():
            if presence.status != PresenceStatus.OFFLINE:
                online_users.append(presence)
        
        return online_users
    
    async def get_users_in_document(self, document_id: str) -> List[UserPresence]:
        """Get all users currently viewing a specific document."""
        users = []
        
        for presence in self.local_presence.values():
            if (presence.current_document == document_id and 
                presence.status != PresenceStatus.OFFLINE):
                users.append(presence)
        
        return users
    
    async def broadcast_presence_update(self, user_id: str, 
                                      callback: Optional[callable] = None) -> bool:
        """Broadcast presence update to relevant rooms."""
        presence = await self.get_user_presence(user_id)
        if not presence:
            return False
        
        # Notify all rooms the user is in
        for room_id in presence.active_rooms:
            if callback:
                await callback(room_id, presence.to_dict())
        
        return True
    
    async def _store_presence_redis(self, presence: UserPresence):
        """Store presence data in Redis."""
        try:
            import json
            data = json.dumps(presence.to_dict())
            await self.redis.hset(f"presence:{presence.user_id}", "data", data)
            await self.redis.expire(f"presence:{presence.user_id}", 3600)  # 1 hour TTL
        except Exception as e:
            logging.error(f"Error storing presence in Redis: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale presence data."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_presence()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in presence cleanup: {e}")
    
    async def _cleanup_stale_presence(self):
        """Clean up stale presence data."""
        now = datetime.utcnow()
        stale_users = []
        
        for user_id, presence in list(self.local_presence.items()):
            time_since_activity = now - presence.last_seen
            
            # Mark as idle if inactive for a while
            if (time_since_activity > self.idle_threshold and 
                presence.status == PresenceStatus.ONLINE):
                presence.status = PresenceStatus.IDLE
                await self._store_presence_redis(presence)
                logging.debug(f"User {presence.username} marked as idle")
            
            # Mark as offline if inactive too long
            elif time_since_activity > self.offline_threshold:
                stale_users.append(user_id)
        
        # Clean up offline users
        for user_id in stale_users:
            await self.set_user_offline(user_id)
    
    def get_presence_stats(self) -> Dict[str, Any]:
        """Get presence statistics."""
        online_count = 0
        away_count = 0
        busy_count = 0
        idle_count = 0
        
        for presence in self.local_presence.values():
            if presence.status == PresenceStatus.ONLINE:
                online_count += 1
            elif presence.status == PresenceStatus.AWAY:
                away_count += 1
            elif presence.status == PresenceStatus.BUSY:
                busy_count += 1
            elif presence.status == PresenceStatus.IDLE:
                idle_count += 1
        
        return {
            'total_users': len(self.local_presence),
            'online': online_count,
            'away': away_count,
            'busy': busy_count,
            'idle': idle_count,
            'active_rooms': len(self.room_members),
            'total_connections': sum(len(conns) for conns in self.user_connections.values())
        }


# Global instance
presence_manager = PresenceManager()