"""
Shared cursor positions and selections for real-time collaboration.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json

import redis.asyncio as redis
from pydantic import BaseModel


class CursorEventType(Enum):
    MOVE = "move"
    SELECT = "select"
    CLICK = "click"
    SCROLL = "scroll"
    HIGHLIGHT = "highlight"


@dataclass
class CursorPosition:
    """Represents a cursor position in a document."""
    x: float
    y: float
    page: int = 0
    line: Optional[int] = None
    column: Optional[int] = None
    element_id: Optional[str] = None
    offset: Optional[int] = None


@dataclass
class TextSelection:
    """Represents a text selection range."""
    start_position: CursorPosition
    end_position: CursorPosition
    selected_text: Optional[str] = None
    context: Optional[str] = None  # Surrounding text context


@dataclass
class CursorState:
    """Complete cursor state for a user."""
    user_id: str
    username: str
    room_id: str
    position: Optional[CursorPosition] = None
    selection: Optional[TextSelection] = None
    color: str = "#4A90E2"  # Default blue color
    last_update: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'room_id': self.room_id,
            'position': asdict(self.position) if self.position else None,
            'selection': {
                'start_position': asdict(self.selection.start_position),
                'end_position': asdict(self.selection.end_position),
                'selected_text': self.selection.selected_text,
                'context': self.selection.context
            } if self.selection else None,
            'color': self.color,
            'last_update': self.last_update.isoformat(),
            'is_active': self.is_active,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CursorState':
        """Create from dictionary."""
        position = None
        if data.get('position'):
            position = CursorPosition(**data['position'])
        
        selection = None
        if data.get('selection'):
            sel_data = data['selection']
            selection = TextSelection(
                start_position=CursorPosition(**sel_data['start_position']),
                end_position=CursorPosition(**sel_data['end_position']),
                selected_text=sel_data.get('selected_text'),
                context=sel_data.get('context')
            )
        
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            room_id=data['room_id'],
            position=position,
            selection=selection,
            color=data.get('color', '#4A90E2'),
            last_update=datetime.fromisoformat(data['last_update']),
            is_active=data.get('is_active', True),
            metadata=data.get('metadata', {})
        )


@dataclass
class CursorEvent:
    """Cursor movement/selection event."""
    event_id: str
    event_type: CursorEventType
    cursor_state: CursorState
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'cursor_state': self.cursor_state.to_dict(),
            'timestamp': self.timestamp.isoformat()
        }


class CursorManager:
    """Manages shared cursor positions and selections."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.active_cursors: Dict[str, Dict[str, CursorState]] = {}  # room_id -> user_id -> cursor_state
        self.user_colors: Dict[str, str] = {}  # user_id -> color
        self.available_colors = [
            "#4A90E2",  # Blue
            "#7ED321",  # Green
            "#F5A623",  # Orange
            "#D0021B",  # Red
            "#9013FE",  # Purple
            "#50E3C2",  # Teal
            "#B8E986",  # Light Green
            "#4BD5EE",  # Light Blue
            "#F8E71C",  # Yellow
            "#BD10E0",  # Magenta
            "#B29900",  # Gold
            "#00C200",  # Forest Green
        ]
        self.color_index = 0
        self.cursor_timeout = timedelta(minutes=5)
        self.cleanup_task = None
        
    async def start(self):
        """Start the cursor manager."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("Cursor manager started")
    
    async def stop(self):
        """Stop the cursor manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        await self.redis.close()
        logging.info("Cursor manager stopped")
    
    def _assign_user_color(self, user_id: str) -> str:
        """Assign a unique color to a user."""
        if user_id not in self.user_colors:
            color = self.available_colors[self.color_index % len(self.available_colors)]
            self.user_colors[user_id] = color
            self.color_index += 1
            return color
        return self.user_colors[user_id]
    
    async def update_cursor_position(self, user_id: str, username: str, room_id: str,
                                   position: CursorPosition, metadata: Optional[Dict[str, Any]] = None) -> CursorState:
        """Update a user's cursor position."""
        color = self._assign_user_color(user_id)
        
        cursor_state = CursorState(
            user_id=user_id,
            username=username,
            room_id=room_id,
            position=position,
            color=color,
            last_update=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in active cursors
        if room_id not in self.active_cursors:
            self.active_cursors[room_id] = {}
        
        self.active_cursors[room_id][user_id] = cursor_state
        
        # Store in Redis for persistence
        await self._store_cursor_redis(cursor_state)
        
        return cursor_state
    
    async def update_text_selection(self, user_id: str, username: str, room_id: str,
                                  selection: TextSelection, metadata: Optional[Dict[str, Any]] = None) -> CursorState:
        """Update a user's text selection."""
        color = self._assign_user_color(user_id)
        
        # Get existing cursor state if available
        cursor_state = None
        if room_id in self.active_cursors and user_id in self.active_cursors[room_id]:
            cursor_state = self.active_cursors[room_id][user_id]
            cursor_state.selection = selection
            cursor_state.last_update = datetime.utcnow()
            if metadata:
                cursor_state.metadata.update(metadata)
        else:
            cursor_state = CursorState(
                user_id=user_id,
                username=username,
                room_id=room_id,
                selection=selection,
                color=color,
                last_update=datetime.utcnow(),
                metadata=metadata or {}
            )
        
        # Store in active cursors
        if room_id not in self.active_cursors:
            self.active_cursors[room_id] = {}
        
        self.active_cursors[room_id][user_id] = cursor_state
        
        # Store in Redis
        await self._store_cursor_redis(cursor_state)
        
        return cursor_state
    
    async def clear_selection(self, user_id: str, room_id: str) -> Optional[CursorState]:
        """Clear a user's text selection."""
        if room_id in self.active_cursors and user_id in self.active_cursors[room_id]:
            cursor_state = self.active_cursors[room_id][user_id]
            cursor_state.selection = None
            cursor_state.last_update = datetime.utcnow()
            
            # Update in Redis
            await self._store_cursor_redis(cursor_state)
            
            return cursor_state
        
        return None
    
    async def set_cursor_inactive(self, user_id: str, room_id: str) -> bool:
        """Mark a user's cursor as inactive."""
        if room_id in self.active_cursors and user_id in self.active_cursors[room_id]:
            cursor_state = self.active_cursors[room_id][user_id]
            cursor_state.is_active = False
            cursor_state.last_update = datetime.utcnow()
            
            # Update in Redis
            await self._store_cursor_redis(cursor_state)
            
            return True
        
        return False
    
    async def remove_cursor(self, user_id: str, room_id: str) -> bool:
        """Remove a user's cursor from a room."""
        removed = False
        
        if room_id in self.active_cursors and user_id in self.active_cursors[room_id]:
            del self.active_cursors[room_id][user_id]
            removed = True
            
            # Clean up empty rooms
            if not self.active_cursors[room_id]:
                del self.active_cursors[room_id]
        
        # Remove from Redis
        await self.redis.delete(f"cursor:{room_id}:{user_id}")
        
        return removed
    
    async def get_room_cursors(self, room_id: str, include_inactive: bool = False) -> List[CursorState]:
        """Get all cursor states for a room."""
        cursors = []
        
        # Check active cursors
        if room_id in self.active_cursors:
            for cursor_state in self.active_cursors[room_id].values():
                if include_inactive or cursor_state.is_active:
                    cursors.append(cursor_state)
        
        # Also check Redis for any missed cursors
        try:
            cursor_keys = await self.redis.keys(f"cursor:{room_id}:*")
            for key in cursor_keys:
                cursor_data = await self.redis.get(key)
                if cursor_data:
                    try:
                        data = json.loads(cursor_data)
                        cursor_state = CursorState.from_dict(data)
                        
                        # Add to active cursors if not already there
                        if cursor_state.user_id not in self.active_cursors.get(room_id, {}):
                            if room_id not in self.active_cursors:
                                self.active_cursors[room_id] = {}
                            self.active_cursors[room_id][cursor_state.user_id] = cursor_state
                        
                        if include_inactive or cursor_state.is_active:
                            if cursor_state not in cursors:
                                cursors.append(cursor_state)
                    except json.JSONDecodeError:
                        logging.warning(f"Invalid cursor data in Redis key: {key}")
        except Exception as e:
            logging.error(f"Error fetching cursors from Redis: {e}")
        
        return cursors
    
    async def get_user_cursor(self, user_id: str, room_id: str) -> Optional[CursorState]:
        """Get a specific user's cursor state."""
        # Check active cursors first
        if room_id in self.active_cursors and user_id in self.active_cursors[room_id]:
            return self.active_cursors[room_id][user_id]
        
        # Check Redis
        cursor_data = await self.redis.get(f"cursor:{room_id}:{user_id}")
        if cursor_data:
            try:
                data = json.loads(cursor_data)
                cursor_state = CursorState.from_dict(data)
                
                # Add to active cursors
                if room_id not in self.active_cursors:
                    self.active_cursors[room_id] = {}
                self.active_cursors[room_id][user_id] = cursor_state
                
                return cursor_state
            except json.JSONDecodeError:
                logging.warning(f"Invalid cursor data for user {user_id} in room {room_id}")
        
        return None
    
    async def get_cursors_at_position(self, room_id: str, position: CursorPosition, 
                                    tolerance: float = 5.0) -> List[CursorState]:
        """Get all cursors near a specific position."""
        cursors = await self.get_room_cursors(room_id, include_inactive=False)
        nearby_cursors = []
        
        for cursor_state in cursors:
            if cursor_state.position and cursor_state.position.page == position.page:
                distance = ((cursor_state.position.x - position.x) ** 2 + 
                           (cursor_state.position.y - position.y) ** 2) ** 0.5
                if distance <= tolerance:
                    nearby_cursors.append(cursor_state)
        
        return nearby_cursors
    
    async def get_overlapping_selections(self, room_id: str, selection: TextSelection) -> List[CursorState]:
        """Get cursors with selections that overlap with given selection."""
        cursors = await self.get_room_cursors(room_id, include_inactive=False)
        overlapping = []
        
        for cursor_state in cursors:
            if cursor_state.selection and self._selections_overlap(cursor_state.selection, selection):
                overlapping.append(cursor_state)
        
        return overlapping
    
    def _selections_overlap(self, sel1: TextSelection, sel2: TextSelection) -> bool:
        """Check if two text selections overlap."""
        # Simple overlap check - assumes both selections are on the same page
        if (sel1.start_position.page != sel2.start_position.page or
            sel1.end_position.page != sel2.end_position.page):
            return False
        
        # Check if selections overlap in text space (simplified)
        if sel1.start_position.line is not None and sel1.start_position.column is not None:
            # Line-based comparison
            start1 = sel1.start_position.line * 1000 + (sel1.start_position.column or 0)
            end1 = sel1.end_position.line * 1000 + (sel1.end_position.column or 0)
            start2 = sel2.start_position.line * 1000 + (sel2.start_position.column or 0)
            end2 = sel2.end_position.line * 1000 + (sel2.end_position.column or 0)
            
            return not (end1 < start2 or end2 < start1)
        else:
            # Coordinate-based comparison
            return not (sel1.end_position.y < sel2.start_position.y or 
                       sel2.end_position.y < sel1.start_position.y)
    
    async def create_cursor_event(self, cursor_state: CursorState, 
                                event_type: CursorEventType) -> CursorEvent:
        """Create a cursor event for broadcasting."""
        event = CursorEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            cursor_state=cursor_state,
            timestamp=datetime.utcnow()
        )
        
        # Store event in Redis for audit trail (optional)
        await self.redis.lpush(
            f"cursor_events:{cursor_state.room_id}",
            json.dumps(event.to_dict())
        )
        await self.redis.ltrim(f"cursor_events:{cursor_state.room_id}", 0, 999)  # Keep last 1000 events
        
        return event
    
    async def _store_cursor_redis(self, cursor_state: CursorState):
        """Store cursor state in Redis."""
        try:
            data = json.dumps(cursor_state.to_dict())
            await self.redis.setex(
                f"cursor:{cursor_state.room_id}:{cursor_state.user_id}",
                300,  # 5 minutes TTL
                data
            )
        except Exception as e:
            logging.error(f"Error storing cursor in Redis: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale cursor data."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_cursors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in cursor cleanup: {e}")
    
    async def _cleanup_stale_cursors(self):
        """Remove stale cursor data."""
        now = datetime.utcnow()
        stale_cursors = []
        
        for room_id, room_cursors in list(self.active_cursors.items()):
            for user_id, cursor_state in list(room_cursors.items()):
                time_since_update = now - cursor_state.last_update
                
                if time_since_update > self.cursor_timeout:
                    stale_cursors.append((room_id, user_id))
        
        # Remove stale cursors
        for room_id, user_id in stale_cursors:
            await self.remove_cursor(user_id, room_id)
            logging.debug(f"Removed stale cursor for user {user_id} in room {room_id}")
    
    def get_cursor_stats(self) -> Dict[str, Any]:
        """Get cursor management statistics."""
        total_cursors = 0
        active_cursors = 0
        rooms_with_cursors = len(self.active_cursors)
        
        for room_cursors in self.active_cursors.values():
            total_cursors += len(room_cursors)
            active_cursors += sum(1 for cursor in room_cursors.values() if cursor.is_active)
        
        return {
            'total_cursors': total_cursors,
            'active_cursors': active_cursors,
            'rooms_with_cursors': rooms_with_cursors,
            'assigned_colors': len(self.user_colors)
        }


# Global instance
cursor_manager = CursorManager()