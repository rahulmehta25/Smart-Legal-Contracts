"""
Event Management System for Real-time Updates
"""

import asyncio
from typing import Dict, Any, AsyncGenerator, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class EventType(Enum):
    """Event types for the system"""
    DOCUMENT_PROCESSING = "document_processing"
    ANALYSIS_PROGRESS = "analysis_progress"
    COMMENT_UPDATE = "comment_update"
    COLLABORATION_UPDATE = "collaboration_update"
    SYSTEM_UPDATE = "system_update"


@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    source: str = "system"


class EventManager:
    """Manage and distribute events for real-time updates"""
    
    def __init__(self):
        self.subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self.event_history: List[Event] = []
    
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        # Add to history
        self.event_history.append(event)
        
        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Send to subscribers
        channel_key = f"{event.event_type.value}"
        if channel_key in self.subscribers:
            for queue in self.subscribers[channel_key].copy():
                try:
                    await queue.put(event)
                except:
                    # Remove dead queue
                    self.subscribers[channel_key].discard(queue)
    
    async def subscribe(self, channel: str) -> AsyncGenerator[Event, None]:
        """Subscribe to events for a channel"""
        queue = asyncio.Queue()
        
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        
        self.subscribers[channel].add(queue)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            # Cleanup on disconnect
            if channel in self.subscribers:
                self.subscribers[channel].discard(queue)
    
    def get_event_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """Get recent event history"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]