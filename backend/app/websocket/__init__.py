"""
WebSocket module for real-time features.

This module provides comprehensive WebSocket functionality for:
- Real-time analysis progress tracking
- Live collaborative annotations
- User presence indicators
- Real-time notifications
- Document sharing and collaboration
"""

from .connection_manager import ConnectionManager
from .events import WebSocketEvent, EventType
from .handlers import WebSocketHandlers
from .rooms import RoomManager, Room
from .server import WebSocketServer

__all__ = [
    "ConnectionManager",
    "WebSocketEvent", 
    "EventType",
    "WebSocketHandlers",
    "RoomManager",
    "Room",
    "WebSocketServer"
]