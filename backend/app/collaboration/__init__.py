"""
Real-time collaboration platform for arbitration detection system.

This module provides comprehensive collaboration features including:
- WebSocket-based real-time communication
- User presence tracking
- Shared cursor positions
- Collaborative annotations
- Threaded comments system
- CRDT-based conflict resolution
- Document co-editing with operational transformation
"""

from .websocket_server import CollaborationWebSocketServer
from .presence import PresenceManager
from .cursors import CursorManager
from .annotations import AnnotationManager
from .comments import CommentManager
from .conflict_resolution import CRDTManager
from .collaborative_editor import CollaborativeEditor

__all__ = [
    'CollaborationWebSocketServer',
    'PresenceManager',
    'CursorManager',
    'AnnotationManager',
    'CommentManager',
    'CRDTManager',
    'CollaborativeEditor'
]