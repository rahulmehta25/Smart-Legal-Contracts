"""
GraphQL Subscription Resolvers for Real-time Updates
"""

import strawberry
import asyncio
from typing import AsyncGenerator, Optional
from strawberry.types import Info

from ..types import (
    DocumentProcessingUpdate,
    AnalysisUpdate,
    CommentUpdate,
    CollaborationUpdate,
    SystemStats
)
from ..utils.auth import get_current_user, require_auth
from ...core.events import EventManager, EventType


@strawberry.type
class Subscription:
    """Root Subscription type for real-time updates"""
    
    @strawberry.subscription
    @require_auth
    async def document_processing(
        self,
        info: Info,
        document_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[DocumentProcessingUpdate, None]:
        """Subscribe to document processing updates"""
        try:
            current_user = await get_current_user(info)
            event_manager = info.context.get("event_manager", EventManager())
            
            # Create subscription channel
            channel = f"document_processing_{document_id}" if document_id else "document_processing_*"
            
            async for event in event_manager.subscribe(channel):
                if event.event_type == EventType.DOCUMENT_PROCESSING:
                    # Check if user has permission to see this document
                    if await self._can_access_document(current_user, event.data.get("document_id")):
                        yield DocumentProcessingUpdate(
                            document_id=event.data["document_id"],
                            status=event.data["status"],
                            progress=event.data.get("progress"),
                            message=event.data.get("message"),
                            error_message=event.data.get("error_message")
                        )
                        
        except Exception as e:
            # Log error and end subscription
            pass
    
    @strawberry.subscription
    @require_auth
    async def analysis_progress(
        self,
        info: Info,
        document_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[AnalysisUpdate, None]:
        """Subscribe to analysis progress updates"""
        try:
            current_user = await get_current_user(info)
            event_manager = info.context.get("event_manager", EventManager())
            
            channel = f"analysis_progress_{document_id}" if document_id else "analysis_progress_*"
            
            async for event in event_manager.subscribe(channel):
                if event.event_type == EventType.ANALYSIS_PROGRESS:
                    if await self._can_access_document(current_user, event.data.get("document_id")):
                        yield AnalysisUpdate(
                            analysis_id=event.data["analysis_id"],
                            document_id=event.data["document_id"],
                            status=event.data["status"],
                            progress=event.data.get("progress"),
                            results=event.data.get("results")
                        )
                        
        except Exception as e:
            pass
    
    @strawberry.subscription
    @require_auth
    async def document_comments(
        self,
        info: Info,
        document_id: strawberry.ID
    ) -> AsyncGenerator[CommentUpdate, None]:
        """Subscribe to comment updates on a specific document"""
        try:
            current_user = await get_current_user(info)
            
            # Check if user can access this document
            if not await self._can_access_document(current_user, document_id):
                return
            
            event_manager = info.context.get("event_manager", EventManager())
            channel = f"document_comments_{document_id}"
            
            async for event in event_manager.subscribe(channel):
                if event.event_type == EventType.COMMENT_UPDATE:
                    # For now, yield placeholder as Comment model doesn't exist
                    # In real implementation, this would convert event.data to CommentUpdate
                    pass
                    
        except Exception as e:
            pass
    
    @strawberry.subscription
    @require_auth
    async def document_collaboration(
        self,
        info: Info,
        document_id: strawberry.ID
    ) -> AsyncGenerator[CollaborationUpdate, None]:
        """Subscribe to collaboration updates on a specific document"""
        try:
            current_user = await get_current_user(info)
            
            # Check if user can access this document
            if not await self._can_access_document(current_user, document_id):
                return
            
            event_manager = info.context.get("event_manager", EventManager())
            channel = f"document_collaboration_{document_id}"
            
            async for event in event_manager.subscribe(channel):
                if event.event_type == EventType.COLLABORATION_UPDATE:
                    yield CollaborationUpdate(
                        document_id=event.data["document_id"],
                        user_id=event.data["user_id"],
                        action=event.data["action"],
                        data=str(event.data.get("data", ""))
                    )
                    
        except Exception as e:
            pass
    
    @strawberry.subscription
    @require_auth(role="ADMIN")
    async def system_stats(self, info: Info) -> AsyncGenerator[SystemStats, None]:
        """Subscribe to system statistics updates (admin only)"""
        try:
            current_user = await get_current_user(info)
            
            # Admin-only subscription
            if current_user.role != "ADMIN":
                return
            
            # Emit stats every 30 seconds
            while True:
                try:
                    session = info.context["session"]
                    loaders = info.context["loaders"]
                    
                    # Get current stats
                    doc_stats = await loaders["document_stats"].load("global")
                    detection_stats = await loaders["detection_stats"].load("global")
                    pattern_stats = await loaders["pattern_stats"].load("global")
                    
                    yield SystemStats(
                        documents=doc_stats,
                        detections=detection_stats,
                        patterns=pattern_stats,
                        uptime=self._get_uptime(),
                        version="1.0.0"
                    )
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    break
                    
        except Exception as e:
            pass
    
    async def _can_access_document(self, user, document_id: str) -> bool:
        """Check if user can access a specific document"""
        try:
            # For now, allow all authenticated users
            # In real implementation, check document ownership/permissions
            return user is not None
        except Exception:
            return False
    
    def _get_uptime(self) -> str:
        """Get server uptime string"""
        try:
            # Would calculate actual uptime from server start time
            return "Unknown"
        except Exception:
            return "Unknown"


# WebSocket connection management for subscriptions
class WebSocketManager:
    """Manage WebSocket connections for GraphQL subscriptions"""
    
    def __init__(self):
        self.connections = {}
        self.user_connections = {}
    
    async def connect(self, websocket, user_id: str):
        """Add new WebSocket connection"""
        await websocket.accept()
        connection_id = id(websocket)
        
        self.connections[connection_id] = {
            'websocket': websocket,
            'user_id': user_id,
            'subscriptions': set()
        }
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
    
    async def disconnect(self, websocket):
        """Remove WebSocket connection"""
        connection_id = id(websocket)
        
        if connection_id in self.connections:
            user_id = self.connections[connection_id]['user_id']
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove connection
            del self.connections[connection_id]
    
    async def add_subscription(self, websocket, subscription_id: str):
        """Add subscription to connection"""
        connection_id = id(websocket)
        if connection_id in self.connections:
            self.connections[connection_id]['subscriptions'].add(subscription_id)
    
    async def remove_subscription(self, websocket, subscription_id: str):
        """Remove subscription from connection"""
        connection_id = id(websocket)
        if connection_id in self.connections:
            self.connections[connection_id]['subscriptions'].discard(subscription_id)
    
    async def broadcast_to_document(self, document_id: str, message: dict):
        """Broadcast message to all connections subscribed to a document"""
        for connection in self.connections.values():
            if f"document_{document_id}" in connection['subscriptions']:
                try:
                    await connection['websocket'].send_json(message)
                except Exception:
                    # Connection is dead, will be cleaned up later
                    pass
    
    async def broadcast_to_user(self, user_id: str, message: dict):
        """Broadcast message to all connections for a specific user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                if connection_id in self.connections:
                    try:
                        await self.connections[connection_id]['websocket'].send_json(message)
                    except Exception:
                        # Connection is dead, will be cleaned up later
                        pass
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connections"""
        for connection in self.connections.values():
            try:
                await connection['websocket'].send_json(message)
            except Exception:
                # Connection is dead, will be cleaned up later
                pass
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.connections)
    
    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user"""
        return len(self.user_connections.get(user_id, set()))


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Subscription event handlers
async def handle_document_processing_event(document_id: str, status: str, progress: float = None, message: str = None, error_message: str = None):
    """Handle document processing events"""
    event_data = {
        "type": "document_processing",
        "data": {
            "document_id": document_id,
            "status": status,
            "progress": progress,
            "message": message,
            "error_message": error_message
        }
    }
    
    await websocket_manager.broadcast_to_document(document_id, event_data)


async def handle_analysis_progress_event(analysis_id: str, document_id: str, status: str, progress: float = None, results=None):
    """Handle analysis progress events"""
    event_data = {
        "type": "analysis_progress", 
        "data": {
            "analysis_id": analysis_id,
            "document_id": document_id,
            "status": status,
            "progress": progress,
            "results": results
        }
    }
    
    await websocket_manager.broadcast_to_document(document_id, event_data)


async def handle_comment_event(document_id: str, comment, action: str):
    """Handle comment events"""
    event_data = {
        "type": "comment_update",
        "data": {
            "document_id": document_id,
            "comment": comment,
            "action": action
        }
    }
    
    await websocket_manager.broadcast_to_document(document_id, event_data)


async def handle_collaboration_event(document_id: str, user_id: str, action: str, data=None):
    """Handle collaboration events"""
    event_data = {
        "type": "collaboration_update",
        "data": {
            "document_id": document_id,
            "user_id": user_id,
            "action": action,
            "data": data
        }
    }
    
    await websocket_manager.broadcast_to_document(document_id, event_data)