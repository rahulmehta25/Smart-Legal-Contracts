"""
Main WebSocket Server for real-time features.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
import socketio
from starlette.middleware.cors import CORSMiddleware

from .connection_manager import ConnectionManager
from .events import WebSocketEvent, EventType, create_system_event, create_notification_event
from .handlers import WebSocketHandlers
from .rooms import RoomManager, RoomType, RoomSettings
from ..core.security import verify_token, get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)


class WebSocketServer:
    """Main WebSocket server managing real-time features."""
    
    def __init__(self, app: Optional[FastAPI] = None):
        # Core components
        self.connection_manager = ConnectionManager()
        self.room_manager = RoomManager()
        self.handlers = WebSocketHandlers(self.connection_manager, self.room_manager)
        
        # Socket.IO server for advanced features
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        # Setup Socket.IO handlers
        self._setup_socketio_handlers()
        
        # Background tasks
        self.analysis_broadcaster: Optional[asyncio.Task] = None
        self.notification_processor: Optional[asyncio.Task] = None
        self.system_monitor: Optional[asyncio.Task] = None
        
        # Server statistics
        self.stats = {
            "started_at": datetime.utcnow(),
            "total_connections": 0,
            "total_events_processed": 0,
            "total_rooms_created": 0,
            "uptime_seconds": 0
        }
        
        # Configuration
        self.config = {
            "max_connections_per_user": 10,
            "max_message_size": 10 * 1024,  # 10KB
            "rate_limit_messages": 100,  # per minute
            "heartbeat_interval": 30,
            "cleanup_interval": 300,
            "enable_namespaces": True,
            "enable_rooms": True,
            "enable_presence": True,
            "enable_analysis_streaming": True,
            "enable_notifications": True
        }
        
        if app:
            self.setup_routes(app)
    
    def _setup_socketio_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle Socket.IO connection."""
            try:
                # Extract token from auth
                token = auth.get('token') if auth else None
                if not token:
                    logger.warning(f"Socket.IO connection {sid} rejected: no token")
                    return False
                
                # Verify token
                user = await verify_token(token)
                if not user:
                    logger.warning(f"Socket.IO connection {sid} rejected: invalid token")
                    return False
                
                # Create WebSocket-like connection for Socket.IO
                connection_id = await self.connection_manager.connect(
                    websocket=sid,  # Use sid as websocket identifier
                    user_id=str(user.id),
                    username=user.username,
                    avatar_url=getattr(user, 'avatar_url', None),
                    metadata={"socket_io": True, "sid": sid}
                )
                
                # Store connection mapping
                await self.sio.save_session(sid, {
                    "connection_id": connection_id,
                    "user_id": str(user.id),
                    "username": user.username,
                    "avatar_url": getattr(user, 'avatar_url', None)
                })
                
                # Send welcome event
                welcome_event = WebSocketEvent(
                    event_type=EventType.CONNECT,
                    user_id=str(user.id),
                    data={
                        "connection_id": connection_id,
                        "features": ["socket_io", "namespaces", "rooms", "binary"],
                        "server_time": datetime.utcnow().isoformat()
                    }
                )
                
                await self.sio.emit('websocket_event', welcome_event.to_dict(), to=sid)
                
                logger.info(f"Socket.IO user {user.username} connected: {sid}")
                self.stats["total_connections"] += 1
                
            except Exception as e:
                logger.error(f"Socket.IO connection error: {e}")
                return False
        
        @self.sio.event
        async def disconnect(sid):
            """Handle Socket.IO disconnection."""
            try:
                session = await self.sio.get_session(sid)
                connection_id = session.get("connection_id")
                user_id = session.get("user_id")
                
                if connection_id:
                    await self.connection_manager.disconnect(connection_id, "socket_io_disconnect")
                
                logger.info(f"Socket.IO user {user_id} disconnected: {sid}")
                
            except Exception as e:
                logger.error(f"Socket.IO disconnect error: {e}")
        
        @self.sio.event
        async def websocket_event(sid, data):
            """Handle generic WebSocket events via Socket.IO."""
            try:
                session = await self.sio.get_session(sid)
                connection_id = session.get("connection_id")
                
                if not connection_id:
                    return {"error": "No connection found"}
                
                # Parse event
                event = WebSocketEvent.from_dict(data)
                
                # Handle event
                await self.handlers.handle_event(connection_id, event)
                
                self.stats["total_events_processed"] += 1
                return {"status": "processed"}
                
            except Exception as e:
                logger.error(f"Socket.IO event error: {e}")
                return {"error": str(e)}
        
        @self.sio.event
        async def join_room(sid, data):
            """Handle room joining via Socket.IO."""
            try:
                session = await self.sio.get_session(sid)
                connection_id = session.get("connection_id")
                user_id = session.get("user_id")
                username = session.get("username")
                
                if not all([connection_id, user_id, username]):
                    return {"error": "Session data incomplete"}
                
                room_id = data.get("room_id")
                if not room_id:
                    return {"error": "room_id required"}
                
                # Join room in room manager
                success, error, room = self.room_manager.join_room(
                    room_id=room_id,
                    user_id=user_id,
                    username=username,
                    connection_id=connection_id,
                    auto_create=data.get("auto_create", True)
                )
                
                if success and error != "Approval required":
                    # Join Socket.IO room
                    await self.sio.enter_room(sid, room_id)
                    
                    # Join connection manager room
                    await self.connection_manager.join_room(connection_id, room_id)
                    
                    # Send room state
                    room_state = room.to_dict(include_sensitive=True) if room else {}
                    
                    return {
                        "status": "joined",
                        "room_state": room_state
                    }
                else:
                    return {"error": error or "Failed to join room"}
                
            except Exception as e:
                logger.error(f"Socket.IO join_room error: {e}")
                return {"error": str(e)}
        
        @self.sio.event
        async def leave_room(sid, data):
            """Handle room leaving via Socket.IO."""
            try:
                session = await self.sio.get_session(sid)
                connection_id = session.get("connection_id")
                user_id = session.get("user_id")
                
                room_id = data.get("room_id")
                if not room_id:
                    return {"error": "room_id required"}
                
                # Leave Socket.IO room
                await self.sio.leave_room(sid, room_id)
                
                # Leave room in managers
                if connection_id:
                    await self.connection_manager.leave_room(connection_id, room_id)
                
                if user_id:
                    self.room_manager.leave_room(room_id, user_id, connection_id)
                
                return {"status": "left"}
                
            except Exception as e:
                logger.error(f"Socket.IO leave_room error: {e}")
                return {"error": str(e)}
    
    def setup_routes(self, app: FastAPI):
        """Setup WebSocket and HTTP routes."""
        
        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
            await self.handlers.handle_websocket_connection(websocket, token)
        
        # Socket.IO ASGI app mount
        socket_app = socketio.ASGIApp(self.sio, other_asgi_app=app)
        
        # HTTP endpoints for WebSocket management
        @app.get("/api/websocket/stats")
        async def get_websocket_stats(user: User = Depends(get_current_user)):
            """Get WebSocket server statistics."""
            return {
                "server": self.get_server_stats(),
                "connections": self.connection_manager.get_statistics(),
                "rooms": self.room_manager.get_statistics()
            }
        
        @app.get("/api/websocket/rooms")
        async def get_public_rooms(user: User = Depends(get_current_user)):
            """Get list of public rooms."""
            return {"rooms": self.room_manager.get_public_rooms()}
        
        @app.post("/api/websocket/rooms")
        async def create_room(
            room_data: Dict[str, Any],
            user: User = Depends(get_current_user)
        ):
            """Create a new room."""
            try:
                room = self.room_manager.create_room(
                    room_id=room_data.get("room_id"),
                    room_type=RoomType(room_data.get("room_type", "document")),
                    name=room_data.get("name"),
                    description=room_data.get("description"),
                    owner_id=str(user.id),
                    settings=RoomSettings(**room_data.get("settings", {})) if room_data.get("settings") else None,
                    metadata=room_data.get("metadata")
                )
                
                self.stats["total_rooms_created"] += 1
                return {"room": room.to_dict(include_sensitive=False)}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/api/websocket/rooms/{room_id}")
        async def get_room_info(room_id: str, user: User = Depends(get_current_user)):
            """Get room information."""
            room = self.room_manager.get_room(room_id)
            if not room:
                raise HTTPException(status_code=404, detail="Room not found")
            
            # Check if user has access to detailed info
            include_sensitive = room.has_permission(str(user.id), room_manager.RoomPermission.READ)
            
            return {"room": room.to_dict(include_sensitive=include_sensitive)}
        
        @app.post("/api/websocket/broadcast")
        async def broadcast_message(
            broadcast_data: Dict[str, Any],
            user: User = Depends(get_current_user)
        ):
            """Broadcast message to rooms or users (admin only)."""
            # This would need proper admin authentication
            if not getattr(user, 'is_admin', False):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            event = WebSocketEvent.from_dict(broadcast_data.get("event", {}))
            
            if broadcast_data.get("target_rooms"):
                for room_id in broadcast_data["target_rooms"]:
                    await self.connection_manager.send_to_room(room_id, event.to_dict())
            elif broadcast_data.get("target_users"):
                for user_id in broadcast_data["target_users"]:
                    await self.connection_manager.send_to_user(user_id, event.to_dict())
            else:
                # Broadcast to all
                await self.connection_manager.broadcast(event.to_dict())
            
            return {"status": "broadcasted"}
        
        @app.post("/api/websocket/analysis/{analysis_id}/progress")
        async def update_analysis_progress(
            analysis_id: str,
            progress_data: Dict[str, Any],
            user: User = Depends(get_current_user)
        ):
            """Update analysis progress for real-time streaming."""
            event = WebSocketEvent(
                event_type=EventType.ANALYSIS_PROGRESS,
                user_id=str(user.id),
                data={
                    "analysis_id": analysis_id,
                    "progress": progress_data.get("progress", 0.0),
                    "stage": progress_data.get("stage", "processing"),
                    "details": progress_data.get("details", {})
                }
            )
            
            # Send to user's connections
            await self.connection_manager.send_to_user(str(user.id), event.to_dict())
            
            # Send to any rooms user specified
            if progress_data.get("room_id"):
                await self.connection_manager.send_to_room(
                    progress_data["room_id"], event.to_dict()
                )
            
            return {"status": "progress_updated"}
        
        @app.post("/api/websocket/notifications")
        async def send_notification(
            notification_data: Dict[str, Any],
            user: User = Depends(get_current_user)
        ):
            """Send notification to user."""
            event = create_notification_event(
                user_id=str(user.id),
                title=notification_data["title"],
                message=notification_data["message"],
                notification_type=notification_data.get("type", "info"),
                action_url=notification_data.get("action_url"),
                expires_at=notification_data.get("expires_at")
            )
            
            await self.connection_manager.send_to_user(str(user.id), event.to_dict())
            
            return {"status": "notification_sent"}
    
    async def start_background_tasks(self):
        """Start background tasks for real-time features."""
        if self.config.get("enable_analysis_streaming"):
            self.analysis_broadcaster = asyncio.create_task(self._analysis_broadcaster())
        
        if self.config.get("enable_notifications"):
            self.notification_processor = asyncio.create_task(self._notification_processor())
        
        self.system_monitor = asyncio.create_task(self._system_monitor())
        
        logger.info("WebSocket server background tasks started")
    
    async def _analysis_broadcaster(self):
        """Broadcast analysis updates to connected clients."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Get active analyses from handlers
                active_analyses = self.handlers.get_active_analyses()
                
                # Broadcast progress for analyses that need updates
                for analysis_id, analysis_info in active_analyses.items():
                    user_id = analysis_info.get("user_id")
                    if user_id:
                        # Create progress event (this would be triggered by actual analysis progress)
                        pass
                        
            except Exception as e:
                logger.error(f"Analysis broadcaster error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _notification_processor(self):
        """Process and send pending notifications."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # This would integrate with a notification queue/database
                # For now, just a placeholder
                
            except Exception as e:
                logger.error(f"Notification processor error: {e}")
                await asyncio.sleep(10)
    
    async def _system_monitor(self):
        """Monitor system health and send alerts."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update uptime
                self.stats["uptime_seconds"] = (
                    datetime.utcnow() - self.stats["started_at"]
                ).total_seconds()
                
                # Check system health
                conn_stats = self.connection_manager.get_statistics()
                room_stats = self.room_manager.get_statistics()
                
                # Log statistics
                logger.info(
                    f"WebSocket Stats - "
                    f"Connections: {conn_stats['active_connections']}, "
                    f"Users: {conn_stats['active_users']}, "
                    f"Rooms: {room_stats['active_rooms']}, "
                    f"Uptime: {self.stats['uptime_seconds']:.0f}s"
                )
                
                # Send system health event to admin users if needed
                # This would need admin user identification
                
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(30)
    
    async def send_system_broadcast(
        self,
        message: str,
        severity: str = "info",
        target_users: Optional[List[str]] = None,
        maintenance_window: Optional[Dict[str, str]] = None
    ):
        """Send system-wide broadcast message."""
        event = create_system_event(
            event_type=EventType.SYSTEM_UPDATE,
            message=message,
            severity=severity,
            affected_users=target_users,
            maintenance_window=maintenance_window
        )
        
        if target_users:
            for user_id in target_users:
                await self.connection_manager.send_to_user(user_id, event.to_dict())
        else:
            await self.connection_manager.broadcast(event.to_dict())
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.utcnow() - self.stats["started_at"]).total_seconds(),
            "config": self.config.copy()
        }
    
    async def shutdown(self):
        """Shutdown WebSocket server gracefully."""
        logger.info("Shutting down WebSocket server...")
        
        # Cancel background tasks
        tasks = [self.analysis_broadcaster, self.notification_processor, self.system_monitor]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Send shutdown notification
        await self.send_system_broadcast(
            "Server is shutting down for maintenance",
            severity="warning"
        )
        
        # Wait a bit for messages to be sent
        await asyncio.sleep(2)
        
        # Shutdown components
        await self.connection_manager.shutdown()
        await self.room_manager.shutdown()
        
        logger.info("WebSocket server shutdown complete")


# Global WebSocket server instance
websocket_server = WebSocketServer()


@asynccontextmanager
async def websocket_lifespan(app: FastAPI):
    """Lifespan context manager for WebSocket server."""
    # Startup
    await websocket_server.start_background_tasks()
    logger.info("WebSocket server started")
    
    yield
    
    # Shutdown
    await websocket_server.shutdown()


def create_websocket_app() -> FastAPI:
    """Create FastAPI app with WebSocket support."""
    app = FastAPI(
        title="Arbitration Detection WebSocket API",
        description="Real-time WebSocket API for arbitration detection system",
        version="1.0.0",
        lifespan=websocket_lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup WebSocket server routes
    websocket_server.setup_routes(app)
    
    return app


# For Socket.IO ASGI app
def create_socketio_app() -> socketio.ASGIApp:
    """Create Socket.IO ASGI app."""
    return socketio.ASGIApp(websocket_server.sio)