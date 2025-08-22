"""
WebRTC server for peer-to-peer video/audio communication.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

import socketio
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import redis.asyncio as redis


class MediaType(Enum):
    AUDIO = "audio"
    VIDEO = "video"
    SCREEN = "screen"
    DATA = "data"


class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class MediaTrack:
    """Media track information."""
    track_id: str
    user_id: str
    media_type: MediaType
    enabled: bool = True
    muted: bool = False
    quality: str = "high"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebRTCPeer:
    """WebRTC peer connection information."""
    peer_id: str
    user_id: str
    username: str
    room_id: str
    connection: RTCPeerConnection
    state: ConnectionState
    media_tracks: List[MediaTrack]
    ice_candidates: List[Dict[str, Any]]
    created_at: datetime
    last_activity: datetime
    is_presenter: bool = False
    bandwidth_limit: Optional[int] = None  # kbps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'peer_id': self.peer_id,
            'user_id': self.user_id,
            'username': self.username,
            'room_id': self.room_id,
            'state': self.state.value,
            'media_tracks': [track.to_dict() for track in self.media_tracks],
            'ice_candidates': self.ice_candidates,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_presenter': self.is_presenter,
            'bandwidth_limit': self.bandwidth_limit
        }


@dataclass
class WebRTCRoom:
    """WebRTC room for managing multiple peer connections."""
    room_id: str
    meeting_id: str
    max_participants: int
    peers: Dict[str, WebRTCPeer]
    media_server_config: Dict[str, Any]
    recording_enabled: bool = False
    screen_sharing_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_active_peers(self) -> List[WebRTCPeer]:
        """Get active peer connections."""
        return [peer for peer in self.peers.values() 
                if peer.state == ConnectionState.CONNECTED]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'room_id': self.room_id,
            'meeting_id': self.meeting_id,
            'max_participants': self.max_participants,
            'peers': {pid: peer.to_dict() for pid, peer in self.peers.items()},
            'media_server_config': self.media_server_config,
            'recording_enabled': self.recording_enabled,
            'screen_sharing_enabled': self.screen_sharing_enabled,
            'created_at': self.created_at.isoformat(),
            'active_peer_count': len(self.get_active_peers())
        }


class WebRTCServer:
    """WebRTC server for managing peer-to-peer connections."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.rooms: Dict[str, WebRTCRoom] = {}
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.ice_servers = [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            # Add TURN servers for production
            # {"urls": "turn:your-turn-server.com", "username": "user", "credential": "pass"}
        ]
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins="*",
            logger=True
        )
        self.cleanup_task = None
        self.setup_handlers()
        
    async def start(self):
        """Start the WebRTC server."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logging.info("WebRTC server started")
    
    async def stop(self):
        """Stop the WebRTC server."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all peer connections
        for pc in self.peer_connections.values():
            await pc.close()
        
        await self.redis.close()
        logging.info("WebRTC server stopped")
    
    def setup_handlers(self):
        """Setup WebRTC event handlers."""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection."""
            try:
                token = auth.get('token') if auth else None
                if not token:
                    return False
                
                # Verify token and get user info
                # user = await verify_token(token)
                # For now, we'll use dummy user data
                user_id = auth.get('user_id', f'user_{sid}')
                username = auth.get('username', f'User {sid[:8]}')
                
                await self.sio.save_session(sid, {
                    'user_id': user_id,
                    'username': username
                })
                
                logging.info(f"WebRTC client connected: {username} ({sid})")
                return True
                
            except Exception as e:
                logging.error(f"WebRTC connection failed: {e}")
                return False
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            try:
                session = await self.sio.get_session(sid)
                user_id = session.get('user_id')
                
                # Clean up peer connections
                await self._cleanup_peer_connections(user_id)
                
                logging.info(f"WebRTC client disconnected: {user_id}")
                
            except Exception as e:
                logging.error(f"WebRTC disconnect error: {e}")
        
        @self.sio.event
        async def join_room(sid, data):
            """Handle joining a WebRTC room."""
            return await self.handle_join_room(sid, data)
        
        @self.sio.event
        async def leave_room(sid, data):
            """Handle leaving a WebRTC room."""
            return await self.handle_leave_room(sid, data)
        
        @self.sio.event
        async def offer(sid, data):
            """Handle WebRTC offer."""
            return await self.handle_offer(sid, data)
        
        @self.sio.event
        async def answer(sid, data):
            """Handle WebRTC answer."""
            return await self.handle_answer(sid, data)
        
        @self.sio.event
        async def ice_candidate(sid, data):
            """Handle ICE candidate."""
            return await self.handle_ice_candidate(sid, data)
        
        @self.sio.event
        async def toggle_media(sid, data):
            """Handle media toggle (mute/unmute)."""
            return await self.handle_toggle_media(sid, data)
        
        @self.sio.event
        async def screen_share(sid, data):
            """Handle screen sharing."""
            return await self.handle_screen_share(sid, data)
    
    async def create_room(self, room_id: str, meeting_id: str, max_participants: int = 50,
                         recording_enabled: bool = False) -> WebRTCRoom:
        """Create a new WebRTC room."""
        if room_id in self.rooms:
            return self.rooms[room_id]
        
        room = WebRTCRoom(
            room_id=room_id,
            meeting_id=meeting_id,
            max_participants=max_participants,
            peers={},
            media_server_config={
                'ice_servers': self.ice_servers,
                'video_codecs': ['VP8', 'VP9', 'H264'],
                'audio_codecs': ['OPUS', 'PCMU']
            },
            recording_enabled=recording_enabled
        )
        
        self.rooms[room_id] = room
        
        # Store in Redis for persistence
        await self._store_room_info(room)
        
        logging.info(f"Created WebRTC room: {room_id}")
        return room
    
    async def handle_join_room(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user joining a WebRTC room."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            username = session['username']
            
            room_id = data['room_id']
            meeting_id = data.get('meeting_id', room_id)
            
            # Get or create room
            if room_id not in self.rooms:
                room = await self.create_room(room_id, meeting_id)
            else:
                room = self.rooms[room_id]
            
            # Check room capacity
            if len(room.get_active_peers()) >= room.max_participants:
                return {'success': False, 'error': 'Room is full'}
            
            # Create peer connection
            peer_id = f"{user_id}_{sid}"
            pc = RTCPeerConnection(configuration={
                "iceServers": self.ice_servers
            })
            
            # Store peer connection
            self.peer_connections[peer_id] = pc
            
            # Create peer object
            peer = WebRTCPeer(
                peer_id=peer_id,
                user_id=user_id,
                username=username,
                room_id=room_id,
                connection=pc,
                state=ConnectionState.CONNECTING,
                media_tracks=[],
                ice_candidates=[],
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow()
            )
            
            room.peers[peer_id] = peer
            
            # Join socket.io room
            await self.sio.enter_room(sid, room_id)
            
            # Setup peer connection event handlers
            await self._setup_peer_connection_handlers(peer, sid)
            
            # Notify other peers
            await self.sio.emit('peer_joined', {
                'peer': peer.to_dict(),
                'room_stats': {
                    'active_peers': len(room.get_active_peers()),
                    'max_participants': room.max_participants
                }
            }, room=room_id, skip_sid=sid)
            
            # Send existing peers to new peer
            existing_peers = [p.to_dict() for p in room.get_active_peers() if p.peer_id != peer_id]
            
            return {
                'success': True,
                'room': room.to_dict(),
                'your_peer_id': peer_id,
                'existing_peers': existing_peers,
                'ice_servers': self.ice_servers
            }
            
        except Exception as e:
            logging.error(f"Error joining WebRTC room: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_leave_room(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user leaving a WebRTC room."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room_id = data['room_id']
            
            # Find and remove peer
            peer_id = None
            if room_id in self.rooms:
                room = self.rooms[room_id]
                for pid, peer in list(room.peers.items()):
                    if peer.user_id == user_id:
                        peer_id = pid
                        # Close peer connection
                        if pid in self.peer_connections:
                            await self.peer_connections[pid].close()
                            del self.peer_connections[pid]
                        
                        # Remove from room
                        del room.peers[pid]
                        break
            
            # Leave socket.io room
            await self.sio.leave_room(sid, room_id)
            
            # Notify other peers
            if peer_id:
                await self.sio.emit('peer_left', {
                    'peer_id': peer_id,
                    'user_id': user_id
                }, room=room_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error leaving WebRTC room: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_offer(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebRTC offer."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            
            offer = data['offer']
            target_peer_id = data['target_peer_id']
            source_peer_id = data['source_peer_id']
            
            # Forward offer to target peer
            await self.sio.emit('offer', {
                'offer': offer,
                'source_peer_id': source_peer_id
            }, to=target_peer_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling offer: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_answer(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebRTC answer."""
        try:
            answer = data['answer']
            target_peer_id = data['target_peer_id']
            source_peer_id = data['source_peer_id']
            
            # Forward answer to target peer
            await self.sio.emit('answer', {
                'answer': answer,
                'source_peer_id': source_peer_id
            }, to=target_peer_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling answer: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_ice_candidate(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ICE candidate."""
        try:
            candidate = data['candidate']
            target_peer_id = data['target_peer_id']
            source_peer_id = data['source_peer_id']
            
            # Forward ICE candidate to target peer
            await self.sio.emit('ice_candidate', {
                'candidate': candidate,
                'source_peer_id': source_peer_id
            }, to=target_peer_id)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling ICE candidate: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_toggle_media(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle media toggle (mute/unmute)."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            
            media_type = data['media_type']  # 'audio' or 'video'
            enabled = data['enabled']
            room_id = data['room_id']
            
            # Find peer and update media state
            if room_id in self.rooms:
                room = self.rooms[room_id]
                for peer in room.peers.values():
                    if peer.user_id == user_id:
                        # Update media track state
                        for track in peer.media_tracks:
                            if track.media_type.value == media_type:
                                track.enabled = enabled
                                track.muted = not enabled
                        
                        peer.last_activity = datetime.utcnow()
                        break
            
            # Broadcast media state change
            await self.sio.emit('media_state_changed', {
                'user_id': user_id,
                'media_type': media_type,
                'enabled': enabled
            }, room=room_id, skip_sid=sid)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling media toggle: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_screen_share(self, sid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle screen sharing."""
        try:
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            username = session['username']
            
            room_id = data['room_id']
            is_sharing = data['is_sharing']
            
            # Update room state
            if room_id in self.rooms:
                room = self.rooms[room_id]
                
                if is_sharing:
                    # Set as presenter
                    for peer in room.peers.values():
                        peer.is_presenter = (peer.user_id == user_id)
                        
                        # Add screen track for presenter
                        if peer.user_id == user_id:
                            screen_track = MediaTrack(
                                track_id=f"screen_{user_id}",
                                user_id=user_id,
                                media_type=MediaType.SCREEN,
                                enabled=True
                            )
                            peer.media_tracks.append(screen_track)
                else:
                    # Remove presenter status and screen tracks
                    for peer in room.peers.values():
                        if peer.user_id == user_id:
                            peer.is_presenter = False
                            peer.media_tracks = [
                                track for track in peer.media_tracks
                                if track.media_type != MediaType.SCREEN
                            ]
            
            # Broadcast screen share state
            await self.sio.emit('screen_share_changed', {
                'user_id': user_id,
                'username': username,
                'is_sharing': is_sharing
            }, room=room_id, skip_sid=sid)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Error handling screen share: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _setup_peer_connection_handlers(self, peer: WebRTCPeer, sid: str):
        """Setup event handlers for a peer connection."""
        pc = peer.connection
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "connected":
                peer.state = ConnectionState.CONNECTED
                peer.last_activity = datetime.utcnow()
                logging.info(f"Peer {peer.peer_id} connected")
            elif pc.connectionState == "failed":
                peer.state = ConnectionState.FAILED
                logging.warning(f"Peer {peer.peer_id} connection failed")
            elif pc.connectionState == "closed":
                peer.state = ConnectionState.CLOSED
                logging.info(f"Peer {peer.peer_id} connection closed")
        
        @pc.on("track")
        def on_track(track):
            logging.info(f"Received track {track.kind} from peer {peer.peer_id}")
            
            # Add track to peer's media tracks
            media_track = MediaTrack(
                track_id=str(uuid.uuid4()),
                user_id=peer.user_id,
                media_type=MediaType.AUDIO if track.kind == "audio" else MediaType.VIDEO,
                enabled=True
            )
            peer.media_tracks.append(media_track)
        
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                # Store ICE candidate
                candidate_data = {
                    'candidate': candidate.candidate,
                    'sdpMid': candidate.sdpMid,
                    'sdpMLineIndex': candidate.sdpMLineIndex
                }
                peer.ice_candidates.append(candidate_data)
                
                # Send to client
                await self.sio.emit('ice_candidate', {
                    'candidate': candidate_data,
                    'peer_id': peer.peer_id
                }, to=sid)
    
    async def _cleanup_peer_connections(self, user_id: str):
        """Clean up peer connections for a user."""
        peers_to_remove = []
        
        for room in self.rooms.values():
            for peer_id, peer in list(room.peers.items()):
                if peer.user_id == user_id:
                    peers_to_remove.append((room.room_id, peer_id))
        
        for room_id, peer_id in peers_to_remove:
            if room_id in self.rooms and peer_id in self.rooms[room_id].peers:
                # Close connection
                if peer_id in self.peer_connections:
                    await self.peer_connections[peer_id].close()
                    del self.peer_connections[peer_id]
                
                # Remove from room
                del self.rooms[room_id].peers[peer_id]
                
                # Notify other peers
                await self.sio.emit('peer_left', {
                    'peer_id': peer_id,
                    'user_id': user_id
                }, room=room_id)
    
    async def _store_room_info(self, room: WebRTCRoom):
        """Store room information in Redis."""
        try:
            data = json.dumps(room.to_dict())
            await self.redis.set(f"webrtc_room:{room.room_id}", data)
            await self.redis.expire(f"webrtc_room:{room.room_id}", 86400)  # 24 hours
        except Exception as e:
            logging.error(f"Error storing WebRTC room info: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in WebRTC cleanup: {e}")
    
    async def _cleanup_stale_connections(self):
        """Clean up stale peer connections."""
        stale_threshold = datetime.utcnow() - timedelta(minutes=10)
        stale_peers = []
        
        for room_id, room in list(self.rooms.items()):
            for peer_id, peer in list(room.peers.items()):
                if (peer.last_activity < stale_threshold or 
                    peer.state in [ConnectionState.FAILED, ConnectionState.CLOSED]):
                    stale_peers.append((room_id, peer_id))
        
        # Clean up stale peers
        for room_id, peer_id in stale_peers:
            if room_id in self.rooms and peer_id in self.rooms[room_id].peers:
                peer = self.rooms[room_id].peers[peer_id]
                
                # Close connection
                if peer_id in self.peer_connections:
                    await self.peer_connections[peer_id].close()
                    del self.peer_connections[peer_id]
                
                # Remove from room
                del self.rooms[room_id].peers[peer_id]
                
                logging.info(f"Cleaned up stale peer: {peer_id}")
        
        # Clean up empty rooms
        empty_rooms = [room_id for room_id, room in self.rooms.items() 
                      if not room.get_active_peers()]
        
        for room_id in empty_rooms:
            del self.rooms[room_id]
            await self.redis.delete(f"webrtc_room:{room_id}")
            logging.info(f"Cleaned up empty room: {room_id}")
    
    def get_webrtc_stats(self) -> Dict[str, Any]:
        """Get WebRTC server statistics."""
        total_rooms = len(self.rooms)
        total_peers = sum(len(room.peers) for room in self.rooms.values())
        active_peers = sum(len(room.get_active_peers()) for room in self.rooms.values())
        
        return {
            'total_rooms': total_rooms,
            'total_peers': total_peers,
            'active_peers': active_peers,
            'peer_connections': len(self.peer_connections),
            'rooms': {room_id: room.to_dict() for room_id, room in self.rooms.items()}
        }


# Global instance
webrtc_server = WebRTCServer()