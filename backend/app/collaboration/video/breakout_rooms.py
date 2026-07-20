"""
Breakout rooms manager for organizing sub-discussions during meetings.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import random

import redis.asyncio as redis


class BreakoutRoomStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class AssignmentMethod(Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SELF_SELECT = "self_select"
    RANDOM = "random"


class BreakoutRoomType(Enum):
    DISCUSSION = "discussion"
    COLLABORATION = "collaboration"
    PRESENTATION = "presentation"
    PROBLEM_SOLVING = "problem_solving"
    BRAINSTORMING = "brainstorming"


@dataclass
class BreakoutParticipant:
    """Participant in a breakout room."""
    user_id: str
    username: str
    role: str = "participant"  # host, facilitator, participant
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    is_online: bool = False
    camera_enabled: bool = True
    microphone_enabled: bool = True
    screen_sharing: bool = False
    hand_raised: bool = False
    speaking_time: float = 0.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None,
            'left_at': self.left_at.isoformat() if self.left_at else None,
            'is_online': self.is_online,
            'camera_enabled': self.camera_enabled,
            'microphone_enabled': self.microphone_enabled,
            'screen_sharing': self.screen_sharing,
            'hand_raised': self.hand_raised,
            'speaking_time': self.speaking_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreakoutParticipant':
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            role=data.get('role', 'participant'),
            joined_at=datetime.fromisoformat(data['joined_at']) if data.get('joined_at') else None,
            left_at=datetime.fromisoformat(data['left_at']) if data.get('left_at') else None,
            is_online=data.get('is_online', False),
            camera_enabled=data.get('camera_enabled', True),
            microphone_enabled=data.get('microphone_enabled', True),
            screen_sharing=data.get('screen_sharing', False),
            hand_raised=data.get('hand_raised', False),
            speaking_time=data.get('speaking_time', 0.0)
        )


@dataclass
class BreakoutRoomSettings:
    """Settings for breakout rooms."""
    max_participants: int = 8
    auto_join_audio: bool = True
    allow_participants_to_return: bool = True
    timer_duration: Optional[int] = None  # minutes
    record_sessions: bool = False
    enable_screen_sharing: bool = True
    enable_chat: bool = True
    enable_file_sharing: bool = True
    moderator_assistance: bool = True
    close_rooms_automatically: bool = False
    pre_assign_participants: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreakoutRoomSettings':
        return cls(**data)


@dataclass
class BreakoutRoom:
    """A single breakout room."""
    room_id: str
    name: str
    main_meeting_id: str
    room_type: BreakoutRoomType
    status: BreakoutRoomStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_by: str = ""
    facilitator_id: Optional[str] = None
    participants: Dict[str, BreakoutParticipant] = field(default_factory=dict)
    settings: BreakoutRoomSettings = field(default_factory=BreakoutRoomSettings)
    topic: Optional[str] = None
    instructions: Optional[str] = None
    resources: List[str] = field(default_factory=list)  # URLs or file references
    chat_messages: List[Dict[str, Any]] = field(default_factory=list)
    shared_notes: str = ""
    timer_end_time: Optional[datetime] = None
    help_requested: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_active_participants(self) -> List[BreakoutParticipant]:
        """Get currently active participants."""
        return [p for p in self.participants.values() if p.is_online]
    
    def get_duration(self) -> Optional[timedelta]:
        """Get room duration."""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def is_timer_expired(self) -> bool:
        """Check if timer has expired."""
        if self.timer_end_time:
            return datetime.utcnow() > self.timer_end_time
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'room_id': self.room_id,
            'name': self.name,
            'main_meeting_id': self.main_meeting_id,
            'room_type': self.room_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'created_by': self.created_by,
            'facilitator_id': self.facilitator_id,
            'participants': {uid: p.to_dict() for uid, p in self.participants.items()},
            'settings': self.settings.to_dict(),
            'topic': self.topic,
            'instructions': self.instructions,
            'resources': self.resources,
            'chat_messages': self.chat_messages,
            'shared_notes': self.shared_notes,
            'timer_end_time': self.timer_end_time.isoformat() if self.timer_end_time else None,
            'help_requested': self.help_requested,
            'metadata': self.metadata,
            'active_participant_count': len(self.get_active_participants()),
            'duration_minutes': self.get_duration().total_seconds() / 60 if self.get_duration() else None,
            'timer_expired': self.is_timer_expired()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreakoutRoom':
        return cls(
            room_id=data['room_id'],
            name=data['name'],
            main_meeting_id=data['main_meeting_id'],
            room_type=BreakoutRoomType(data['room_type']),
            status=BreakoutRoomStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            ended_at=datetime.fromisoformat(data['ended_at']) if data.get('ended_at') else None,
            created_by=data.get('created_by', ''),
            facilitator_id=data.get('facilitator_id'),
            participants={uid: BreakoutParticipant.from_dict(p) for uid, p in data.get('participants', {}).items()},
            settings=BreakoutRoomSettings.from_dict(data.get('settings', {})),
            topic=data.get('topic'),
            instructions=data.get('instructions'),
            resources=data.get('resources', []),
            chat_messages=data.get('chat_messages', []),
            shared_notes=data.get('shared_notes', ''),
            timer_end_time=datetime.fromisoformat(data['timer_end_time']) if data.get('timer_end_time') else None,
            help_requested=data.get('help_requested', False),
            metadata=data.get('metadata', {})
        )


@dataclass
class BreakoutSession:
    """A complete breakout session with multiple rooms."""
    session_id: str
    main_meeting_id: str
    title: str
    description: Optional[str]
    created_by: str
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    assignment_method: AssignmentMethod = AssignmentMethod.MANUAL
    rooms: Dict[str, BreakoutRoom] = field(default_factory=dict)
    global_settings: BreakoutRoomSettings = field(default_factory=BreakoutRoomSettings)
    unassigned_participants: List[str] = field(default_factory=list)
    session_timer: Optional[int] = None  # minutes
    timer_end_time: Optional[datetime] = None
    
    def get_total_participants(self) -> int:
        """Get total number of participants across all rooms."""
        return sum(len(room.participants) for room in self.rooms.values())
    
    def get_active_rooms(self) -> List[BreakoutRoom]:
        """Get currently active rooms."""
        return [room for room in self.rooms.values() if room.status == BreakoutRoomStatus.ACTIVE]
    
    def is_session_timer_expired(self) -> bool:
        """Check if session timer has expired."""
        if self.timer_end_time:
            return datetime.utcnow() > self.timer_end_time
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'main_meeting_id': self.main_meeting_id,
            'title': self.title,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'assignment_method': self.assignment_method.value,
            'rooms': {rid: room.to_dict() for rid, room in self.rooms.items()},
            'global_settings': self.global_settings.to_dict(),
            'unassigned_participants': self.unassigned_participants,
            'session_timer': self.session_timer,
            'timer_end_time': self.timer_end_time.isoformat() if self.timer_end_time else None,
            'total_participants': self.get_total_participants(),
            'active_room_count': len(self.get_active_rooms()),
            'session_timer_expired': self.is_session_timer_expired()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreakoutSession':
        return cls(
            session_id=data['session_id'],
            main_meeting_id=data['main_meeting_id'],
            title=data['title'],
            description=data.get('description'),
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            ended_at=datetime.fromisoformat(data['ended_at']) if data.get('ended_at') else None,
            assignment_method=AssignmentMethod(data.get('assignment_method', 'manual')),
            rooms={rid: BreakoutRoom.from_dict(room) for rid, room in data.get('rooms', {}).items()},
            global_settings=BreakoutRoomSettings.from_dict(data.get('global_settings', {})),
            unassigned_participants=data.get('unassigned_participants', []),
            session_timer=data.get('session_timer'),
            timer_end_time=datetime.fromisoformat(data['timer_end_time']) if data.get('timer_end_time') else None
        )


class BreakoutRoomManager:
    """Manages breakout rooms for video conferences."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.active_sessions: Dict[str, BreakoutSession] = {}
        self.event_callbacks: List[callable] = []
        self.timer_task = None
        
    async def start(self):
        """Start the breakout room manager."""
        # Load active sessions from Redis
        await self._load_active_sessions()
        
        # Start timer monitoring task
        self.timer_task = asyncio.create_task(self._timer_monitoring_loop())
        
        logging.info("Breakout room manager started")
    
    async def stop(self):
        """Stop the breakout room manager."""
        # End all active sessions
        for session in list(self.active_sessions.values()):
            await self.end_session(session.session_id, session.created_by)
        
        # Stop timer task
        if self.timer_task:
            self.timer_task.cancel()
            try:
                await self.timer_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.close()
        logging.info("Breakout room manager stopped")
    
    def add_event_callback(self, callback: callable):
        """Add a callback for breakout room events."""
        self.event_callbacks.append(callback)
    
    async def create_session(self, main_meeting_id: str, title: str, created_by: str,
                           description: Optional[str] = None,
                           assignment_method: AssignmentMethod = AssignmentMethod.MANUAL,
                           global_settings: Optional[BreakoutRoomSettings] = None,
                           session_timer: Optional[int] = None) -> BreakoutSession:
        """Create a new breakout session."""
        session_id = str(uuid.uuid4())
        
        session = BreakoutSession(
            session_id=session_id,
            main_meeting_id=main_meeting_id,
            title=title,
            description=description,
            created_by=created_by,
            created_at=datetime.utcnow(),
            assignment_method=assignment_method,
            global_settings=global_settings or BreakoutRoomSettings(),
            session_timer=session_timer
        )
        
        self.active_sessions[session_id] = session
        
        # Store in Redis
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "session_created", created_by, {
            'title': title,
            'assignment_method': assignment_method.value,
            'session_timer': session_timer
        })
        
        logging.info(f"Created breakout session: {title} ({session_id})")
        return session
    
    async def create_room(self, session_id: str, name: str, room_type: BreakoutRoomType,
                        topic: Optional[str] = None, instructions: Optional[str] = None,
                        facilitator_id: Optional[str] = None,
                        room_settings: Optional[BreakoutRoomSettings] = None) -> Optional[BreakoutRoom]:
        """Create a breakout room in a session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        room_id = str(uuid.uuid4())
        
        room = BreakoutRoom(
            room_id=room_id,
            name=name,
            main_meeting_id=session.main_meeting_id,
            room_type=room_type,
            status=BreakoutRoomStatus.CREATED,
            created_at=datetime.utcnow(),
            created_by=session.created_by,
            facilitator_id=facilitator_id,
            topic=topic,
            instructions=instructions,
            settings=room_settings or session.global_settings
        )
        
        session.rooms[room_id] = room
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "room_created", session.created_by, {
            'room_id': room_id,
            'room_name': name,
            'room_type': room_type.value,
            'facilitator_id': facilitator_id
        })
        
        logging.info(f"Created breakout room: {name} ({room_id})")
        return room
    
    async def assign_participants(self, session_id: str, assignments: Dict[str, List[str]]) -> bool:
        """Manually assign participants to rooms."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        for room_id, user_ids in assignments.items():
            if room_id not in session.rooms:
                continue
            
            room = session.rooms[room_id]
            
            for user_id in user_ids:
                # Remove from unassigned if present
                if user_id in session.unassigned_participants:
                    session.unassigned_participants.remove(user_id)
                
                # Add to room if not already present
                if user_id not in room.participants:
                    participant = BreakoutParticipant(
                        user_id=user_id,
                        username=f"User {user_id}",  # This should be fetched from user service
                        role="participant"
                    )
                    room.participants[user_id] = participant
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "participants_assigned", session.created_by, {
            'assignments': assignments
        })
        
        return True
    
    async def auto_assign_participants(self, session_id: str, participant_ids: List[str],
                                     num_rooms: Optional[int] = None) -> bool:
        """Automatically assign participants to rooms."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Create rooms if needed
        if num_rooms and len(session.rooms) < num_rooms:
            for i in range(len(session.rooms), num_rooms):
                await self.create_room(
                    session_id=session_id,
                    name=f"Room {i + 1}",
                    room_type=BreakoutRoomType.DISCUSSION
                )
        
        rooms = list(session.rooms.values())
        if not rooms:
            return False
        
        # Shuffle participants for random assignment
        if session.assignment_method == AssignmentMethod.RANDOM:
            random.shuffle(participant_ids)
        
        # Assign participants to rooms
        for i, user_id in enumerate(participant_ids):
            room = rooms[i % len(rooms)]
            
            participant = BreakoutParticipant(
                user_id=user_id,
                username=f"User {user_id}",  # This should be fetched from user service
                role="participant"
            )
            
            room.participants[user_id] = participant
            
            # Remove from unassigned
            if user_id in session.unassigned_participants:
                session.unassigned_participants.remove(user_id)
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "participants_auto_assigned", session.created_by, {
            'participant_count': len(participant_ids),
            'room_count': len(rooms),
            'assignment_method': session.assignment_method.value
        })
        
        return True
    
    async def start_session(self, session_id: str, user_id: str) -> bool:
        """Start all rooms in a breakout session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check if user can start the session
        if user_id != session.created_by:
            return False
        
        session.started_at = datetime.utcnow()
        
        # Set session timer if configured
        if session.session_timer:
            session.timer_end_time = session.started_at + timedelta(minutes=session.session_timer)
        
        # Start all rooms
        for room in session.rooms.values():
            room.status = BreakoutRoomStatus.ACTIVE
            room.started_at = session.started_at
            
            # Set room timer if configured
            if room.settings.timer_duration:
                room.timer_end_time = room.started_at + timedelta(minutes=room.settings.timer_duration)
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "session_started", user_id, {
            'room_count': len(session.rooms),
            'total_participants': session.get_total_participants(),
            'session_timer': session.session_timer
        })
        
        logging.info(f"Started breakout session: {session.title} ({session_id})")
        return True
    
    async def end_session(self, session_id: str, user_id: str) -> bool:
        """End a breakout session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check if user can end the session
        if user_id != session.created_by:
            return False
        
        session.ended_at = datetime.utcnow()
        
        # Close all rooms
        for room in session.rooms.values():
            room.status = BreakoutRoomStatus.CLOSED
            room.ended_at = session.ended_at
            
            # Mark all participants as offline
            for participant in room.participants.values():
                if participant.is_online:
                    participant.is_online = False
                    participant.left_at = session.ended_at
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Store final session state
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "session_ended", user_id, {
            'duration_minutes': (session.ended_at - session.started_at).total_seconds() / 60 if session.started_at else 0,
            'room_count': len(session.rooms),
            'total_participants': session.get_total_participants()
        })
        
        logging.info(f"Ended breakout session: {session.title} ({session_id})")
        return True
    
    async def join_room(self, session_id: str, room_id: str, user_id: str, username: str) -> Optional[BreakoutParticipant]:
        """Join a breakout room."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        if room_id not in session.rooms:
            return None
        
        room = session.rooms[room_id]
        
        # Check if room is active
        if room.status != BreakoutRoomStatus.ACTIVE:
            return None
        
        # Check room capacity
        if len(room.get_active_participants()) >= room.settings.max_participants:
            return None
        
        # Get or create participant
        if user_id in room.participants:
            participant = room.participants[user_id]
        else:
            participant = BreakoutParticipant(
                user_id=user_id,
                username=username,
                role="participant"
            )
            room.participants[user_id] = participant
        
        # Update participant status
        participant.is_online = True
        participant.joined_at = datetime.utcnow()
        participant.camera_enabled = room.settings.auto_join_audio
        participant.microphone_enabled = room.settings.auto_join_audio
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "participant_joined_room", user_id, {
            'room_id': room_id,
            'room_name': room.name,
            'username': username
        })
        
        logging.info(f"User {username} joined breakout room {room.name}")
        return participant
    
    async def leave_room(self, session_id: str, room_id: str, user_id: str) -> bool:
        """Leave a breakout room."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if room_id not in session.rooms:
            return False
        
        room = session.rooms[room_id]
        
        if user_id not in room.participants:
            return False
        
        participant = room.participants[user_id]
        participant.is_online = False
        participant.left_at = datetime.utcnow()
        participant.camera_enabled = False
        participant.microphone_enabled = False
        participant.screen_sharing = False
        participant.hand_raised = False
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "participant_left_room", user_id, {
            'room_id': room_id,
            'room_name': room.name,
            'username': participant.username
        })
        
        logging.info(f"User {participant.username} left breakout room {room.name}")
        return True
    
    async def request_help(self, session_id: str, room_id: str, user_id: str) -> bool:
        """Request help from moderator."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if room_id not in session.rooms:
            return False
        
        room = session.rooms[room_id]
        room.help_requested = True
        
        # Store updated session
        await self._store_session(session)
        
        # Create event
        await self._create_event(session_id, "help_requested", user_id, {
            'room_id': room_id,
            'room_name': room.name
        })
        
        return True
    
    async def update_shared_notes(self, session_id: str, room_id: str, user_id: str, notes: str) -> bool:
        """Update shared notes for a room."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if room_id not in session.rooms:
            return False
        
        room = session.rooms[room_id]
        room.shared_notes = notes
        
        # Store updated session
        await self._store_session(session)
        
        return True
    
    async def add_chat_message(self, session_id: str, room_id: str, user_id: str, username: str, message: str) -> bool:
        """Add a chat message to a room."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if room_id not in session.rooms:
            return False
        
        room = session.rooms[room_id]
        
        chat_message = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'username': username,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        room.chat_messages.append(chat_message)
        
        # Keep only last 50 messages
        if len(room.chat_messages) > 50:
            room.chat_messages = room.chat_messages[-50:]
        
        # Store updated session
        await self._store_session(session)
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[BreakoutSession]:
        """Get session by ID."""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from Redis
        session_data = await self.redis.get(f"breakout_session:{session_id}")
        if session_data:
            try:
                data = json.loads(session_data)
                session = BreakoutSession.from_dict(data)
                
                # Add to active sessions if still active
                if session.ended_at is None:
                    self.active_sessions[session_id] = session
                
                return session
            except Exception as e:
                logging.error(f"Error loading breakout session from Redis: {e}")
        
        return None
    
    async def get_meeting_sessions(self, meeting_id: str) -> List[BreakoutSession]:
        """Get all breakout sessions for a meeting."""
        sessions = []
        
        # Check active sessions
        for session in self.active_sessions.values():
            if session.main_meeting_id == meeting_id:
                sessions.append(session)
        
        # Load from Redis
        session_keys = await self.redis.keys(f"breakout_session:*")
        for key in session_keys:
            session_data = await self.redis.get(key)
            if session_data:
                try:
                    data = json.loads(session_data)
                    session = BreakoutSession.from_dict(data)
                    if (session.main_meeting_id == meeting_id and 
                        session.session_id not in [s.session_id for s in sessions]):
                        sessions.append(session)
                except Exception as e:
                    logging.error(f"Error loading breakout session: {e}")
        
        # Sort by creation time
        sessions.sort(key=lambda s: s.created_at)
        return sessions
    
    async def _store_session(self, session: BreakoutSession):
        """Store session in Redis."""
        try:
            data = json.dumps(session.to_dict())
            await self.redis.set(f"breakout_session:{session.session_id}", data)
            await self.redis.expire(f"breakout_session:{session.session_id}", 86400 * 7)  # 7 days
        except Exception as e:
            logging.error(f"Error storing breakout session: {e}")
    
    async def _create_event(self, session_id: str, event_type: str, user_id: str, data: Dict[str, Any]):
        """Create a breakout room event."""
        event = {
            'event_id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': event_type,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Store event
        try:
            event_data = json.dumps(event)
            await self.redis.lpush(f"breakout_events:{session_id}", event_data)
            await self.redis.ltrim(f"breakout_events:{session_id}", 0, 999)  # Keep last 1000 events
        except Exception as e:
            logging.error(f"Error storing breakout event: {e}")
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logging.error(f"Error in breakout event callback: {e}")
    
    async def _load_active_sessions(self):
        """Load active sessions from Redis."""
        try:
            session_keys = await self.redis.keys("breakout_session:*")
            for key in session_keys:
                session_data = await self.redis.get(key)
                if session_data:
                    try:
                        data = json.loads(session_data)
                        session = BreakoutSession.from_dict(data)
                        
                        # Only load if session is still active
                        if session.ended_at is None:
                            self.active_sessions[session.session_id] = session
                    except Exception as e:
                        logging.error(f"Error loading breakout session: {e}")
        except Exception as e:
            logging.error(f"Error loading active breakout sessions: {e}")
    
    async def _timer_monitoring_loop(self):
        """Monitor timers for sessions and rooms."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_timers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in timer monitoring: {e}")
    
    async def _check_timers(self):
        """Check and handle expired timers."""
        current_time = datetime.utcnow()
        sessions_to_end = []
        rooms_to_close = []
        
        for session in list(self.active_sessions.values()):
            # Check session timer
            if session.is_session_timer_expired():
                sessions_to_end.append((session.session_id, session.created_by))
            
            # Check room timers
            for room in session.rooms.values():
                if room.is_timer_expired() and room.status == BreakoutRoomStatus.ACTIVE:
                    rooms_to_close.append((session.session_id, room.room_id))
        
        # Handle expired timers
        for session_id, created_by in sessions_to_end:
            await self.end_session(session_id, created_by)
            logging.info(f"Auto-ended breakout session {session_id} due to timer expiration")
        
        for session_id, room_id in rooms_to_close:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if room_id in session.rooms:
                    room = session.rooms[room_id]
                    room.status = BreakoutRoomStatus.CLOSED
                    room.ended_at = current_time
                    
                    # Mark participants as offline
                    for participant in room.participants.values():
                        if participant.is_online:
                            participant.is_online = False
                            participant.left_at = current_time
                    
                    await self._store_session(session)
                    logging.info(f"Auto-closed breakout room {room.name} due to timer expiration")
    
    def get_breakout_stats(self) -> Dict[str, Any]:
        """Get breakout room statistics."""
        active_sessions = len(self.active_sessions)
        total_rooms = sum(len(session.rooms) for session in self.active_sessions.values())
        active_rooms = sum(len(session.get_active_rooms()) for session in self.active_sessions.values())
        total_participants = sum(session.get_total_participants() for session in self.active_sessions.values())
        
        return {
            'active_sessions': active_sessions,
            'total_rooms': total_rooms,
            'active_rooms': active_rooms,
            'total_participants': total_participants,
            'session_details': [session.to_dict() for session in self.active_sessions.values()]
        }


# Global instance
breakout_room_manager = BreakoutRoomManager()