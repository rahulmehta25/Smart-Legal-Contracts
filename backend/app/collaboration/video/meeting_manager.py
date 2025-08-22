"""
Meeting manager for organizing and controlling video conferences.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import json

import redis.asyncio as redis
from pydantic import BaseModel, Field


class MeetingStatus(Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    CANCELLED = "cancelled"


class ParticipantRole(Enum):
    HOST = "host"
    CO_HOST = "co_host"
    PRESENTER = "presenter"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class MeetingFeature(Enum):
    SCREEN_SHARING = "screen_sharing"
    RECORDING = "recording"
    TRANSCRIPTION = "transcription"
    BREAKOUT_ROOMS = "breakout_rooms"
    VIRTUAL_BACKGROUNDS = "virtual_backgrounds"
    POLLS = "polls"
    CHAT = "chat"
    REACTIONS = "reactions"


@dataclass
class MeetingParticipant:
    """Meeting participant information."""
    user_id: str
    username: str
    email: Optional[str]
    role: ParticipantRole
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    is_online: bool = False
    camera_enabled: bool = True
    microphone_enabled: bool = True
    screen_sharing: bool = False
    is_speaking: bool = False
    hand_raised: bool = False
    connection_quality: str = "good"  # poor, fair, good, excellent
    bandwidth_usage: int = 0  # kbps
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None,
            'left_at': self.left_at.isoformat() if self.left_at else None,
            'is_online': self.is_online,
            'camera_enabled': self.camera_enabled,
            'microphone_enabled': self.microphone_enabled,
            'screen_sharing': self.screen_sharing,
            'is_speaking': self.is_speaking,
            'hand_raised': self.hand_raised,
            'connection_quality': self.connection_quality,
            'bandwidth_usage': self.bandwidth_usage,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeetingParticipant':
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data.get('email'),
            role=ParticipantRole(data['role']),
            joined_at=datetime.fromisoformat(data['joined_at']) if data.get('joined_at') else None,
            left_at=datetime.fromisoformat(data['left_at']) if data.get('left_at') else None,
            is_online=data.get('is_online', False),
            camera_enabled=data.get('camera_enabled', True),
            microphone_enabled=data.get('microphone_enabled', True),
            screen_sharing=data.get('screen_sharing', False),
            is_speaking=data.get('is_speaking', False),
            hand_raised=data.get('hand_raised', False),
            connection_quality=data.get('connection_quality', 'good'),
            bandwidth_usage=data.get('bandwidth_usage', 0),
            metadata=data.get('metadata', {})
        )


@dataclass
class MeetingSettings:
    """Meeting configuration settings."""
    allow_recording: bool = True
    allow_screen_sharing: bool = True
    require_password: bool = False
    password: Optional[str] = None
    max_participants: int = 100
    auto_mute_on_join: bool = False
    waiting_room_enabled: bool = False
    breakout_rooms_enabled: bool = True
    virtual_backgrounds_enabled: bool = True
    transcription_enabled: bool = False
    auto_transcription_language: str = "en"
    recording_auto_start: bool = False
    participant_permissions: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeetingSettings':
        return cls(**data)


@dataclass
class Meeting:
    """Complete meeting information."""
    meeting_id: str
    title: str
    description: Optional[str]
    host_user_id: str
    room_id: str
    status: MeetingStatus
    scheduled_start: datetime
    scheduled_end: datetime
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    participants: Dict[str, MeetingParticipant] = field(default_factory=dict)
    settings: MeetingSettings = field(default_factory=MeetingSettings)
    enabled_features: Set[MeetingFeature] = field(default_factory=set)
    recording_urls: List[str] = field(default_factory=list)
    transcription_url: Optional[str] = None
    meeting_notes: Optional[str] = None
    poll_results: List[Dict[str, Any]] = field(default_factory=list)
    chat_messages: List[Dict[str, Any]] = field(default_factory=list)
    breakout_rooms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_active_participants(self) -> List[MeetingParticipant]:
        """Get currently active participants."""
        return [p for p in self.participants.values() if p.is_online]
    
    def get_duration(self) -> Optional[timedelta]:
        """Get meeting duration."""
        if self.actual_start and self.actual_end:
            return self.actual_end - self.actual_start
        elif self.actual_start:
            return datetime.utcnow() - self.actual_start
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'meeting_id': self.meeting_id,
            'title': self.title,
            'description': self.description,
            'host_user_id': self.host_user_id,
            'room_id': self.room_id,
            'status': self.status.value,
            'scheduled_start': self.scheduled_start.isoformat(),
            'scheduled_end': self.scheduled_end.isoformat(),
            'actual_start': self.actual_start.isoformat() if self.actual_start else None,
            'actual_end': self.actual_end.isoformat() if self.actual_end else None,
            'participants': {uid: p.to_dict() for uid, p in self.participants.items()},
            'settings': self.settings.to_dict(),
            'enabled_features': [f.value for f in self.enabled_features],
            'recording_urls': self.recording_urls,
            'transcription_url': self.transcription_url,
            'meeting_notes': self.meeting_notes,
            'poll_results': self.poll_results,
            'chat_messages': self.chat_messages,
            'breakout_rooms': self.breakout_rooms,
            'metadata': self.metadata,
            'active_participant_count': len(self.get_active_participants()),
            'duration_minutes': self.get_duration().total_seconds() / 60 if self.get_duration() else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Meeting':
        return cls(
            meeting_id=data['meeting_id'],
            title=data['title'],
            description=data.get('description'),
            host_user_id=data['host_user_id'],
            room_id=data['room_id'],
            status=MeetingStatus(data['status']),
            scheduled_start=datetime.fromisoformat(data['scheduled_start']),
            scheduled_end=datetime.fromisoformat(data['scheduled_end']),
            actual_start=datetime.fromisoformat(data['actual_start']) if data.get('actual_start') else None,
            actual_end=datetime.fromisoformat(data['actual_end']) if data.get('actual_end') else None,
            participants={uid: MeetingParticipant.from_dict(p) for uid, p in data.get('participants', {}).items()},
            settings=MeetingSettings.from_dict(data.get('settings', {})),
            enabled_features={MeetingFeature(f) for f in data.get('enabled_features', [])},
            recording_urls=data.get('recording_urls', []),
            transcription_url=data.get('transcription_url'),
            meeting_notes=data.get('meeting_notes'),
            poll_results=data.get('poll_results', []),
            chat_messages=data.get('chat_messages', []),
            breakout_rooms=data.get('breakout_rooms', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class MeetingEvent:
    """Meeting event for audit trail."""
    event_id: str
    meeting_id: str
    event_type: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'meeting_id': self.meeting_id,
            'event_type': self.event_type,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }


class MeetingManager:
    """Manages video conference meetings."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.active_meetings: Dict[str, Meeting] = {}
        self.scheduled_meetings: Dict[str, Meeting] = {}
        self.event_callbacks: List[callable] = []
        self.auto_cleanup_task = None
        
    async def start(self):
        """Start the meeting manager."""
        # Load scheduled meetings from Redis
        await self._load_scheduled_meetings()
        
        # Start auto-cleanup task
        self.auto_cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
        
        logging.info("Meeting manager started")
    
    async def stop(self):
        """Stop the meeting manager."""
        if self.auto_cleanup_task:
            self.auto_cleanup_task.cancel()
            try:
                await self.auto_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # End all active meetings
        for meeting in list(self.active_meetings.values()):
            await self.end_meeting(meeting.meeting_id, meeting.host_user_id)
        
        await self.redis.close()
        logging.info("Meeting manager stopped")
    
    def add_event_callback(self, callback: callable):
        """Add a callback for meeting events."""
        self.event_callbacks.append(callback)
    
    async def schedule_meeting(self, title: str, host_user_id: str, 
                             scheduled_start: datetime, scheduled_end: datetime,
                             description: Optional[str] = None,
                             settings: Optional[MeetingSettings] = None,
                             invitees: Optional[List[str]] = None) -> Meeting:
        """Schedule a new meeting."""
        meeting_id = str(uuid.uuid4())
        room_id = f"meeting_{meeting_id}"
        
        meeting = Meeting(
            meeting_id=meeting_id,
            title=title,
            description=description,
            host_user_id=host_user_id,
            room_id=room_id,
            status=MeetingStatus.SCHEDULED,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            settings=settings or MeetingSettings()
        )
        
        # Add host as participant
        host_participant = MeetingParticipant(
            user_id=host_user_id,
            username="Host",  # This should be fetched from user service
            email=None,
            role=ParticipantRole.HOST
        )
        meeting.participants[host_user_id] = host_participant
        
        # Add invitees as participants
        if invitees:
            for user_id in invitees:
                participant = MeetingParticipant(
                    user_id=user_id,
                    username=f"User {user_id}",  # This should be fetched from user service
                    email=None,
                    role=ParticipantRole.PARTICIPANT
                )
                meeting.participants[user_id] = participant
        
        # Store meeting
        self.scheduled_meetings[meeting_id] = meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "meeting_scheduled", host_user_id, {
            'title': title,
            'scheduled_start': scheduled_start.isoformat(),
            'scheduled_end': scheduled_end.isoformat(),
            'participant_count': len(meeting.participants)
        })
        
        logging.info(f"Scheduled meeting: {title} ({meeting_id})")
        return meeting
    
    async def start_meeting(self, meeting_id: str, user_id: str) -> Optional[Meeting]:
        """Start a scheduled meeting."""
        # Get meeting from scheduled or active
        meeting = self.scheduled_meetings.get(meeting_id) or self.active_meetings.get(meeting_id)
        if not meeting:
            return None
        
        # Check if user can start the meeting
        if not await self._can_start_meeting(meeting, user_id):
            return None
        
        # Move to active meetings
        if meeting_id in self.scheduled_meetings:
            del self.scheduled_meetings[meeting_id]
        
        meeting.status = MeetingStatus.ACTIVE
        meeting.actual_start = datetime.utcnow()
        
        self.active_meetings[meeting_id] = meeting
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "meeting_started", user_id, {
            'actual_start': meeting.actual_start.isoformat()
        })
        
        logging.info(f"Started meeting: {meeting.title} ({meeting_id})")
        return meeting
    
    async def end_meeting(self, meeting_id: str, user_id: str) -> bool:
        """End an active meeting."""
        if meeting_id not in self.active_meetings:
            return False
        
        meeting = self.active_meetings[meeting_id]
        
        # Check if user can end the meeting
        if not await self._can_end_meeting(meeting, user_id):
            return False
        
        # Update meeting status
        meeting.status = MeetingStatus.ENDED
        meeting.actual_end = datetime.utcnow()
        
        # Mark all participants as offline
        for participant in meeting.participants.values():
            if participant.is_online:
                participant.is_online = False
                participant.left_at = meeting.actual_end
        
        # Remove from active meetings
        del self.active_meetings[meeting_id]
        
        # Store final meeting state
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "meeting_ended", user_id, {
            'actual_end': meeting.actual_end.isoformat(),
            'duration_minutes': meeting.get_duration().total_seconds() / 60 if meeting.get_duration() else 0
        })
        
        logging.info(f"Ended meeting: {meeting.title} ({meeting_id})")
        return True
    
    async def join_meeting(self, meeting_id: str, user_id: str, username: str,
                          password: Optional[str] = None) -> Optional[MeetingParticipant]:
        """Join a meeting."""
        meeting = self.active_meetings.get(meeting_id)
        if not meeting:
            # Check if it's a scheduled meeting that needs to be started
            scheduled_meeting = self.scheduled_meetings.get(meeting_id)
            if scheduled_meeting:
                meeting = await self.start_meeting(meeting_id, user_id)
                if not meeting:
                    return None
            else:
                return None
        
        # Check password if required
        if meeting.settings.require_password and meeting.settings.password:
            if password != meeting.settings.password:
                return None
        
        # Check participant limit
        if len(meeting.get_active_participants()) >= meeting.settings.max_participants:
            return None
        
        # Get or create participant
        if user_id in meeting.participants:
            participant = meeting.participants[user_id]
        else:
            participant = MeetingParticipant(
                user_id=user_id,
                username=username,
                email=None,
                role=ParticipantRole.PARTICIPANT
            )
            meeting.participants[user_id] = participant
        
        # Update participant status
        participant.is_online = True
        participant.joined_at = datetime.utcnow()
        participant.camera_enabled = not meeting.settings.auto_mute_on_join
        participant.microphone_enabled = not meeting.settings.auto_mute_on_join
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "participant_joined", user_id, {
            'username': username,
            'role': participant.role.value
        })
        
        logging.info(f"User {username} joined meeting {meeting.title}")
        return participant
    
    async def leave_meeting(self, meeting_id: str, user_id: str) -> bool:
        """Leave a meeting."""
        if meeting_id not in self.active_meetings:
            return False
        
        meeting = self.active_meetings[meeting_id]
        
        if user_id not in meeting.participants:
            return False
        
        participant = meeting.participants[user_id]
        participant.is_online = False
        participant.left_at = datetime.utcnow()
        participant.camera_enabled = False
        participant.microphone_enabled = False
        participant.screen_sharing = False
        participant.hand_raised = False
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "participant_left", user_id, {
            'username': participant.username
        })
        
        logging.info(f"User {participant.username} left meeting {meeting.title}")
        return True
    
    async def update_participant_media(self, meeting_id: str, user_id: str,
                                     camera_enabled: Optional[bool] = None,
                                     microphone_enabled: Optional[bool] = None,
                                     screen_sharing: Optional[bool] = None) -> bool:
        """Update participant media state."""
        if meeting_id not in self.active_meetings:
            return False
        
        meeting = self.active_meetings[meeting_id]
        
        if user_id not in meeting.participants:
            return False
        
        participant = meeting.participants[user_id]
        changes = {}
        
        if camera_enabled is not None:
            participant.camera_enabled = camera_enabled
            changes['camera_enabled'] = camera_enabled
        
        if microphone_enabled is not None:
            participant.microphone_enabled = microphone_enabled
            changes['microphone_enabled'] = microphone_enabled
        
        if screen_sharing is not None:
            participant.screen_sharing = screen_sharing
            changes['screen_sharing'] = screen_sharing
            
            # Only one person can share screen at a time
            if screen_sharing:
                for other_participant in meeting.participants.values():
                    if other_participant.user_id != user_id:
                        other_participant.screen_sharing = False
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        if changes:
            await self._create_event(meeting_id, "participant_media_updated", user_id, changes)
        
        return True
    
    async def raise_hand(self, meeting_id: str, user_id: str, raised: bool = True) -> bool:
        """Raise or lower hand in meeting."""
        if meeting_id not in self.active_meetings:
            return False
        
        meeting = self.active_meetings[meeting_id]
        
        if user_id not in meeting.participants:
            return False
        
        participant = meeting.participants[user_id]
        participant.hand_raised = raised
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "hand_raised" if raised else "hand_lowered", user_id, {
            'username': participant.username
        })
        
        return True
    
    async def add_chat_message(self, meeting_id: str, user_id: str, username: str,
                             message: str, recipient_id: Optional[str] = None) -> bool:
        """Add a chat message to the meeting."""
        if meeting_id not in self.active_meetings:
            return False
        
        meeting = self.active_meetings[meeting_id]
        
        chat_message = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'username': username,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            'recipient_id': recipient_id,  # None for public messages
            'is_private': recipient_id is not None
        }
        
        meeting.chat_messages.append(chat_message)
        
        # Keep only last 100 messages
        if len(meeting.chat_messages) > 100:
            meeting.chat_messages = meeting.chat_messages[-100:]
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        return True
    
    async def update_meeting_settings(self, meeting_id: str, user_id: str,
                                    settings: MeetingSettings) -> bool:
        """Update meeting settings."""
        meeting = self.active_meetings.get(meeting_id) or self.scheduled_meetings.get(meeting_id)
        if not meeting:
            return False
        
        # Check if user can update settings
        if not await self._can_update_settings(meeting, user_id):
            return False
        
        meeting.settings = settings
        
        # Store updated meeting
        await self._store_meeting(meeting)
        
        # Create event
        await self._create_event(meeting_id, "settings_updated", user_id, {
            'settings': settings.to_dict()
        })
        
        return True
    
    async def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Get meeting by ID."""
        # Check active meetings first
        if meeting_id in self.active_meetings:
            return self.active_meetings[meeting_id]
        
        # Check scheduled meetings
        if meeting_id in self.scheduled_meetings:
            return self.scheduled_meetings[meeting_id]
        
        # Try to load from Redis
        meeting_data = await self.redis.get(f"meeting:{meeting_id}")
        if meeting_data:
            try:
                data = json.loads(meeting_data)
                meeting = Meeting.from_dict(data)
                
                # Add to appropriate collection
                if meeting.status == MeetingStatus.ACTIVE:
                    self.active_meetings[meeting_id] = meeting
                elif meeting.status == MeetingStatus.SCHEDULED:
                    self.scheduled_meetings[meeting_id] = meeting
                
                return meeting
            except Exception as e:
                logging.error(f"Error loading meeting from Redis: {e}")
        
        return None
    
    async def get_user_meetings(self, user_id: str, include_ended: bool = False) -> List[Meeting]:
        """Get all meetings for a user."""
        meetings = []
        
        # Check active meetings
        for meeting in self.active_meetings.values():
            if user_id in meeting.participants:
                meetings.append(meeting)
        
        # Check scheduled meetings
        for meeting in self.scheduled_meetings.values():
            if user_id in meeting.participants:
                meetings.append(meeting)
        
        # Load from Redis if needed
        if include_ended:
            meeting_keys = await self.redis.keys(f"meeting:*")
            for key in meeting_keys:
                meeting_data = await self.redis.get(key)
                if meeting_data:
                    try:
                        data = json.loads(meeting_data)
                        meeting = Meeting.from_dict(data)
                        if (user_id in meeting.participants and 
                            meeting.meeting_id not in [m.meeting_id for m in meetings]):
                            meetings.append(meeting)
                    except Exception as e:
                        logging.error(f"Error loading meeting from Redis: {e}")
        
        # Sort by scheduled start time
        meetings.sort(key=lambda m: m.scheduled_start)
        return meetings
    
    async def _can_start_meeting(self, meeting: Meeting, user_id: str) -> bool:
        """Check if user can start a meeting."""
        if user_id == meeting.host_user_id:
            return True
        
        participant = meeting.participants.get(user_id)
        if participant and participant.role in [ParticipantRole.HOST, ParticipantRole.CO_HOST]:
            return True
        
        return False
    
    async def _can_end_meeting(self, meeting: Meeting, user_id: str) -> bool:
        """Check if user can end a meeting."""
        if user_id == meeting.host_user_id:
            return True
        
        participant = meeting.participants.get(user_id)
        if participant and participant.role in [ParticipantRole.HOST, ParticipantRole.CO_HOST]:
            return True
        
        return False
    
    async def _can_update_settings(self, meeting: Meeting, user_id: str) -> bool:
        """Check if user can update meeting settings."""
        if user_id == meeting.host_user_id:
            return True
        
        participant = meeting.participants.get(user_id)
        if participant and participant.role in [ParticipantRole.HOST, ParticipantRole.CO_HOST]:
            return True
        
        return False
    
    async def _store_meeting(self, meeting: Meeting):
        """Store meeting in Redis."""
        try:
            data = json.dumps(meeting.to_dict())
            await self.redis.set(f"meeting:{meeting.meeting_id}", data)
            await self.redis.expire(f"meeting:{meeting.meeting_id}", 86400 * 7)  # 7 days
        except Exception as e:
            logging.error(f"Error storing meeting: {e}")
    
    async def _create_event(self, meeting_id: str, event_type: str, user_id: str, data: Dict[str, Any]):
        """Create a meeting event."""
        event = MeetingEvent(
            event_id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            event_type=event_type,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            data=data
        )
        
        # Store event
        try:
            event_data = json.dumps(event.to_dict())
            await self.redis.lpush(f"meeting_events:{meeting_id}", event_data)
            await self.redis.ltrim(f"meeting_events:{meeting_id}", 0, 999)  # Keep last 1000 events
        except Exception as e:
            logging.error(f"Error storing meeting event: {e}")
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logging.error(f"Error in meeting event callback: {e}")
    
    async def _load_scheduled_meetings(self):
        """Load scheduled meetings from Redis."""
        try:
            meeting_keys = await self.redis.keys("meeting:*")
            for key in meeting_keys:
                meeting_data = await self.redis.get(key)
                if meeting_data:
                    try:
                        data = json.loads(meeting_data)
                        meeting = Meeting.from_dict(data)
                        
                        if meeting.status == MeetingStatus.SCHEDULED:
                            self.scheduled_meetings[meeting.meeting_id] = meeting
                        elif meeting.status == MeetingStatus.ACTIVE:
                            self.active_meetings[meeting.meeting_id] = meeting
                    except Exception as e:
                        logging.error(f"Error loading meeting: {e}")
        except Exception as e:
            logging.error(f"Error loading scheduled meetings: {e}")
    
    async def _auto_cleanup_loop(self):
        """Periodic cleanup of ended meetings."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_ended_meetings()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in meeting cleanup: {e}")
    
    async def _cleanup_ended_meetings(self):
        """Clean up ended meetings."""
        # Auto-end meetings that are past their scheduled end time
        current_time = datetime.utcnow()
        meetings_to_end = []
        
        for meeting_id, meeting in list(self.active_meetings.items()):
            # End meetings that are 30 minutes past scheduled end
            if (meeting.scheduled_end + timedelta(minutes=30)) < current_time:
                meetings_to_end.append((meeting_id, meeting.host_user_id))
        
        for meeting_id, host_user_id in meetings_to_end:
            await self.end_meeting(meeting_id, host_user_id)
            logging.info(f"Auto-ended meeting {meeting_id}")
    
    def get_meeting_stats(self) -> Dict[str, Any]:
        """Get meeting statistics."""
        total_active = len(self.active_meetings)
        total_scheduled = len(self.scheduled_meetings)
        total_participants = sum(len(m.get_active_participants()) for m in self.active_meetings.values())
        
        return {
            'active_meetings': total_active,
            'scheduled_meetings': total_scheduled,
            'total_participants': total_participants,
            'active_meeting_details': [m.to_dict() for m in self.active_meetings.values()]
        }


# Global instance
meeting_manager = MeetingManager()