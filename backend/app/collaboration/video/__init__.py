"""
Video conferencing system for real-time collaboration.

This module provides comprehensive video conferencing features including:
- WebRTC peer-to-peer video/audio communication
- Screen sharing with annotation overlay
- Recording capabilities
- Virtual backgrounds
- Breakout rooms for sub-discussions
- Meeting transcription and notes
"""

from .webrtc_server import WebRTCServer
from .meeting_manager import MeetingManager
from .recording_service import RecordingService
from .transcription_service import TranscriptionService
from .breakout_rooms import BreakoutRoomManager

__all__ = [
    'WebRTCServer',
    'MeetingManager', 
    'RecordingService',
    'TranscriptionService',
    'BreakoutRoomManager'
]