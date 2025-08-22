"""
Recording service for video conferences with multiple format support.
"""
import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import subprocess
import tempfile

import redis.asyncio as redis
import boto3
from aiortc.contrib.media import MediaRecorder
import ffmpeg


class RecordingFormat(Enum):
    MP4 = "mp4"
    WEBM = "webm"
    MKV = "mkv"
    MOV = "mov"


class RecordingQuality(Enum):
    LOW = "low"      # 480p, 1Mbps
    MEDIUM = "medium"  # 720p, 2Mbps
    HIGH = "high"     # 1080p, 4Mbps
    AUTO = "auto"     # Adaptive based on bandwidth


class RecordingStatus(Enum):
    PENDING = "pending"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class RecordingLayout(Enum):
    GALLERY = "gallery"      # Grid layout with all participants
    SPEAKER = "speaker"      # Focus on active speaker
    SCREEN_SHARE = "screen_share"  # Focus on shared screen
    CUSTOM = "custom"        # Custom layout configuration


@dataclass
class RecordingSettings:
    """Recording configuration settings."""
    format: RecordingFormat = RecordingFormat.MP4
    quality: RecordingQuality = RecordingQuality.HIGH
    layout: RecordingLayout = RecordingLayout.GALLERY
    include_audio: bool = True
    include_video: bool = True
    include_screen_share: bool = True
    include_chat: bool = True
    separate_audio_tracks: bool = False
    auto_upload_to_cloud: bool = True
    retention_days: int = 90
    password_protected: bool = False
    password: Optional[str] = None
    custom_layout_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'format': self.format.value,
            'quality': self.quality.value,
            'layout': self.layout.value,
            'include_audio': self.include_audio,
            'include_video': self.include_video,
            'include_screen_share': self.include_screen_share,
            'include_chat': self.include_chat,
            'separate_audio_tracks': self.separate_audio_tracks,
            'auto_upload_to_cloud': self.auto_upload_to_cloud,
            'retention_days': self.retention_days,
            'password_protected': self.password_protected,
            'password': self.password,
            'custom_layout_config': self.custom_layout_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecordingSettings':
        return cls(
            format=RecordingFormat(data.get('format', 'mp4')),
            quality=RecordingQuality(data.get('quality', 'high')),
            layout=RecordingLayout(data.get('layout', 'gallery')),
            include_audio=data.get('include_audio', True),
            include_video=data.get('include_video', True),
            include_screen_share=data.get('include_screen_share', True),
            include_chat=data.get('include_chat', True),
            separate_audio_tracks=data.get('separate_audio_tracks', False),
            auto_upload_to_cloud=data.get('auto_upload_to_cloud', True),
            retention_days=data.get('retention_days', 90),
            password_protected=data.get('password_protected', False),
            password=data.get('password'),
            custom_layout_config=data.get('custom_layout_config', {})
        )


@dataclass
class Recording:
    """Recording information and metadata."""
    recording_id: str
    meeting_id: str
    title: str
    started_by: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RecordingStatus = RecordingStatus.PENDING
    settings: RecordingSettings = field(default_factory=RecordingSettings)
    file_size: int = 0  # bytes
    duration: int = 0  # seconds
    local_file_path: Optional[str] = None
    cloud_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    download_url: Optional[str] = None
    processing_progress: int = 0  # percentage
    error_message: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    chat_transcript: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string."""
        if self.duration <= 0:
            return "00:00:00"
        
        hours = self.duration // 3600
        minutes = (self.duration % 3600) // 60
        seconds = self.duration % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recording_id': self.recording_id,
            'meeting_id': self.meeting_id,
            'title': self.title,
            'started_by': self.started_by,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'settings': self.settings.to_dict(),
            'file_size': self.file_size,
            'duration': self.duration,
            'duration_formatted': self.get_duration_formatted(),
            'local_file_path': self.local_file_path,
            'cloud_url': self.cloud_url,
            'thumbnail_url': self.thumbnail_url,
            'download_url': self.download_url,
            'processing_progress': self.processing_progress,
            'error_message': self.error_message,
            'participants': self.participants,
            'chat_transcript': self.chat_transcript,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recording':
        return cls(
            recording_id=data['recording_id'],
            meeting_id=data['meeting_id'],
            title=data['title'],
            started_by=data['started_by'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            status=RecordingStatus(data['status']),
            settings=RecordingSettings.from_dict(data.get('settings', {})),
            file_size=data.get('file_size', 0),
            duration=data.get('duration', 0),
            local_file_path=data.get('local_file_path'),
            cloud_url=data.get('cloud_url'),
            thumbnail_url=data.get('thumbnail_url'),
            download_url=data.get('download_url'),
            processing_progress=data.get('processing_progress', 0),
            error_message=data.get('error_message'),
            participants=data.get('participants', []),
            chat_transcript=data.get('chat_transcript', []),
            metadata=data.get('metadata', {})
        )


class RecordingService:
    """Service for managing video conference recordings."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 storage_path: str = "/tmp/recordings",
                 aws_bucket: Optional[str] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.storage_path = storage_path
        self.aws_bucket = aws_bucket
        self.active_recordings: Dict[str, Recording] = {}
        self.media_recorders: Dict[str, MediaRecorder] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.s3_client = None
        
        # Initialize AWS S3 client if bucket is provided
        if aws_bucket:
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                logging.warning(f"Failed to initialize S3 client: {e}")
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        self.processing_task = None
        
    async def start(self):
        """Start the recording service."""
        self.processing_task = asyncio.create_task(self._processing_loop())
        logging.info("Recording service started")
    
    async def stop(self):
        """Stop the recording service."""
        # Stop all active recordings
        for recording_id in list(self.active_recordings.keys()):
            await self.stop_recording(recording_id)
        
        # Stop processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.close()
        logging.info("Recording service stopped")
    
    async def start_recording(self, meeting_id: str, title: str, started_by: str,
                            settings: Optional[RecordingSettings] = None,
                            participants: Optional[List[str]] = None) -> Recording:
        """Start recording a meeting."""
        recording_id = str(uuid.uuid4())
        
        recording = Recording(
            recording_id=recording_id,
            meeting_id=meeting_id,
            title=title,
            started_by=started_by,
            start_time=datetime.utcnow(),
            settings=settings or RecordingSettings(),
            participants=participants or []
        )
        
        # Generate file path
        timestamp = recording.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{meeting_id}_{timestamp}.{recording.settings.format.value}"
        recording.local_file_path = os.path.join(self.storage_path, filename)
        
        # Initialize media recorder
        try:
            # Create media recorder based on settings
            recorder_options = self._get_recorder_options(recording.settings)
            
            # Note: In a real implementation, you would set up the actual media streams
            # For now, we'll create a placeholder recorder
            media_recorder = MediaRecorder(recording.local_file_path, format=recording.settings.format.value)
            self.media_recorders[recording_id] = media_recorder
            
            # Start recording
            recording.status = RecordingStatus.RECORDING
            self.active_recordings[recording_id] = recording
            
            # Store in Redis
            await self._store_recording(recording)
            
            logging.info(f"Started recording: {title} ({recording_id})")
            
        except Exception as e:
            recording.status = RecordingStatus.FAILED
            recording.error_message = str(e)
            await self._store_recording(recording)
            logging.error(f"Failed to start recording: {e}")
        
        return recording
    
    async def stop_recording(self, recording_id: str) -> Optional[Recording]:
        """Stop an active recording."""
        if recording_id not in self.active_recordings:
            return None
        
        recording = self.active_recordings[recording_id]
        
        try:
            # Stop media recorder
            if recording_id in self.media_recorders:
                media_recorder = self.media_recorders[recording_id]
                await media_recorder.stop()
                del self.media_recorders[recording_id]
            
            # Update recording info
            recording.end_time = datetime.utcnow()
            recording.status = RecordingStatus.PROCESSING
            
            # Calculate duration
            if recording.start_time and recording.end_time:
                duration = recording.end_time - recording.start_time
                recording.duration = int(duration.total_seconds())
            
            # Get file size
            if recording.local_file_path and os.path.exists(recording.local_file_path):
                recording.file_size = os.path.getsize(recording.local_file_path)
            
            # Remove from active recordings
            del self.active_recordings[recording_id]
            
            # Add to processing queue
            await self.processing_queue.put(recording)
            
            # Store updated recording
            await self._store_recording(recording)
            
            logging.info(f"Stopped recording: {recording.title} ({recording_id})")
            
        except Exception as e:
            recording.status = RecordingStatus.FAILED
            recording.error_message = str(e)
            await self._store_recording(recording)
            logging.error(f"Failed to stop recording: {e}")
        
        return recording
    
    async def get_recording(self, recording_id: str) -> Optional[Recording]:
        """Get recording by ID."""
        # Check active recordings first
        if recording_id in self.active_recordings:
            return self.active_recordings[recording_id]
        
        # Load from Redis
        recording_data = await self.redis.get(f"recording:{recording_id}")
        if recording_data:
            try:
                data = json.loads(recording_data)
                return Recording.from_dict(data)
            except Exception as e:
                logging.error(f"Error loading recording from Redis: {e}")
        
        return None
    
    async def get_meeting_recordings(self, meeting_id: str) -> List[Recording]:
        """Get all recordings for a meeting."""
        recordings = []
        
        # Check active recordings
        for recording in self.active_recordings.values():
            if recording.meeting_id == meeting_id:
                recordings.append(recording)
        
        # Load from Redis
        recording_keys = await self.redis.keys(f"recording:*")
        for key in recording_keys:
            recording_data = await self.redis.get(key)
            if recording_data:
                try:
                    data = json.loads(recording_data)
                    recording = Recording.from_dict(data)
                    if (recording.meeting_id == meeting_id and 
                        recording.recording_id not in [r.recording_id for r in recordings]):
                        recordings.append(recording)
                except Exception as e:
                    logging.error(f"Error loading recording: {e}")
        
        # Sort by start time
        recordings.sort(key=lambda r: r.start_time)
        return recordings
    
    async def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording."""
        recording = await self.get_recording(recording_id)
        if not recording:
            return False
        
        try:
            # Delete local file
            if recording.local_file_path and os.path.exists(recording.local_file_path):
                os.remove(recording.local_file_path)
            
            # Delete thumbnail
            if recording.thumbnail_url and recording.thumbnail_url.startswith('file://'):
                thumbnail_path = recording.thumbnail_url.replace('file://', '')
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
            
            # Delete from cloud storage
            if recording.cloud_url and self.s3_client and self.aws_bucket:
                # Extract S3 key from URL
                s3_key = recording.cloud_url.split('/')[-1]
                self.s3_client.delete_object(Bucket=self.aws_bucket, Key=s3_key)
            
            # Update status
            recording.status = RecordingStatus.DELETED
            await self._store_recording(recording)
            
            logging.info(f"Deleted recording: {recording_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting recording: {e}")
            return False
    
    async def generate_download_url(self, recording_id: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a temporary download URL for a recording."""
        recording = await self.get_recording(recording_id)
        if not recording or recording.status != RecordingStatus.COMPLETED:
            return None
        
        try:
            if recording.cloud_url and self.s3_client and self.aws_bucket:
                # Generate presigned URL for S3
                s3_key = recording.cloud_url.split('/')[-1]
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.aws_bucket, 'Key': s3_key},
                    ExpiresIn=expires_in
                )
                return url
            elif recording.local_file_path and os.path.exists(recording.local_file_path):
                # For local files, return a file URL (in production, you'd serve through a web server)
                return f"file://{recording.local_file_path}"
            
        except Exception as e:
            logging.error(f"Error generating download URL: {e}")
        
        return None
    
    def _get_recorder_options(self, settings: RecordingSettings) -> Dict[str, Any]:
        """Get recorder options based on settings."""
        options = {}
        
        # Video settings
        if settings.include_video:
            if settings.quality == RecordingQuality.LOW:
                options['video_size'] = '854x480'
                options['video_bitrate'] = '1M'
            elif settings.quality == RecordingQuality.MEDIUM:
                options['video_size'] = '1280x720'
                options['video_bitrate'] = '2M'
            elif settings.quality == RecordingQuality.HIGH:
                options['video_size'] = '1920x1080'
                options['video_bitrate'] = '4M'
        
        # Audio settings
        if settings.include_audio:
            options['audio_bitrate'] = '128k'
            options['audio_codec'] = 'aac'
        
        return options
    
    async def _process_recording(self, recording: Recording):
        """Process a completed recording."""
        try:
            recording.processing_progress = 10
            await self._store_recording(recording)
            
            # Generate thumbnail
            await self._generate_thumbnail(recording)
            recording.processing_progress = 30
            await self._store_recording(recording)
            
            # Optimize video if needed
            await self._optimize_video(recording)
            recording.processing_progress = 70
            await self._store_recording(recording)
            
            # Upload to cloud storage
            if recording.settings.auto_upload_to_cloud:
                await self._upload_to_cloud(recording)
            
            recording.processing_progress = 100
            recording.status = RecordingStatus.COMPLETED
            
            # Generate download URL
            recording.download_url = await self.generate_download_url(recording.recording_id)
            
            await self._store_recording(recording)
            
            logging.info(f"Completed processing recording: {recording.recording_id}")
            
        except Exception as e:
            recording.status = RecordingStatus.FAILED
            recording.error_message = str(e)
            await self._store_recording(recording)
            logging.error(f"Error processing recording: {e}")
    
    async def _generate_thumbnail(self, recording: Recording):
        """Generate thumbnail for recording."""
        if not recording.local_file_path or not os.path.exists(recording.local_file_path):
            return
        
        try:
            # Generate thumbnail at 10% of video duration
            duration = recording.duration or 60
            timestamp = max(1, int(duration * 0.1))
            
            thumbnail_path = recording.local_file_path.replace(
                f".{recording.settings.format.value}", "_thumb.jpg"
            )
            
            # Use ffmpeg to extract thumbnail
            (
                ffmpeg
                .input(recording.local_file_path, ss=timestamp)
                .output(thumbnail_path, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            if os.path.exists(thumbnail_path):
                recording.thumbnail_url = f"file://{thumbnail_path}"
            
        except Exception as e:
            logging.warning(f"Failed to generate thumbnail: {e}")
    
    async def _optimize_video(self, recording: Recording):
        """Optimize video file for web delivery."""
        if not recording.local_file_path or not os.path.exists(recording.local_file_path):
            return
        
        try:
            # Create optimized version
            optimized_path = recording.local_file_path.replace(
                f".{recording.settings.format.value}", 
                f"_optimized.{recording.settings.format.value}"
            )
            
            # Use ffmpeg for optimization
            (
                ffmpeg
                .input(recording.local_file_path)
                .output(
                    optimized_path,
                    vcodec='libx264',
                    acodec='aac',
                    preset='medium',
                    crf=23,
                    movflags='faststart'  # Enable progressive download
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Replace original with optimized version
            if os.path.exists(optimized_path):
                os.replace(optimized_path, recording.local_file_path)
                recording.file_size = os.path.getsize(recording.local_file_path)
            
        except Exception as e:
            logging.warning(f"Failed to optimize video: {e}")
    
    async def _upload_to_cloud(self, recording: Recording):
        """Upload recording to cloud storage."""
        if not self.s3_client or not self.aws_bucket or not recording.local_file_path:
            return
        
        try:
            # Generate S3 key
            timestamp = recording.start_time.strftime("%Y/%m/%d")
            s3_key = f"recordings/{timestamp}/{recording.recording_id}.{recording.settings.format.value}"
            
            # Upload file
            self.s3_client.upload_file(
                recording.local_file_path,
                self.aws_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': f'video/{recording.settings.format.value}',
                    'Metadata': {
                        'meeting_id': recording.meeting_id,
                        'recording_id': recording.recording_id,
                        'started_by': recording.started_by
                    }
                }
            )
            
            # Set cloud URL
            recording.cloud_url = f"https://{self.aws_bucket}.s3.amazonaws.com/{s3_key}"
            
            # Upload thumbnail if exists
            if recording.thumbnail_url and recording.thumbnail_url.startswith('file://'):
                thumbnail_path = recording.thumbnail_url.replace('file://', '')
                if os.path.exists(thumbnail_path):
                    thumbnail_s3_key = s3_key.replace(f".{recording.settings.format.value}", "_thumb.jpg")
                    self.s3_client.upload_file(
                        thumbnail_path,
                        self.aws_bucket,
                        thumbnail_s3_key,
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )
                    recording.thumbnail_url = f"https://{self.aws_bucket}.s3.amazonaws.com/{thumbnail_s3_key}"
            
            logging.info(f"Uploaded recording to S3: {s3_key}")
            
        except Exception as e:
            logging.error(f"Failed to upload to cloud: {e}")
            raise
    
    async def _store_recording(self, recording: Recording):
        """Store recording in Redis."""
        try:
            data = json.dumps(recording.to_dict())
            await self.redis.set(f"recording:{recording.recording_id}", data)
            await self.redis.expire(f"recording:{recording.recording_id}", 86400 * recording.settings.retention_days)
        except Exception as e:
            logging.error(f"Error storing recording: {e}")
    
    async def _processing_loop(self):
        """Main processing loop for recordings."""
        while True:
            try:
                # Wait for recording to process
                recording = await self.processing_queue.get()
                await self._process_recording(recording)
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in recording processing loop: {e}")
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get recording service statistics."""
        return {
            'active_recordings': len(self.active_recordings),
            'pending_processing': self.processing_queue.qsize(),
            'storage_path': self.storage_path,
            'cloud_storage_enabled': self.s3_client is not None,
            'aws_bucket': self.aws_bucket
        }


# Global instance
recording_service = RecordingService()