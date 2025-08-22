"""
Transcription service for real-time speech-to-text and meeting notes.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import io
import tempfile
import os

import redis.asyncio as redis
import speech_recognition as sr
from pydub import AudioSegment
import openai
import boto3


class TranscriptionProvider(Enum):
    GOOGLE = "google"
    AWS_TRANSCRIBE = "aws_transcribe"
    AZURE = "azure"
    OPENAI_WHISPER = "openai_whisper"
    LOCAL_WHISPER = "local_whisper"


class TranscriptionLanguage(Enum):
    ENGLISH = "en-US"
    SPANISH = "es-ES"
    FRENCH = "fr-FR"
    GERMAN = "de-DE"
    ITALIAN = "it-IT"
    PORTUGUESE = "pt-BR"
    CHINESE = "zh-CN"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"
    AUTO_DETECT = "auto"


class SpeakerRole(Enum):
    HOST = "host"
    PARTICIPANT = "participant"
    PRESENTER = "presenter"
    UNKNOWN = "unknown"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with metadata."""
    segment_id: str
    speaker_id: Optional[str]
    speaker_name: Optional[str]
    speaker_role: SpeakerRole
    text: str
    confidence: float
    start_time: float  # seconds from start of recording
    end_time: float
    language: str
    is_final: bool = True
    alternatives: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # positive, negative, neutral
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'speaker_id': self.speaker_id,
            'speaker_name': self.speaker_name,
            'speaker_role': self.speaker_role.value,
            'text': self.text,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'language': self.language,
            'is_final': self.is_final,
            'alternatives': self.alternatives,
            'keywords': self.keywords,
            'sentiment': self.sentiment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSegment':
        return cls(
            segment_id=data['segment_id'],
            speaker_id=data.get('speaker_id'),
            speaker_name=data.get('speaker_name'),
            speaker_role=SpeakerRole(data.get('speaker_role', 'unknown')),
            text=data['text'],
            confidence=data['confidence'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            language=data['language'],
            is_final=data.get('is_final', True),
            alternatives=data.get('alternatives', []),
            keywords=data.get('keywords', []),
            sentiment=data.get('sentiment')
        )


@dataclass
class TranscriptionSettings:
    """Transcription configuration settings."""
    provider: TranscriptionProvider = TranscriptionProvider.OPENAI_WHISPER
    language: TranscriptionLanguage = TranscriptionLanguage.AUTO_DETECT
    enable_speaker_diarization: bool = True
    enable_punctuation: bool = True
    enable_word_timestamps: bool = True
    enable_profanity_filter: bool = False
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    real_time_transcription: bool = True
    save_audio_segments: bool = False
    custom_vocabulary: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider.value,
            'language': self.language.value,
            'enable_speaker_diarization': self.enable_speaker_diarization,
            'enable_punctuation': self.enable_punctuation,
            'enable_word_timestamps': self.enable_word_timestamps,
            'enable_profanity_filter': self.enable_profanity_filter,
            'enable_sentiment_analysis': self.enable_sentiment_analysis,
            'enable_keyword_extraction': self.enable_keyword_extraction,
            'real_time_transcription': self.real_time_transcription,
            'save_audio_segments': self.save_audio_segments,
            'custom_vocabulary': self.custom_vocabulary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSettings':
        return cls(
            provider=TranscriptionProvider(data.get('provider', 'openai_whisper')),
            language=TranscriptionLanguage(data.get('language', 'auto')),
            enable_speaker_diarization=data.get('enable_speaker_diarization', True),
            enable_punctuation=data.get('enable_punctuation', True),
            enable_word_timestamps=data.get('enable_word_timestamps', True),
            enable_profanity_filter=data.get('enable_profanity_filter', False),
            enable_sentiment_analysis=data.get('enable_sentiment_analysis', True),
            enable_keyword_extraction=data.get('enable_keyword_extraction', True),
            real_time_transcription=data.get('real_time_transcription', True),
            save_audio_segments=data.get('save_audio_segments', False),
            custom_vocabulary=data.get('custom_vocabulary', [])
        )


@dataclass
class MeetingTranscript:
    """Complete transcript for a meeting."""
    transcript_id: str
    meeting_id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    settings: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    segments: List[TranscriptionSegment] = field(default_factory=list)
    speakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # speaker_id -> info
    summary: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    key_topics: List[str] = field(default_factory=list)
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    speaking_time: Dict[str, float] = field(default_factory=dict)  # speaker_id -> seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> timedelta:
        """Get transcript duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.utcnow() - self.start_time
    
    def get_full_text(self) -> str:
        """Get full transcript text."""
        return " ".join(segment.text for segment in self.segments if segment.is_final)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transcript_id': self.transcript_id,
            'meeting_id': self.meeting_id,
            'title': self.title,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'settings': self.settings.to_dict(),
            'segments': [segment.to_dict() for segment in self.segments],
            'speakers': self.speakers,
            'summary': self.summary,
            'action_items': self.action_items,
            'key_topics': self.key_topics,
            'sentiment_analysis': self.sentiment_analysis,
            'word_count': self.word_count,
            'speaking_time': self.speaking_time,
            'metadata': self.metadata,
            'duration_minutes': self.get_duration().total_seconds() / 60,
            'full_text': self.get_full_text()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeetingTranscript':
        return cls(
            transcript_id=data['transcript_id'],
            meeting_id=data['meeting_id'],
            title=data['title'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            settings=TranscriptionSettings.from_dict(data.get('settings', {})),
            segments=[TranscriptionSegment.from_dict(seg) for seg in data.get('segments', [])],
            speakers=data.get('speakers', {}),
            summary=data.get('summary'),
            action_items=data.get('action_items', []),
            key_topics=data.get('key_topics', []),
            sentiment_analysis=data.get('sentiment_analysis', {}),
            word_count=data.get('word_count', 0),
            speaking_time=data.get('speaking_time', {}),
            metadata=data.get('metadata', {})
        )


class TranscriptionService:
    """Service for real-time transcription and meeting analysis."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.active_transcripts: Dict[str, MeetingTranscript] = {}
        self.audio_processors: Dict[str, sr.Recognizer] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize AI services
        self.openai_client = None
        self.aws_transcribe = None
        
        try:
            # Initialize OpenAI (for Whisper and GPT)
            self.openai_client = openai.OpenAI()
        except Exception as e:
            logging.warning(f"OpenAI client not initialized: {e}")
        
        try:
            # Initialize AWS Transcribe
            self.aws_transcribe = boto3.client('transcribe')
        except Exception as e:
            logging.warning(f"AWS Transcribe client not initialized: {e}")
        
        self.processing_task = None
        
    async def start(self):
        """Start the transcription service."""
        self.processing_task = asyncio.create_task(self._processing_loop())
        logging.info("Transcription service started")
    
    async def stop(self):
        """Stop the transcription service."""
        # Stop all active transcriptions
        for transcript_id in list(self.active_transcripts.keys()):
            await self.stop_transcription(transcript_id)
        
        # Stop processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.close()
        logging.info("Transcription service stopped")
    
    async def start_transcription(self, meeting_id: str, title: str,
                                settings: Optional[TranscriptionSettings] = None) -> MeetingTranscript:
        """Start transcribing a meeting."""
        transcript_id = str(uuid.uuid4())
        
        transcript = MeetingTranscript(
            transcript_id=transcript_id,
            meeting_id=meeting_id,
            title=title,
            start_time=datetime.utcnow(),
            settings=settings or TranscriptionSettings()
        )
        
        self.active_transcripts[transcript_id] = transcript
        
        # Initialize audio processor
        if transcript.settings.real_time_transcription:
            self.audio_processors[transcript_id] = sr.Recognizer()
        
        # Store in Redis
        await self._store_transcript(transcript)
        
        logging.info(f"Started transcription: {title} ({transcript_id})")
        return transcript
    
    async def stop_transcription(self, transcript_id: str) -> Optional[MeetingTranscript]:
        """Stop transcribing a meeting."""
        if transcript_id not in self.active_transcripts:
            return None
        
        transcript = self.active_transcripts[transcript_id]
        transcript.end_time = datetime.utcnow()
        
        # Calculate statistics
        transcript.word_count = len(transcript.get_full_text().split())
        
        # Calculate speaking time for each speaker
        for segment in transcript.segments:
            if segment.speaker_id:
                duration = segment.end_time - segment.start_time
                if segment.speaker_id not in transcript.speaking_time:
                    transcript.speaking_time[segment.speaker_id] = 0
                transcript.speaking_time[segment.speaker_id] += duration
        
        # Remove from active transcripts
        del self.active_transcripts[transcript_id]
        
        # Clean up audio processor
        if transcript_id in self.audio_processors:
            del self.audio_processors[transcript_id]
        
        # Add to processing queue for analysis
        await self.processing_queue.put(transcript)
        
        # Store updated transcript
        await self._store_transcript(transcript)
        
        logging.info(f"Stopped transcription: {transcript.title} ({transcript_id})")
        return transcript
    
    async def add_audio_chunk(self, transcript_id: str, audio_data: bytes,
                            speaker_id: Optional[str] = None, speaker_name: Optional[str] = None) -> Optional[TranscriptionSegment]:
        """Process an audio chunk for real-time transcription."""
        if transcript_id not in self.active_transcripts:
            return None
        
        transcript = self.active_transcripts[transcript_id]
        
        try:
            # Convert audio data to the format expected by speech recognition
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(audio_data),
                sample_width=2,
                frame_rate=16000,
                channels=1
            )
            
            # Convert to wav format for speech recognition
            wav_data = io.BytesIO()
            audio_segment.export(wav_data, format="wav")
            wav_data.seek(0)
            
            # Perform speech recognition
            with sr.AudioFile(wav_data) as source:
                audio = self.recognizer.record(source)
            
            # Use the configured provider
            text, confidence = await self._transcribe_audio(audio, transcript.settings)
            
            if text:
                # Create transcription segment
                current_time = (datetime.utcnow() - transcript.start_time).total_seconds()
                
                segment = TranscriptionSegment(
                    segment_id=str(uuid.uuid4()),
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    speaker_role=SpeakerRole.PARTICIPANT,
                    text=text,
                    confidence=confidence,
                    start_time=current_time - (len(text.split()) * 0.5),  # Rough estimate
                    end_time=current_time,
                    language=transcript.settings.language.value,
                    is_final=True
                )
                
                # Add speaker diarization if enabled
                if transcript.settings.enable_speaker_diarization and speaker_id:
                    if speaker_id not in transcript.speakers:
                        transcript.speakers[speaker_id] = {
                            'name': speaker_name or f"Speaker {len(transcript.speakers) + 1}",
                            'role': SpeakerRole.PARTICIPANT.value,
                            'first_spoke_at': current_time
                        }
                
                # Add sentiment analysis if enabled
                if transcript.settings.enable_sentiment_analysis:
                    segment.sentiment = await self._analyze_sentiment(text)
                
                # Add keyword extraction if enabled
                if transcript.settings.enable_keyword_extraction:
                    segment.keywords = await self._extract_keywords(text)
                
                transcript.segments.append(segment)
                
                # Store updated transcript
                await self._store_transcript(transcript)
                
                return segment
                
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")
        
        return None
    
    async def get_transcript(self, transcript_id: str) -> Optional[MeetingTranscript]:
        """Get transcript by ID."""
        # Check active transcripts first
        if transcript_id in self.active_transcripts:
            return self.active_transcripts[transcript_id]
        
        # Load from Redis
        transcript_data = await self.redis.get(f"transcript:{transcript_id}")
        if transcript_data:
            try:
                data = json.loads(transcript_data)
                return MeetingTranscript.from_dict(data)
            except Exception as e:
                logging.error(f"Error loading transcript from Redis: {e}")
        
        return None
    
    async def get_meeting_transcripts(self, meeting_id: str) -> List[MeetingTranscript]:
        """Get all transcripts for a meeting."""
        transcripts = []
        
        # Check active transcripts
        for transcript in self.active_transcripts.values():
            if transcript.meeting_id == meeting_id:
                transcripts.append(transcript)
        
        # Load from Redis
        transcript_keys = await self.redis.keys(f"transcript:*")
        for key in transcript_keys:
            transcript_data = await self.redis.get(key)
            if transcript_data:
                try:
                    data = json.loads(transcript_data)
                    transcript = MeetingTranscript.from_dict(data)
                    if (transcript.meeting_id == meeting_id and 
                        transcript.transcript_id not in [t.transcript_id for t in transcripts]):
                        transcripts.append(transcript)
                except Exception as e:
                    logging.error(f"Error loading transcript: {e}")
        
        # Sort by start time
        transcripts.sort(key=lambda t: t.start_time)
        return transcripts
    
    async def search_transcripts(self, query: str, meeting_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search transcripts for specific text."""
        results = []
        
        # Get transcripts to search
        if meeting_id:
            transcripts = await self.get_meeting_transcripts(meeting_id)
        else:
            # Search all transcripts (in production, you might want to limit this)
            transcript_keys = await self.redis.keys("transcript:*")
            transcripts = []
            for key in transcript_keys:
                transcript_data = await self.redis.get(key)
                if transcript_data:
                    try:
                        data = json.loads(transcript_data)
                        transcripts.append(MeetingTranscript.from_dict(data))
                    except Exception as e:
                        continue
        
        # Search through segments
        query_lower = query.lower()
        for transcript in transcripts:
            for segment in transcript.segments:
                if query_lower in segment.text.lower():
                    results.append({
                        'transcript_id': transcript.transcript_id,
                        'meeting_id': transcript.meeting_id,
                        'title': transcript.title,
                        'segment': segment.to_dict(),
                        'context_before': self._get_context_before(transcript, segment),
                        'context_after': self._get_context_after(transcript, segment)
                    })
        
        return results
    
    async def _transcribe_audio(self, audio: sr.AudioData, settings: TranscriptionSettings) -> tuple[str, float]:
        """Transcribe audio using the configured provider."""
        try:
            if settings.provider == TranscriptionProvider.OPENAI_WHISPER and self.openai_client:
                return await self._transcribe_with_whisper(audio, settings)
            elif settings.provider == TranscriptionProvider.GOOGLE:
                return await self._transcribe_with_google(audio, settings)
            elif settings.provider == TranscriptionProvider.AWS_TRANSCRIBE and self.aws_transcribe:
                return await self._transcribe_with_aws(audio, settings)
            else:
                # Fallback to local speech recognition
                return await self._transcribe_with_local(audio, settings)
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return "", 0.0
    
    async def _transcribe_with_whisper(self, audio: sr.AudioData, settings: TranscriptionSettings) -> tuple[str, float]:
        """Transcribe using OpenAI Whisper."""
        try:
            # Convert audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio.get_wav_data())
                temp_file_path = temp_file.name
            
            try:
                # Use OpenAI Whisper API
                with open(temp_file_path, "rb") as audio_file:
                    response = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=settings.language.value if settings.language != TranscriptionLanguage.AUTO_DETECT else None,
                        response_format="verbose_json",
                        timestamp_granularities=["word"] if settings.enable_word_timestamps else ["segment"]
                    )
                
                text = response.text
                confidence = 0.9  # Whisper doesn't provide confidence scores
                
                return text, confidence
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}")
            return "", 0.0
    
    async def _transcribe_with_google(self, audio: sr.AudioData, settings: TranscriptionSettings) -> tuple[str, float]:
        """Transcribe using Google Speech Recognition."""
        try:
            text = self.recognizer.recognize_google(
                audio,
                language=settings.language.value if settings.language != TranscriptionLanguage.AUTO_DETECT else None,
                show_all=False
            )
            return text, 0.8  # Google doesn't provide confidence in free tier
        except sr.UnknownValueError:
            return "", 0.0
        except sr.RequestError as e:
            logging.error(f"Google Speech Recognition error: {e}")
            return "", 0.0
    
    async def _transcribe_with_aws(self, audio: sr.AudioData, settings: TranscriptionSettings) -> tuple[str, float]:
        """Transcribe using AWS Transcribe."""
        # AWS Transcribe implementation would go here
        # This is a placeholder as AWS Transcribe requires streaming setup
        return "", 0.0
    
    async def _transcribe_with_local(self, audio: sr.AudioData, settings: TranscriptionSettings) -> tuple[str, float]:
        """Transcribe using local speech recognition."""
        try:
            text = self.recognizer.recognize_sphinx(audio)
            return text, 0.6  # Sphinx typically has lower accuracy
        except sr.UnknownValueError:
            return "", 0.0
        except sr.RequestError as e:
            logging.error(f"Sphinx recognition error: {e}")
            return "", 0.0
    
    async def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        # Simple keyword-based sentiment analysis
        # In production, you'd use a proper sentiment analysis model
        positive_words = ["good", "great", "excellent", "happy", "pleased", "agree", "yes"]
        negative_words = ["bad", "terrible", "awful", "sad", "angry", "disagree", "no"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction based on word frequency
        # In production, you'd use NLP libraries like spaCy or NLTK
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique keywords
        return list(set(keywords))[:5]  # Return top 5 keywords
    
    async def _generate_summary(self, transcript: MeetingTranscript) -> str:
        """Generate meeting summary using AI."""
        if not self.openai_client:
            return "Summary generation not available (OpenAI client not configured)"
        
        try:
            full_text = transcript.get_full_text()
            if len(full_text) < 100:
                return "Meeting too short to generate summary"
            
            # Use GPT to generate summary
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a meeting assistant. Summarize the following meeting transcript in 2-3 concise paragraphs, highlighting key points, decisions made, and action items."},
                    {"role": "user", "content": full_text[:4000]}  # Limit to avoid token limits
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return "Error generating summary"
    
    async def _extract_action_items(self, transcript: MeetingTranscript) -> List[str]:
        """Extract action items from transcript."""
        if not self.openai_client:
            return []
        
        try:
            full_text = transcript.get_full_text()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract action items from this meeting transcript. Return each action item as a separate line. Only include specific, actionable tasks that were assigned or agreed upon."},
                    {"role": "user", "content": full_text[:4000]}
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            action_items = response.choices[0].message.content.strip().split('\n')
            return [item.strip('- ').strip() for item in action_items if item.strip()]
            
        except Exception as e:
            logging.error(f"Error extracting action items: {e}")
            return []
    
    def _get_context_before(self, transcript: MeetingTranscript, segment: TranscriptionSegment, context_size: int = 2) -> List[str]:
        """Get context segments before the current segment."""
        segment_index = next((i for i, s in enumerate(transcript.segments) if s.segment_id == segment.segment_id), -1)
        if segment_index == -1:
            return []
        
        start_index = max(0, segment_index - context_size)
        return [s.text for s in transcript.segments[start_index:segment_index]]
    
    def _get_context_after(self, transcript: MeetingTranscript, segment: TranscriptionSegment, context_size: int = 2) -> List[str]:
        """Get context segments after the current segment."""
        segment_index = next((i for i, s in enumerate(transcript.segments) if s.segment_id == segment.segment_id), -1)
        if segment_index == -1:
            return []
        
        end_index = min(len(transcript.segments), segment_index + context_size + 1)
        return [s.text for s in transcript.segments[segment_index + 1:end_index]]
    
    async def _store_transcript(self, transcript: MeetingTranscript):
        """Store transcript in Redis."""
        try:
            data = json.dumps(transcript.to_dict())
            await self.redis.set(f"transcript:{transcript.transcript_id}", data)
            await self.redis.expire(f"transcript:{transcript.transcript_id}", 86400 * 90)  # 90 days
        except Exception as e:
            logging.error(f"Error storing transcript: {e}")
    
    async def _processing_loop(self):
        """Main processing loop for completed transcripts."""
        while True:
            try:
                # Wait for transcript to process
                transcript = await self.processing_queue.get()
                
                # Generate summary
                transcript.summary = await self._generate_summary(transcript)
                
                # Extract action items
                transcript.action_items = await self._extract_action_items(transcript)
                
                # Store processed transcript
                await self._store_transcript(transcript)
                
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in transcription processing loop: {e}")
    
    def get_transcription_stats(self) -> Dict[str, Any]:
        """Get transcription service statistics."""
        total_segments = sum(len(t.segments) for t in self.active_transcripts.values())
        
        return {
            'active_transcripts': len(self.active_transcripts),
            'pending_processing': self.processing_queue.qsize(),
            'total_active_segments': total_segments,
            'openai_available': self.openai_client is not None,
            'aws_transcribe_available': self.aws_transcribe is not None
        }


# Global instance
transcription_service = TranscriptionService()