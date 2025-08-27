"""
AI Meeting Assistant for automated summaries and action items.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import re

import redis.asyncio as redis
import openai
from transformers import pipeline, AutoTokenizer, AutoModel


class MeetingPhase(Enum):
    INTRODUCTION = "introduction"
    DISCUSSION = "discussion"
    DECISION_MAKING = "decision_making"
    ACTION_PLANNING = "action_planning"
    WRAP_UP = "wrap_up"


class AssistantCapability(Enum):
    SUMMARIZATION = "summarization"
    ACTION_EXTRACTION = "action_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_TRACKING = "topic_tracking"
    SPEAKER_ANALYTICS = "speaker_analytics"
    DECISION_TRACKING = "decision_tracking"
    FOLLOW_UP_GENERATION = "follow_up_generation"


@dataclass
class MeetingInsight:
    """AI-generated insight about the meeting."""
    insight_id: str
    insight_type: str
    title: str
    content: str
    confidence: float
    timestamp: datetime
    participants_involved: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    priority: str = "medium"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type,
            'title': self.title,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'participants_involved': self.participants_involved,
            'keywords': self.keywords,
            'sentiment': self.sentiment,
            'priority': self.priority
        }


@dataclass
class ActionItem:
    """AI-extracted action item."""
    action_id: str
    description: str
    assignee: Optional[str]
    due_date: Optional[datetime]
    priority: str
    status: str = "pending"
    confidence: float = 0.0
    extracted_from: str = ""  # Source text
    related_decisions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'description': self.description,
            'assignee': self.assignee,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'priority': self.priority,
            'status': self.status,
            'confidence': self.confidence,
            'extracted_from': self.extracted_from,
            'related_decisions': self.related_decisions
        }


@dataclass
class MeetingDecision:
    """AI-identified decision made during meeting."""
    decision_id: str
    description: str
    decision_maker: Optional[str]
    rationale: Optional[str]
    impact_level: str
    confidence: float
    timestamp: datetime
    supporting_participants: List[str] = field(default_factory=list)
    opposing_participants: List[str] = field(default_factory=list)
    related_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'description': self.description,
            'decision_maker': self.decision_maker,
            'rationale': self.rationale,
            'impact_level': self.impact_level,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'supporting_participants': self.supporting_participants,
            'opposing_participants': self.opposing_participants,
            'related_actions': self.related_actions
        }


@dataclass
class MeetingAnalysis:
    """Comprehensive AI analysis of a meeting."""
    analysis_id: str
    meeting_id: str
    generated_at: datetime
    summary: str
    key_topics: List[str]
    action_items: List[ActionItem]
    decisions: List[MeetingDecision]
    insights: List[MeetingInsight]
    participant_analytics: Dict[str, Any]
    sentiment_timeline: List[Dict[str, Any]]
    meeting_effectiveness_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_id': self.analysis_id,
            'meeting_id': self.meeting_id,
            'generated_at': self.generated_at.isoformat(),
            'summary': self.summary,
            'key_topics': self.key_topics,
            'action_items': [item.to_dict() for item in self.action_items],
            'decisions': [decision.to_dict() for decision in self.decisions],
            'insights': [insight.to_dict() for insight in self.insights],
            'participant_analytics': self.participant_analytics,
            'sentiment_timeline': self.sentiment_timeline,
            'meeting_effectiveness_score': self.meeting_effectiveness_score,
            'recommendations': self.recommendations
        }


class MeetingAssistant:
    """AI-powered meeting assistant for real-time analysis and insights."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.openai_client = None
        self.active_meetings: Dict[str, Dict[str, Any]] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize AI models
        try:
            self.openai_client = openai.OpenAI()
            
            # Initialize local NLP models
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
            
        except Exception as e:
            logging.warning(f"Some AI models not available: {e}")
        
        self.processing_task = None
        
    async def start(self):
        """Start the meeting assistant."""
        self.processing_task = asyncio.create_task(self._processing_loop())
        logging.info("Meeting assistant started")
    
    async def stop(self):
        """Stop the meeting assistant."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.close()
        logging.info("Meeting assistant stopped")
    
    async def start_meeting_analysis(self, meeting_id: str, meeting_title: str,
                                   participants: List[str], capabilities: List[AssistantCapability]) -> str:
        """Start AI analysis for a meeting."""
        analysis_session = {
            'meeting_id': meeting_id,
            'meeting_title': meeting_title,
            'participants': participants,
            'capabilities': [cap.value for cap in capabilities],
            'start_time': datetime.utcnow(),
            'transcript_segments': [],
            'real_time_insights': [],
            'current_phase': MeetingPhase.INTRODUCTION,
            'topic_evolution': [],
            'participant_contributions': {p: {'words': 0, 'sentiment_scores': [], 'topics': []} for p in participants},
            'decision_markers': [],
            'action_keywords_detected': []
        }
        
        self.active_meetings[meeting_id] = analysis_session
        
        # Store in Redis
        await self._store_meeting_session(analysis_session)
        
        logging.info(f"Started AI analysis for meeting: {meeting_title}")
        return meeting_id
    
    async def process_transcript_segment(self, meeting_id: str, segment: Dict[str, Any]) -> List[MeetingInsight]:
        """Process a real-time transcript segment and generate insights."""
        if meeting_id not in self.active_meetings:
            return []
        
        session = self.active_meetings[meeting_id]
        session['transcript_segments'].append(segment)
        
        insights = []
        text = segment.get('text', '')
        speaker = segment.get('speaker_name', 'Unknown')
        timestamp = datetime.fromisoformat(segment.get('timestamp', datetime.utcnow().isoformat()))
        
        # Update participant contributions
        if speaker in session['participant_contributions']:
            session['participant_contributions'][speaker]['words'] += len(text.split())
        
        # Real-time sentiment analysis
        if AssistantCapability.SENTIMENT_ANALYSIS.value in session['capabilities']:
            sentiment_insight = await self._analyze_segment_sentiment(text, speaker, timestamp)
            if sentiment_insight:
                insights.append(sentiment_insight)
                session['participant_contributions'][speaker]['sentiment_scores'].append(sentiment_insight.sentiment)
        
        # Topic tracking
        if AssistantCapability.TOPIC_TRACKING.value in session['capabilities']:
            topic_insights = await self._track_topics(text, timestamp, session)
            insights.extend(topic_insights)
        
        # Action item detection
        if AssistantCapability.ACTION_EXTRACTION.value in session['capabilities']:
            action_insights = await self._detect_action_items(text, speaker, timestamp)
            insights.extend(action_insights)
        
        # Decision tracking
        if AssistantCapability.DECISION_TRACKING.value in session['capabilities']:
            decision_insights = await self._detect_decisions(text, speaker, timestamp, session)
            insights.extend(decision_insights)
        
        # Meeting phase detection
        await self._detect_meeting_phase(text, session)
        
        # Store insights
        session['real_time_insights'].extend(insights)
        
        # Store updated session
        await self._store_meeting_session(session)
        
        return insights
    
    async def generate_meeting_summary(self, meeting_id: str) -> Optional[MeetingAnalysis]:
        """Generate comprehensive meeting analysis."""
        if meeting_id not in self.active_meetings:
            # Try to load from Redis
            session_data = await self.redis.get(f"meeting_session:{meeting_id}")
            if not session_data:
                return None
            session = json.loads(session_data)
        else:
            session = self.active_meetings[meeting_id]
        
        # Compile full transcript
        full_transcript = " ".join([seg.get('text', '') for seg in session['transcript_segments']])
        
        if not full_transcript.strip():
            return None
        
        analysis_id = str(uuid.uuid4())
        analysis = MeetingAnalysis(
            analysis_id=analysis_id,
            meeting_id=meeting_id,
            generated_at=datetime.utcnow(),
            summary="",
            key_topics=[],
            action_items=[],
            decisions=[],
            insights=session['real_time_insights'],
            participant_analytics=session['participant_contributions'],
            sentiment_timeline=[],
            meeting_effectiveness_score=0.0,
            recommendations=[]
        )
        
        # Generate summary
        if AssistantCapability.SUMMARIZATION.value in session['capabilities']:
            analysis.summary = await self._generate_summary(full_transcript, session)
        
        # Extract key topics
        analysis.key_topics = await self._extract_key_topics(full_transcript)
        
        # Extract action items
        if AssistantCapability.ACTION_EXTRACTION.value in session['capabilities']:
            analysis.action_items = await self._extract_action_items(full_transcript, session)
        
        # Extract decisions
        if AssistantCapability.DECISION_TRACKING.value in session['capabilities']:
            analysis.decisions = await self._extract_decisions(full_transcript, session)
        
        # Analyze participant engagement
        if AssistantCapability.SPEAKER_ANALYTICS.value in session['capabilities']:
            analysis.participant_analytics = await self._analyze_participant_engagement(session)
        
        # Generate sentiment timeline
        if AssistantCapability.SENTIMENT_ANALYSIS.value in session['capabilities']:
            analysis.sentiment_timeline = await self._generate_sentiment_timeline(session)
        
        # Calculate meeting effectiveness
        analysis.meeting_effectiveness_score = await self._calculate_effectiveness_score(session, analysis)
        
        # Generate recommendations
        analysis.recommendations = await self._generate_recommendations(session, analysis)
        
        # Store analysis
        await self._store_meeting_analysis(analysis)
        
        logging.info(f"Generated meeting analysis for: {meeting_id}")
        return analysis
    
    async def get_real_time_insights(self, meeting_id: str, since: Optional[datetime] = None) -> List[MeetingInsight]:
        """Get real-time insights for an active meeting."""
        if meeting_id not in self.active_meetings:
            return []
        
        session = self.active_meetings[meeting_id]
        insights = session['real_time_insights']
        
        if since:
            insights = [insight for insight in insights if insight.timestamp > since]
        
        return insights
    
    async def suggest_next_steps(self, meeting_id: str) -> List[str]:
        """Suggest next steps based on current meeting state."""
        if meeting_id not in self.active_meetings:
            return []
        
        session = self.active_meetings[meeting_id]
        current_phase = session['current_phase']
        
        suggestions = []
        
        if current_phase == MeetingPhase.INTRODUCTION:
            suggestions = [
                "Consider setting clear agenda items",
                "Ensure all participants have introduced themselves",
                "Review the meeting objectives"
            ]
        elif current_phase == MeetingPhase.DISCUSSION:
            suggestions = [
                "Facilitate balanced participation from all attendees",
                "Document key points being discussed",
                "Consider time-boxing discussion topics"
            ]
        elif current_phase == MeetingPhase.DECISION_MAKING:
            suggestions = [
                "Summarize options before making decisions",
                "Ensure consensus or document dissenting views",
                "Clarify decision criteria"
            ]
        elif current_phase == MeetingPhase.ACTION_PLANNING:
            suggestions = [
                "Assign clear owners to action items",
                "Set specific deadlines",
                "Define success criteria"
            ]
        elif current_phase == MeetingPhase.WRAP_UP:
            suggestions = [
                "Summarize key decisions and action items",
                "Schedule follow-up meetings if needed",
                "Share meeting notes with participants"
            ]
        
        return suggestions
    
    async def _analyze_segment_sentiment(self, text: str, speaker: str, timestamp: datetime) -> Optional[MeetingInsight]:
        """Analyze sentiment of a transcript segment."""
        try:
            if not hasattr(self, 'sentiment_analyzer') or not text.strip():
                return None
            
            result = self.sentiment_analyzer(text)[0]
            sentiment = result['label'].lower()
            confidence = result['score']
            
            # Only report significant sentiment changes or strong emotions
            if confidence > 0.8 and sentiment in ['negative', 'positive']:
                return MeetingInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="sentiment",
                    title=f"{sentiment.title()} sentiment detected from {speaker}",
                    content=f"Speaker expressed {sentiment} sentiment: \"{text[:100]}...\"",
                    confidence=confidence,
                    timestamp=timestamp,
                    participants_involved=[speaker],
                    sentiment=sentiment,
                    priority="medium" if sentiment == "negative" else "low"
                )
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
        
        return None
    
    async def _track_topics(self, text: str, timestamp: datetime, session: Dict[str, Any]) -> List[MeetingInsight]:
        """Track topic evolution during the meeting."""
        insights = []
        
        try:
            if not hasattr(self, 'ner_pipeline'):
                return insights
            
            # Extract named entities as potential topics
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                if entity['entity_group'] in ['ORG', 'PERSON', 'MISC'] and entity['score'] > 0.9:
                    topic = entity['word']
                    
                    # Check if this is a new topic
                    existing_topics = [t['topic'] for t in session['topic_evolution']]
                    if topic not in existing_topics:
                        session['topic_evolution'].append({
                            'topic': topic,
                            'first_mentioned': timestamp.isoformat(),
                            'frequency': 1
                        })
                        
                        insights.append(MeetingInsight(
                            insight_id=str(uuid.uuid4()),
                            insight_type="new_topic",
                            title=f"New topic introduced: {topic}",
                            content=f"The topic '{topic}' was first mentioned in the discussion.",
                            confidence=entity['score'],
                            timestamp=timestamp,
                            keywords=[topic],
                            priority="low"
                        ))
        except Exception as e:
            logging.error(f"Error tracking topics: {e}")
        
        return insights
    
    async def _detect_action_items(self, text: str, speaker: str, timestamp: datetime) -> List[MeetingInsight]:
        """Detect potential action items in real-time."""
        insights = []
        
        # Action item indicators
        action_patterns = [
            r"(?:will|shall|should|need to|must|have to|going to)\s+(\w+(?:\s+\w+){0,10})",
            r"(?:action|task|todo|follow.?up):\s*(.+)",
            r"(?:assigned to|responsible for|owner)\s+(\w+)",
            r"(?:by|due|deadline|before)\s+(\w+(?:\s+\w+){0,3})"
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action_text = match.group(1) if match.groups() else match.group(0)
                
                insights.append(MeetingInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="potential_action",
                    title="Potential action item detected",
                    content=f"Possible action from {speaker}: {action_text}",
                    confidence=0.7,
                    timestamp=timestamp,
                    participants_involved=[speaker],
                    keywords=["action", "task"],
                    priority="medium"
                ))
        
        return insights
    
    async def _detect_decisions(self, text: str, speaker: str, timestamp: datetime, 
                              session: Dict[str, Any]) -> List[MeetingInsight]:
        """Detect decision-making moments."""
        insights = []
        
        # Decision indicators
        decision_patterns = [
            r"(?:decided|agreed|concluded|determined|resolved)\s+(?:to|that)\s+(.+)",
            r"(?:decision|choice|option):\s*(.+)",
            r"(?:we will|let's|going with|final decision)\s+(.+)"
        ]
        
        for pattern in decision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                decision_text = match.group(1) if match.groups() else match.group(0)
                
                session['decision_markers'].append({
                    'text': decision_text,
                    'speaker': speaker,
                    'timestamp': timestamp.isoformat()
                })
                
                insights.append(MeetingInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type="decision",
                    title="Decision point identified",
                    content=f"Decision by {speaker}: {decision_text}",
                    confidence=0.8,
                    timestamp=timestamp,
                    participants_involved=[speaker],
                    keywords=["decision", "agreed"],
                    priority="high"
                ))
        
        return insights
    
    async def _detect_meeting_phase(self, text: str, session: Dict[str, Any]):
        """Detect current meeting phase based on content."""
        phase_indicators = {
            MeetingPhase.INTRODUCTION: ["welcome", "introduction", "agenda", "started", "begin"],
            MeetingPhase.DISCUSSION: ["discuss", "think", "opinion", "perspective", "issue"],
            MeetingPhase.DECISION_MAKING: ["decide", "agree", "choose", "vote", "consensus"],
            MeetingPhase.ACTION_PLANNING: ["action", "next steps", "follow up", "assign", "responsible"],
            MeetingPhase.WRAP_UP: ["summary", "conclude", "end", "thanks", "next meeting"]
        }
        
        text_lower = text.lower()
        
        for phase, keywords in phase_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                if session['current_phase'] != phase:
                    session['current_phase'] = phase
                    logging.info(f"Meeting phase changed to: {phase.value}")
                break
    
    async def _generate_summary(self, transcript: str, session: Dict[str, Any]) -> str:
        """Generate meeting summary."""
        try:
            if self.openai_client:
                # Use GPT for high-quality summary
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a meeting assistant. Create a concise meeting summary highlighting key points, decisions, and outcomes."},
                        {"role": "user", "content": f"Summarize this meeting transcript:\n\n{transcript[:4000]}"}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                return response.choices[0].message.content
            
            elif hasattr(self, 'summarizer'):
                # Use local summarization model
                chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
                summaries = []
                
                for chunk in chunks[:3]:  # Limit to avoid memory issues
                    summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                
                return " ".join(summaries)
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
        
        return "Summary generation unavailable"
    
    async def _extract_key_topics(self, transcript: str) -> List[str]:
        """Extract key topics from transcript."""
        try:
            if hasattr(self, 'ner_pipeline'):
                entities = self.ner_pipeline(transcript[:2000])  # Limit for performance
                topics = []
                
                for entity in entities:
                    if entity['entity_group'] in ['ORG', 'MISC', 'PERSON'] and entity['score'] > 0.8:
                        topics.append(entity['word'])
                
                # Return unique topics
                return list(set(topics))[:10]  # Limit to top 10
        except Exception as e:
            logging.error(f"Error extracting topics: {e}")
        
        return []
    
    async def _extract_action_items(self, transcript: str, session: Dict[str, Any]) -> List[ActionItem]:
        """Extract action items from full transcript."""
        action_items = []
        
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract action items from this meeting transcript. Return each action item with assignee (if mentioned) and priority level."},
                        {"role": "user", "content": f"Extract action items from:\n\n{transcript[:3000]}"}
                    ],
                    max_tokens=400,
                    temperature=0.2
                )
                
                # Parse GPT response into action items
                ai_response = response.choices[0].message.content
                lines = ai_response.split('\n')
                
                for line in lines:
                    if line.strip() and ('action' in line.lower() or 'task' in line.lower()):
                        action_items.append(ActionItem(
                            action_id=str(uuid.uuid4()),
                            description=line.strip(),
                            assignee=None,  # Would need more sophisticated parsing
                            due_date=None,
                            priority="medium",
                            confidence=0.8
                        ))
        
        except Exception as e:
            logging.error(f"Error extracting action items: {e}")
        
        return action_items[:10]  # Limit to top 10
    
    async def _extract_decisions(self, transcript: str, session: Dict[str, Any]) -> List[MeetingDecision]:
        """Extract decisions from transcript."""
        decisions = []
        
        for marker in session.get('decision_markers', []):
            decisions.append(MeetingDecision(
                decision_id=str(uuid.uuid4()),
                description=marker['text'],
                decision_maker=marker['speaker'],
                rationale=None,
                impact_level="medium",
                confidence=0.7,
                timestamp=datetime.fromisoformat(marker['timestamp'])
            ))
        
        return decisions
    
    async def _analyze_participant_engagement(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze participant engagement levels."""
        analytics = {}
        
        for participant, data in session['participant_contributions'].items():
            word_count = data['words']
            sentiment_scores = data['sentiment_scores']
            
            avg_sentiment = sum([1 if s == 'positive' else -1 if s == 'negative' else 0 
                               for s in sentiment_scores]) / max(len(sentiment_scores), 1)
            
            analytics[participant] = {
                'word_count': word_count,
                'engagement_level': 'high' if word_count > 100 else 'medium' if word_count > 50 else 'low',
                'avg_sentiment': avg_sentiment,
                'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
            }
        
        return analytics
    
    async def _generate_sentiment_timeline(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sentiment timeline for the meeting."""
        timeline = []
        
        for i, segment in enumerate(session['transcript_segments']):
            if i % 10 == 0:  # Sample every 10th segment
                timestamp = segment.get('timestamp')
                speaker = segment.get('speaker_name')
                
                # Get sentiment for this timepoint
                participant_data = session['participant_contributions'].get(speaker, {})
                sentiment_scores = participant_data.get('sentiment_scores', [])
                
                if sentiment_scores:
                    latest_sentiment = sentiment_scores[-1]
                    timeline.append({
                        'timestamp': timestamp,
                        'sentiment': latest_sentiment,
                        'speaker': speaker
                    })
        
        return timeline
    
    async def _calculate_effectiveness_score(self, session: Dict[str, Any], analysis: MeetingAnalysis) -> float:
        """Calculate meeting effectiveness score."""
        score = 50.0  # Base score
        
        # Positive factors
        if len(analysis.action_items) > 0:
            score += 20
        
        if len(analysis.decisions) > 0:
            score += 15
        
        # Balanced participation
        word_counts = [data['words'] for data in session['participant_contributions'].values()]
        if word_counts and max(word_counts) / max(min(word_counts), 1) < 3:  # Not too dominated by one person
            score += 10
        
        # Good sentiment
        avg_sentiment = sum([sum([1 if s == 'positive' else -1 if s == 'negative' else 0 
                                for s in data['sentiment_scores']]) 
                           for data in session['participant_contributions'].values()])
        if avg_sentiment > 0:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    async def _generate_recommendations(self, session: Dict[str, Any], analysis: MeetingAnalysis) -> List[str]:
        """Generate recommendations for future meetings."""
        recommendations = []
        
        # Check for issues and suggest improvements
        if analysis.meeting_effectiveness_score < 60:
            recommendations.append("Consider setting clearer objectives for future meetings")
        
        if len(analysis.action_items) == 0:
            recommendations.append("Ensure meetings conclude with specific action items and owners")
        
        # Check participation balance
        word_counts = [data['words'] for data in session['participant_contributions'].values()]
        if word_counts and max(word_counts) / max(min(word_counts), 1) > 4:
            recommendations.append("Encourage more balanced participation from all attendees")
        
        # Check meeting duration
        duration = (datetime.utcnow() - datetime.fromisoformat(session['start_time'].isoformat())).total_seconds() / 60
        if duration > 90:
            recommendations.append("Consider shorter meeting durations or breaks for longer sessions")
        
        return recommendations
    
    async def _store_meeting_session(self, session: Dict[str, Any]):
        """Store meeting session in Redis."""
        try:
            # Convert datetime objects to ISO strings for JSON serialization
            session_copy = session.copy()
            session_copy['start_time'] = session_copy['start_time'].isoformat()
            
            data = json.dumps(session_copy, default=str)
            await self.redis.set(f"meeting_session:{session['meeting_id']}", data)
            await self.redis.expire(f"meeting_session:{session['meeting_id']}", 86400)  # 24 hours
        except Exception as e:
            logging.error(f"Error storing meeting session: {e}")
    
    async def _store_meeting_analysis(self, analysis: MeetingAnalysis):
        """Store meeting analysis in Redis."""
        try:
            data = json.dumps(analysis.to_dict())
            await self.redis.set(f"meeting_analysis:{analysis.meeting_id}", data)
            await self.redis.expire(f"meeting_analysis:{analysis.meeting_id}", 86400 * 7)  # 7 days
        except Exception as e:
            logging.error(f"Error storing meeting analysis: {e}")
    
    async def _processing_loop(self):
        """Main processing loop for background analysis."""
        while True:
            try:
                await asyncio.sleep(30)  # Process every 30 seconds
                # Additional background processing can be added here
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in meeting assistant processing loop: {e}")
    
    def get_assistant_stats(self) -> Dict[str, Any]:
        """Get meeting assistant statistics."""
        return {
            'active_meetings': len(self.active_meetings),
            'openai_available': self.openai_client is not None,
            'local_models_loaded': hasattr(self, 'summarizer'),
            'pending_analysis': self.processing_queue.qsize()
        }


# Global instance
meeting_assistant = MeetingAssistant()