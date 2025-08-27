"""
User behavior tracking and analytics system.
Tracks user interactions, feature usage, and conversion funnels.
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
import numpy as np
from enum import Enum


class EventType(Enum):
    """Types of user events"""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    FORM_SUBMIT = "form_submit"
    FEATURE_USE = "feature_use"
    ERROR = "error"
    CONVERSION = "conversion"
    CUSTOM = "custom"


class UserSegment(Enum):
    """User segmentation categories"""
    NEW = "new"
    ACTIVE = "active"
    RETURNING = "returning"
    POWER = "power"
    CHURNED = "churned"
    TRIAL = "trial"
    PAID = "paid"
    ENTERPRISE = "enterprise"


@dataclass
class UserEvent:
    """User interaction event"""
    user_id: str
    event_type: EventType
    event_name: str
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    device_info: Optional[Dict[str, str]] = None
    location: Optional[Dict[str, Any]] = None


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[UserEvent] = field(default_factory=list)
    page_views: int = 0
    duration_seconds: float = 0
    bounce: bool = False


class UserBehaviorAnalytics:
    """
    Comprehensive user behavior analytics system.
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize user behavior analytics.
        
        Args:
            session_timeout_minutes: Minutes of inactivity before session ends
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Event storage
        self.events: List[UserEvent] = []
        self.max_events = 100000
        
        # Session tracking
        self.sessions: Dict[str, UserSession] = {}
        self.completed_sessions: List[UserSession] = []
        
        # User profiles
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Feature usage tracking
        self.feature_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Funnel tracking
        self.funnels: Dict[str, List[str]] = {}
        self.funnel_data: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # A/B test tracking
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Cohort analysis
        self.cohorts: Dict[str, List[str]] = defaultdict(list)
    
    def track_event(self, event: UserEvent):
        """
        Track a user event.
        
        Args:
            event: UserEvent to track
        """
        # Store event
        self.events.append(event)
        
        # Trim events if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Update session
        self._update_session(event)
        
        # Update user profile
        self._update_user_profile(event)
        
        # Track feature usage
        if event.event_type == EventType.FEATURE_USE:
            self.feature_usage[event.event_name][event.user_id] += 1
        
        # Check funnel progression
        self._check_funnel_progression(event)
        
        # Update A/B test metrics
        self._update_ab_test_metrics(event)
    
    def _update_session(self, event: UserEvent):
        """Update or create session for event"""
        session_id = event.session_id or self._generate_session_id(event.user_id)
        
        if session_id not in self.sessions:
            # Create new session
            self.sessions[session_id] = UserSession(
                session_id=session_id,
                user_id=event.user_id,
                start_time=event.timestamp
            )
        
        session = self.sessions[session_id]
        session.events.append(event)
        
        # Update session metrics
        if event.event_type == EventType.PAGE_VIEW:
            session.page_views += 1
        
        # Check if session should be closed
        if session.end_time is None:
            if event.timestamp - session.start_time > self.session_timeout:
                self._close_session(session_id)
    
    def _close_session(self, session_id: str):
        """Close a session and calculate metrics"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.end_time = session.events[-1].timestamp if session.events else session.start_time
        session.duration_seconds = (session.end_time - session.start_time).total_seconds()
        session.bounce = session.page_views <= 1
        
        # Move to completed sessions
        self.completed_sessions.append(session)
        del self.sessions[session_id]
    
    def _update_user_profile(self, event: UserEvent):
        """Update user profile based on event"""
        user_id = event.user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'first_seen': event.timestamp,
                'last_seen': event.timestamp,
                'total_events': 0,
                'total_sessions': 0,
                'segment': UserSegment.NEW.value,
                'features_used': set(),
                'conversion_events': [],
                'device_types': set(),
                'locations': set()
            }
        
        profile = self.user_profiles[user_id]
        profile['last_seen'] = event.timestamp
        profile['total_events'] += 1
        
        if event.event_type == EventType.FEATURE_USE:
            profile['features_used'].add(event.event_name)
        
        if event.event_type == EventType.CONVERSION:
            profile['conversion_events'].append(event.event_name)
        
        if event.device_info:
            profile['device_types'].add(event.device_info.get('type', 'unknown'))
        
        if event.location:
            profile['locations'].add(event.location.get('country', 'unknown'))
        
        # Update segment
        profile['segment'] = self._calculate_user_segment(profile)
    
    def _calculate_user_segment(self, profile: Dict[str, Any]) -> str:
        """Calculate user segment based on profile"""
        days_since_first = (datetime.utcnow() - profile['first_seen']).days
        days_since_last = (datetime.utcnow() - profile['last_seen']).days
        
        if days_since_first <= 7:
            return UserSegment.NEW.value
        elif days_since_last > 30:
            return UserSegment.CHURNED.value
        elif profile['total_events'] > 1000:
            return UserSegment.POWER.value
        elif days_since_last <= 7:
            return UserSegment.ACTIVE.value
        else:
            return UserSegment.RETURNING.value
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.md5(f"{user_id}_{timestamp}".encode()).hexdigest()[:16]
    
    def define_funnel(self, funnel_name: str, steps: List[str]):
        """
        Define a conversion funnel.
        
        Args:
            funnel_name: Name of the funnel
            steps: Ordered list of event names in the funnel
        """
        self.funnels[funnel_name] = steps
    
    def _check_funnel_progression(self, event: UserEvent):
        """Check if event progresses any funnels"""
        for funnel_name, steps in self.funnels.items():
            if event.event_name in steps:
                user_progress = self.funnel_data[funnel_name][event.user_id]
                
                # Check if this is the next step in the funnel
                if not user_progress:
                    if event.event_name == steps[0]:
                        user_progress.append(event.event_name)
                else:
                    current_index = steps.index(user_progress[-1])
                    event_index = steps.index(event.event_name)
                    
                    if event_index == current_index + 1:
                        user_progress.append(event.event_name)
    
    def get_funnel_metrics(self, funnel_name: str) -> Dict[str, Any]:
        """
        Get conversion metrics for a funnel.
        
        Args:
            funnel_name: Name of the funnel
        
        Returns:
            Funnel conversion metrics
        """
        if funnel_name not in self.funnels:
            return {'error': 'Funnel not found'}
        
        steps = self.funnels[funnel_name]
        user_progress = self.funnel_data[funnel_name]
        
        # Calculate conversion rates
        step_counts = {step: 0 for step in steps}
        
        for user_id, progress in user_progress.items():
            for step in progress:
                step_counts[step] += 1
        
        # Calculate conversion rates between steps
        conversions = []
        for i in range(len(steps) - 1):
            if step_counts[steps[i]] > 0:
                rate = (step_counts[steps[i + 1]] / step_counts[steps[i]]) * 100
            else:
                rate = 0
            
            conversions.append({
                'from': steps[i],
                'to': steps[i + 1],
                'conversion_rate': rate,
                'drop_off_rate': 100 - rate
            })
        
        return {
            'funnel_name': funnel_name,
            'steps': steps,
            'step_counts': step_counts,
            'conversions': conversions,
            'overall_conversion': (step_counts[steps[-1]] / step_counts[steps[0]] * 100) 
                                 if step_counts[steps[0]] > 0 else 0
        }
    
    def create_ab_test(self, test_name: str, variants: List[str],
                      metrics: List[str], allocation: Optional[Dict[str, float]] = None):
        """
        Create an A/B test.
        
        Args:
            test_name: Name of the test
            variants: List of variant names
            metrics: Metrics to track
            allocation: Traffic allocation per variant (defaults to equal)
        """
        if allocation is None:
            allocation = {v: 1.0 / len(variants) for v in variants}
        
        self.ab_tests[test_name] = {
            'variants': variants,
            'metrics': metrics,
            'allocation': allocation,
            'assignments': {},
            'results': {v: {m: [] for m in metrics} for v in variants}
        }
    
    def assign_ab_test_variant(self, test_name: str, user_id: str) -> str:
        """
        Assign user to A/B test variant.
        
        Args:
            test_name: Name of the test
            user_id: User identifier
        
        Returns:
            Assigned variant
        """
        if test_name not in self.ab_tests:
            return 'control'
        
        test = self.ab_tests[test_name]
        
        # Check if already assigned
        if user_id in test['assignments']:
            return test['assignments'][user_id]
        
        # Assign based on allocation
        rand = hash(f"{test_name}_{user_id}") % 100 / 100
        cumulative = 0
        
        for variant, allocation in test['allocation'].items():
            cumulative += allocation
            if rand < cumulative:
                test['assignments'][user_id] = variant
                return variant
        
        # Default to last variant
        variant = test['variants'][-1]
        test['assignments'][user_id] = variant
        return variant
    
    def _update_ab_test_metrics(self, event: UserEvent):
        """Update A/B test metrics based on event"""
        for test_name, test in self.ab_tests.items():
            if event.user_id in test['assignments']:
                variant = test['assignments'][event.user_id]
                
                for metric in test['metrics']:
                    if metric == event.event_name:
                        test['results'][variant][metric].append(event.timestamp)
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """
        Get A/B test results with statistical significance.
        
        Args:
            test_name: Name of the test
        
        Returns:
            Test results with statistics
        """
        if test_name not in self.ab_tests:
            return {'error': 'Test not found'}
        
        test = self.ab_tests[test_name]
        results = {}
        
        for variant in test['variants']:
            variant_users = [u for u, v in test['assignments'].items() if v == variant]
            
            results[variant] = {
                'users': len(variant_users),
                'metrics': {}
            }
            
            for metric in test['metrics']:
                events = test['results'][variant][metric]
                results[variant]['metrics'][metric] = {
                    'count': len(events),
                    'conversion_rate': (len(events) / len(variant_users) * 100) 
                                     if variant_users else 0
                }
        
        # Calculate statistical significance (simplified)
        if len(test['variants']) == 2:
            control = test['variants'][0]
            treatment = test['variants'][1]
            
            for metric in test['metrics']:
                control_rate = results[control]['metrics'][metric]['conversion_rate']
                treatment_rate = results[treatment]['metrics'][metric]['conversion_rate']
                
                # Simple lift calculation
                if control_rate > 0:
                    lift = ((treatment_rate - control_rate) / control_rate) * 100
                else:
                    lift = 0
                
                results[treatment]['metrics'][metric]['lift'] = lift
        
        return {
            'test_name': test_name,
            'variants': test['variants'],
            'results': results,
            'total_users': len(test['assignments'])
        }
    
    def create_cohort(self, cohort_name: str, user_filter: Callable[[Dict[str, Any]], bool]):
        """
        Create a user cohort based on filter criteria.
        
        Args:
            cohort_name: Name of the cohort
            user_filter: Function to filter users
        """
        cohort_users = []
        
        for user_id, profile in self.user_profiles.items():
            if user_filter(profile):
                cohort_users.append(user_id)
        
        self.cohorts[cohort_name] = cohort_users
    
    def get_cohort_analysis(self, cohort_name: str, 
                           metric: str = 'retention') -> Dict[str, Any]:
        """
        Analyze cohort behavior over time.
        
        Args:
            cohort_name: Name of the cohort
            metric: Metric to analyze
        
        Returns:
            Cohort analysis results
        """
        if cohort_name not in self.cohorts:
            return {'error': 'Cohort not found'}
        
        cohort_users = self.cohorts[cohort_name]
        
        if metric == 'retention':
            # Calculate retention by week
            retention_by_week = {}
            
            for week in range(12):  # 12 weeks
                week_start = datetime.utcnow() - timedelta(weeks=week+1)
                week_end = datetime.utcnow() - timedelta(weeks=week)
                
                active_users = set()
                for event in self.events:
                    if (event.user_id in cohort_users and 
                        week_start <= event.timestamp < week_end):
                        active_users.add(event.user_id)
                
                retention_by_week[f'week_{week}'] = {
                    'active_users': len(active_users),
                    'retention_rate': (len(active_users) / len(cohort_users) * 100) 
                                     if cohort_users else 0
                }
            
            return {
                'cohort_name': cohort_name,
                'cohort_size': len(cohort_users),
                'metric': metric,
                'retention_by_week': retention_by_week
            }
        
        return {'error': f'Metric {metric} not supported'}
    
    def get_user_journey(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user journey timeline.
        
        Args:
            user_id: User identifier
            limit: Maximum number of events
        
        Returns:
            List of user events in chronological order
        """
        user_events = [e for e in self.events if e.user_id == user_id]
        user_events.sort(key=lambda x: x.timestamp)
        
        journey = []
        for event in user_events[:limit]:
            journey.append({
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'event_name': event.event_name,
                'properties': event.properties,
                'session_id': event.session_id
            })
        
        return journey
    
    def get_feature_adoption(self) -> Dict[str, Any]:
        """
        Get feature adoption metrics.
        
        Returns:
            Feature adoption statistics
        """
        adoption_metrics = {}
        
        for feature, users in self.feature_usage.items():
            total_usage = sum(users.values())
            unique_users = len(users)
            
            adoption_metrics[feature] = {
                'unique_users': unique_users,
                'total_usage': total_usage,
                'avg_usage_per_user': total_usage / unique_users if unique_users > 0 else 0,
                'adoption_rate': (unique_users / len(self.user_profiles) * 100) 
                               if self.user_profiles else 0
            }
        
        # Sort by adoption rate
        sorted_features = sorted(
            adoption_metrics.items(),
            key=lambda x: x[1]['adoption_rate'],
            reverse=True
        )
        
        return {
            'total_features': len(adoption_metrics),
            'total_users': len(self.user_profiles),
            'features': dict(sorted_features)
        }
    
    def get_user_segments(self) -> Dict[str, Any]:
        """
        Get user segmentation analysis.
        
        Returns:
            User segment distribution
        """
        segment_counts = Counter()
        
        for profile in self.user_profiles.values():
            segment_counts[profile['segment']] += 1
        
        total_users = len(self.user_profiles)
        
        segments = {}
        for segment, count in segment_counts.items():
            segments[segment] = {
                'count': count,
                'percentage': (count / total_users * 100) if total_users > 0 else 0
            }
        
        return {
            'total_users': total_users,
            'segments': segments
        }
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """
        Get session-level metrics.
        
        Returns:
            Session statistics
        """
        all_sessions = self.completed_sessions + list(self.sessions.values())
        
        if not all_sessions:
            return {'message': 'No session data available'}
        
        durations = [s.duration_seconds for s in self.completed_sessions if s.duration_seconds > 0]
        page_views = [s.page_views for s in all_sessions]
        bounce_count = sum(1 for s in self.completed_sessions if s.bounce)
        
        return {
            'total_sessions': len(all_sessions),
            'active_sessions': len(self.sessions),
            'completed_sessions': len(self.completed_sessions),
            'avg_duration_seconds': np.mean(durations) if durations else 0,
            'median_duration_seconds': np.median(durations) if durations else 0,
            'avg_page_views': np.mean(page_views) if page_views else 0,
            'bounce_rate': (bounce_count / len(self.completed_sessions) * 100) 
                         if self.completed_sessions else 0
        }