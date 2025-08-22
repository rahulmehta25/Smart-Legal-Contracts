"""
Timeline visualizer for document version history and changes over time.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..version_tracker import DocumentVersion, DocumentVersionTracker


@dataclass
class TimelineEvent:
    """Represents an event in the document timeline."""
    timestamp: datetime
    event_type: str
    title: str
    description: str
    author: str
    version_id: str
    metadata: Dict[str, Any] = None


class TimelineVisualizer:
    """Creates timeline visualizations of document changes over time."""
    
    def create_version_timeline(self, version_tracker: DocumentVersionTracker,
                              branch_name: str = None) -> Dict[str, Any]:
        """
        Create timeline of version history.
        
        Args:
            version_tracker: Version tracker instance
            branch_name: Branch to show timeline for
            
        Returns:
            Timeline visualization data
        """
        history = version_tracker.get_version_history(branch_name)
        
        timeline_events = []
        for version in history:
            event = TimelineEvent(
                timestamp=version.timestamp,
                event_type="version_created",
                title=f"Version {version.version_id[:8]}",
                description=version.message,
                author=version.author.name,
                version_id=version.version_id,
                metadata={
                    'status': version.status.value,
                    'tags': version.tags,
                    'content_hash': version.content_hash
                }
            )
            timeline_events.append(event)
        
        # Sort by timestamp
        timeline_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        html_timeline = self._generate_html_timeline(timeline_events)
        
        return {
            'timeline_events': [self._event_to_dict(e) for e in timeline_events],
            'html_visualization': html_timeline,
            'statistics': {
                'total_versions': len(timeline_events),
                'date_range': {
                    'earliest': min(e.timestamp for e in timeline_events) if timeline_events else None,
                    'latest': max(e.timestamp for e in timeline_events) if timeline_events else None
                },
                'unique_authors': len(set(e.author for e in timeline_events))
            }
        }
    
    def _event_to_dict(self, event: TimelineEvent) -> Dict[str, Any]:
        """Convert timeline event to dictionary."""
        return {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'title': event.title,
            'description': event.description,
            'author': event.author,
            'version_id': event.version_id,
            'metadata': event.metadata or {}
        }
    
    def _generate_html_timeline(self, events: List[TimelineEvent]) -> str:
        """Generate HTML timeline visualization."""
        events_html = []
        
        for i, event in enumerate(events):
            # Format timestamp
            formatted_time = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            relative_time = self._get_relative_time(event.timestamp)
            
            # Determine event icon
            event_icon = self._get_event_icon(event.event_type)
            
            # Create event HTML
            event_html = f"""
            <div class="timeline-event" data-version-id="{event.version_id}">
                <div class="timeline-marker">
                    <div class="timeline-icon">{event_icon}</div>
                </div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <h3 class="timeline-title">{event.title}</h3>
                        <div class="timeline-meta">
                            <span class="timeline-author">{event.author}</span>
                            <span class="timeline-time" title="{formatted_time}">{relative_time}</span>
                        </div>
                    </div>
                    <div class="timeline-description">{event.description}</div>
                    {self._generate_metadata_display(event.metadata)}
                </div>
            </div>
            """
            events_html.append(event_html)
        
        css_styles = self._generate_timeline_css()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Timeline</title>
            <style>{css_styles}</style>
        </head>
        <body>
            <div class="timeline-container">
                <div class="timeline-header">
                    <h1>Document Version Timeline</h1>
                    <div class="timeline-stats">
                        <span>Total versions: {len(events)}</span>
                    </div>
                </div>
                <div class="timeline">
                    {''.join(events_html)}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_relative_time(self, timestamp: datetime) -> str:
        """Get human-readable relative time."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    def _get_event_icon(self, event_type: str) -> str:
        """Get icon for event type."""
        icons = {
            'version_created': '📝',
            'branch_created': '🌿',
            'merge_completed': '🔀',
            'tag_added': '🏷️'
        }
        return icons.get(event_type, '📄')
    
    def _generate_metadata_display(self, metadata: Dict[str, Any]) -> str:
        """Generate HTML for event metadata."""
        if not metadata:
            return ""
        
        metadata_items = []
        
        if 'status' in metadata:
            status_color = {
                'draft': '#6c757d',
                'reviewed': '#ffc107', 
                'approved': '#28a745',
                'published': '#007bff'
            }.get(metadata['status'], '#6c757d')
            
            metadata_items.append(
                f'<span class="status-badge" style="background-color: {status_color};">'
                f'{metadata["status"]}</span>'
            )
        
        if 'tags' in metadata and metadata['tags']:
            for tag in metadata['tags'][:3]:  # Show max 3 tags
                metadata_items.append(f'<span class="tag-badge">{tag}</span>')
        
        if metadata_items:
            return f'<div class="timeline-metadata">{"".join(metadata_items)}</div>'
        
        return ""
    
    def _generate_timeline_css(self) -> str:
        """Generate CSS for timeline visualization."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        .timeline-container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .timeline-header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .timeline-header h1 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .timeline-stats {
            color: #6c757d;
            font-size: 14px;
        }
        
        .timeline {
            position: relative;
            padding-left: 40px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 20px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #dee2e6;
        }
        
        .timeline-event {
            position: relative;
            margin-bottom: 30px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .timeline-event:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .timeline-marker {
            position: absolute;
            left: -35px;
            top: 20px;
            width: 30px;
            height: 30px;
            background: #007bff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 3px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .timeline-icon {
            font-size: 14px;
        }
        
        .timeline-content {
            padding: 20px;
        }
        
        .timeline-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        
        .timeline-title {
            margin: 0;
            color: #2c3e50;
            font-size: 18px;
        }
        
        .timeline-meta {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            font-size: 12px;
            color: #6c757d;
        }
        
        .timeline-author {
            font-weight: 500;
            margin-bottom: 2px;
        }
        
        .timeline-time {
            font-style: italic;
        }
        
        .timeline-description {
            color: #495057;
            line-height: 1.5;
            margin-bottom: 10px;
        }
        
        .timeline-metadata {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .status-badge {
            color: white;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        
        .tag-badge {
            background: #e9ecef;
            color: #495057;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        
        @media (max-width: 768px) {
            .timeline-container {
                padding: 10px;
            }
            
            .timeline {
                padding-left: 20px;
            }
            
            .timeline-marker {
                left: -25px;
                width: 20px;
                height: 20px;
            }
            
            .timeline-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }
            
            .timeline-meta {
                flex-direction: row;
                gap: 10px;
            }
        }
        """