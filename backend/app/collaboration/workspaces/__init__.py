"""
Collaborative workspaces for organizing project rooms, team dashboards, and shared resources.

This module provides comprehensive workspace management including:
- Project rooms with document collections
- Team dashboards with activity feeds
- Shared folders and permissions
- Task assignment and tracking
- Notification system
- Calendar integration
"""

from .workspace_manager import WorkspaceManager
from .project_rooms import ProjectRoomManager
from .team_dashboard import TeamDashboard
from .shared_folders import SharedFolderManager
from .task_tracker import TaskTracker
from .notification_system import NotificationSystem

__all__ = [
    'WorkspaceManager',
    'ProjectRoomManager',
    'TeamDashboard',
    'SharedFolderManager',
    'TaskTracker',
    'NotificationSystem'
]