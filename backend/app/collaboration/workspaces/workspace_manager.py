"""
Workspace manager for organizing collaborative environments.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

import redis.asyncio as redis


class WorkspaceType(Enum):
    PROJECT = "project"
    TEAM = "team"
    DEPARTMENT = "department"
    ORGANIZATION = "organization"
    CLIENT = "client"
    TEMPORARY = "temporary"


class WorkspaceVisibility(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    INVITE_ONLY = "invite_only"


class MemberRole(Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    VIEWER = "viewer"
    GUEST = "guest"


class PermissionType(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"
    MANAGE_MEMBERS = "manage_members"
    CREATE_ROOMS = "create_rooms"
    MANAGE_SETTINGS = "manage_settings"


@dataclass
class WorkspaceMember:
    """Member of a workspace."""
    user_id: str
    username: str
    email: Optional[str]
    role: MemberRole
    permissions: Set[PermissionType]
    joined_at: datetime
    last_active: Optional[datetime] = None
    avatar_url: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    is_online: bool = False
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    def has_permission(self, permission: PermissionType) -> bool:
        """Check if member has specific permission."""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'permissions': [p.value for p in self.permissions],
            'joined_at': self.joined_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'avatar_url': self.avatar_url,
            'title': self.title,
            'department': self.department,
            'is_online': self.is_online,
            'notification_preferences': self.notification_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceMember':
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data.get('email'),
            role=MemberRole(data['role']),
            permissions={PermissionType(p) for p in data.get('permissions', [])},
            joined_at=datetime.fromisoformat(data['joined_at']),
            last_active=datetime.fromisoformat(data['last_active']) if data.get('last_active') else None,
            avatar_url=data.get('avatar_url'),
            title=data.get('title'),
            department=data.get('department'),
            is_online=data.get('is_online', False),
            notification_preferences=data.get('notification_preferences', {})
        )


@dataclass
class WorkspaceSettings:
    """Workspace configuration settings."""
    allow_guest_access: bool = False
    require_approval_for_join: bool = True
    auto_archive_inactive: bool = False
    inactive_threshold_days: int = 90
    enable_notifications: bool = True
    enable_file_sharing: bool = True
    enable_video_calls: bool = True
    enable_screen_sharing: bool = True
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: ['pdf', 'doc', 'docx', 'txt', 'jpg', 'png'])
    default_room_permissions: Dict[str, bool] = field(default_factory=dict)
    integration_settings: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceSettings':
        return cls(**data)


@dataclass
class WorkspaceStats:
    """Workspace statistics."""
    total_members: int
    active_members_today: int
    total_rooms: int
    total_documents: int
    total_tasks: int
    completed_tasks: int
    total_messages: int
    storage_used_mb: int
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_members': self.total_members,
            'active_members_today': self.active_members_today,
            'total_rooms': self.total_rooms,
            'total_documents': self.total_documents,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'total_messages': self.total_messages,
            'storage_used_mb': self.storage_used_mb,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'task_completion_rate': (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        }


@dataclass
class Workspace:
    """A collaborative workspace."""
    workspace_id: str
    name: str
    description: Optional[str]
    workspace_type: WorkspaceType
    visibility: WorkspaceVisibility
    owner_id: str
    created_at: datetime
    updated_at: datetime
    members: Dict[str, WorkspaceMember] = field(default_factory=dict)
    settings: WorkspaceSettings = field(default_factory=WorkspaceSettings)
    tags: List[str] = field(default_factory=list)
    avatar_url: Optional[str] = None
    banner_url: Optional[str] = None
    project_rooms: List[str] = field(default_factory=list)  # Room IDs
    shared_folders: List[str] = field(default_factory=list)  # Folder IDs
    integrations: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    
    def get_active_members(self) -> List[WorkspaceMember]:
        """Get currently active members."""
        return [m for m in self.members.values() if m.is_online]
    
    def get_member_by_role(self, role: MemberRole) -> List[WorkspaceMember]:
        """Get members by role."""
        return [m for m in self.members.values() if m.role == role]
    
    def get_stats(self) -> WorkspaceStats:
        """Calculate workspace statistics."""
        active_today = sum(1 for m in self.members.values() 
                          if m.last_active and m.last_active.date() == datetime.utcnow().date())
        
        last_activity = None
        if self.members:
            last_activities = [m.last_active for m in self.members.values() if m.last_active]
            if last_activities:
                last_activity = max(last_activities)
        
        return WorkspaceStats(
            total_members=len(self.members),
            active_members_today=active_today,
            total_rooms=len(self.project_rooms),
            total_documents=0,  # This would be calculated from actual documents
            total_tasks=0,      # This would be calculated from task system
            completed_tasks=0,  # This would be calculated from task system
            total_messages=0,   # This would be calculated from message system
            storage_used_mb=0,  # This would be calculated from file system
            last_activity=last_activity
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workspace_id': self.workspace_id,
            'name': self.name,
            'description': self.description,
            'workspace_type': self.workspace_type.value,
            'visibility': self.visibility.value,
            'owner_id': self.owner_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'members': {uid: m.to_dict() for uid, m in self.members.items()},
            'settings': self.settings.to_dict(),
            'tags': self.tags,
            'avatar_url': self.avatar_url,
            'banner_url': self.banner_url,
            'project_rooms': self.project_rooms,
            'shared_folders': self.shared_folders,
            'integrations': self.integrations,
            'custom_fields': self.custom_fields,
            'is_archived': self.is_archived,
            'archived_at': self.archived_at.isoformat() if self.archived_at else None,
            'stats': self.get_stats().to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workspace':
        return cls(
            workspace_id=data['workspace_id'],
            name=data['name'],
            description=data.get('description'),
            workspace_type=WorkspaceType(data['workspace_type']),
            visibility=WorkspaceVisibility(data['visibility']),
            owner_id=data['owner_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            members={uid: WorkspaceMember.from_dict(m) for uid, m in data.get('members', {}).items()},
            settings=WorkspaceSettings.from_dict(data.get('settings', {})),
            tags=data.get('tags', []),
            avatar_url=data.get('avatar_url'),
            banner_url=data.get('banner_url'),
            project_rooms=data.get('project_rooms', []),
            shared_folders=data.get('shared_folders', []),
            integrations=data.get('integrations', {}),
            custom_fields=data.get('custom_fields', {}),
            is_archived=data.get('is_archived', False),
            archived_at=datetime.fromisoformat(data['archived_at']) if data.get('archived_at') else None
        )


class WorkspaceManager:
    """Manages collaborative workspaces."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.workspaces: Dict[str, Workspace] = {}
        self.user_workspaces: Dict[str, Set[str]] = {}  # user_id -> workspace_ids
        self.event_callbacks: List[callable] = []
        self.cleanup_task = None
        
        # Default permissions for roles
        self.role_permissions = {
            MemberRole.OWNER: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE,
                PermissionType.SHARE, PermissionType.ADMIN, PermissionType.MANAGE_MEMBERS,
                PermissionType.CREATE_ROOMS, PermissionType.MANAGE_SETTINGS
            },
            MemberRole.ADMIN: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE,
                PermissionType.SHARE, PermissionType.MANAGE_MEMBERS,
                PermissionType.CREATE_ROOMS, PermissionType.MANAGE_SETTINGS
            },
            MemberRole.MODERATOR: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE,
                PermissionType.CREATE_ROOMS
            },
            MemberRole.MEMBER: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE
            },
            MemberRole.VIEWER: {
                PermissionType.READ
            },
            MemberRole.GUEST: {
                PermissionType.READ
            }
        }
        
    async def start(self):
        """Start the workspace manager."""
        # Load workspaces from Redis
        await self._load_workspaces()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logging.info("Workspace manager started")
    
    async def stop(self):
        """Stop the workspace manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.redis.close()
        logging.info("Workspace manager stopped")
    
    def add_event_callback(self, callback: callable):
        """Add a callback for workspace events."""
        self.event_callbacks.append(callback)
    
    async def create_workspace(self, name: str, owner_id: str, 
                             workspace_type: WorkspaceType = WorkspaceType.PROJECT,
                             visibility: WorkspaceVisibility = WorkspaceVisibility.PRIVATE,
                             description: Optional[str] = None,
                             settings: Optional[WorkspaceSettings] = None) -> Workspace:
        """Create a new workspace."""
        workspace_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            workspace_type=workspace_type,
            visibility=visibility,
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
            settings=settings or WorkspaceSettings()
        )
        
        # Add owner as member
        owner_member = WorkspaceMember(
            user_id=owner_id,
            username="Owner",  # This should be fetched from user service
            email=None,
            role=MemberRole.OWNER,
            permissions=self.role_permissions[MemberRole.OWNER],
            joined_at=now,
            last_active=now
        )
        workspace.members[owner_id] = owner_member
        
        # Store workspace
        self.workspaces[workspace_id] = workspace
        
        # Update user workspace mapping
        if owner_id not in self.user_workspaces:
            self.user_workspaces[owner_id] = set()
        self.user_workspaces[owner_id].add(workspace_id)
        
        # Store in Redis
        await self._store_workspace(workspace)
        
        # Create event
        await self._create_event(workspace_id, "workspace_created", owner_id, {
            'name': name,
            'workspace_type': workspace_type.value,
            'visibility': visibility.value
        })
        
        logging.info(f"Created workspace: {name} ({workspace_id})")
        return workspace
    
    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        # Check cache first
        if workspace_id in self.workspaces:
            return self.workspaces[workspace_id]
        
        # Load from Redis
        workspace_data = await self.redis.get(f"workspace:{workspace_id}")
        if workspace_data:
            try:
                data = json.loads(workspace_data)
                workspace = Workspace.from_dict(data)
                self.workspaces[workspace_id] = workspace
                return workspace
            except Exception as e:
                logging.error(f"Error loading workspace from Redis: {e}")
        
        return None
    
    async def update_workspace(self, workspace_id: str, user_id: str,
                             name: Optional[str] = None,
                             description: Optional[str] = None,
                             visibility: Optional[WorkspaceVisibility] = None,
                             settings: Optional[WorkspaceSettings] = None,
                             tags: Optional[List[str]] = None,
                             custom_fields: Optional[Dict[str, Any]] = None) -> Optional[Workspace]:
        """Update workspace properties."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        # Check permissions
        if not await self._can_manage_workspace(workspace, user_id):
            return None
        
        changes = {}
        
        if name is not None:
            changes['name'] = {'old': workspace.name, 'new': name}
            workspace.name = name
        
        if description is not None:
            changes['description'] = {'old': workspace.description, 'new': description}
            workspace.description = description
        
        if visibility is not None:
            changes['visibility'] = {'old': workspace.visibility.value, 'new': visibility.value}
            workspace.visibility = visibility
        
        if settings is not None:
            changes['settings'] = {'updated': True}
            workspace.settings = settings
        
        if tags is not None:
            changes['tags'] = {'old': workspace.tags, 'new': tags}
            workspace.tags = tags
        
        if custom_fields is not None:
            changes['custom_fields'] = {'updated': True}
            workspace.custom_fields.update(custom_fields)
        
        if changes:
            workspace.updated_at = datetime.utcnow()
            await self._store_workspace(workspace)
            
            # Create event
            await self._create_event(workspace_id, "workspace_updated", user_id, changes)
        
        return workspace
    
    async def add_member(self, workspace_id: str, user_id: str, username: str,
                        inviter_id: str, role: MemberRole = MemberRole.MEMBER,
                        email: Optional[str] = None) -> Optional[WorkspaceMember]:
        """Add a member to workspace."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        # Check permissions
        if not await self._can_manage_members(workspace, inviter_id):
            return None
        
        # Check if user is already a member
        if user_id in workspace.members:
            return workspace.members[user_id]
        
        # Create member
        member = WorkspaceMember(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions.get(role, set()),
            joined_at=datetime.utcnow()
        )
        
        workspace.members[user_id] = member
        workspace.updated_at = datetime.utcnow()
        
        # Update user workspace mapping
        if user_id not in self.user_workspaces:
            self.user_workspaces[user_id] = set()
        self.user_workspaces[user_id].add(workspace_id)
        
        # Store updated workspace
        await self._store_workspace(workspace)
        
        # Create event
        await self._create_event(workspace_id, "member_added", inviter_id, {
            'new_member_id': user_id,
            'new_member_username': username,
            'role': role.value
        })
        
        logging.info(f"Added member {username} to workspace {workspace.name}")
        return member
    
    async def remove_member(self, workspace_id: str, user_id: str, remover_id: str) -> bool:
        """Remove a member from workspace."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Check permissions
        if not await self._can_manage_members(workspace, remover_id):
            return False
        
        # Can't remove owner
        if user_id == workspace.owner_id:
            return False
        
        # Remove member
        if user_id in workspace.members:
            member = workspace.members[user_id]
            del workspace.members[user_id]
            workspace.updated_at = datetime.utcnow()
            
            # Update user workspace mapping
            if user_id in self.user_workspaces:
                self.user_workspaces[user_id].discard(workspace_id)
            
            # Store updated workspace
            await self._store_workspace(workspace)
            
            # Create event
            await self._create_event(workspace_id, "member_removed", remover_id, {
                'removed_member_id': user_id,
                'removed_member_username': member.username
            })
            
            logging.info(f"Removed member {member.username} from workspace {workspace.name}")
            return True
        
        return False
    
    async def update_member_role(self, workspace_id: str, user_id: str, new_role: MemberRole,
                               updater_id: str) -> Optional[WorkspaceMember]:
        """Update a member's role."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        # Check permissions
        if not await self._can_manage_members(workspace, updater_id):
            return None
        
        # Can't change owner role
        if user_id == workspace.owner_id:
            return None
        
        # Update member role
        if user_id in workspace.members:
            member = workspace.members[user_id]
            old_role = member.role
            member.role = new_role
            member.permissions = self.role_permissions.get(new_role, set())
            workspace.updated_at = datetime.utcnow()
            
            # Store updated workspace
            await self._store_workspace(workspace)
            
            # Create event
            await self._create_event(workspace_id, "member_role_updated", updater_id, {
                'member_id': user_id,
                'member_username': member.username,
                'old_role': old_role.value,
                'new_role': new_role.value
            })
            
            return member
        
        return None
    
    async def update_member_activity(self, workspace_id: str, user_id: str) -> bool:
        """Update member's last activity timestamp."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace or user_id not in workspace.members:
            return False
        
        member = workspace.members[user_id]
        member.last_active = datetime.utcnow()
        member.is_online = True
        
        # Store updated workspace (async to avoid blocking)
        asyncio.create_task(self._store_workspace(workspace))
        
        return True
    
    async def set_member_offline(self, workspace_id: str, user_id: str) -> bool:
        """Set member as offline."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace or user_id not in workspace.members:
            return False
        
        member = workspace.members[user_id]
        member.is_online = False
        
        # Store updated workspace
        await self._store_workspace(workspace)
        
        return True
    
    async def get_user_workspaces(self, user_id: str, include_archived: bool = False) -> List[Workspace]:
        """Get all workspaces for a user."""
        workspaces = []
        
        # Get workspace IDs for user
        workspace_ids = self.user_workspaces.get(user_id, set())
        
        # Load workspace IDs from Redis if not in cache
        if not workspace_ids:
            user_workspace_data = await self.redis.smembers(f"user_workspaces:{user_id}")
            workspace_ids = set(user_workspace_data)
            self.user_workspaces[user_id] = workspace_ids
        
        # Load workspaces
        for workspace_id in workspace_ids:
            workspace = await self.get_workspace(workspace_id)
            if workspace:
                if include_archived or not workspace.is_archived:
                    workspaces.append(workspace)
        
        # Sort by last activity
        workspaces.sort(key=lambda w: w.updated_at, reverse=True)
        return workspaces
    
    async def search_workspaces(self, query: str, user_id: Optional[str] = None,
                              workspace_type: Optional[WorkspaceType] = None,
                              visibility: Optional[WorkspaceVisibility] = None,
                              tags: Optional[List[str]] = None) -> List[Workspace]:
        """Search workspaces."""
        results = []
        
        # Get workspaces to search
        if user_id:
            workspaces = await self.get_user_workspaces(user_id)
        else:
            # Load all public workspaces
            workspace_keys = await self.redis.keys("workspace:*")
            workspaces = []
            for key in workspace_keys:
                workspace_data = await self.redis.get(key)
                if workspace_data:
                    try:
                        data = json.loads(workspace_data)
                        workspace = Workspace.from_dict(data)
                        if workspace.visibility == WorkspaceVisibility.PUBLIC:
                            workspaces.append(workspace)
                    except Exception:
                        continue
        
        # Apply filters
        query_lower = query.lower()
        for workspace in workspaces:
            # Text search
            if query_lower in workspace.name.lower() or (workspace.description and query_lower in workspace.description.lower()):
                match = True
            else:
                match = False
            
            # Type filter
            if workspace_type and workspace.workspace_type != workspace_type:
                match = False
            
            # Visibility filter
            if visibility and workspace.visibility != visibility:
                match = False
            
            # Tags filter
            if tags and not any(tag in workspace.tags for tag in tags):
                match = False
            
            if match:
                results.append(workspace)
        
        return results
    
    async def archive_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Archive a workspace."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Check permissions
        if not await self._can_manage_workspace(workspace, user_id):
            return False
        
        workspace.is_archived = True
        workspace.archived_at = datetime.utcnow()
        workspace.updated_at = datetime.utcnow()
        
        # Store updated workspace
        await self._store_workspace(workspace)
        
        # Create event
        await self._create_event(workspace_id, "workspace_archived", user_id, {
            'name': workspace.name
        })
        
        logging.info(f"Archived workspace: {workspace.name} ({workspace_id})")
        return True
    
    async def unarchive_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Unarchive a workspace."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Check permissions
        if not await self._can_manage_workspace(workspace, user_id):
            return False
        
        workspace.is_archived = False
        workspace.archived_at = None
        workspace.updated_at = datetime.utcnow()
        
        # Store updated workspace
        await self._store_workspace(workspace)
        
        # Create event
        await self._create_event(workspace_id, "workspace_unarchived", user_id, {
            'name': workspace.name
        })
        
        logging.info(f"Unarchived workspace: {workspace.name} ({workspace_id})")
        return True
    
    async def delete_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Delete a workspace (only owner can delete)."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Only owner can delete
        if user_id != workspace.owner_id:
            return False
        
        # Remove from all members' workspace lists
        for member_id in workspace.members.keys():
            if member_id in self.user_workspaces:
                self.user_workspaces[member_id].discard(workspace_id)
            await self.redis.srem(f"user_workspaces:{member_id}", workspace_id)
        
        # Remove from cache
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
        
        # Delete from Redis
        await self.redis.delete(f"workspace:{workspace_id}")
        
        # Create event
        await self._create_event(workspace_id, "workspace_deleted", user_id, {
            'name': workspace.name
        })
        
        logging.info(f"Deleted workspace: {workspace.name} ({workspace_id})")
        return True
    
    async def _can_manage_workspace(self, workspace: Workspace, user_id: str) -> bool:
        """Check if user can manage workspace settings."""
        if user_id == workspace.owner_id:
            return True
        
        member = workspace.members.get(user_id)
        return member and member.has_permission(PermissionType.MANAGE_SETTINGS)
    
    async def _can_manage_members(self, workspace: Workspace, user_id: str) -> bool:
        """Check if user can manage workspace members."""
        if user_id == workspace.owner_id:
            return True
        
        member = workspace.members.get(user_id)
        return member and member.has_permission(PermissionType.MANAGE_MEMBERS)
    
    async def _store_workspace(self, workspace: Workspace):
        """Store workspace in Redis."""
        try:
            data = json.dumps(workspace.to_dict())
            await self.redis.set(f"workspace:{workspace.workspace_id}", data)
            
            # Update user workspace mappings
            for member_id in workspace.members.keys():
                await self.redis.sadd(f"user_workspaces:{member_id}", workspace.workspace_id)
            
        except Exception as e:
            logging.error(f"Error storing workspace: {e}")
    
    async def _create_event(self, workspace_id: str, event_type: str, user_id: str, data: Dict[str, Any]):
        """Create a workspace event."""
        event = {
            'event_id': str(uuid.uuid4()),
            'workspace_id': workspace_id,
            'event_type': event_type,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Store event
        try:
            event_data = json.dumps(event)
            await self.redis.lpush(f"workspace_events:{workspace_id}", event_data)
            await self.redis.ltrim(f"workspace_events:{workspace_id}", 0, 999)  # Keep last 1000 events
        except Exception as e:
            logging.error(f"Error storing workspace event: {e}")
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logging.error(f"Error in workspace event callback: {e}")
    
    async def _load_workspaces(self):
        """Load workspaces from Redis."""
        try:
            workspace_keys = await self.redis.keys("workspace:*")
            for key in workspace_keys:
                workspace_data = await self.redis.get(key)
                if workspace_data:
                    try:
                        data = json.loads(workspace_data)
                        workspace = Workspace.from_dict(data)
                        self.workspaces[workspace.workspace_id] = workspace
                        
                        # Update user workspace mappings
                        for member_id in workspace.members.keys():
                            if member_id not in self.user_workspaces:
                                self.user_workspaces[member_id] = set()
                            self.user_workspaces[member_id].add(workspace.workspace_id)
                            
                    except Exception as e:
                        logging.error(f"Error loading workspace: {e}")
        except Exception as e:
            logging.error(f"Error loading workspaces: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of workspace data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_inactive_workspaces()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in workspace cleanup: {e}")
    
    async def _cleanup_inactive_workspaces(self):
        """Clean up inactive workspaces."""
        threshold = datetime.utcnow() - timedelta(days=90)
        workspaces_to_archive = []
        
        for workspace in self.workspaces.values():
            if (workspace.settings.auto_archive_inactive and
                not workspace.is_archived and
                workspace.updated_at < threshold):
                workspaces_to_archive.append(workspace.workspace_id)
        
        # Auto-archive inactive workspaces
        for workspace_id in workspaces_to_archive:
            await self.archive_workspace(workspace_id, "system")
            logging.info(f"Auto-archived inactive workspace: {workspace_id}")
    
    def get_workspace_stats(self) -> Dict[str, Any]:
        """Get workspace manager statistics."""
        total_workspaces = len(self.workspaces)
        archived_workspaces = sum(1 for w in self.workspaces.values() if w.is_archived)
        total_members = sum(len(w.members) for w in self.workspaces.values())
        
        workspace_types = {}
        for workspace in self.workspaces.values():
            workspace_type = workspace.workspace_type.value
            workspace_types[workspace_type] = workspace_types.get(workspace_type, 0) + 1
        
        return {
            'total_workspaces': total_workspaces,
            'active_workspaces': total_workspaces - archived_workspaces,
            'archived_workspaces': archived_workspaces,
            'total_members': total_members,
            'workspace_types': workspace_types,
            'users_with_workspaces': len(self.user_workspaces)
        }


# Global instance
workspace_manager = WorkspaceManager()