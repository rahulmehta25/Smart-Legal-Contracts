"""
Role-Based Access Control (RBAC) System

Comprehensive permissions management with:
- Hierarchical roles
- Granular permissions
- Dynamic role assignment
- Permission inheritance
- Resource-based permissions
- Audit logging
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis
import json
from abc import ABC, abstractmethod


class Permission(str, Enum):
    """Granular system permissions"""
    
    # Document Management
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_SHARE = "document:share"
    DOCUMENT_EXPORT = "document:export"
    DOCUMENT_BULK_UPLOAD = "document:bulk_upload"
    
    # Analysis Operations
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"
    ANALYSIS_APPROVE = "analysis:approve"
    ANALYSIS_PUBLISH = "analysis:publish"
    ANALYSIS_BULK_PROCESS = "analysis:bulk_process"
    
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_IMPERSONATE = "user:impersonate"
    USER_MANAGE_ROLES = "user:manage_roles"
    USER_RESET_PASSWORD = "user:reset_password"
    USER_VIEW_ACTIVITY = "user:view_activity"
    
    # Organization Management
    ORG_CREATE = "org:create"
    ORG_READ = "org:read"
    ORG_UPDATE = "org:update"
    ORG_DELETE = "org:delete"
    ORG_MANAGE_MEMBERS = "org:manage_members"
    ORG_MANAGE_BILLING = "org:manage_billing"
    ORG_VIEW_USAGE = "org:view_usage"
    
    # API Management
    API_KEY_CREATE = "api_key:create"
    API_KEY_READ = "api_key:read"
    API_KEY_REVOKE = "api_key:revoke"
    API_KEY_MANAGE = "api_key:manage"
    API_RATE_LIMIT_OVERRIDE = "api:rate_limit_override"
    
    # System Administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"
    SYSTEM_BACKUP = "system:backup"
    SYSTEM_RESTORE = "system:restore"
    SYSTEM_MAINTENANCE = "system:maintenance"
    SYSTEM_MONITOR = "system:monitor"
    
    # Security Management
    SECURITY_VIEW_LOGS = "security:view_logs"
    SECURITY_MANAGE_2FA = "security:manage_2fa"
    SECURITY_VIEW_SESSIONS = "security:view_sessions"
    SECURITY_TERMINATE_SESSIONS = "security:terminate_sessions"
    SECURITY_IP_WHITELIST = "security:ip_whitelist"
    
    # Billing and Subscriptions
    BILLING_VIEW = "billing:view"
    BILLING_MANAGE = "billing:manage"
    BILLING_DOWNLOAD_INVOICES = "billing:download_invoices"
    
    # Analytics and Reporting
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    ANALYTICS_ADVANCED = "analytics:advanced"
    
    # Compliance and Legal
    COMPLIANCE_VIEW = "compliance:view"
    COMPLIANCE_MANAGE = "compliance:manage"
    COMPLIANCE_AUDIT = "compliance:audit"
    
    # Integration Management
    INTEGRATION_CREATE = "integration:create"
    INTEGRATION_MANAGE = "integration:manage"
    INTEGRATION_DELETE = "integration:delete"


class Role(str, Enum):
    """System roles with hierarchical structure"""
    
    # Hierarchical roles (higher roles inherit lower role permissions)
    SUPER_ADMIN = "super_admin"          # Level 7 - All permissions
    ADMIN = "admin"                      # Level 6 - Organization administration
    MANAGER = "manager"                  # Level 5 - Team and resource management
    SENIOR_ANALYST = "senior_analyst"    # Level 4 - Advanced analysis operations
    ANALYST = "analyst"                  # Level 3 - Standard analysis operations
    REVIEWER = "reviewer"                # Level 2 - Review and approval
    VIEWER = "viewer"                    # Level 1 - Read-only access
    
    # Special purpose roles
    API_USER = "api_user"               # Programmatic API access
    BILLING_ADMIN = "billing_admin"     # Billing and subscription management
    COMPLIANCE_OFFICER = "compliance_officer"  # Compliance and audit
    INTEGRATION_MANAGER = "integration_manager"  # Integration management
    
    # Client roles
    CLIENT_ADMIN = "client_admin"       # Client organization admin
    CLIENT_USER = "client_user"         # Client organization user
    
    # Temporary/Limited roles
    GUEST = "guest"                     # Limited temporary access
    DEMO_USER = "demo_user"            # Demo/trial access


@dataclass
class RoleDefinition:
    """Role definition with permissions and metadata"""
    name: Role
    display_name: str
    description: str
    permissions: Set[Permission]
    level: int  # Hierarchical level (higher inherits lower)
    can_be_assigned_by: List[Role] = field(default_factory=list)
    max_duration_days: Optional[int] = None  # Role expiration
    requires_approval: bool = False
    is_system_role: bool = True


@dataclass
class UserRole:
    """User role assignment with metadata"""
    user_id: str
    role: Role
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePermission:
    """Resource-specific permission"""
    resource_type: str  # e.g., "document", "analysis", "organization"
    resource_id: str
    permission: Permission
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None


class PermissionChecker(ABC):
    """Abstract base class for permission checking strategies"""
    
    @abstractmethod
    def has_permission(self, user_roles: List[Role], 
                      user_permissions: Set[Permission],
                      required_permission: Permission,
                      resource_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> bool:
        pass


class StandardPermissionChecker(PermissionChecker):
    """Standard permission checking implementation"""
    
    def has_permission(self, user_roles: List[Role],
                      user_permissions: Set[Permission],
                      required_permission: Permission,
                      resource_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> bool:
        return required_permission in user_permissions


class ContextualPermissionChecker(PermissionChecker):
    """Contextual permission checking with business logic"""
    
    def has_permission(self, user_roles: List[Role],
                      user_permissions: Set[Permission],
                      required_permission: Permission,
                      resource_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> bool:
        
        # First check standard permissions
        if required_permission in user_permissions:
            return True
        
        # Apply contextual business rules
        if context:
            # Example: Allow document creators to update their own documents
            if (required_permission == Permission.DOCUMENT_UPDATE and
                context.get('is_owner') == True):
                return True
            
            # Example: Allow team members to read team documents
            if (required_permission == Permission.DOCUMENT_READ and
                context.get('is_team_member') == True):
                return True
        
        return False


class RoleManager:
    """
    Comprehensive role and permission management system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize role manager
        
        Args:
            redis_client: Redis client for caching and storage
        """
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Permission checker strategy
        self.permission_checker: PermissionChecker = ContextualPermissionChecker()
        
        # Initialize role definitions
        self.role_definitions = self._initialize_role_definitions()
        
        # Cache role definitions in Redis
        self._cache_role_definitions()
    
    def _initialize_role_definitions(self) -> Dict[Role, RoleDefinition]:
        """Initialize all role definitions with their permissions"""
        
        # Define permissions for each role
        role_permissions = {
            Role.SUPER_ADMIN: set(Permission),  # All permissions
            
            Role.ADMIN: {
                # Document permissions (all except bulk operations)
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_DELETE,
                Permission.DOCUMENT_SHARE, Permission.DOCUMENT_EXPORT,
                
                # Analysis permissions (all except bulk)
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_DELETE,
                Permission.ANALYSIS_APPROVE, Permission.ANALYSIS_PUBLISH,
                
                # User management
                Permission.USER_CREATE, Permission.USER_READ,
                Permission.USER_UPDATE, Permission.USER_MANAGE_ROLES,
                Permission.USER_RESET_PASSWORD, Permission.USER_VIEW_ACTIVITY,
                
                # Organization management
                Permission.ORG_READ, Permission.ORG_UPDATE,
                Permission.ORG_MANAGE_MEMBERS, Permission.ORG_VIEW_USAGE,
                
                # API management
                Permission.API_KEY_CREATE, Permission.API_KEY_READ,
                Permission.API_KEY_REVOKE, Permission.API_KEY_MANAGE,
                
                # Security
                Permission.SECURITY_VIEW_LOGS, Permission.SECURITY_VIEW_SESSIONS,
                Permission.SECURITY_TERMINATE_SESSIONS,
                
                # Analytics
                Permission.ANALYTICS_VIEW, Permission.ANALYTICS_EXPORT,
            },
            
            Role.MANAGER: {
                # Document permissions
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_SHARE,
                Permission.DOCUMENT_EXPORT,
                
                # Analysis permissions
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_APPROVE,
                
                # Limited user management
                Permission.USER_READ, Permission.USER_UPDATE,
                
                # Organization viewing
                Permission.ORG_READ, Permission.ORG_VIEW_USAGE,
                
                # Analytics
                Permission.ANALYTICS_VIEW,
            },
            
            Role.SENIOR_ANALYST: {
                # Document permissions
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_SHARE,
                
                # Analysis permissions
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_APPROVE,
                
                # Basic analytics
                Permission.ANALYTICS_VIEW,
            },
            
            Role.ANALYST: {
                # Document permissions
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE,
                
                # Analysis permissions
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.ANALYSIS_UPDATE,
            },
            
            Role.REVIEWER: {
                # Limited document access
                Permission.DOCUMENT_READ,
                
                # Review permissions
                Permission.ANALYSIS_READ, Permission.ANALYSIS_APPROVE,
            },
            
            Role.VIEWER: {
                # Read-only access
                Permission.DOCUMENT_READ, Permission.ANALYSIS_READ,
            },
            
            Role.API_USER: {
                # Programmatic access
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.API_KEY_READ,
            },
            
            Role.BILLING_ADMIN: {
                # Billing permissions
                Permission.BILLING_VIEW, Permission.BILLING_MANAGE,
                Permission.BILLING_DOWNLOAD_INVOICES,
                Permission.ORG_MANAGE_BILLING,
            },
            
            Role.COMPLIANCE_OFFICER: {
                # Compliance permissions
                Permission.COMPLIANCE_VIEW, Permission.COMPLIANCE_MANAGE,
                Permission.COMPLIANCE_AUDIT,
                Permission.SECURITY_VIEW_LOGS, Permission.SYSTEM_AUDIT,
                Permission.DOCUMENT_READ, Permission.ANALYSIS_READ,
            },
            
            Role.CLIENT_ADMIN: {
                # Client organization management
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_SHARE,
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
                Permission.USER_READ, Permission.USER_UPDATE,
                Permission.ORG_READ, Permission.ORG_VIEW_USAGE,
                Permission.BILLING_VIEW,
            },
            
            Role.CLIENT_USER: {
                # Basic client access
                Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
                Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
            },
            
            Role.GUEST: {
                # Very limited access
                Permission.DOCUMENT_READ, Permission.ANALYSIS_READ,
            },
            
            Role.DEMO_USER: {
                # Demo access
                Permission.DOCUMENT_READ, Permission.ANALYSIS_READ,
                Permission.DOCUMENT_CREATE, Permission.ANALYSIS_CREATE,
            }
        }
        
        # Create role definitions
        definitions = {}
        
        for role, permissions in role_permissions.items():
            # Determine hierarchy level
            level_map = {
                Role.SUPER_ADMIN: 7,
                Role.ADMIN: 6,
                Role.MANAGER: 5,
                Role.SENIOR_ANALYST: 4,
                Role.ANALYST: 3,
                Role.REVIEWER: 2,
                Role.VIEWER: 1,
                Role.API_USER: 0,
                Role.BILLING_ADMIN: 0,
                Role.COMPLIANCE_OFFICER: 0,
                Role.INTEGRATION_MANAGER: 0,
                Role.CLIENT_ADMIN: 3,
                Role.CLIENT_USER: 2,
                Role.GUEST: 0,
                Role.DEMO_USER: 0
            }
            
            # Who can assign this role
            assignment_permissions = {
                Role.SUPER_ADMIN: [Role.SUPER_ADMIN],
                Role.ADMIN: [Role.SUPER_ADMIN],
                Role.MANAGER: [Role.SUPER_ADMIN, Role.ADMIN],
                Role.SENIOR_ANALYST: [Role.SUPER_ADMIN, Role.ADMIN, Role.MANAGER],
                Role.ANALYST: [Role.SUPER_ADMIN, Role.ADMIN, Role.MANAGER],
                Role.REVIEWER: [Role.SUPER_ADMIN, Role.ADMIN, Role.MANAGER],
                Role.VIEWER: [Role.SUPER_ADMIN, Role.ADMIN, Role.MANAGER],
                Role.CLIENT_ADMIN: [Role.SUPER_ADMIN, Role.ADMIN],
                Role.CLIENT_USER: [Role.SUPER_ADMIN, Role.ADMIN, Role.CLIENT_ADMIN],
            }
            
            definitions[role] = RoleDefinition(
                name=role,
                display_name=role.value.replace('_', ' ').title(),
                description=f"Role: {role.value}",
                permissions=permissions,
                level=level_map.get(role, 0),
                can_be_assigned_by=assignment_permissions.get(role, [Role.SUPER_ADMIN]),
                requires_approval=role in [Role.ADMIN, Role.SUPER_ADMIN]
            )
        
        return definitions
    
    def _cache_role_definitions(self):
        """Cache role definitions in Redis for fast access"""
        try:
            for role, definition in self.role_definitions.items():
                cache_key = f"role_def:{role.value}"
                cache_data = {
                    "permissions": [p.value for p in definition.permissions],
                    "level": definition.level,
                    "display_name": definition.display_name,
                    "description": definition.description
                }
                self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))
        except Exception:
            pass  # Continue without caching if Redis unavailable
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role including inherited permissions"""
        if role not in self.role_definitions:
            return set()
        
        role_def = self.role_definitions[role]
        permissions = role_def.permissions.copy()
        
        # Add inherited permissions from lower-level roles
        current_level = role_def.level
        for other_role, other_def in self.role_definitions.items():
            if other_def.level < current_level and other_def.is_system_role:
                permissions.update(other_def.permissions)
        
        return permissions
    
    def get_user_permissions(self, user_roles: List[Role]) -> Set[Permission]:
        """Get all permissions for a user based on their roles"""
        all_permissions = set()
        
        for role in user_roles:
            role_permissions = self.get_role_permissions(role)
            all_permissions.update(role_permissions)
        
        return all_permissions
    
    def has_permission(self, user_roles: List[Role],
                      required_permission: Permission,
                      resource_id: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if user has required permission
        
        Args:
            user_roles: User's assigned roles
            required_permission: Permission to check
            resource_id: Optional resource identifier
            context: Additional context for permission checking
            
        Returns:
            True if user has permission
        """
        user_permissions = self.get_user_permissions(user_roles)
        
        return self.permission_checker.has_permission(
            user_roles=user_roles,
            user_permissions=user_permissions,
            required_permission=required_permission,
            resource_id=resource_id,
            context=context
        )
    
    def has_any_permission(self, user_roles: List[Role],
                          required_permissions: List[Permission],
                          resource_id: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has any of the required permissions"""
        for permission in required_permissions:
            if self.has_permission(user_roles, permission, resource_id, context):
                return True
        return False
    
    def has_all_permissions(self, user_roles: List[Role],
                           required_permissions: List[Permission],
                           resource_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has all required permissions"""
        for permission in required_permissions:
            if not self.has_permission(user_roles, permission, resource_id, context):
                return False
        return True
    
    def can_assign_role(self, assigner_roles: List[Role], target_role: Role) -> bool:
        """Check if user can assign a specific role"""
        if target_role not in self.role_definitions:
            return False
        
        target_def = self.role_definitions[target_role]
        
        for assigner_role in assigner_roles:
            if assigner_role in target_def.can_be_assigned_by:
                return True
        
        return False
    
    def get_assignable_roles(self, user_roles: List[Role]) -> List[Role]:
        """Get list of roles that user can assign to others"""
        assignable = []
        
        for role, definition in self.role_definitions.items():
            if any(user_role in definition.can_be_assigned_by for user_role in user_roles):
                assignable.append(role)
        
        return assignable
    
    def assign_role(self, user_id: str, role: Role, 
                   assigned_by: str, duration_days: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> UserRole:
        """
        Assign role to user
        
        Args:
            user_id: Target user ID
            role: Role to assign
            assigned_by: User ID who is assigning the role
            duration_days: Optional role expiration in days
            metadata: Additional role metadata
            
        Returns:
            UserRole assignment record
        """
        expires_at = None
        if duration_days:
            expires_at = datetime.utcnow() + timedelta(days=duration_days)
        
        user_role = UserRole(
            user_id=user_id,
            role=role,
            granted_by=assigned_by,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Store in Redis (in production, use persistent database)
        try:
            role_key = f"user_roles:{user_id}"
            role_data = {
                "role": role.value,
                "granted_by": assigned_by,
                "granted_at": user_role.granted_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "is_active": True,
                "metadata": json.dumps(metadata or {})
            }
            
            self.redis_client.hset(f"{role_key}:{role.value}", mapping=role_data)
            
            # Add to user's role list
            self.redis_client.sadd(role_key, role.value)
            
            # Set expiration if specified
            if expires_at:
                ttl = int((expires_at - datetime.utcnow()).total_seconds())
                self.redis_client.expire(f"{role_key}:{role.value}", ttl)
        
        except Exception:
            pass  # Continue without Redis if unavailable
        
        return user_role
    
    def revoke_role(self, user_id: str, role: Role, revoked_by: str) -> bool:
        """
        Revoke role from user
        
        Args:
            user_id: Target user ID
            role: Role to revoke
            revoked_by: User ID who is revoking the role
            
        Returns:
            True if role was revoked
        """
        try:
            role_key = f"user_roles:{user_id}"
            
            # Mark role as inactive
            self.redis_client.hset(
                f"{role_key}:{role.value}",
                "is_active", 
                False
            )
            
            # Remove from user's active role list
            self.redis_client.srem(role_key, role.value)
            
            return True
        except Exception:
            return False
    
    def get_user_roles(self, user_id: str) -> List[UserRole]:
        """Get all active roles for a user"""
        try:
            role_key = f"user_roles:{user_id}"
            role_names = self.redis_client.smembers(role_key)
            
            user_roles = []
            for role_name in role_names:
                role_data = self.redis_client.hgetall(f"{role_key}:{role_name}")
                
                if role_data and role_data.get("is_active") == "True":
                    # Check if role has expired
                    expires_at = role_data.get("expires_at")
                    if expires_at:
                        expires_dt = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expires_dt:
                            # Role expired, remove it
                            self.revoke_role(user_id, Role(role_name), "system")
                            continue
                    
                    user_role = UserRole(
                        user_id=user_id,
                        role=Role(role_name),
                        granted_by=role_data.get("granted_by", ""),
                        granted_at=datetime.fromisoformat(role_data.get("granted_at", "1970-01-01")),
                        expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
                        is_active=True,
                        metadata=json.loads(role_data.get("metadata", "{}"))
                    )
                    user_roles.append(user_role)
            
            return user_roles
        except Exception:
            return []
    
    def get_role_hierarchy(self) -> Dict[str, Any]:
        """Get role hierarchy visualization"""
        hierarchy = {}
        
        # Group roles by level
        levels = {}
        for role, definition in self.role_definitions.items():
            level = definition.level
            if level not in levels:
                levels[level] = []
            levels[level].append({
                "role": role.value,
                "display_name": definition.display_name,
                "permissions_count": len(definition.permissions),
                "can_be_assigned_by": [r.value for r in definition.can_be_assigned_by]
            })
        
        # Sort by level
        for level in sorted(levels.keys(), reverse=True):
            hierarchy[f"Level {level}"] = levels[level]
        
        return hierarchy