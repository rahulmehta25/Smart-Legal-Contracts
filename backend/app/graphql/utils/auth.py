"""
Authentication and Authorization utilities for GraphQL
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List
from strawberry.types import Info
from functools import wraps

from ...models.user import User
from ...core.config import settings


class AuthenticationError(Exception):
    """Authentication error"""
    pass


class AuthorizationError(Exception):
    """Authorization error"""
    pass


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_access_token(username: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode = {
        "sub": username,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        getattr(settings, 'SECRET_KEY', 'secret'), 
        algorithm="HS256"
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode JWT access token"""
    try:
        payload = jwt.decode(
            token, 
            getattr(settings, 'SECRET_KEY', 'secret'), 
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid token")


async def get_current_user(info: Info) -> Optional[User]:
    """Get current user from GraphQL context"""
    try:
        request = info.context.get("request")
        if not request:
            return None
        
        # Try to get token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
        
        # Extract token
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                return None
        except ValueError:
            return None
        
        # Decode token
        payload = decode_access_token(token)
        username = payload.get("sub")
        
        if not username:
            return None
        
        # Get user from database
        session = info.context.get("session")
        if not session:
            return None
        
        user = session.query(User).filter(User.username == username).first()
        return user
        
    except Exception:
        return None


def require_auth(role: Optional[str] = None):
    """Decorator to require authentication and optionally specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Info argument
            info = None
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            if not info:
                raise AuthenticationError("No context available")
            
            # Get current user
            user = await get_current_user(info)
            if not user:
                raise AuthenticationError("Authentication required")
            
            if not user.is_active:
                raise AuthenticationError("Account disabled")
            
            # Check role if specified
            if role:
                user_role = getattr(user, 'role', 'USER')
                if user_role != role and user_role != 'ADMIN':
                    raise AuthorizationError(f"Role {role} required")
            
            # Add user to context
            info.context["current_user"] = user
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class PermissionChecker:
    """Check permissions for GraphQL operations"""
    
    def __init__(self):
        self.role_hierarchy = {
            'VIEWER': 0,
            'USER': 1,
            'ANALYST': 2,
            'ADMIN': 3
        }
    
    def has_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource/action"""
        if not user or not user.is_active:
            return False
        
        user_role = getattr(user, 'role', 'USER')
        
        # Admin can do everything
        if user_role == 'ADMIN':
            return True
        
        # Define permission matrix
        permissions = {
            'document': {
                'read': ['VIEWER', 'USER', 'ANALYST', 'ADMIN'],
                'create': ['USER', 'ANALYST', 'ADMIN'],
                'update': ['USER', 'ANALYST', 'ADMIN'],
                'delete': ['USER', 'ANALYST', 'ADMIN'],
            },
            'analysis': {
                'read': ['VIEWER', 'USER', 'ANALYST', 'ADMIN'],
                'create': ['USER', 'ANALYST', 'ADMIN'],
                'update': ['ANALYST', 'ADMIN'],
                'delete': ['ANALYST', 'ADMIN'],
            },
            'pattern': {
                'read': ['ANALYST', 'ADMIN'],
                'create': ['ADMIN'],
                'update': ['ADMIN'],
                'delete': ['ADMIN'],
            },
            'user': {
                'read': ['USER', 'ANALYST', 'ADMIN'],
                'create': ['ADMIN'],
                'update': ['ADMIN'],
                'delete': ['ADMIN'],
            },
            'system': {
                'read': ['ADMIN'],
                'update': ['ADMIN'],
                'delete': ['ADMIN'],
            }
        }
        
        resource_permissions = permissions.get(resource, {})
        allowed_roles = resource_permissions.get(action, [])
        
        return user_role in allowed_roles
    
    def can_access_document(self, user: User, document_id: str) -> bool:
        """Check if user can access specific document"""
        # For now, allow all authenticated users
        # In real implementation, check document ownership/sharing
        return user is not None and user.is_active
    
    def can_modify_document(self, user: User, document_id: str) -> bool:
        """Check if user can modify specific document"""
        # For now, allow all users with update permission
        # In real implementation, check document ownership
        return self.has_permission(user, 'document', 'update')


# Global permission checker
permission_checker = PermissionChecker()


def check_permission(resource: str, action: str):
    """Decorator to check permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Info argument
            info = None
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            if not info:
                raise AuthenticationError("No context available")
            
            # Get current user
            user = await get_current_user(info)
            if not user:
                raise AuthenticationError("Authentication required")
            
            # Check permission
            if not permission_checker.has_permission(user, resource, action):
                raise AuthorizationError(f"Permission denied for {action} on {resource}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_document_access(document_id_arg: str = "document_id"):
    """Decorator to check document access"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Info argument and document ID
            info = None
            document_id = None
            
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            # Get document ID from kwargs
            document_id = kwargs.get(document_id_arg)
            
            if not document_id:
                # Try to get from input argument
                input_arg = kwargs.get('input')
                if input_arg and hasattr(input_arg, document_id_arg):
                    document_id = getattr(input_arg, document_id_arg)
            
            if not info or not document_id:
                raise AuthenticationError("Missing context or document ID")
            
            # Get current user
            user = await get_current_user(info)
            if not user:
                raise AuthenticationError("Authentication required")
            
            # Check document access
            if not permission_checker.can_access_document(user, document_id):
                raise AuthorizationError("Access denied to document")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityContext:
    """Security context for GraphQL operations"""
    
    def __init__(self, user: Optional[User] = None, permissions: List[str] = None):
        self.user = user
        self.permissions = permissions or []
        self.is_authenticated = user is not None
        self.is_admin = user and getattr(user, 'role', '') == 'ADMIN'
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions or self.is_admin
    
    def can_access_resource(self, resource_id: str, resource_type: str) -> bool:
        """Check if context can access specific resource"""
        if self.is_admin:
            return True
        
        # Implement resource-specific access logic
        return self.is_authenticated


def create_security_context(info: Info) -> SecurityContext:
    """Create security context from GraphQL info"""
    user = info.context.get("current_user")
    
    if not user:
        return SecurityContext()
    
    # Determine user permissions based on role
    role = getattr(user, 'role', 'USER')
    permissions = []
    
    if role == 'ADMIN':
        permissions = ['*']  # All permissions
    elif role == 'ANALYST':
        permissions = ['read:*', 'create:analysis', 'update:analysis', 'read:patterns']
    elif role == 'USER':
        permissions = ['read:documents', 'create:documents', 'create:analysis']
    elif role == 'VIEWER':
        permissions = ['read:documents']
    
    return SecurityContext(user, permissions)


# Field-level authorization
def authorize_field(info: Info, field_name: str, parent_type: str = None) -> bool:
    """Authorize access to specific field"""
    security_context = create_security_context(info)
    
    # Define field authorization rules
    protected_fields = {
        'User.email': ['read:user_details'],
        'User.hashedPassword': [],  # Never accessible
        'Document.metadata': ['read:document_metadata'],
        'ArbitrationAnalysis.metadata': ['read:analysis_metadata'],
        'SystemStats': ['read:system_stats'],
        'Pattern': ['read:patterns'],
    }
    
    field_key = f"{parent_type}.{field_name}" if parent_type else field_name
    required_permissions = protected_fields.get(field_key, [])
    
    # If no permissions required, allow access
    if not required_permissions:
        return True
    
    # Check if user has any of the required permissions
    return any(
        security_context.has_permission(perm) 
        for perm in required_permissions
    )


# OAuth2 scopes
class OAuth2Scopes:
    """OAuth2 scope definitions"""
    
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    
    READ_ANALYSES = "read:analyses"
    WRITE_ANALYSES = "write:analyses"
    DELETE_ANALYSES = "delete:analyses"
    
    READ_PATTERNS = "read:patterns"
    WRITE_PATTERNS = "write:patterns"
    DELETE_PATTERNS = "delete:patterns"
    
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    DELETE_USERS = "delete:users"
    
    READ_SYSTEM = "read:system"
    WRITE_SYSTEM = "write:system"
    
    ADMIN = "admin"


def require_scopes(*required_scopes):
    """Decorator to require OAuth2 scopes"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Info argument
            info = None
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            if not info:
                raise AuthenticationError("No context available")
            
            # Get token scopes from context
            token_scopes = info.context.get("token_scopes", [])
            
            # Check if any required scope is present
            if not any(scope in token_scopes for scope in required_scopes):
                raise AuthorizationError(f"Required scopes: {', '.join(required_scopes)}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator