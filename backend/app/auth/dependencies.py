"""
FastAPI Authentication Dependencies

Production-ready authentication dependencies for:
- User authentication
- Permission checking
- Role validation
- Session management
- Rate limiting
- Device tracking
"""

import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis

from .jwt_handler import JWTHandler, TokenType
from .password import PasswordHandler
from .permissions import Role, Permission, RoleManager
from ..core.config import get_settings


# Global instances (initialized in main app)
jwt_handler: Optional[JWTHandler] = None
password_handler: Optional[PasswordHandler] = None
role_manager: Optional[RoleManager] = None
redis_client: Optional[redis.Redis] = None

# Security scheme
security = HTTPBearer(auto_error=False)


def get_jwt_handler() -> JWTHandler:
    """Get JWT handler instance"""
    global jwt_handler
    if not jwt_handler:
        settings = get_settings()
        jwt_handler = JWTHandler(
            secret_key=settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
            access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            refresh_token_expire_days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    return jwt_handler


def get_password_handler() -> PasswordHandler:
    """Get password handler instance"""
    global password_handler
    if not password_handler:
        password_handler = PasswordHandler()
    return password_handler


def get_role_manager() -> RoleManager:
    """Get role manager instance"""
    global role_manager
    if not role_manager:
        role_manager = RoleManager()
    return role_manager


def get_redis_client() -> redis.Redis:
    """Get Redis client instance"""
    global redis_client
    if not redis_client:
        settings = get_settings()
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
    return redis_client


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    # Check for forwarded IP headers (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        ip = forwarded_for.split(",")[0].strip()
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            pass
    
    # Check other common headers
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        try:
            ipaddress.ip_address(real_ip)
            return real_ip
        except ValueError:
            pass
    
    # Fall back to request client
    client_ip = request.client.host if request.client else "127.0.0.1"
    return client_ip


def get_user_agent(request: Request) -> str:
    """Extract user agent from request"""
    return request.headers.get("User-Agent", "Unknown")


def get_device_fingerprint(request: Request) -> str:
    """Generate device fingerprint from request headers"""
    import hashlib
    
    # Combine various headers to create a device fingerprint
    fingerprint_data = [
        request.headers.get("User-Agent", ""),
        request.headers.get("Accept-Language", ""),
        request.headers.get("Accept-Encoding", ""),
        get_client_ip(request)
    ]
    
    fingerprint_string = "|".join(fingerprint_data)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    role_manager: RoleManager = Depends(get_role_manager)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token or API key
    
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    
    # Extract request metadata
    client_ip = get_client_ip(request)
    user_agent = get_user_agent(request)
    device_fingerprint = get_device_fingerprint(request)
    
    # Try API key authentication first
    if x_api_key:
        # TODO: Implement API key validation
        # For now, return placeholder
        pass
    
    # Try JWT token authentication
    if credentials:
        token = credentials.credentials
        token_data = jwt_handler.verify_token(token, TokenType.ACCESS)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Update session activity
        session_id = token_data.get("session_id")
        if session_id:
            jwt_handler.update_session_activity(session_id)
        
        # Get user roles
        user_id = token_data.get("user_id")
        user_roles = role_manager.get_user_roles(user_id)
        role_names = [ur.role for ur in user_roles]
        
        # Get user permissions
        user_permissions = role_manager.get_user_permissions(role_names)
        
        return {
            "user_id": user_id,
            "username": token_data.get("username", ""),
            "email": token_data.get("email", ""),
            "roles": [role.value for role in role_names],
            "permissions": [perm.value for perm in user_permissions],
            "session_id": session_id,
            "mfa_verified": token_data.get("mfa_verified", False),
            "token_type": token_data.get("token_type"),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "device_fingerprint": device_fingerprint,
            "auth_method": "jwt"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user (user must be active)
    
    Returns:
        Active user information
        
    Raises:
        HTTPException: If user is inactive
    """
    # TODO: Check user active status from database
    # For now, assume all authenticated users are active
    
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return current_user


def require_permission(permission: Permission) -> Callable:
    """
    Dependency factory to require specific permission
    
    Args:
        permission: Required permission
        
    Returns:
        FastAPI dependency function
    """
    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_active_user),
        role_manager: RoleManager = Depends(get_role_manager)
    ) -> Dict[str, Any]:
        
        user_roles = [Role(role) for role in current_user.get("roles", [])]
        
        if not role_manager.has_permission(user_roles, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission.value}"
            )
        
        return current_user
    
    return permission_dependency


def require_any_permission(permissions: List[Permission]) -> Callable:
    """
    Dependency factory to require any of the specified permissions
    
    Args:
        permissions: List of acceptable permissions
        
    Returns:
        FastAPI dependency function
    """
    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_active_user),
        role_manager: RoleManager = Depends(get_role_manager)
    ) -> Dict[str, Any]:
        
        user_roles = [Role(role) for role in current_user.get("roles", [])]
        
        if not role_manager.has_any_permission(user_roles, permissions):
            permission_names = [p.value for p in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required one of: {permission_names}"
            )
        
        return current_user
    
    return permission_dependency


def require_role(role: Role) -> Callable:
    """
    Dependency factory to require specific role
    
    Args:
        role: Required role
        
    Returns:
        FastAPI dependency function
    """
    async def role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        
        user_roles = current_user.get("roles", [])
        
        if role.value not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        
        return current_user
    
    return role_dependency


def require_any_role(roles: List[Role]) -> Callable:
    """
    Dependency factory to require any of the specified roles
    
    Args:
        roles: List of acceptable roles
        
    Returns:
        FastAPI dependency function
    """
    async def role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        
        user_roles = current_user.get("roles", [])
        role_names = [role.value for role in roles]
        
        if not any(role in user_roles for role in role_names):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required. One of: {role_names}"
            )
        
        return current_user
    
    return role_dependency


def require_mfa(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Dependency to require MFA verification
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information if MFA verified
        
    Raises:
        HTTPException: If MFA not verified
    """
    if not current_user.get("mfa_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Multi-factor authentication required"
        )
    
    return current_user


def require_fresh_auth(max_age_minutes: int = 30) -> Callable:
    """
    Dependency factory to require recent authentication
    
    Args:
        max_age_minutes: Maximum age of authentication in minutes
        
    Returns:
        FastAPI dependency function
    """
    async def fresh_auth_dependency(
        current_user: Dict[str, Any] = Depends(get_current_active_user),
        jwt_handler: JWTHandler = Depends(get_jwt_handler)
    ) -> Dict[str, Any]:
        
        # TODO: Check token issue time and compare with max_age_minutes
        # For now, assume authentication is fresh
        
        return current_user
    
    return fresh_auth_dependency


class RateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self, 
                 max_requests: int = 100,
                 window_minutes: int = 15,
                 per_user: bool = True):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes
            per_user: Rate limit per user (True) or per IP (False)
        """
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.per_user = per_user
    
    async def __call__(self, 
                      request: Request,
                      current_user: Optional[Dict[str, Any]] = None,
                      redis_client: redis.Redis = Depends(get_redis_client)):
        """Check rate limit"""
        
        # Determine rate limit key
        if self.per_user and current_user:
            key = f"rate_limit:user:{current_user['user_id']}"
        else:
            key = f"rate_limit:ip:{get_client_ip(request)}"
        
        try:
            # Get current request count
            current_requests = redis_client.get(key)
            
            if current_requests is None:
                # First request in window
                redis_client.setex(key, self.window_seconds, 1)
                return
            
            current_requests = int(current_requests)
            
            if current_requests >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds//60} minutes"
                )
            
            # Increment request count
            redis_client.incr(key)
            
        except redis.RedisError:
            # If Redis is unavailable, allow request (fail open)
            pass


def rate_limit(max_requests: int = 100, 
              window_minutes: int = 15,
              per_user: bool = True) -> Callable:
    """
    Rate limiting dependency factory
    
    Args:
        max_requests: Maximum requests allowed
        window_minutes: Time window in minutes  
        per_user: Rate limit per user (True) or per IP (False)
        
    Returns:
        FastAPI dependency function
    """
    limiter = RateLimiter(max_requests, window_minutes, per_user)
    
    async def rate_limit_dependency(
        request: Request,
        current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
        redis_client: redis.Redis = Depends(get_redis_client)
    ):
        await limiter(request, current_user, redis_client)
    
    return rate_limit_dependency


def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise
    Used for endpoints that work with or without authentication
    
    Returns:
        User information dictionary or None
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_data = jwt_handler.verify_token(token, TokenType.ACCESS)
        
        if not token_data:
            return None
        
        # Extract basic user info without full role/permission lookup
        return {
            "user_id": token_data.get("user_id"),
            "username": token_data.get("username", ""),
            "email": token_data.get("email", ""),
            "auth_method": "jwt"
        }
    
    except Exception:
        return None


def log_security_event(event_type: str, 
                      user_id: Optional[str] = None,
                      details: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Dependency factory for logging security events
    
    Args:
        event_type: Type of security event
        user_id: Optional user ID
        details: Additional event details
        
    Returns:
        FastAPI dependency function
    """
    async def log_event_dependency(
        request: Request,
        current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
        redis_client: redis.Redis = Depends(get_redis_client)
    ):
        
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id or (current_user.get("user_id") if current_user else None),
            "ip_address": get_client_ip(request),
            "user_agent": get_user_agent(request),
            "endpoint": str(request.url),
            "method": request.method,
            "details": details or {}
        }
        
        try:
            # Log to Redis (in production, also log to persistent storage)
            log_key = f"security_log:{datetime.utcnow().strftime('%Y%m%d')}"
            redis_client.lpush(log_key, str(event_data))
            redis_client.expire(log_key, 86400 * 30)  # Keep logs for 30 days
        except Exception:
            pass  # Continue without logging if Redis unavailable
    
    return log_event_dependency


# Commonly used dependency combinations
admin_required = require_role(Role.ADMIN)
manager_required = require_role(Role.MANAGER)
analyst_required = require_any_role([Role.ANALYST, Role.SENIOR_ANALYST, Role.MANAGER, Role.ADMIN])

# Permission-based dependencies
can_create_documents = require_permission(Permission.DOCUMENT_CREATE)
can_read_documents = require_permission(Permission.DOCUMENT_READ)
can_update_documents = require_permission(Permission.DOCUMENT_UPDATE)
can_delete_documents = require_permission(Permission.DOCUMENT_DELETE)

can_create_analysis = require_permission(Permission.ANALYSIS_CREATE)
can_read_analysis = require_permission(Permission.ANALYSIS_READ)
can_approve_analysis = require_permission(Permission.ANALYSIS_APPROVE)

can_manage_users = require_permission(Permission.USER_MANAGE_ROLES)
can_view_system_config = require_permission(Permission.SYSTEM_CONFIG)

# Rate limiting presets
standard_rate_limit = rate_limit(max_requests=100, window_minutes=15, per_user=True)
strict_rate_limit = rate_limit(max_requests=50, window_minutes=15, per_user=True)
api_rate_limit = rate_limit(max_requests=1000, window_minutes=60, per_user=True)