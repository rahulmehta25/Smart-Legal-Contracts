"""
Authentication Module

Comprehensive authentication system with:
- JWT token management
- OAuth2 integration
- Role-based access control (RBAC)
- Multi-factor authentication (MFA)
- Brute force protection
- Session management
"""

from .jwt_handler import JWTHandler
from .password import PasswordHandler
from .dependencies import (
    get_current_user, 
    get_current_active_user,
    require_permission,
    require_any_permission,
    require_role,
    require_mfa
)
from .permissions import Permission, Role, RoleManager

__all__ = [
    "JWTHandler",
    "PasswordHandler", 
    "get_current_user",
    "get_current_active_user",
    "require_permission",
    "require_any_permission", 
    "require_role",
    "require_mfa",
    "Permission",
    "Role",
    "RoleManager"
]