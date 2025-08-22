"""
Enhanced Authentication System with OAuth2, JWT, MFA, and RBAC
Implements OWASP authentication best practices
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import secrets
import hashlib
import hmac
import base64
import time
import uuid
import re
from dataclasses import dataclass
from functools import wraps

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.totp import TOTP
import pyotp
from fastapi import HTTPException, status, Depends, Header
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import redis
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2


# OWASP Compliant Password Policy
PASSWORD_MIN_LENGTH = 12
PASSWORD_REGEX = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]'
)


class UserRole(str, Enum):
    """User roles for RBAC"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    CLIENT = "client"
    API_USER = "api_user"
    READ_ONLY = "read_only"


class Permission(str, Enum):
    """Granular permissions"""
    # Document permissions
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    
    # Analysis permissions
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"
    ANALYSIS_APPROVE = "analysis:approve"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"
    
    # API key management
    API_KEY_CREATE = "api_key:create"
    API_KEY_REVOKE = "api_key:revoke"
    API_KEY_MANAGE = "api_key:manage"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"
    SYSTEM_BACKUP = "system:backup"


# Role-Permission Mapping (RBAC)
ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: [p.value for p in Permission],  # All permissions
    UserRole.ADMIN: [
        Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ, 
        Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_DELETE,
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_APPROVE,
        Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
        Permission.API_KEY_CREATE, Permission.API_KEY_REVOKE,
        Permission.SYSTEM_AUDIT
    ],
    UserRole.ANALYST: [
        Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPDATE,
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE
    ],
    UserRole.REVIEWER: [
        Permission.DOCUMENT_READ,
        Permission.ANALYSIS_READ, Permission.ANALYSIS_APPROVE
    ],
    UserRole.CLIENT: [
        Permission.DOCUMENT_READ,
        Permission.ANALYSIS_READ
    ],
    UserRole.API_USER: [
        Permission.DOCUMENT_CREATE, Permission.DOCUMENT_READ,
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ
    ],
    UserRole.READ_ONLY: [
        Permission.DOCUMENT_READ,
        Permission.ANALYSIS_READ
    ]
}


@dataclass
class TokenData:
    """JWT Token payload structure"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    token_type: str = "access"
    mfa_verified: bool = False
    api_key_id: Optional[str] = None


class AuthenticationManager:
    """
    Comprehensive authentication manager implementing:
    - OAuth2 + JWT
    - Multi-factor authentication (MFA)
    - Role-based access control (RBAC)
    - API key management
    - Session management
    - Brute force protection
    """
    
    def __init__(self, 
                 secret_key: str,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7,
                 redis_client: Optional[redis.Redis] = None):
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        
        # Password hashing with Argon2 (OWASP recommended)
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__rounds=4,
            argon2__memory_cost=102400,
            argon2__parallelism=8
        )
        
        # OAuth2 scheme
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
        self.http_bearer = HTTPBearer()
        
        # Redis for session management and rate limiting
        self.redis_client = redis_client or redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # Brute force protection settings
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
    
    # ========== Password Management ==========
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        Validate password against OWASP requirements
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
        
        if not PASSWORD_REGEX.match(password):
            return False, "Password must contain uppercase, lowercase, digit, and special character"
        
        # Check against common passwords (would use a proper list in production)
        common_passwords = ["Password123!", "Admin123!", "Welcome123!"]
        if password in common_passwords:
            return False, "Password is too common"
        
        return True, ""
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt using Argon2"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    # ========== JWT Token Management ==========
    
    def create_access_token(self, token_data: TokenData) -> str:
        """Create JWT access token with user data and permissions"""
        payload = {
            "sub": token_data.username,
            "user_id": token_data.user_id,
            "email": token_data.email,
            "roles": token_data.roles,
            "permissions": token_data.permissions,
            "session_id": token_data.session_id,
            "token_type": "access",
            "mfa_verified": token_data.mfa_verified,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.access_token_expire,
            "jti": str(uuid.uuid4())  # JWT ID for token revocation
        }
        
        if token_data.api_key_id:
            payload["api_key_id"] = token_data.api_key_id
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, token_data: TokenData) -> str:
        """Create JWT refresh token"""
        payload = {
            "sub": token_data.username,
            "user_id": token_data.user_id,
            "session_id": token_data.session_id,
            "token_type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_token_expire,
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Check if token is revoked
            if self.is_token_revoked(payload.get("jti")):
                return None
            
            # Check session validity
            if not self.is_session_valid(payload.get("session_id")):
                return None
            
            return payload
            
        except JWTError:
            return None
    
    def revoke_token(self, jti: str, expiry: int = 86400):
        """Revoke a token by adding its JTI to blacklist"""
        self.redis_client.setex(f"revoked_token:{jti}", expiry, "1")
    
    def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        return self.redis_client.exists(f"revoked_token:{jti}") > 0
    
    # ========== Multi-Factor Authentication (MFA) ==========
    
    def generate_mfa_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        # Store encrypted in database (implementation depends on your ORM)
        self.redis_client.hset(f"mfa_secret:{user_id}", "secret", secret)
        return secret
    
    def generate_qr_code(self, username: str, secret: str, issuer: str = "ArbitrationSystem") -> str:
        """Generate QR code provisioning URI for authenticator apps"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )
    
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        secret = self.redis_client.hget(f"mfa_secret:{user_id}", "secret")
        if not secret:
            return False
        
        totp = pyotp.TOTP(secret)
        # Allow 1 window tolerance for time sync issues
        return totp.verify(token, valid_window=1)
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        
        # Hash and store codes
        for code in codes:
            hashed = hashlib.sha256(code.encode()).hexdigest()
            self.redis_client.sadd(f"backup_codes:{user_id}", hashed)
        
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code"""
        hashed = hashlib.sha256(code.encode()).hexdigest()
        return self.redis_client.srem(f"backup_codes:{user_id}", hashed) > 0
    
    # ========== API Key Management ==========
    
    def generate_api_key(self, user_id: str, name: str, 
                        permissions: List[str], 
                        expires_in_days: Optional[int] = None) -> Dict[str, Any]:
        """Generate API key for enterprise clients"""
        
        # Generate cryptographically secure API key
        key_id = str(uuid.uuid4())
        key_secret = secrets.token_urlsafe(32)
        api_key = f"ak_{key_id}_{key_secret}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store key metadata
        key_data = {
            "key_id": key_id,
            "user_id": user_id,
            "name": name,
            "key_hash": key_hash,
            "permissions": ",".join(permissions),
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "usage_count": 0
        }
        
        if expires_in_days:
            key_data["expires_at"] = (
                datetime.utcnow() + timedelta(days=expires_in_days)
            ).isoformat()
        
        # Store in Redis (in production, use persistent database)
        self.redis_client.hset(f"api_key:{key_id}", mapping=key_data)
        
        return {
            "api_key": api_key,
            "key_id": key_id,
            "expires_at": key_data.get("expires_at")
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated data"""
        
        # Parse API key
        if not api_key.startswith("ak_"):
            return None
        
        parts = api_key.split("_")
        if len(parts) != 3:
            return None
        
        key_id = parts[1]
        key_secret = parts[2]
        
        # Retrieve key data
        key_data = self.redis_client.hgetall(f"api_key:{key_id}")
        if not key_data:
            return None
        
        # Verify key hash
        provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if not hmac.compare_digest(provided_hash, key_data.get("key_hash", "")):
            return None
        
        # Check expiration
        if expires_at := key_data.get("expires_at"):
            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                return None
        
        # Update usage statistics
        self.redis_client.hincrby(f"api_key:{key_id}", "usage_count", 1)
        self.redis_client.hset(f"api_key:{key_id}", 
                              "last_used", 
                              datetime.utcnow().isoformat())
        
        return {
            "key_id": key_id,
            "user_id": key_data["user_id"],
            "permissions": key_data["permissions"].split(","),
            "name": key_data["name"]
        }
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        return self.redis_client.delete(f"api_key:{key_id}") > 0
    
    # ========== Session Management ==========
    
    def create_session(self, user_id: str, 
                      user_agent: str = "", 
                      ip_address: str = "") -> str:
        """Create user session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "user_agent": user_agent,
            "ip_address": ip_address,
            "active": "1"
        }
        
        # Store session with 24 hour expiry
        self.redis_client.hset(f"session:{session_id}", mapping=session_data)
        self.redis_client.expire(f"session:{session_id}", 86400)
        
        return session_id
    
    def update_session_activity(self, session_id: str):
        """Update session last activity time"""
        self.redis_client.hset(
            f"session:{session_id}", 
            "last_activity", 
            datetime.utcnow().isoformat()
        )
        self.redis_client.expire(f"session:{session_id}", 86400)
    
    def is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and active"""
        session_data = self.redis_client.hgetall(f"session:{session_id}")
        return session_data and session_data.get("active") == "1"
    
    def terminate_session(self, session_id: str):
        """Terminate a user session"""
        self.redis_client.hset(f"session:{session_id}", "active", "0")
    
    def terminate_all_sessions(self, user_id: str):
        """Terminate all sessions for a user"""
        # In production, maintain a user->sessions index
        for key in self.redis_client.scan_iter("session:*"):
            session_data = self.redis_client.hgetall(key)
            if session_data.get("user_id") == user_id:
                self.redis_client.hset(key, "active", "0")
    
    # ========== Brute Force Protection ==========
    
    def check_login_attempts(self, identifier: str) -> bool:
        """Check if login attempts exceeded"""
        attempts_key = f"login_attempts:{identifier}"
        attempts = self.redis_client.get(attempts_key)
        
        if attempts and int(attempts) >= self.max_login_attempts:
            return False
        return True
    
    def record_failed_login(self, identifier: str):
        """Record failed login attempt"""
        attempts_key = f"login_attempts:{identifier}"
        self.redis_client.incr(attempts_key)
        self.redis_client.expire(attempts_key, self.lockout_duration)
    
    def reset_login_attempts(self, identifier: str):
        """Reset login attempts on successful login"""
        self.redis_client.delete(f"login_attempts:{identifier}")
    
    # ========== RBAC Authorization ==========
    
    def get_user_permissions(self, roles: List[str]) -> List[str]:
        """Get all permissions for user roles"""
        permissions = set()
        for role in roles:
            if role in ROLE_PERMISSIONS:
                permissions.update(ROLE_PERMISSIONS[role])
        return list(permissions)
    
    def has_permission(self, user_permissions: List[str], 
                       required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def has_any_permission(self, user_permissions: List[str], 
                          required_permissions: List[str]) -> bool:
        """Check if user has any of the required permissions"""
        return any(perm in user_permissions for perm in required_permissions)
    
    def has_all_permissions(self, user_permissions: List[str], 
                           required_permissions: List[str]) -> bool:
        """Check if user has all required permissions"""
        return all(perm in user_permissions for perm in required_permissions)


# ========== FastAPI Dependencies ==========

auth_manager = None  # Initialize in main application

def get_auth_manager() -> AuthenticationManager:
    """Get authentication manager instance"""
    global auth_manager
    if not auth_manager:
        from app.core.config import settings
        auth_manager = AuthenticationManager(
            secret_key=settings.secret_key,
            algorithm=settings.algorithm,
            access_token_expire_minutes=settings.access_token_expire_minutes
        )
    return auth_manager


async def get_current_user(
    authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token or API key
    """
    auth = get_auth_manager()
    
    # Check API key first
    if x_api_key:
        api_data = auth.validate_api_key(x_api_key)
        if api_data:
            return {
                "user_id": api_data["user_id"],
                "username": f"api_key_{api_data['name']}",
                "permissions": api_data["permissions"],
                "auth_method": "api_key"
            }
    
    # Check JWT token
    if authorization:
        token_data = auth.decode_token(authorization.credentials)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update session activity
        auth.update_session_activity(token_data.get("session_id"))
        
        return {
            "user_id": token_data["user_id"],
            "username": token_data["sub"],
            "email": token_data.get("email"),
            "roles": token_data.get("roles", []),
            "permissions": token_data.get("permissions", []),
            "mfa_verified": token_data.get("mfa_verified", False),
            "auth_method": "jwt"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: str):
    """
    Decorator to require specific permission
    """
    def permission_checker(current_user: Dict = Depends(get_current_user)):
        auth = get_auth_manager()
        if not auth.has_permission(current_user.get("permissions", []), permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        return current_user
    return permission_checker


def require_any_permission(permissions: List[str]):
    """
    Decorator to require any of the specified permissions
    """
    def permission_checker(current_user: Dict = Depends(get_current_user)):
        auth = get_auth_manager()
        if not auth.has_any_permission(current_user.get("permissions", []), permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required one of: {permissions}"
            )
        return current_user
    return permission_checker


def require_mfa(current_user: Dict = Depends(get_current_user)):
    """
    Require MFA verification for sensitive operations
    """
    if not current_user.get("mfa_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MFA verification required for this operation"
        )
    return current_user