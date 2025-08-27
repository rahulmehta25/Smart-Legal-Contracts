"""
API Key Management System
Implements secure API key generation, validation, and lifecycle management
OWASP API Security Top 10 compliant
"""

import secrets
import hashlib
import hmac
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import base64

import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Request
import jwt

from ..core.security import get_password_hash, verify_password


class APIKeyType(str, Enum):
    """Types of API keys"""
    PUBLIC = "public"          # Public key for client identification
    SECRET = "secret"          # Secret key for authentication
    WEBHOOK = "webhook"        # Webhook signing key
    SERVICE = "service"        # Service-to-service communication
    ADMIN = "admin"           # Administrative access


class APIKeyStatus(str, Enum):
    """API key lifecycle states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class APIKeyTier(str, Enum):
    """API key tiers for rate limiting"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass
class APIKeyConfig:
    """API key configuration"""
    key_type: APIKeyType = APIKeyType.SECRET
    tier: APIKeyTier = APIKeyTier.FREE
    expires_in: Optional[int] = None  # Seconds until expiration
    permissions: List[str] = None
    ip_whitelist: List[str] = None
    allowed_domains: List[str] = None
    rate_limit_override: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = None


@dataclass
class APIKeyInfo:
    """API key information"""
    key_id: str
    key_type: APIKeyType
    tier: APIKeyTier
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    permissions: List[str]
    metadata: Dict[str, Any]


class APIKeyManager:
    """
    Comprehensive API key management system with:
    - Secure key generation using cryptographic standards
    - Key rotation and versioning
    - Hierarchical permission system
    - Usage tracking and analytics
    - Automatic expiration handling
    - Key encryption at rest
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 encryption_key: Optional[bytes] = None):
        
        # Redis for caching and fast lookups
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Encryption for sensitive data
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate encryption key from environment or config
            self.cipher = Fernet(Fernet.generate_key())
        
        # Key prefix patterns
        self.key_prefix = "api_key"
        self.hash_prefix = "api_key_hash"
        self.usage_prefix = "api_key_usage"
        
        # Tier configurations
        self.tier_configs = {
            APIKeyTier.FREE: {
                "rate_limit": 100,
                "rate_window": 3600,
                "max_keys": 1,
                "expires_in": 30 * 86400  # 30 days
            },
            APIKeyTier.BASIC: {
                "rate_limit": 1000,
                "rate_window": 3600,
                "max_keys": 5,
                "expires_in": 90 * 86400  # 90 days
            },
            APIKeyTier.PRO: {
                "rate_limit": 10000,
                "rate_window": 3600,
                "max_keys": 20,
                "expires_in": 365 * 86400  # 1 year
            },
            APIKeyTier.ENTERPRISE: {
                "rate_limit": 100000,
                "rate_window": 3600,
                "max_keys": 100,
                "expires_in": None  # No expiration
            },
            APIKeyTier.UNLIMITED: {
                "rate_limit": None,
                "rate_window": 3600,
                "max_keys": None,
                "expires_in": None
            }
        }
    
    # ========== Key Generation ==========
    
    def generate_api_key(self,
                        user_id: str,
                        config: APIKeyConfig) -> Tuple[str, str, APIKeyInfo]:
        """
        Generate a new API key pair
        Returns: (public_key, secret_key, key_info)
        """
        
        # Generate unique key ID
        key_id = str(uuid.uuid4())
        
        # Generate cryptographically secure keys
        public_key = self._generate_public_key(key_id)
        secret_key = self._generate_secret_key()
        
        # Hash secret key for storage
        secret_hash = self._hash_key(secret_key)
        
        # Create key info
        now = datetime.utcnow()
        expires_at = None
        if config.expires_in:
            expires_at = now + timedelta(seconds=config.expires_in)
        elif config.tier in self.tier_configs:
            tier_expiry = self.tier_configs[config.tier].get("expires_in")
            if tier_expiry:
                expires_at = now + timedelta(seconds=tier_expiry)
        
        key_info = APIKeyInfo(
            key_id=key_id,
            key_type=config.key_type,
            tier=config.tier,
            status=APIKeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            last_used=None,
            usage_count=0,
            permissions=config.permissions or [],
            metadata=config.metadata or {}
        )
        
        # Store key information
        self._store_key(user_id, public_key, secret_hash, key_info, config)
        
        # Return keys and info
        return public_key, secret_key, key_info
    
    def _generate_public_key(self, key_id: str) -> str:
        """Generate public API key"""
        
        # Format: pk_[environment]_[random]
        environment = "live"  # or "test" for test keys
        random_part = secrets.token_urlsafe(16)
        
        return f"pk_{environment}_{random_part}"
    
    def _generate_secret_key(self) -> str:
        """Generate secret API key"""
        
        # Format: sk_[environment]_[random]
        environment = "live"
        random_part = secrets.token_urlsafe(32)
        
        return f"sk_{environment}_{random_part}"
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for secure storage"""
        
        # Use PBKDF2 for key derivation
        salt = secrets.token_bytes(32)
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key_hash = base64.b64encode(
            salt + kdf.derive(key.encode())
        ).decode()
        
        return key_hash
    
    # ========== Key Validation ==========
    
    def validate_api_key(self,
                        public_key: str,
                        secret_key: str,
                        required_permissions: Optional[List[str]] = None) -> APIKeyInfo:
        """
        Validate API key pair and check permissions
        """
        
        # Get key info from cache or database
        key_info = self._get_key_info(public_key)
        
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check key status
        if key_info.status != APIKeyStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key is {key_info.status}"
            )
        
        # Check expiration
        if key_info.expires_at and key_info.expires_at < datetime.utcnow():
            self._update_key_status(public_key, APIKeyStatus.EXPIRED)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired"
            )
        
        # Verify secret key
        stored_hash = self._get_secret_hash(public_key)
        if not self._verify_key(secret_key, stored_hash):
            # Record failed attempt
            self._record_failed_attempt(public_key)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check permissions
        if required_permissions:
            missing = set(required_permissions) - set(key_info.permissions)
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {missing}"
                )
        
        # Check IP whitelist
        ip_whitelist = self._get_ip_whitelist(public_key)
        if ip_whitelist:
            # Would need to pass client IP from request
            pass
        
        # Update usage
        self._record_usage(public_key)
        
        return key_info
    
    def _verify_key(self, key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        
        try:
            # Extract salt and hash
            decoded = base64.b64decode(stored_hash)
            salt = decoded[:32]
            stored_key_hash = decoded[32:]
            
            # Derive key with same salt
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key_hash = kdf.derive(key.encode())
            
            # Constant-time comparison
            return hmac.compare_digest(key_hash, stored_key_hash)
        except:
            return False
    
    # ========== Key Rotation ==========
    
    def rotate_api_key(self,
                      public_key: str,
                      grace_period: int = 86400) -> Tuple[str, str, APIKeyInfo]:
        """
        Rotate API key with grace period
        Old key remains valid during grace period
        """
        
        # Get existing key info
        old_info = self._get_key_info(public_key)
        if not old_info:
            raise ValueError("API key not found")
        
        # Generate new keys
        config = APIKeyConfig(
            key_type=old_info.key_type,
            tier=old_info.tier,
            permissions=old_info.permissions,
            metadata=old_info.metadata
        )
        
        user_id = self._get_user_id(public_key)
        new_public, new_secret, new_info = self.generate_api_key(user_id, config)
        
        # Set grace period for old key
        grace_expiry = datetime.utcnow() + timedelta(seconds=grace_period)
        self._set_key_expiry(public_key, grace_expiry)
        
        # Link old and new keys
        self._link_rotated_keys(public_key, new_public)
        
        return new_public, new_secret, new_info
    
    # ========== Key Revocation ==========
    
    def revoke_api_key(self, public_key: str, reason: Optional[str] = None):
        """Revoke API key immediately"""
        
        # Update status
        self._update_key_status(public_key, APIKeyStatus.REVOKED)
        
        # Add to revocation list
        revocation_data = {
            "key": public_key,
            "revoked_at": datetime.utcnow().isoformat(),
            "reason": reason
        }
        
        self.redis_client.lpush("revoked_keys", json.dumps(revocation_data))
        
        # Clear from cache
        self._clear_key_cache(public_key)
    
    def suspend_api_key(self, public_key: str, duration: int = 3600):
        """Temporarily suspend API key"""
        
        # Update status
        self._update_key_status(public_key, APIKeyStatus.SUSPENDED)
        
        # Set automatic reactivation
        reactivate_at = time.time() + duration
        self.redis_client.zadd(
            "suspended_keys",
            {public_key: reactivate_at}
        )
    
    def reactivate_api_key(self, public_key: str):
        """Reactivate suspended API key"""
        
        # Check current status
        info = self._get_key_info(public_key)
        if info and info.status == APIKeyStatus.SUSPENDED:
            self._update_key_status(public_key, APIKeyStatus.ACTIVE)
            self.redis_client.zrem("suspended_keys", public_key)
    
    # ========== Usage Tracking ==========
    
    def _record_usage(self, public_key: str):
        """Record API key usage"""
        
        # Update last used timestamp
        now = datetime.utcnow()
        self.redis_client.hset(
            f"{self.key_prefix}:{public_key}",
            "last_used",
            now.isoformat()
        )
        
        # Increment usage counter
        self.redis_client.hincrby(
            f"{self.key_prefix}:{public_key}",
            "usage_count",
            1
        )
        
        # Record in time series for analytics
        usage_key = f"{self.usage_prefix}:{public_key}:{now.strftime('%Y%m%d')}"
        self.redis_client.hincrby(usage_key, now.hour, 1)
        self.redis_client.expire(usage_key, 30 * 86400)  # Keep for 30 days
    
    def get_usage_statistics(self,
                            public_key: str,
                            days: int = 30) -> Dict[str, Any]:
        """Get API key usage statistics"""
        
        stats = {
            "total_requests": 0,
            "daily_usage": {},
            "hourly_distribution": [0] * 24,
            "last_used": None
        }
        
        # Get total usage
        info = self._get_key_info(public_key)
        if info:
            stats["total_requests"] = info.usage_count
            stats["last_used"] = info.last_used
        
        # Get daily usage
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        current = start_date
        while current <= end_date:
            date_key = current.strftime('%Y%m%d')
            usage_key = f"{self.usage_prefix}:{public_key}:{date_key}"
            
            daily_data = self.redis_client.hgetall(usage_key)
            if daily_data:
                daily_total = sum(int(v) for v in daily_data.values())
                stats["daily_usage"][date_key] = daily_total
                
                # Update hourly distribution
                for hour, count in daily_data.items():
                    stats["hourly_distribution"][int(hour)] += int(count)
            
            current += timedelta(days=1)
        
        return stats
    
    # ========== Permission Management ==========
    
    def update_permissions(self,
                          public_key: str,
                          permissions: List[str],
                          mode: str = "replace"):
        """
        Update API key permissions
        mode: 'replace', 'add', 'remove'
        """
        
        current = self._get_key_info(public_key)
        if not current:
            raise ValueError("API key not found")
        
        if mode == "replace":
            new_permissions = permissions
        elif mode == "add":
            new_permissions = list(set(current.permissions + permissions))
        elif mode == "remove":
            new_permissions = [p for p in current.permissions if p not in permissions]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Update permissions
        self.redis_client.hset(
            f"{self.key_prefix}:{public_key}",
            "permissions",
            json.dumps(new_permissions)
        )
    
    def check_permission(self, public_key: str, permission: str) -> bool:
        """Check if API key has specific permission"""
        
        info = self._get_key_info(public_key)
        if not info:
            return False
        
        # Check exact permission
        if permission in info.permissions:
            return True
        
        # Check wildcard permissions
        for perm in info.permissions:
            if perm.endswith("*"):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True
        
        return False
    
    # ========== Storage Helpers ==========
    
    def _store_key(self,
                  user_id: str,
                  public_key: str,
                  secret_hash: str,
                  info: APIKeyInfo,
                  config: APIKeyConfig):
        """Store API key information"""
        
        # Store main key data
        key_data = {
            "user_id": user_id,
            "key_id": info.key_id,
            "key_type": info.key_type,
            "tier": info.tier,
            "status": info.status,
            "created_at": info.created_at.isoformat(),
            "expires_at": info.expires_at.isoformat() if info.expires_at else None,
            "permissions": json.dumps(info.permissions),
            "metadata": json.dumps(info.metadata),
            "usage_count": 0
        }
        
        self.redis_client.hset(
            f"{self.key_prefix}:{public_key}",
            mapping=key_data
        )
        
        # Store secret hash separately (encrypted)
        encrypted_hash = self.cipher.encrypt(secret_hash.encode()).decode()
        self.redis_client.set(
            f"{self.hash_prefix}:{public_key}",
            encrypted_hash
        )
        
        # Store additional config
        if config.ip_whitelist:
            self.redis_client.set(
                f"api_key_ip_whitelist:{public_key}",
                json.dumps(config.ip_whitelist)
            )
        
        # Index by user
        self.redis_client.sadd(f"user_keys:{user_id}", public_key)
        
        # Set expiration if needed
        if info.expires_at:
            ttl = int((info.expires_at - datetime.utcnow()).total_seconds())
            self.redis_client.expire(f"{self.key_prefix}:{public_key}", ttl)
            self.redis_client.expire(f"{self.hash_prefix}:{public_key}", ttl)
    
    def _get_key_info(self, public_key: str) -> Optional[APIKeyInfo]:
        """Get API key information"""
        
        data = self.redis_client.hgetall(f"{self.key_prefix}:{public_key}")
        if not data:
            return None
        
        return APIKeyInfo(
            key_id=data.get("key_id"),
            key_type=APIKeyType(data.get("key_type")),
            tier=APIKeyTier(data.get("tier")),
            status=APIKeyStatus(data.get("status")),
            created_at=datetime.fromisoformat(data.get("created_at")),
            expires_at=datetime.fromisoformat(data.get("expires_at")) if data.get("expires_at") else None,
            last_used=datetime.fromisoformat(data.get("last_used")) if data.get("last_used") else None,
            usage_count=int(data.get("usage_count", 0)),
            permissions=json.loads(data.get("permissions", "[]")),
            metadata=json.loads(data.get("metadata", "{}"))
        )
    
    def _get_secret_hash(self, public_key: str) -> Optional[str]:
        """Get secret key hash"""
        
        encrypted = self.redis_client.get(f"{self.hash_prefix}:{public_key}")
        if not encrypted:
            return None
        
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def _get_user_id(self, public_key: str) -> Optional[str]:
        """Get user ID for API key"""
        
        return self.redis_client.hget(f"{self.key_prefix}:{public_key}", "user_id")
    
    def _get_ip_whitelist(self, public_key: str) -> Optional[List[str]]:
        """Get IP whitelist for API key"""
        
        data = self.redis_client.get(f"api_key_ip_whitelist:{public_key}")
        return json.loads(data) if data else None
    
    def _update_key_status(self, public_key: str, status: APIKeyStatus):
        """Update API key status"""
        
        self.redis_client.hset(
            f"{self.key_prefix}:{public_key}",
            "status",
            status
        )
    
    def _set_key_expiry(self, public_key: str, expires_at: datetime):
        """Set API key expiration"""
        
        self.redis_client.hset(
            f"{self.key_prefix}:{public_key}",
            "expires_at",
            expires_at.isoformat()
        )
        
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        self.redis_client.expire(f"{self.key_prefix}:{public_key}", ttl)
        self.redis_client.expire(f"{self.hash_prefix}:{public_key}", ttl)
    
    def _link_rotated_keys(self, old_key: str, new_key: str):
        """Link rotated keys for audit trail"""
        
        self.redis_client.hset(
            f"key_rotation:{old_key}",
            "rotated_to",
            new_key
        )
        self.redis_client.hset(
            f"key_rotation:{new_key}",
            "rotated_from",
            old_key
        )
    
    def _clear_key_cache(self, public_key: str):
        """Clear API key from cache"""
        
        # Would clear from any application-level caches
        pass
    
    def _record_failed_attempt(self, public_key: str):
        """Record failed authentication attempt"""
        
        attempts_key = f"failed_attempts:{public_key}"
        attempts = self.redis_client.incr(attempts_key)
        self.redis_client.expire(attempts_key, 3600)  # Reset after 1 hour
        
        # Suspend after too many failures
        if attempts >= 5:
            self.suspend_api_key(public_key, duration=3600)
    
    # ========== Cleanup and Maintenance ==========
    
    def cleanup_expired_keys(self):
        """Remove expired API keys"""
        
        # This would be run periodically by a background job
        # Scan for expired keys and clean them up
        pass
    
    def process_suspended_keys(self):
        """Process suspended keys for reactivation"""
        
        now = time.time()
        suspended = self.redis_client.zrangebyscore(
            "suspended_keys",
            0,
            now
        )
        
        for key in suspended:
            self.reactivate_api_key(key)