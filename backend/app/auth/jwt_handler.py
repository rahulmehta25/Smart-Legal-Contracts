"""
JWT Token Management

Production-ready JWT handling with:
- Token signing and verification
- Token blacklisting
- Refresh token rotation
- Claims validation
- Token introspection
"""

import uuid
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

import jwt
from jwt.exceptions import (
    InvalidTokenError, 
    ExpiredSignatureError, 
    InvalidSignatureError,
    InvalidKeyError,
    InvalidIssuerError,
    InvalidAudienceError
)
import redis
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, status


class TokenType(str, Enum):
    """Token types for different use cases"""
    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    API_KEY = "api_key"
    TEMPORARY = "temporary"


@dataclass
class TokenClaims:
    """Standard JWT claims with custom extensions"""
    # Standard JWT claims
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration time
    iat: int  # Issued at
    nbf: int  # Not before
    jti: str  # JWT ID
    
    # Custom claims
    token_type: TokenType
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    mfa_verified: bool = False
    api_key_id: Optional[str] = None
    scope: Optional[str] = None  # OAuth2 scopes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT payload"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class JWTHandler:
    """
    Production-ready JWT token handler with comprehensive security features
    """
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 private_key: Optional[str] = None,
                 public_key: Optional[str] = None,
                 algorithm: str = "HS256",
                 issuer: str = "arbitration-system",
                 audience: str = "arbitration-api",
                 access_token_expire_minutes: int = 15,
                 refresh_token_expire_days: int = 7,
                 redis_client: Optional[redis.Redis] = None):
        """
        Initialize JWT handler
        
        Args:
            secret_key: Secret key for HMAC algorithms (HS256, HS384, HS512)
            private_key: Private key for RSA/ECDSA algorithms (RS256, ES256, etc.)
            public_key: Public key for token verification
            algorithm: JWT signing algorithm
            issuer: Token issuer identifier
            audience: Token audience identifier
            access_token_expire_minutes: Access token expiration time
            refresh_token_expire_days: Refresh token expiration time
            redis_client: Redis client for token blacklisting
        """
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        
        # Set up signing keys based on algorithm
        if algorithm.startswith('HS'):
            if not secret_key:
                raise ValueError("Secret key required for HMAC algorithms")
            self.signing_key = secret_key
            self.verification_key = secret_key
        elif algorithm.startswith(('RS', 'PS')):
            if not private_key:
                raise ValueError("Private key required for RSA algorithms")
            self.signing_key = private_key
            self.verification_key = public_key or self._extract_public_key(private_key)
        elif algorithm.startswith('ES'):
            if not private_key:
                raise ValueError("Private key required for ECDSA algorithms")
            self.signing_key = private_key
            self.verification_key = public_key or self._extract_public_key(private_key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Redis for token blacklisting and session management
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        # Token blacklist TTL (set to max token lifetime)
        self.blacklist_ttl = max(
            int(self.access_token_expire.total_seconds()),
            int(self.refresh_token_expire.total_seconds())
        )
    
    def _extract_public_key(self, private_key: str) -> str:
        """Extract public key from private key for asymmetric algorithms"""
        try:
            private_key_obj = serialization.load_pem_private_key(
                private_key.encode(), 
                password=None
            )
            public_key_obj = private_key_obj.public_key()
            return public_key_obj.public_key_pem().decode()
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")
    
    def generate_keypair(self) -> tuple[str, str]:
        """Generate RSA key pair for production use"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        return private_pem, public_pem
    
    def create_token(self, claims: TokenClaims) -> str:
        """
        Create JWT token with claims
        
        Args:
            claims: Token claims
            
        Returns:
            Encoded JWT token
        """
        try:
            payload = claims.to_dict()
            
            # Ensure required claims
            payload.update({
                'iss': self.issuer,
                'aud': self.audience,
                'iat': int(datetime.utcnow().timestamp()),
                'nbf': int(datetime.utcnow().timestamp()),
                'jti': str(uuid.uuid4())
            })
            
            return jwt.encode(
                payload, 
                self.signing_key, 
                algorithm=self.algorithm
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Token creation failed: {str(e)}"
            )
    
    def create_access_token(self, 
                           user_id: str,
                           username: str,
                           email: str,
                           roles: List[str],
                           permissions: List[str],
                           session_id: str,
                           mfa_verified: bool = False,
                           device_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> str:
        """Create access token with user information"""
        
        expiry = datetime.utcnow() + self.access_token_expire
        
        claims = TokenClaims(
            sub=username,
            iss=self.issuer,
            aud=self.audience,
            exp=int(expiry.timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            nbf=int(datetime.utcnow().timestamp()),
            jti=str(uuid.uuid4()),
            token_type=TokenType.ACCESS,
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            device_id=device_id,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified
        )
        
        return self.create_token(claims)
    
    def create_refresh_token(self, 
                            user_id: str,
                            username: str,
                            session_id: str,
                            device_id: Optional[str] = None) -> str:
        """Create refresh token for token renewal"""
        
        expiry = datetime.utcnow() + self.refresh_token_expire
        
        claims = TokenClaims(
            sub=username,
            iss=self.issuer,
            aud=self.audience,
            exp=int(expiry.timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            nbf=int(datetime.utcnow().timestamp()),
            jti=str(uuid.uuid4()),
            token_type=TokenType.REFRESH,
            user_id=user_id,
            username=username,
            email="",  # Not needed for refresh tokens
            roles=[],
            permissions=[],
            session_id=session_id,
            device_id=device_id
        )
        
        return self.create_token(claims)
    
    def create_email_verification_token(self, 
                                       user_id: str, 
                                       email: str,
                                       expires_in_hours: int = 24) -> str:
        """Create email verification token"""
        
        expiry = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        claims = TokenClaims(
            sub=user_id,
            iss=self.issuer,
            aud=self.audience,
            exp=int(expiry.timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            nbf=int(datetime.utcnow().timestamp()),
            jti=str(uuid.uuid4()),
            token_type=TokenType.EMAIL_VERIFICATION,
            user_id=user_id,
            username="",
            email=email,
            roles=[],
            permissions=[],
            session_id=""
        )
        
        return self.create_token(claims)
    
    def create_password_reset_token(self, 
                                   user_id: str, 
                                   email: str,
                                   expires_in_hours: int = 1) -> str:
        """Create password reset token"""
        
        expiry = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        claims = TokenClaims(
            sub=user_id,
            iss=self.issuer,
            aud=self.audience,
            exp=int(expiry.timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            nbf=int(datetime.utcnow().timestamp()),
            jti=str(uuid.uuid4()),
            token_type=TokenType.PASSWORD_RESET,
            user_id=user_id,
            username="",
            email=email,
            roles=[],
            permissions=[],
            session_id=""
        )
        
        return self.create_token(claims)
    
    def verify_token(self, token: str, 
                    expected_type: Optional[TokenType] = None,
                    verify_claims: bool = True) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token to verify
            expected_type: Expected token type
            verify_claims: Whether to verify standard claims
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.verification_key,
                algorithms=[self.algorithm],
                issuer=self.issuer if verify_claims else None,
                audience=self.audience if verify_claims else None,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_iss": verify_claims,
                    "verify_aud": verify_claims
                }
            )
            
            # Check if token is blacklisted
            jti = payload.get('jti')
            if jti and self.is_token_blacklisted(jti):
                return None
            
            # Verify token type if specified
            if expected_type and payload.get('token_type') != expected_type.value:
                return None
            
            # Verify session is still valid for access/refresh tokens
            token_type = payload.get('token_type')
            if token_type in [TokenType.ACCESS.value, TokenType.REFRESH.value]:
                session_id = payload.get('session_id')
                if session_id and not self.is_session_valid(session_id):
                    return None
            
            return payload
            
        except ExpiredSignatureError:
            return None
        except (InvalidTokenError, InvalidSignatureError, 
                InvalidKeyError, InvalidIssuerError, InvalidAudienceError):
            return None
        except Exception:
            return None
    
    def decode_token_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for inspection only)
        
        WARNING: Only use for debugging/logging purposes
        """
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None
    
    def blacklist_token(self, jti: str, ttl: Optional[int] = None) -> bool:
        """
        Add token to blacklist
        
        Args:
            jti: JWT ID to blacklist
            ttl: Time to live in seconds (default: token max lifetime)
            
        Returns:
            True if successfully blacklisted
        """
        try:
            ttl = ttl or self.blacklist_ttl
            self.redis_client.setex(f"blacklist:{jti}", ttl, "1")
            return True
        except Exception:
            return False
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        try:
            return self.redis_client.exists(f"blacklist:{jti}") > 0
        except Exception:
            return True  # Fail safe: treat as blacklisted if Redis unavailable
    
    def blacklist_all_user_tokens(self, user_id: str) -> bool:
        """Blacklist all tokens for a user (logout from all devices)"""
        try:
            # Set a flag that all tokens issued before this time are invalid
            timestamp = int(datetime.utcnow().timestamp())
            self.redis_client.setex(
                f"user_logout:{user_id}", 
                self.blacklist_ttl, 
                str(timestamp)
            )
            return True
        except Exception:
            return False
    
    def is_user_logged_out(self, user_id: str, token_iat: int) -> bool:
        """Check if user has been logged out globally"""
        try:
            logout_time = self.redis_client.get(f"user_logout:{user_id}")
            if logout_time and int(logout_time) > token_iat:
                return True
            return False
        except Exception:
            return False
    
    def create_session(self, user_id: str, 
                      device_id: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> str:
        """Create user session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "device_id": device_id or "",
            "ip_address": ip_address or "",
            "user_agent": user_agent or "",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "is_active": "1"
        }
        
        try:
            # Store session with expiry
            self.redis_client.hset(f"session:{session_id}", mapping=session_data)
            self.redis_client.expire(f"session:{session_id}", 
                                   int(self.refresh_token_expire.total_seconds()))
            
            # Add to user's session list
            self.redis_client.sadd(f"user_sessions:{user_id}", session_id)
            
            return session_id
        except Exception:
            return ""
    
    def is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and active"""
        try:
            session_data = self.redis_client.hgetall(f"session:{session_id}")
            return session_data and session_data.get("is_active") == "1"
        except Exception:
            return False
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""
        try:
            self.redis_client.hset(
                f"session:{session_id}",
                "last_activity", 
                datetime.utcnow().isoformat()
            )
            return True
        except Exception:
            return False
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a specific session"""
        try:
            self.redis_client.hset(f"session:{session_id}", "is_active", "0")
            return True
        except Exception:
            return False
    
    def terminate_all_user_sessions(self, user_id: str) -> bool:
        """Terminate all sessions for a user"""
        try:
            session_ids = self.redis_client.smembers(f"user_sessions:{user_id}")
            for session_id in session_ids:
                self.redis_client.hset(f"session:{session_id}", "is_active", "0")
            return True
        except Exception:
            return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        try:
            session_ids = self.redis_client.smembers(f"user_sessions:{user_id}")
            sessions = []
            
            for session_id in session_ids:
                session_data = self.redis_client.hgetall(f"session:{session_id}")
                if session_data and session_data.get("is_active") == "1":
                    sessions.append({
                        "session_id": session_id,
                        **session_data
                    })
            
            return sessions
        except Exception:
            return []
    
    def introspect_token(self, token: str) -> Dict[str, Any]:
        """
        OAuth2-style token introspection
        
        Returns:
            Token metadata including active status, claims, etc.
        """
        payload = self.verify_token(token, verify_claims=False)
        
        if not payload:
            return {"active": False}
        
        # Check if token is still valid
        exp = payload.get('exp', 0)
        now = int(datetime.utcnow().timestamp())
        is_active = exp > now and not self.is_token_blacklisted(payload.get('jti', ''))
        
        result = {
            "active": is_active,
            "token_type": payload.get('token_type'),
            "scope": payload.get('scope', ''),
            "client_id": payload.get('aud'),
            "username": payload.get('username'),
            "user_id": payload.get('user_id'),
            "exp": exp,
            "iat": payload.get('iat'),
            "iss": payload.get('iss'),
            "sub": payload.get('sub')
        }
        
        return result
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Refresh access token using refresh token
        
        Returns:
            New token pair or None if refresh token invalid
        """
        # Verify refresh token
        payload = self.verify_token(refresh_token, TokenType.REFRESH)
        if not payload:
            return None
        
        user_id = payload.get('user_id')
        username = payload.get('username')
        session_id = payload.get('session_id')
        device_id = payload.get('device_id')
        
        if not all([user_id, username, session_id]):
            return None
        
        # TODO: Fetch user's current roles and permissions from database
        # For now, using empty lists - this should be replaced with actual user lookup
        roles = []
        permissions = []
        email = ""
        
        # Create new token pair
        new_access_token = self.create_access_token(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            device_id=device_id
        )
        
        new_refresh_token = self.create_refresh_token(
            user_id=user_id,
            username=username,
            session_id=session_id,
            device_id=device_id
        )
        
        # Blacklist old refresh token (refresh token rotation)
        old_jti = payload.get('jti')
        if old_jti:
            self.blacklist_token(old_jti)
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }