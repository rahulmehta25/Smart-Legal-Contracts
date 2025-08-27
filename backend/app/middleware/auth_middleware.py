"""
Authentication Middleware

Production-ready authentication middleware with:
- Automatic token validation
- Session management
- Device fingerprinting
- Security event logging
- Performance optimization
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

import redis
from ..auth.jwt_handler import JWTHandler, TokenType
from ..auth.permissions import RoleManager
from ..core.config import get_settings


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware that handles:
    - Automatic token validation for protected routes
    - Session management and activity tracking
    - Device fingerprinting
    - Security logging
    - Performance caching
    """
    
    def __init__(self, app, jwt_handler: Optional[JWTHandler] = None, 
                 redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.settings = get_settings()
        self.jwt_handler = jwt_handler
        self.redis_client = redis_client
        self.role_manager = RoleManager(redis_client)
        
        # Routes that don't require authentication
        self.public_routes = {
            "/docs", "/redoc", "/openapi.json", "/health",
            "/auth/login", "/auth/register", "/auth/refresh", 
            "/auth/forgot-password", "/auth/reset-password",
            "/auth/verify-email", "/metrics"
        }
        
        # Routes that require authentication
        self.protected_prefixes = {"/api", "/admin", "/auth/me", "/auth/logout"}
        
        # Cache for user data to reduce database hits
        self.user_cache_ttl = 300  # 5 minutes
    
    def _is_public_route(self, path: str) -> bool:
        """Check if route is public (doesn't require authentication)"""
        
        # Exact matches
        if path in self.public_routes:
            return True
        
        # Prefix matches for public routes
        public_prefixes = {"/health", "/metrics", "/docs", "/redoc", "/static"}
        for prefix in public_prefixes:
            if path.startswith(prefix):
                return True
        
        # Check if it's a protected route
        for prefix in self.protected_prefixes:
            if path.startswith(prefix):
                return False
        
        # Default to public for unknown routes
        return True
    
    def _extract_client_ip(self, request: Request) -> str:
        """Extract client IP with proxy support"""
        
        # Check forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to request client
        return request.client.host if request.client else "127.0.0.1"
    
    def _generate_device_fingerprint(self, request: Request) -> str:
        """Generate device fingerprint from request headers"""
        import hashlib
        
        fingerprint_data = [
            request.headers.get("User-Agent", ""),
            request.headers.get("Accept-Language", ""),
            request.headers.get("Accept-Encoding", ""),
            self._extract_client_ip(request)
        ]
        
        fingerprint_string = "|".join(fingerprint_data)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers or query params"""
        
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check query parameter (less secure, for special cases)
        token = request.query_params.get("token")
        if token:
            return token
        
        return None
    
    def _get_cached_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from cache"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"auth_user_data:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                import json
                return json.loads(cached_data)
        except Exception:
            pass
        
        return None
    
    def _cache_user_data(self, user_id: str, user_data: Dict[str, Any]):
        """Cache user data for performance"""
        if not self.redis_client:
            return
        
        try:
            import json
            cache_key = f"auth_user_data:{user_id}"
            self.redis_client.setex(
                cache_key, 
                self.user_cache_ttl, 
                json.dumps(user_data)
            )
        except Exception:
            pass
    
    def _validate_token_and_get_user(self, token: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate token and return user data
        
        Returns:
            Tuple of (user_data, error_message)
        """
        if not self.jwt_handler:
            return None, "Authentication not configured"
        
        # Verify token
        token_data = self.jwt_handler.verify_token(token, TokenType.ACCESS)
        if not token_data:
            return None, "Invalid or expired token"
        
        user_id = token_data.get("user_id")
        if not user_id:
            return None, "Invalid token payload"
        
        # Check cached user data first
        cached_user = self._get_cached_user_data(user_id)
        if cached_user:
            # Update cached data with fresh token info
            cached_user.update({
                "session_id": token_data.get("session_id"),
                "mfa_verified": token_data.get("mfa_verified", False),
                "token_type": token_data.get("token_type"),
                "auth_method": "jwt"
            })
            return cached_user, None
        
        # Get user roles and permissions
        user_roles = self.role_manager.get_user_roles(user_id)
        role_names = [ur.role for ur in user_roles]
        permissions = self.role_manager.get_user_permissions(role_names)
        
        user_data = {
            "user_id": user_id,
            "username": token_data.get("username", ""),
            "email": token_data.get("email", ""),
            "roles": [role.value for role in role_names],
            "permissions": [perm.value for perm in permissions],
            "session_id": token_data.get("session_id"),
            "mfa_verified": token_data.get("mfa_verified", False),
            "token_type": token_data.get("token_type"),
            "auth_method": "jwt"
        }
        
        # Cache user data
        self._cache_user_data(user_id, user_data)
        
        return user_data, None
    
    def _log_security_event(self, event_type: str, request: Request, 
                           user_data: Optional[Dict[str, Any]] = None,
                           details: Optional[Dict[str, Any]] = None):
        """Log security events"""
        if not self.redis_client:
            return
        
        try:
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "user_id": user_data.get("user_id") if user_data else None,
                "username": user_data.get("username") if user_data else None,
                "ip_address": self._extract_client_ip(request),
                "user_agent": request.headers.get("User-Agent", ""),
                "endpoint": str(request.url),
                "method": request.method,
                "details": details or {}
            }
            
            # Log to Redis
            log_key = f"auth_log:{datetime.utcnow().strftime('%Y%m%d')}"
            self.redis_client.lpush(log_key, str(event_data))
            self.redis_client.expire(log_key, 86400 * 7)  # Keep for 7 days
        
        except Exception:
            pass  # Don't fail request if logging fails
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        
        start_time = time.time()
        path = request.url.path
        
        # Skip authentication for public routes
        if self._is_public_route(path):
            response = await call_next(request)
            return response
        
        # Extract token
        token = self._extract_token(request)
        
        if not token:
            self._log_security_event(
                "auth_token_missing",
                request,
                details={"path": path}
            )
            
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Authentication required",
                    "error_code": "MISSING_TOKEN"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate token and get user data
        user_data, error = self._validate_token_and_get_user(token)
        
        if error or not user_data:
            self._log_security_event(
                "auth_token_invalid",
                request,
                details={
                    "path": path,
                    "error": error,
                    "token_prefix": token[:10] if token else None
                }
            )
            
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={
                    "detail": error or "Authentication failed",
                    "error_code": "INVALID_TOKEN"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Add user data to request state
        request.state.current_user = user_data
        request.state.client_ip = self._extract_client_ip(request)
        request.state.device_fingerprint = self._generate_device_fingerprint(request)
        
        # Update session activity if session exists
        if self.jwt_handler and user_data.get("session_id"):
            try:
                self.jwt_handler.update_session_activity(user_data["session_id"])
            except Exception:
                pass  # Don't fail request if session update fails
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log successful authentication
            processing_time = time.time() - start_time
            if processing_time > 1.0:  # Log slow requests
                self._log_security_event(
                    "auth_slow_request",
                    request,
                    user_data,
                    details={
                        "processing_time": processing_time,
                        "status_code": response.status_code
                    }
                )
            
            # Add security headers
            response.headers["X-User-ID"] = user_data["user_id"]
            response.headers["X-Session-ID"] = user_data.get("session_id", "")
            
            return response
        
        except Exception as e:
            self._log_security_event(
                "auth_request_error",
                request,
                user_data,
                details={"error": str(e)}
            )
            raise


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API Key authentication middleware for programmatic access
    """
    
    def __init__(self, app, jwt_handler: Optional[JWTHandler] = None):
        super().__init__(app)
        self.jwt_handler = jwt_handler
        
        # Routes that support API key authentication
        self.api_key_routes = {"/api"}
    
    def _is_api_key_route(self, path: str) -> bool:
        """Check if route supports API key authentication"""
        for prefix in self.api_key_routes:
            if path.startswith(prefix):
                return True
        return False
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from headers"""
        return request.headers.get("X-API-Key")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process API key authentication"""
        
        path = request.url.path
        
        # Only process API key routes
        if not self._is_api_key_route(path):
            return await call_next(request)
        
        # Check if regular JWT token is present
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Let regular auth middleware handle this
            return await call_next(request)
        
        # Extract API key
        api_key = self._extract_api_key(request)
        if not api_key:
            # No API key, let regular auth middleware handle
            return await call_next(request)
        
        # Validate API key
        if self.jwt_handler:
            api_data = self.jwt_handler.validate_api_key(api_key)
            if api_data:
                # Create user context from API key
                user_data = {
                    "user_id": api_data["user_id"],
                    "username": f"api_key_{api_data['name']}",
                    "email": "",
                    "roles": [],
                    "permissions": api_data["permissions"],
                    "session_id": "",
                    "mfa_verified": True,  # API keys bypass MFA
                    "auth_method": "api_key",
                    "api_key_id": api_data["key_id"]
                }
                
                request.state.current_user = user_data
                return await call_next(request)
        
        # Invalid API key
        return JSONResponse(
            status_code=HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Invalid API key",
                "error_code": "INVALID_API_KEY"
            }
        )