"""
Security Headers Middleware
Implements comprehensive HTTP security headers
OWASP Secure Headers Project compliant
"""

import hashlib
import secrets
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class CSPDirective(str, Enum):
    """Content Security Policy directives"""
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    FONT_SRC = "font-src"
    CONNECT_SRC = "connect-src"
    MEDIA_SRC = "media-src"
    OBJECT_SRC = "object-src"
    FRAME_SRC = "frame-src"
    FRAME_ANCESTORS = "frame-ancestors"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    REPORT_URI = "report-uri"
    REPORT_TO = "report-to"
    UPGRADE_INSECURE = "upgrade-insecure-requests"
    BLOCK_ALL_MIXED = "block-all-mixed-content"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security headers middleware implementing:
    - Content Security Policy (CSP)
    - HTTP Strict Transport Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    - Cross-Origin policies
    - Cache control
    """
    
    def __init__(self,
                 app,
                 enable_csp: bool = True,
                 enable_hsts: bool = True,
                 csp_report_only: bool = False,
                 custom_csp: Optional[Dict[str, str]] = None):
        
        super().__init__(app)
        
        self.enable_csp = enable_csp
        self.enable_hsts = enable_hsts
        self.csp_report_only = csp_report_only
        
        # Default CSP policy
        self.csp_policy = {
            CSPDirective.DEFAULT_SRC: "'self'",
            CSPDirective.SCRIPT_SRC: "'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
            CSPDirective.STYLE_SRC: "'self' 'unsafe-inline' https://fonts.googleapis.com",
            CSPDirective.IMG_SRC: "'self' data: https:",
            CSPDirective.FONT_SRC: "'self' https://fonts.gstatic.com",
            CSPDirective.CONNECT_SRC: "'self' https://api.example.com wss://ws.example.com",
            CSPDirective.MEDIA_SRC: "'self'",
            CSPDirective.OBJECT_SRC: "'none'",
            CSPDirective.FRAME_SRC: "'self'",
            CSPDirective.FRAME_ANCESTORS: "'self'",
            CSPDirective.BASE_URI: "'self'",
            CSPDirective.FORM_ACTION: "'self'",
            CSPDirective.UPGRADE_INSECURE: "",
            CSPDirective.BLOCK_ALL_MIXED: ""
        }
        
        # Override with custom CSP if provided
        if custom_csp:
            self.csp_policy.update(custom_csp)
        
        # HSTS configuration
        self.hsts_max_age = 31536000  # 1 year
        self.hsts_include_subdomains = True
        self.hsts_preload = True
        
        # Permissions Policy
        self.permissions_policy = {
            "accelerometer": "()",
            "camera": "()",
            "geolocation": "()",
            "gyroscope": "()",
            "magnetometer": "()",
            "microphone": "()",
            "payment": "()",
            "usb": "()"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Apply security headers to response"""
        
        # Generate nonce for CSP
        nonce = self._generate_nonce() if self.enable_csp else None
        
        # Store nonce in request state for use in templates
        if nonce:
            request.state.csp_nonce = nonce
        
        # Process request
        response = await call_next(request)
        
        # Apply security headers
        self._apply_security_headers(response, request, nonce)
        
        return response
    
    def _apply_security_headers(self,
                               response: Response,
                               request: Request,
                               nonce: Optional[str] = None):
        """Apply all security headers"""
        
        # Content Security Policy
        if self.enable_csp:
            csp_header = self._build_csp_header(nonce)
            header_name = "Content-Security-Policy-Report-Only" if self.csp_report_only else "Content-Security-Policy"
            response.headers[header_name] = csp_header
        
        # HTTP Strict Transport Security
        if self.enable_hsts and request.url.scheme == "https":
            hsts_header = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_header += "; includeSubDomains"
            if self.hsts_preload:
                hsts_header += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_header
        
        # X-Frame-Options (clickjacking protection)
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        
        # X-Content-Type-Options (MIME sniffing protection)
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection (XSS filter)
        # Note: Deprecated in modern browsers but still useful for older ones
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy (formerly Feature-Policy)
        permissions = self._build_permissions_policy()
        response.headers["Permissions-Policy"] = permissions
        
        # Cross-Origin headers
        self._apply_cors_headers(response, request)
        
        # Cache control for sensitive content
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        
        # Remove sensitive headers
        self._remove_sensitive_headers(response)
    
    def _build_csp_header(self, nonce: Optional[str] = None) -> str:
        """Build Content Security Policy header"""
        
        directives = []
        
        for directive, value in self.csp_policy.items():
            if value:
                # Add nonce to script-src if provided
                if directive == CSPDirective.SCRIPT_SRC and nonce:
                    value = f"{value} 'nonce-{nonce}'"
                
                # Add nonce to style-src if provided
                if directive == CSPDirective.STYLE_SRC and nonce:
                    value = f"{value} 'nonce-{nonce}'"
                
                if value.strip():
                    directives.append(f"{directive} {value}")
                else:
                    directives.append(directive)
        
        return "; ".join(directives)
    
    def _build_permissions_policy(self) -> str:
        """Build Permissions-Policy header"""
        
        policies = []
        for feature, allowlist in self.permissions_policy.items():
            policies.append(f"{feature}={allowlist}")
        
        return ", ".join(policies)
    
    def _apply_cors_headers(self, response: Response, request: Request):
        """Apply CORS headers with security considerations"""
        
        # Only apply CORS for API endpoints
        if not request.url.path.startswith("/api"):
            return
        
        # Get origin from request
        origin = request.headers.get("Origin")
        
        if origin and self._is_allowed_origin(origin):
            # Allow specific origin (not wildcard for security)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            
            # Preflight request headers
            if request.method == "OPTIONS":
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
                response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
        
        # Always vary on Origin for caching
        response.headers["Vary"] = "Origin"
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed for CORS"""
        
        allowed_origins = [
            "https://app.example.com",
            "https://www.example.com",
            "http://localhost:3000",  # Development
            "http://localhost:8080"   # Development
        ]
        
        return origin in allowed_origins
    
    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint handles sensitive data"""
        
        sensitive_paths = [
            "/api/auth",
            "/api/users",
            "/api/payments",
            "/api/admin",
            "/api/documents"
        ]
        
        return any(path.startswith(p) for p in sensitive_paths)
    
    def _remove_sensitive_headers(self, response: Response):
        """Remove headers that might leak sensitive information"""
        
        headers_to_remove = [
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version"
        ]
        
        for header in headers_to_remove:
            response.headers.pop(header, None)
    
    def _generate_nonce(self) -> str:
        """Generate CSP nonce"""
        return secrets.token_urlsafe(16)


class CSPReportHandler:
    """Handle Content Security Policy violation reports"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    async def handle_report(self, request: Request) -> JSONResponse:
        """Process CSP violation report"""
        
        try:
            # Parse report
            report_data = await request.json()
            
            # Extract violation details
            if "csp-report" in report_data:
                violation = report_data["csp-report"]
            else:
                violation = report_data
            
            # Log violation
            self._log_violation(violation)
            
            # Analyze for potential attacks
            if self._is_potential_attack(violation):
                self._handle_potential_attack(violation, request)
            
            return JSONResponse(
                status_code=204,
                content={}
            )
        except Exception as e:
            print(f"CSP report error: {e}")
            return JSONResponse(
                status_code=204,
                content={}
            )
    
    def _log_violation(self, violation: Dict[str, Any]):
        """Log CSP violation"""
        
        violation_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "document_uri": violation.get("document-uri"),
            "referrer": violation.get("referrer"),
            "violated_directive": violation.get("violated-directive"),
            "effective_directive": violation.get("effective-directive"),
            "original_policy": violation.get("original-policy"),
            "blocked_uri": violation.get("blocked-uri"),
            "status_code": violation.get("status-code"),
            "source_file": violation.get("source-file"),
            "line_number": violation.get("line-number"),
            "column_number": violation.get("column-number")
        }
        
        # Store in Redis
        self.redis_client.lpush(
            "csp_violations",
            json.dumps(violation_data)
        )
        self.redis_client.ltrim("csp_violations", 0, 10000)  # Keep last 10000
    
    def _is_potential_attack(self, violation: Dict[str, Any]) -> bool:
        """Check if violation indicates potential attack"""
        
        blocked_uri = violation.get("blocked-uri", "")
        
        # Check for XSS patterns
        xss_patterns = [
            "javascript:",
            "data:text/html",
            "vbscript:",
            "<script",
            "onerror=",
            "onload="
        ]
        
        for pattern in xss_patterns:
            if pattern in blocked_uri.lower():
                return True
        
        # Check for suspicious domains
        if blocked_uri.startswith("http"):
            # Would check against threat intelligence
            pass
        
        return False
    
    def _handle_potential_attack(self, violation: Dict[str, Any], request: Request):
        """Handle potential attack detected via CSP"""
        
        # Get client IP
        client_ip = request.client.host
        
        # Record attack attempt
        attack_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "ip": client_ip,
            "type": "csp_violation",
            "details": violation
        }
        
        self.redis_client.lpush(
            "security_incidents",
            json.dumps(attack_data)
        )
        
        # Increment violation count for IP
        violation_key = f"csp_violations:{client_ip}"
        count = self.redis_client.incr(violation_key)
        self.redis_client.expire(violation_key, 3600)  # 1 hour window
        
        # Block IP if too many violations
        if count > 10:
            # Would trigger IP blocking
            pass


class SecurityHeadersConfig:
    """Configuration for security headers"""
    
    def __init__(self):
        self.profiles = {
            "strict": self._strict_profile(),
            "moderate": self._moderate_profile(),
            "relaxed": self._relaxed_profile(),
            "api": self._api_profile()
        }
    
    def _strict_profile(self) -> Dict[str, Any]:
        """Strict security profile"""
        
        return {
            "csp": {
                CSPDirective.DEFAULT_SRC: "'none'",
                CSPDirective.SCRIPT_SRC: "'self'",
                CSPDirective.STYLE_SRC: "'self'",
                CSPDirective.IMG_SRC: "'self'",
                CSPDirective.FONT_SRC: "'self'",
                CSPDirective.CONNECT_SRC: "'self'",
                CSPDirective.FRAME_ANCESTORS: "'none'",
                CSPDirective.BASE_URI: "'none'",
                CSPDirective.FORM_ACTION: "'self'",
                CSPDirective.UPGRADE_INSECURE: "",
                CSPDirective.BLOCK_ALL_MIXED: ""
            },
            "hsts": {
                "max_age": 63072000,  # 2 years
                "include_subdomains": True,
                "preload": True
            },
            "x_frame_options": "DENY",
            "referrer_policy": "no-referrer"
        }
    
    def _moderate_profile(self) -> Dict[str, Any]:
        """Moderate security profile"""
        
        return {
            "csp": {
                CSPDirective.DEFAULT_SRC: "'self'",
                CSPDirective.SCRIPT_SRC: "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
                CSPDirective.STYLE_SRC: "'self' 'unsafe-inline' https://fonts.googleapis.com",
                CSPDirective.IMG_SRC: "'self' data: https:",
                CSPDirective.FONT_SRC: "'self' https://fonts.gstatic.com",
                CSPDirective.CONNECT_SRC: "'self' https:",
                CSPDirective.FRAME_ANCESTORS: "'self'",
                CSPDirective.UPGRADE_INSECURE: ""
            },
            "hsts": {
                "max_age": 31536000,  # 1 year
                "include_subdomains": True,
                "preload": False
            },
            "x_frame_options": "SAMEORIGIN",
            "referrer_policy": "strict-origin-when-cross-origin"
        }
    
    def _relaxed_profile(self) -> Dict[str, Any]:
        """Relaxed security profile"""
        
        return {
            "csp": {
                CSPDirective.DEFAULT_SRC: "*",
                CSPDirective.SCRIPT_SRC: "* 'unsafe-inline' 'unsafe-eval'",
                CSPDirective.STYLE_SRC: "* 'unsafe-inline'",
                CSPDirective.IMG_SRC: "*",
                CSPDirective.FRAME_ANCESTORS: "*"
            },
            "hsts": {
                "max_age": 86400,  # 1 day
                "include_subdomains": False,
                "preload": False
            },
            "x_frame_options": "SAMEORIGIN",
            "referrer_policy": "origin-when-cross-origin"
        }
    
    def _api_profile(self) -> Dict[str, Any]:
        """API-specific security profile"""
        
        return {
            "csp": None,  # CSP not typically used for APIs
            "hsts": {
                "max_age": 31536000,
                "include_subdomains": True,
                "preload": True
            },
            "x_frame_options": "DENY",
            "referrer_policy": "no-referrer",
            "cors": {
                "allow_credentials": True,
                "max_age": 86400
            }
        }
    
    def get_profile(self, name: str) -> Dict[str, Any]:
        """Get security profile by name"""
        return self.profiles.get(name, self._moderate_profile())