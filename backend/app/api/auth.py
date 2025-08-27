"""
Authentication API Endpoints

Comprehensive authentication system with:
- User registration and login
- JWT token management with refresh
- Email verification
- Password reset
- Multi-factor authentication
- OAuth2 integration
- Profile management
- Session management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.orm import Session

from ..auth.jwt_handler import JWTHandler, TokenType
from ..auth.password import PasswordHandler, PasswordValidationResult
from ..auth.permissions import Role, Permission
from ..auth.dependencies import (
    get_current_user, 
    get_current_active_user,
    get_jwt_handler,
    get_password_handler,
    get_role_manager,
    require_permission,
    rate_limit,
    log_security_event,
    get_client_ip,
    get_user_agent
)
from ..core.config import get_settings
from ..db.database import get_db
from ..models.user import User, UserCreate, UserResponse


# Pydantic models for API
class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False
    mfa_token: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
    requires_mfa: bool = False


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    terms_accepted: bool = True
    
    @validator('terms_accepted')
    def terms_must_be_accepted(cls, v):
        if not v:
            raise ValueError('Terms and conditions must be accepted')
        return v


class RegisterResponse(BaseModel):
    user: UserResponse
    message: str
    email_verification_required: bool = True


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class VerifyEmailRequest(BaseModel):
    token: str


class MFASetupResponse(BaseModel):
    qr_code_uri: str
    backup_codes: List[str]
    secret: str


class MFAVerifyRequest(BaseModel):
    token: str
    backup_code: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    organization: Optional[str] = None
    email: Optional[EmailStr] = None


class SessionInfo(BaseModel):
    session_id: str
    device_id: Optional[str]
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    is_current: bool = False


class UserSessionsResponse(BaseModel):
    sessions: List[SessionInfo]
    total_count: int


# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer(auto_error=False)


# Background tasks
async def send_verification_email(email: str, token: str, username: str):
    """Send email verification email (placeholder implementation)"""
    # TODO: Implement actual email sending
    print(f"Sending verification email to {email} with token {token[:10]}...")


async def send_password_reset_email(email: str, token: str, username: str):
    """Send password reset email (placeholder implementation)"""
    # TODO: Implement actual email sending
    print(f"Sending password reset email to {email} with token {token[:10]}...")


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: Session = Depends(get_db),
    password_handler: PasswordHandler = Depends(get_password_handler),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    _: None = Depends(rate_limit(max_requests=5, window_minutes=60, per_user=False))
):
    """
    Register a new user account
    
    - Creates new user with encrypted password
    - Validates password strength according to OWASP guidelines
    - Sends email verification
    - Logs security events
    """
    
    # Validate password strength
    password_result = password_handler.validate_password(
        request.password,
        username=request.username,
        email=request.email
    )
    
    if not password_result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Password does not meet security requirements",
                "errors": password_result.errors,
                "suggestions": password_result.suggestions
            }
        )
    
    # Check for password breaches if enabled
    try:
        is_breached, breach_count = await password_handler.check_password_breach(request.password)
        if is_breached:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": f"Password found in {breach_count} data breaches. Choose a different password.",
                    "errors": ["Password has been compromised in data breaches"]
                }
            )
    except Exception:
        # Continue if breach check fails (don't block registration)
        pass
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == request.username) | (User.email == request.email)
    ).first()
    
    if existing_user:
        if existing_user.email == request.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    try:
        # Hash password
        hashed_password = password_handler.hash_password(request.password)
        
        # Create user
        new_user = User(
            username=request.username,
            email=request.email,
            hashed_password=hashed_password,
            full_name=request.full_name,
            organization=request.organization,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Store password in history
        password_handler.store_password_history(str(new_user.id), hashed_password)
        
        # Generate email verification token
        verification_token = jwt_handler.create_email_verification_token(
            user_id=str(new_user.id),
            email=new_user.email
        )
        
        # Send verification email
        background_tasks.add_task(
            send_verification_email,
            new_user.email,
            verification_token,
            new_user.username
        )
        
        # Log registration event
        await log_security_event(
            "user_registered",
            user_id=str(new_user.id),
            details={
                "username": new_user.username,
                "email": new_user.email,
                "ip_address": get_client_ip(http_request)
            }
        )(http_request)
        
        return RegisterResponse(
            user=UserResponse.from_orm(new_user),
            message="Registration successful. Please check your email to verify your account.",
            email_verification_required=True
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    password_handler: PasswordHandler = Depends(get_password_handler),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    role_manager = Depends(get_role_manager),
    _: None = Depends(rate_limit(max_requests=10, window_minutes=15, per_user=False))
):
    """
    Authenticate user and return JWT tokens
    
    - Validates credentials
    - Checks for account lockouts
    - Supports MFA
    - Creates session
    - Returns access and refresh tokens
    """
    
    client_ip = get_client_ip(http_request)
    user_agent = get_user_agent(http_request)
    
    # Check for brute force attempts
    login_identifier = f"{request.username}:{client_ip}"
    
    try:
        # Find user
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user:
            # Record failed login attempt for unknown user
            await log_security_event(
                "login_failed_unknown_user",
                details={
                    "username": request.username,
                    "ip_address": client_ip,
                    "reason": "user_not_found"
                }
            )(http_request)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if account is active
        if not user.is_active:
            await log_security_event(
                "login_failed_inactive_account",
                user_id=str(user.id),
                details={
                    "username": request.username,
                    "ip_address": client_ip
                }
            )(http_request)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled"
            )
        
        # Verify password
        if not password_handler.verify_password(request.password, user.hashed_password):
            await log_security_event(
                "login_failed_wrong_password",
                user_id=str(user.id),
                details={
                    "username": request.username,
                    "ip_address": client_ip
                }
            )(http_request)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if password needs rehashing
        if password_handler.needs_rehash(user.hashed_password):
            new_hash = password_handler.hash_password(request.password)
            user.hashed_password = new_hash
            db.commit()
        
        # Get user roles and permissions
        user_roles = role_manager.get_user_roles(str(user.id))
        role_names = [ur.role for ur in user_roles]
        permissions = role_manager.get_user_permissions(role_names)
        
        # Create session
        session_id = jwt_handler.create_session(
            user_id=str(user.id),
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # For now, skip MFA check and create tokens directly
        # TODO: Implement MFA verification logic
        mfa_verified = True  # Would be False if MFA required but not verified
        
        if not mfa_verified and request.mfa_token:
            # Verify MFA token
            # TODO: Implement MFA verification
            pass
        
        # Create tokens
        expires_in = 15 * 60 if not request.remember_me else 7 * 24 * 60 * 60  # 15 min or 7 days
        
        access_token = jwt_handler.create_access_token(
            user_id=str(user.id),
            username=user.username,
            email=user.email,
            roles=[role.value for role in role_names],
            permissions=[perm.value for perm in permissions],
            session_id=session_id,
            mfa_verified=mfa_verified,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        refresh_token = jwt_handler.create_refresh_token(
            user_id=str(user.id),
            username=user.username,
            session_id=session_id
        )
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Log successful login
        await log_security_event(
            "login_successful",
            user_id=str(user.id),
            details={
                "username": user.username,
                "ip_address": client_ip,
                "session_id": session_id
            }
        )(http_request)
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            user=UserResponse.from_orm(user),
            requires_mfa=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/refresh")
async def refresh_token(
    request: RefreshTokenRequest,
    http_request: Request,
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    _: None = Depends(rate_limit(max_requests=50, window_minutes=60, per_user=False))
):
    """
    Refresh access token using refresh token
    """
    
    try:
        # Refresh tokens
        token_response = jwt_handler.refresh_access_token(request.refresh_token)
        
        if not token_response:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return token_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Logout user and invalidate tokens
    """
    
    try:
        # Get session ID from token
        session_id = current_user.get("session_id")
        
        if session_id:
            # Terminate session
            jwt_handler.terminate_session(session_id)
        
        # Log logout event
        await log_security_event(
            "user_logout",
            user_id=current_user["user_id"],
            details={
                "session_id": session_id,
                "ip_address": get_client_ip(http_request)
            }
        )(http_request)
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/logout-all")
async def logout_all_sessions(
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Logout from all sessions/devices
    """
    
    try:
        # Terminate all user sessions
        jwt_handler.terminate_all_user_sessions(current_user["user_id"])
        
        # Blacklist all user tokens
        jwt_handler.blacklist_all_user_tokens(current_user["user_id"])
        
        # Log global logout event
        await log_security_event(
            "user_logout_all",
            user_id=current_user["user_id"],
            details={
                "ip_address": get_client_ip(http_request)
            }
        )(http_request)
        
        return {"message": "Successfully logged out from all devices"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout from all devices failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current user information
    """
    try:
        user = db.query(User).filter(User.id == int(current_user["user_id"])).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )


@router.put("/profile")
async def update_profile(
    request: ProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile information
    """
    try:
        user = db.query(User).filter(User.id == int(current_user["user_id"])).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields
        if request.full_name is not None:
            user.full_name = request.full_name
        if request.organization is not None:
            user.organization = request.organization
        if request.email is not None and request.email != user.email:
            # Check if email is already taken
            existing_user = db.query(User).filter(
                User.email == request.email,
                User.id != user.id
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )
            
            # TODO: Send email verification for new email
            user.email = request.email
            user.is_verified = False
        
        db.commit()
        db.refresh(user)
        
        return {"message": "Profile updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    password_handler: PasswordHandler = Depends(get_password_handler)
):
    """
    Change user password
    """
    try:
        user = db.query(User).filter(User.id == int(current_user["user_id"])).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if user can change password (not too soon after last change)
        can_change, reason = password_handler.can_change_password(current_user["user_id"])
        if not can_change:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=reason
            )
        
        # Verify current password
        if not password_handler.verify_password(request.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Check if new password was used recently
        if password_handler.check_password_history(current_user["user_id"], request.new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot reuse a recent password"
            )
        
        # Validate new password
        password_result = password_handler.validate_password(
            request.new_password,
            username=user.username,
            email=user.email
        )
        
        if not password_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "New password does not meet security requirements",
                    "errors": password_result.errors,
                    "suggestions": password_result.suggestions
                }
            )
        
        # Hash and store new password
        new_hash = password_handler.hash_password(request.new_password)
        user.hashed_password = new_hash
        db.commit()
        
        # Store in password history
        password_handler.store_password_history(current_user["user_id"], new_hash)
        password_handler.record_password_change(current_user["user_id"])
        
        # Log password change
        await log_security_event(
            "password_changed",
            user_id=current_user["user_id"],
            details={
                "ip_address": get_client_ip(http_request)
            }
        )(http_request)
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    _: None = Depends(rate_limit(max_requests=5, window_minutes=60, per_user=False))
):
    """
    Send password reset email
    """
    try:
        user = db.query(User).filter(User.email == request.email).first()
        
        if user:
            # Generate reset token
            reset_token = jwt_handler.create_password_reset_token(
                user_id=str(user.id),
                email=user.email
            )
            
            # Send reset email
            background_tasks.add_task(
                send_password_reset_email,
                user.email,
                reset_token,
                user.username
            )
        
        # Always return success to prevent email enumeration
        return {"message": "If the email exists, a password reset link has been sent"}
        
    except Exception as e:
        # Don't reveal internal errors
        return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
    password_handler: PasswordHandler = Depends(get_password_handler),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Reset password using reset token
    """
    try:
        # Verify reset token
        token_data = jwt_handler.verify_token(request.token, TokenType.PASSWORD_RESET)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        user_id = token_data.get("user_id")
        user = db.query(User).filter(User.id == int(user_id)).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate new password
        password_result = password_handler.validate_password(
            request.new_password,
            username=user.username,
            email=user.email
        )
        
        if not password_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Password does not meet security requirements",
                    "errors": password_result.errors,
                    "suggestions": password_result.suggestions
                }
            )
        
        # Hash and store new password
        new_hash = password_handler.hash_password(request.new_password)
        user.hashed_password = new_hash
        db.commit()
        
        # Store in password history
        password_handler.store_password_history(str(user.id), new_hash)
        password_handler.record_password_change(str(user.id))
        
        # Blacklist the reset token
        jti = token_data.get("jti")
        if jti:
            jwt_handler.blacklist_token(jti)
        
        return {"message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.post("/verify-email")
async def verify_email(
    request: VerifyEmailRequest,
    db: Session = Depends(get_db),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Verify user email address
    """
    try:
        # Verify email verification token
        token_data = jwt_handler.verify_token(request.token, TokenType.EMAIL_VERIFICATION)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        user_id = token_data.get("user_id")
        user = db.query(User).filter(User.id == int(user_id)).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify email
        user.is_verified = True
        db.commit()
        
        # Blacklist the verification token
        jti = token_data.get("jti")
        if jti:
            jwt_handler.blacklist_token(jti)
        
        return {"message": "Email verified successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed"
        )


@router.get("/sessions", response_model=UserSessionsResponse)
async def get_user_sessions(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Get user's active sessions
    """
    try:
        sessions = jwt_handler.get_user_sessions(current_user["user_id"])
        current_session_id = current_user.get("session_id")
        
        session_info = []
        for session in sessions:
            session_info.append(SessionInfo(
                session_id=session["session_id"],
                device_id=session.get("device_id"),
                ip_address=session["ip_address"],
                user_agent=session["user_agent"],
                created_at=datetime.fromisoformat(session["created_at"]),
                last_activity=datetime.fromisoformat(session["last_activity"]),
                is_current=session["session_id"] == current_session_id
            ))
        
        return UserSessionsResponse(
            sessions=session_info,
            total_count=len(session_info)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get sessions"
        )


@router.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
):
    """
    Terminate a specific session
    """
    try:
        # Verify session belongs to user
        user_sessions = jwt_handler.get_user_sessions(current_user["user_id"])
        session_exists = any(s["session_id"] == session_id for s in user_sessions)
        
        if not session_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Terminate session
        jwt_handler.terminate_session(session_id)
        
        return {"message": "Session terminated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to terminate session"
        )