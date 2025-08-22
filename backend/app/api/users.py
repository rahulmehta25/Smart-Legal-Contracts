from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.user_service import UserService
from app.models.user import (
    UserCreate, UserResponse, UserLogin, Token, User
)

router = APIRouter(prefix="/users", tags=["users"])
security = HTTPBearer()
user_service = UserService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user
    """
    token = credentials.credentials
    user = user_service.get_current_user(db, token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user
    """
    try:
        user = user_service.create_user(db, user_data)
        return UserResponse.from_orm(user)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


@router.post("/login", response_model=Token)
async def login_user(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login user and get access token
    """
    try:
        token = user_service.login_user(db, login_data)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    """
    return UserResponse.from_orm(current_user)


@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of users (requires authentication)
    """
    try:
        users = user_service.get_users(db, skip=skip, limit=limit, active_only=active_only)
        return [UserResponse.from_orm(user) for user in users]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user by ID (requires authentication)
    """
    try:
        user = user_service.get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user: {str(e)}")


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user information
    """
    try:
        # Remove password field if present - use separate endpoint for password changes
        if 'password' in update_data:
            del update_data['password']
        
        updated_user = user_service.update_user(db, current_user.id, update_data)
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse.from_orm(updated_user)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@router.post("/change-password")
async def change_password(
    password_data: dict,  # Should contain 'current_password' and 'new_password'
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    """
    try:
        current_password = password_data.get('current_password')
        new_password = password_data.get('new_password')
        
        if not current_password or not new_password:
            raise HTTPException(
                status_code=400, 
                detail="Both current_password and new_password are required"
            )
        
        success = user_service.change_password(
            db, current_user.id, current_password, new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Current password is incorrect"
            )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error changing password: {str(e)}")


@router.post("/deactivate/{user_id}")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate a user account (requires authentication)
    """
    try:
        # Prevent users from deactivating themselves
        if user_id == current_user.id:
            raise HTTPException(
                status_code=400,
                detail="Cannot deactivate your own account"
            )
        
        success = user_service.deactivate_user(db, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User {user_id} deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating user: {str(e)}")


@router.get("/stats/overview")
async def get_user_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user statistics (requires authentication)
    """
    try:
        stats = user_service.get_user_statistics(db)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


# Public endpoints (no authentication required)
@router.post("/verify-token")
async def verify_token(
    token_data: dict,  # Should contain 'token'
    db: Session = Depends(get_db)
):
    """
    Verify if a token is valid
    """
    try:
        token = token_data.get('token')
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        user = user_service.get_current_user(db, token)
        if not user:
            return {"valid": False, "message": "Invalid token"}
        
        return {
            "valid": True,
            "user_id": user.id,
            "username": user.username,
            "message": "Token is valid"
        }
        
    except Exception as e:
        return {"valid": False, "message": f"Error verifying token: {str(e)}"}