"""
Pydantic schemas for user-related operations
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict
# import uuid


class UserBase(BaseModel):
    """Base user schema with common fields"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(...)
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: bool = Field(True)


class UserCreate(UserBase):
    """Schema for user creation"""
    password: str = Field(..., min_length=8, max_length=100, description="User password")


class UserUpdate(BaseModel):
    """Schema for user updates"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")


class UserResponse(UserBase):
    """Schema for user responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    organization_id: Optional[int] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    """Schema for authentication tokens"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Schema for token data"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: list[str] = []


class UserStatistics(BaseModel):
    """Schema for user statistics"""
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    users_by_organization: dict[str, int]