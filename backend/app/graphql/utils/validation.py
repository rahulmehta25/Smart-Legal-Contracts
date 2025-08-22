"""
Validation utilities for GraphQL inputs
"""

import re
from typing import List
from ..types import DocumentUploadInput, PatternCreateInput, UserCreateInput


def validate_document_upload(input: DocumentUploadInput) -> List[str]:
    """Validate document upload input"""
    errors = []
    
    # Validate filename
    if not input.filename or not input.filename.strip():
        errors.append("Filename is required")
    elif len(input.filename) > 255:
        errors.append("Filename too long (max 255 characters)")
    
    # Validate file type
    allowed_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if input.file_type not in allowed_types:
        errors.append(f"File type {input.file_type} not supported")
    
    # Validate content
    if not input.content or not input.content.strip():
        errors.append("Content is required")
    elif len(input.content) > 10 * 1024 * 1024:  # 10MB
        errors.append("Content too large (max 10MB)")
    
    return errors


def validate_pattern_input(input: PatternCreateInput) -> List[str]:
    """Validate pattern input"""
    errors = []
    
    # Validate pattern name
    if not input.pattern_name or not input.pattern_name.strip():
        errors.append("Pattern name is required")
    elif len(input.pattern_name) > 200:
        errors.append("Pattern name too long (max 200 characters)")
    
    # Validate pattern text
    if not input.pattern_text or not input.pattern_text.strip():
        errors.append("Pattern text is required")
    elif len(input.pattern_text) > 10000:
        errors.append("Pattern text too long (max 10000 characters)")
    
    # Validate regex if pattern type is regex
    if input.pattern_type.value == "regex":
        try:
            re.compile(input.pattern_text)
        except re.error as e:
            errors.append(f"Invalid regex pattern: {str(e)}")
    
    # Validate category
    if not input.category or not input.category.strip():
        errors.append("Category is required")
    elif len(input.category) > 100:
        errors.append("Category too long (max 100 characters)")
    
    # Validate language
    valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    if input.language not in valid_languages:
        errors.append(f"Language {input.language} not supported")
    
    return errors


def validate_user_input(input: UserCreateInput) -> List[str]:
    """Validate user input"""
    errors = []
    
    # Validate email
    if not input.email or not input.email.strip():
        errors.append("Email is required")
    else:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, input.email):
            errors.append("Invalid email format")
        elif len(input.email) > 255:
            errors.append("Email too long (max 255 characters)")
    
    # Validate username
    if not input.username or not input.username.strip():
        errors.append("Username is required")
    else:
        username_pattern = r'^[a-zA-Z0-9_]{3,50}$'
        if not re.match(username_pattern, input.username):
            errors.append("Username must be 3-50 characters and contain only letters, numbers, and underscores")
    
    # Validate password
    if not input.password:
        errors.append("Password is required")
    else:
        if len(input.password) < 8:
            errors.append("Password must be at least 8 characters")
        elif len(input.password) > 128:
            errors.append("Password too long (max 128 characters)")
        
        # Check password strength
        if not re.search(r'[A-Z]', input.password):
            errors.append("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', input.password):
            errors.append("Password must contain at least one lowercase letter")
        if not re.search(r'\d', input.password):
            errors.append("Password must contain at least one digit")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', input.password):
            errors.append("Password must contain at least one special character")
    
    # Validate full name if provided
    if input.full_name and len(input.full_name) > 255:
        errors.append("Full name too long (max 255 characters)")
    
    # Validate organization if provided
    if input.organization and len(input.organization) > 255:
        errors.append("Organization too long (max 255 characters)")
    
    return errors