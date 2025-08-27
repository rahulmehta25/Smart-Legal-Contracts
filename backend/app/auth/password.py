"""
Password Management

OWASP-compliant password handling with:
- Strong password policies
- Secure hashing (Argon2, bcrypt)
- Password strength validation
- Common password checking
- Password history tracking
- Breach detection integration
"""

import re
import hashlib
import secrets
import string
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash
import redis
import requests
import asyncio
import aiohttp
from passlib.context import CryptContext


class PasswordStrength(Enum):
    """Password strength levels"""
    VERY_WEAK = 1
    WEAK = 2
    FAIR = 3
    GOOD = 4
    STRONG = 5


@dataclass
class PasswordPolicy:
    """Password policy configuration"""
    min_length: int = 12
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    max_consecutive_chars: int = 3
    max_repeated_chars: int = 2
    prevent_username_inclusion: bool = True
    prevent_email_inclusion: bool = True
    prevent_common_passwords: bool = True
    prevent_dictionary_words: bool = True
    prevent_keyboard_patterns: bool = True
    password_history_count: int = 5  # Prevent reusing last N passwords
    min_age_hours: int = 1  # Minimum time before password can be changed again
    max_age_days: int = 90  # Maximum password age


@dataclass
class PasswordValidationResult:
    """Password validation result"""
    is_valid: bool
    strength: PasswordStrength
    score: int  # 0-100
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    entropy: float
    time_to_crack: str


class PasswordHandler:
    """
    Production-ready password management with OWASP compliance
    """
    
    def __init__(self, 
                 policy: Optional[PasswordPolicy] = None,
                 redis_client: Optional[redis.Redis] = None,
                 enable_breach_check: bool = True,
                 breach_api_timeout: int = 5):
        """
        Initialize password handler
        
        Args:
            policy: Password policy configuration
            redis_client: Redis client for caching and rate limiting
            enable_breach_check: Enable HaveIBeenPwned API checks
            breach_api_timeout: Timeout for breach API calls
        """
        self.policy = policy or PasswordPolicy()
        self.enable_breach_check = enable_breach_check
        self.breach_api_timeout = breach_api_timeout
        
        # Initialize password context with multiple schemes
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            
            # Argon2 configuration (OWASP recommended)
            argon2__rounds=4,
            argon2__memory_cost=102400,  # 100 MB
            argon2__parallelism=8,
            argon2__hash_len=32,
            
            # Bcrypt configuration (fallback)
            bcrypt__rounds=12
        )
        
        # Argon2 hasher for direct use
        self.argon2_hasher = PasswordHasher(
            time_cost=4,      # Number of iterations
            memory_cost=102400,  # Memory usage in KiB (100MB)
            parallelism=8,    # Number of parallel threads
            hash_len=32,      # Length of hash in bytes
            salt_len=16       # Length of salt in bytes
        )
        
        # Redis for caching and rate limiting
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Common passwords list (top 10k most common passwords)
        self.common_passwords = self._load_common_passwords()
        
        # Keyboard patterns
        self.keyboard_patterns = [
            "qwerty", "asdf", "zxcv", "qazwsx", "123456", "abcdef",
            "qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890",
            "!@#$%^&*()", "qweasd", "adgjmptw"
        ]
    
    def _load_common_passwords(self) -> set:
        """Load common passwords from various sources"""
        # In production, load from a comprehensive database
        # This is a minimal set for demonstration
        return {
            "password", "123456", "123456789", "qwerty", "abc123",
            "Password", "password123", "admin", "letmein", "welcome",
            "monkey", "1234567890", "password1", "123123", "dragon",
            "master", "hello", "login", "welcome123", "admin123",
            "root", "toor", "pass", "test", "guest", "demo",
            "Password123", "Password123!", "Welcome123", "Admin123!",
            "Qwerty123", "Summer2023", "Winter2023", "Spring2023"
        }
    
    def hash_password(self, password: str, use_argon2: bool = True) -> str:
        """
        Hash password using secure algorithm
        
        Args:
            password: Plain text password
            use_argon2: Use Argon2 (preferred) or bcrypt
            
        Returns:
            Hashed password
        """
        if use_argon2:
            return self.argon2_hasher.hash(password)
        else:
            return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed_password: Stored password hash
            
        Returns:
            True if password matches
        """
        try:
            # Try Argon2 first
            if hashed_password.startswith('$argon2'):
                return self.argon2_hasher.verify(hashed_password, password)
            else:
                # Fall back to passlib context (handles bcrypt, etc.)
                return self.pwd_context.verify(password, hashed_password)
        except (VerifyMismatchError, InvalidHash):
            return False
        except Exception:
            return False
    
    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if password hash needs to be updated
        
        Args:
            hashed_password: Stored password hash
            
        Returns:
            True if hash should be updated
        """
        if hashed_password.startswith('$argon2'):
            return self.argon2_hasher.check_needs_rehash(hashed_password)
        else:
            return self.pwd_context.needs_update(hashed_password)
    
    def validate_password(self, 
                         password: str,
                         username: Optional[str] = None,
                         email: Optional[str] = None,
                         user_data: Optional[Dict[str, Any]] = None) -> PasswordValidationResult:
        """
        Comprehensive password validation
        
        Args:
            password: Password to validate
            username: User's username (to prevent inclusion)
            email: User's email (to prevent inclusion)
            user_data: Additional user data for context
            
        Returns:
            Validation result with errors and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        score = 0
        
        # Basic length checks
        if len(password) < self.policy.min_length:
            errors.append(f"Password must be at least {self.policy.min_length} characters long")
        elif len(password) >= self.policy.min_length:
            score += 20
        
        if len(password) > self.policy.max_length:
            errors.append(f"Password must not exceed {self.policy.max_length} characters")
        
        # Character type requirements
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(f'[{re.escape(self.policy.special_chars)}]', password))
        
        if self.policy.require_uppercase and not has_upper:
            errors.append("Password must contain at least one uppercase letter")
        elif has_upper:
            score += 15
        
        if self.policy.require_lowercase and not has_lower:
            errors.append("Password must contain at least one lowercase letter")
        elif has_lower:
            score += 15
        
        if self.policy.require_digits and not has_digit:
            errors.append("Password must contain at least one digit")
        elif has_digit:
            score += 15
        
        if self.policy.require_special_chars and not has_special:
            errors.append(f"Password must contain at least one special character ({self.policy.special_chars})")
        elif has_special:
            score += 15
        
        # Character diversity bonus
        char_types = sum([has_upper, has_lower, has_digit, has_special])
        if char_types >= 3:
            score += 10
        if char_types == 4:
            score += 10
        
        # Check for consecutive characters
        consecutive_count = 0
        for i in range(len(password) - 1):
            if ord(password[i]) + 1 == ord(password[i + 1]):
                consecutive_count += 1
            else:
                consecutive_count = 0
            
            if consecutive_count >= self.policy.max_consecutive_chars:
                errors.append(f"Password cannot contain more than {self.policy.max_consecutive_chars} consecutive characters")
                break
        
        # Check for repeated characters
        char_counts = {}
        for char in password:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_repeated = max(char_counts.values()) if char_counts else 0
        if max_repeated > self.policy.max_repeated_chars:
            errors.append(f"Password cannot repeat the same character more than {self.policy.max_repeated_chars} times")
        
        # Check for username inclusion
        if username and self.policy.prevent_username_inclusion:
            if username.lower() in password.lower():
                errors.append("Password cannot contain your username")
        
        # Check for email inclusion
        if email and self.policy.prevent_email_inclusion:
            email_local = email.split('@')[0].lower()
            if email_local in password.lower():
                errors.append("Password cannot contain your email address")
        
        # Check against common passwords
        if self.policy.prevent_common_passwords:
            if password.lower() in self.common_passwords:
                errors.append("Password is too common. Please choose a more unique password")
            elif password in self.common_passwords:
                errors.append("Password is too common. Please choose a more unique password")
        
        # Check for keyboard patterns
        if self.policy.prevent_keyboard_patterns:
            password_lower = password.lower()
            for pattern in self.keyboard_patterns:
                if pattern in password_lower or pattern[::-1] in password_lower:
                    warnings.append("Password contains keyboard patterns. Consider a more random password")
                    break
        
        # Calculate entropy
        entropy = self._calculate_entropy(password)
        if entropy < 30:
            warnings.append("Password has low entropy. Consider adding more random characters")
        elif entropy >= 50:
            score += 10
        
        # Estimate time to crack
        time_to_crack = self._estimate_crack_time(password, entropy)
        
        # Additional scoring based on length
        if len(password) >= 16:
            score += 10
        if len(password) >= 20:
            score += 5
        
        # Determine strength
        strength = self._calculate_strength(score, entropy, len(password))
        
        # Generate suggestions
        if not errors:
            if score < 70:
                suggestions.append("Consider making your password longer for better security")
            if entropy < 40:
                suggestions.append("Add more variety to your password characters")
            if len(password) < 16:
                suggestions.append("Passwords of 16+ characters are recommended for maximum security")
        
        return PasswordValidationResult(
            is_valid=len(errors) == 0,
            strength=strength,
            score=min(score, 100),
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            entropy=entropy,
            time_to_crack=time_to_crack
        )
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy in bits"""
        # Character set size estimation
        charset_size = 0
        
        if re.search(r'[a-z]', password):
            charset_size += 26
        if re.search(r'[A-Z]', password):
            charset_size += 26
        if re.search(r'\d', password):
            charset_size += 10
        if re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
            charset_size += 32
        
        # Basic entropy calculation
        import math
        entropy = len(password) * math.log2(charset_size) if charset_size > 0 else 0
        
        # Adjust for patterns and repetition
        unique_chars = len(set(password))
        if unique_chars < len(password):
            # Reduce entropy for repeated characters
            repetition_factor = unique_chars / len(password)
            entropy *= repetition_factor
        
        return entropy
    
    def _estimate_crack_time(self, password: str, entropy: float) -> str:
        """Estimate time to crack password"""
        # Assume 1 billion attempts per second (modern GPU)
        attempts_per_second = 1_000_000_000
        
        # Average attempts needed is half the keyspace
        total_combinations = 2 ** entropy
        average_attempts = total_combinations / 2
        
        seconds = average_attempts / attempts_per_second
        
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.1f} days"
        elif seconds < 31536000 * 1000:
            return f"{seconds/31536000:.1f} years"
        else:
            return "Effectively uncrackable"
    
    def _calculate_strength(self, score: int, entropy: float, length: int) -> PasswordStrength:
        """Calculate password strength based on multiple factors"""
        if score >= 90 and entropy >= 60 and length >= 16:
            return PasswordStrength.STRONG
        elif score >= 70 and entropy >= 50 and length >= 12:
            return PasswordStrength.GOOD
        elif score >= 50 and entropy >= 40 and length >= 10:
            return PasswordStrength.FAIR
        elif score >= 30 and entropy >= 25 and length >= 8:
            return PasswordStrength.WEAK
        else:
            return PasswordStrength.VERY_WEAK
    
    def generate_secure_password(self, 
                                length: int = 16,
                                include_uppercase: bool = True,
                                include_lowercase: bool = True,
                                include_digits: bool = True,
                                include_special: bool = True,
                                exclude_ambiguous: bool = True) -> str:
        """
        Generate cryptographically secure password
        
        Args:
            length: Password length
            include_uppercase: Include uppercase letters
            include_lowercase: Include lowercase letters  
            include_digits: Include digits
            include_special: Include special characters
            exclude_ambiguous: Exclude ambiguous characters (0, O, l, 1, etc.)
            
        Returns:
            Generated password
        """
        charset = ""
        
        if include_lowercase:
            chars = string.ascii_lowercase
            if exclude_ambiguous:
                chars = chars.replace('l', '').replace('o', '')
            charset += chars
        
        if include_uppercase:
            chars = string.ascii_uppercase
            if exclude_ambiguous:
                chars = chars.replace('I', '').replace('O', '')
            charset += chars
        
        if include_digits:
            chars = string.digits
            if exclude_ambiguous:
                chars = chars.replace('0', '').replace('1', '')
            charset += chars
        
        if include_special:
            chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            charset += chars
        
        if not charset:
            raise ValueError("At least one character type must be included")
        
        # Generate password ensuring at least one character from each required type
        password = []
        
        # Add required characters
        if include_lowercase:
            password.append(secrets.choice(string.ascii_lowercase))
        if include_uppercase:
            password.append(secrets.choice(string.ascii_uppercase))
        if include_digits:
            password.append(secrets.choice(string.digits))
        if include_special:
            password.append(secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?"))
        
        # Fill remaining length with random characters
        remaining_length = length - len(password)
        for _ in range(remaining_length):
            password.append(secrets.choice(charset))
        
        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    async def check_password_breach(self, password: str) -> Tuple[bool, int]:
        """
        Check if password has been found in data breaches using HaveIBeenPwned API
        
        Args:
            password: Password to check
            
        Returns:
            Tuple of (is_breached, occurrence_count)
        """
        if not self.enable_breach_check:
            return False, 0
        
        try:
            # Hash password with SHA-1 for HaveIBeenPwned API
            sha1_hash = hashlib.sha1(password.encode()).hexdigest().upper()
            prefix = sha1_hash[:5]
            suffix = sha1_hash[5:]
            
            # Check cache first
            cache_key = f"breach_check:{prefix}"
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                # Parse cached response
                for line in cached_result.split('\n'):
                    if line.startswith(suffix):
                        count = int(line.split(':')[1])
                        return True, count
                return False, 0
            
            # Make API request
            url = f"https://api.pwnedpasswords.com/range/{prefix}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.breach_api_timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Cache response for 24 hours
                        self.redis_client.setex(cache_key, 86400, content)
                        
                        # Check if our suffix is in the response
                        for line in content.split('\n'):
                            if line.startswith(suffix):
                                count = int(line.split(':')[1])
                                return True, count
                        
                        return False, 0
                    else:
                        # API error, fail safe
                        return False, 0
        
        except Exception:
            # Network error or API unavailable, fail safe
            return False, 0
    
    def store_password_history(self, user_id: str, password_hash: str) -> bool:
        """
        Store password in user's history to prevent reuse
        
        Args:
            user_id: User identifier
            password_hash: Hashed password
            
        Returns:
            True if stored successfully
        """
        try:
            history_key = f"password_history:{user_id}"
            
            # Add new password to history
            self.redis_client.lpush(history_key, password_hash)
            
            # Trim to keep only the last N passwords
            self.redis_client.ltrim(history_key, 0, self.policy.password_history_count - 1)
            
            # Set expiry (keep history for reasonable time)
            self.redis_client.expire(history_key, 86400 * 365)  # 1 year
            
            return True
        except Exception:
            return False
    
    def check_password_history(self, user_id: str, new_password: str) -> bool:
        """
        Check if password was used recently
        
        Args:
            user_id: User identifier
            new_password: New password to check
            
        Returns:
            True if password was used recently
        """
        try:
            history_key = f"password_history:{user_id}"
            password_hashes = self.redis_client.lrange(history_key, 0, -1)
            
            for old_hash in password_hashes:
                if self.verify_password(new_password, old_hash):
                    return True
            
            return False
        except Exception:
            return False
    
    def can_change_password(self, user_id: str) -> Tuple[bool, str]:
        """
        Check if user can change password (not too soon after last change)
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (can_change, reason_if_not)
        """
        try:
            last_change_key = f"last_password_change:{user_id}"
            last_change = self.redis_client.get(last_change_key)
            
            if not last_change:
                return True, ""
            
            last_change_time = datetime.fromisoformat(last_change)
            min_change_time = last_change_time + timedelta(hours=self.policy.min_age_hours)
            
            if datetime.utcnow() < min_change_time:
                remaining = min_change_time - datetime.utcnow()
                hours = remaining.total_seconds() / 3600
                return False, f"Password can be changed in {hours:.1f} hours"
            
            return True, ""
        except Exception:
            return True, ""
    
    def record_password_change(self, user_id: str) -> bool:
        """
        Record when user changed password
        
        Args:
            user_id: User identifier
            
        Returns:
            True if recorded successfully
        """
        try:
            change_key = f"last_password_change:{user_id}"
            self.redis_client.setex(
                change_key, 
                86400 * 365,  # 1 year
                datetime.utcnow().isoformat()
            )
            return True
        except Exception:
            return False
    
    def is_password_expired(self, user_id: str) -> bool:
        """
        Check if user's password has expired
        
        Args:
            user_id: User identifier
            
        Returns:
            True if password is expired
        """
        try:
            last_change_key = f"last_password_change:{user_id}"
            last_change = self.redis_client.get(last_change_key)
            
            if not last_change:
                # No record, assume password is old and needs change
                return True
            
            last_change_time = datetime.fromisoformat(last_change)
            expiry_time = last_change_time + timedelta(days=self.policy.max_age_days)
            
            return datetime.utcnow() >= expiry_time
        except Exception:
            return False