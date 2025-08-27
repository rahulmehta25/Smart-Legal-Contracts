"""
Input Validation and Sanitization System
Implements comprehensive input validation following OWASP guidelines
Prevents injection attacks and ensures data integrity
"""

import re
import html
import urllib.parse
import mimetypes
import hashlib
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import ipaddress
import phonenumbers
import email_validator
from pathlib import Path
import magic
import yara
import clamav

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputType(str, Enum):
    """Types of input for validation"""
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    USERNAME = "username"
    PASSWORD = "password"
    FILENAME = "filename"
    SQL = "sql"
    HTML = "html"
    JAVASCRIPT = "javascript"
    JSON = "json"
    XML = "xml"
    PATH = "path"
    IP_ADDRESS = "ip"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    UUID = "uuid"
    
class FileType(str, Enum):
    """Allowed file types for upload"""
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    TXT = "text/plain"
    RTF = "application/rtf"
    ODT = "application/vnd.oasis.opendocument.text"

class InputValidator:
    """
    Comprehensive input validation system with:
    - Type-specific validation
    - SQL injection prevention
    - XSS prevention
    - Path traversal prevention
    - File upload security
    - Data sanitization
    """
    
    def __init__(self):
        # Regex patterns for validation
        self.patterns = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "username": re.compile(r'^[a-zA-Z0-9_-]{3,32}$'),
            "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            "safe_string": re.compile(r'^[a-zA-Z0-9\s\-_.,!?@#$%^&*()\[\]{}+=:;"\'<>/\\|`~]*$'),
            "alphanumeric": re.compile(r'^[a-zA-Z0-9]+$'),
            "numeric": re.compile(r'^\d+$'),
            "phone_us": re.compile(r'^(\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$')
        }
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)",
            r"(\'|\"|;|\\x00|\\n|\\r|\\x1a)",
            r"(\bSLEEP\b\s*\(\s*\d+\s*\))",
            r"(\bBENCHMARK\b\s*\()",
            r"(\bWAITFOR\b\s+\bDELAY\b)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"<applet[^>]*>.*?</applet>",
            r"<form[^>]*>.*?</form>",
            r"<input[^>]*>",
            r"eval\s*\(",
            r"expression\s*\(",
            r"vbscript:",
            r"data:text/html",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%252e%252e%252f",
            r"\.\.%c0%af",
            r"\.\.%c1%9c",
            r"/etc/passwd",
            r"C:\\Windows",
            r"..\\",
        ]
        
        # Initialize file type checker
        try:
            self.file_magic = magic.Magic(mime=True)
        except:
            self.file_magic = None
            
        # Initialize antivirus (ClamAV)
        try:
            self.clam_daemon = clamav.ClamdUnixSocket()
        except:
            self.clam_daemon = None
            
        # Initialize YARA rules for malware detection
        self.yara_rules = None
        self._load_yara_rules()
    
    def _load_yara_rules(self):
        """Load YARA rules for malware detection"""
        try:
            # Would load from file in production
            rules = '''
            rule suspicious_pdf {
                strings:
                    $a = "/JS"
                    $b = "/JavaScript"
                    $c = "/Launch"
                    $d = "/EmbeddedFile"
                condition:
                    uint32(0) == 0x46445025 and any of them
            }
            '''
            self.yara_rules = yara.compile(source=rules)
        except:
            pass
    
    # ========== Core Validation Methods ==========
    
    def validate_input(self, 
                      value: Any, 
                      input_type: InputType,
                      max_length: Optional[int] = None,
                      min_length: Optional[int] = None,
                      required: bool = True) -> Any:
        """
        Validate input based on type
        Returns sanitized value or raises ValidationError
        """
        
        # Check if required
        if required and not value:
            raise ValidationError("Input is required")
        
        if not value:
            return value
            
        # Convert to string for validation
        value_str = str(value)
        
        # Check length constraints
        if max_length and len(value_str) > max_length:
            raise ValidationError(f"Input exceeds maximum length of {max_length}")
        
        if min_length and len(value_str) < min_length:
            raise ValidationError(f"Input must be at least {min_length} characters")
        
        # Type-specific validation
        if input_type == InputType.EMAIL:
            return self.validate_email(value_str)
        elif input_type == InputType.URL:
            return self.validate_url(value_str)
        elif input_type == InputType.PHONE:
            return self.validate_phone(value_str)
        elif input_type == InputType.USERNAME:
            return self.validate_username(value_str)
        elif input_type == InputType.PASSWORD:
            return self.validate_password(value_str)
        elif input_type == InputType.FILENAME:
            return self.validate_filename(value_str)
        elif input_type == InputType.PATH:
            return self.validate_path(value_str)
        elif input_type == InputType.IP_ADDRESS:
            return self.validate_ip_address(value_str)
        elif input_type == InputType.UUID:
            return self.validate_uuid(value_str)
        elif input_type == InputType.SQL:
            return self.validate_sql_safe(value_str)
        elif input_type == InputType.HTML:
            return self.sanitize_html(value_str)
        elif input_type == InputType.JAVASCRIPT:
            return self.validate_javascript_safe(value_str)
        else:
            return self.sanitize_string(value_str)
    
    # ========== Specific Validators ==========
    
    def validate_email(self, email: str) -> str:
        """Validate email address"""
        try:
            validated = email_validator.validate_email(email)
            return validated.email
        except email_validator.EmailNotValidError as e:
            raise ValidationError(f"Invalid email: {str(e)}")
    
    def validate_url(self, url: str) -> str:
        """Validate and sanitize URL"""
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        if any(url.lower().startswith(proto) for proto in dangerous_protocols):
            raise ValidationError("Dangerous URL protocol")
        
        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError("Invalid URL format")
            
            # Ensure http/https only
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError("Only HTTP/HTTPS URLs allowed")
            
            # Reconstruct clean URL
            clean_url = urllib.parse.urlunparse(parsed)
            return clean_url
        except Exception as e:
            raise ValidationError(f"Invalid URL: {str(e)}")
    
    def validate_phone(self, phone: str, region: str = "US") -> str:
        """Validate phone number"""
        try:
            parsed = phonenumbers.parse(phone, region)
            if not phonenumbers.is_valid_number(parsed):
                raise ValidationError("Invalid phone number")
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except Exception as e:
            raise ValidationError(f"Invalid phone number: {str(e)}")
    
    def validate_username(self, username: str) -> str:
        """Validate username"""
        if not self.patterns["username"].match(username):
            raise ValidationError("Username must be 3-32 characters, alphanumeric with - and _")
        
        # Check for reserved names
        reserved = ['admin', 'root', 'administrator', 'system', 'null', 'undefined']
        if username.lower() in reserved:
            raise ValidationError("Username is reserved")
        
        return username
    
    def validate_password(self, password: str) -> str:
        """Validate password strength"""
        if len(password) < 12:
            raise ValidationError("Password must be at least 12 characters")
        
        # Check complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        if not all([has_upper, has_lower, has_digit, has_special]):
            raise ValidationError("Password must contain uppercase, lowercase, digit, and special character")
        
        # Check for common patterns
        common_patterns = ['password', '12345', 'qwerty', 'admin', 'letmein']
        if any(pattern in password.lower() for pattern in common_patterns):
            raise ValidationError("Password contains common pattern")
        
        return password
    
    def validate_filename(self, filename: str) -> str:
        """Validate and sanitize filename"""
        # Remove path components
        filename = Path(filename).name
        
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValidationError("Invalid filename")
        
        # Sanitize special characters
        valid_chars = re.compile(r'[^a-zA-Z0-9._-]')
        clean_filename = valid_chars.sub('_', filename)
        
        # Limit length
        if len(clean_filename) > 255:
            name, ext = clean_filename.rsplit('.', 1) if '.' in clean_filename else (clean_filename, '')
            clean_filename = f"{name[:240]}.{ext}" if ext else name[:255]
        
        return clean_filename
    
    def validate_path(self, path_str: str) -> str:
        """Validate file path and prevent traversal"""
        # Check for path traversal patterns
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path_str, re.IGNORECASE):
                raise ValidationError("Path traversal attempt detected")
        
        # Normalize path
        try:
            path = Path(path_str).resolve()
            
            # Ensure path doesn't escape allowed directory
            # In production, compare against allowed base paths
            return str(path)
        except Exception as e:
            raise ValidationError(f"Invalid path: {str(e)}")
    
    def validate_ip_address(self, ip: str) -> str:
        """Validate IP address"""
        try:
            # Parse IP address
            ip_obj = ipaddress.ip_address(ip)
            
            # Check for private/reserved IPs if needed
            if ip_obj.is_private:
                pass  # Allow or deny based on requirements
            
            return str(ip_obj)
        except ValueError:
            raise ValidationError("Invalid IP address")
    
    def validate_uuid(self, uuid_str: str) -> str:
        """Validate UUID format"""
        if not self.patterns["uuid"].match(uuid_str.lower()):
            raise ValidationError("Invalid UUID format")
        return uuid_str.lower()
    
    # ========== SQL Injection Prevention ==========
    
    def validate_sql_safe(self, value: str) -> str:
        """Check for SQL injection attempts"""
        value_upper = value.upper()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value_upper):
                raise ValidationError("Potential SQL injection detected")
        
        # Additional checks for encoded attempts
        decoded = urllib.parse.unquote(value)
        if decoded != value:
            # Check decoded version
            return self.validate_sql_safe(decoded)
        
        return self.sanitize_string(value)
    
    def escape_sql_identifier(self, identifier: str) -> str:
        """Escape SQL identifier (table/column name)"""
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', identifier):
            raise ValidationError("Invalid SQL identifier")
        return identifier
    
    # ========== XSS Prevention ==========
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML to prevent XSS"""
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                raise ValidationError("Potential XSS detected")
        
        # HTML escape
        return html.escape(html_content)
    
    def validate_javascript_safe(self, js_content: str) -> str:
        """Validate JavaScript content for safety"""
        dangerous_functions = ['eval', 'Function', 'setTimeout', 'setInterval', 'execScript']
        
        for func in dangerous_functions:
            if func in js_content:
                raise ValidationError(f"Dangerous JavaScript function: {func}")
        
        return js_content
    
    # ========== File Upload Security ==========
    
    def validate_file_upload(self,
                            file_content: bytes,
                            filename: str,
                            allowed_types: List[FileType],
                            max_size: int = 10485760) -> Dict[str, Any]:
        """
        Validate uploaded file for security
        Returns: dict with validated filename, content type, and scan results
        """
        
        # Validate filename
        clean_filename = self.validate_filename(filename)
        
        # Check file size
        if len(file_content) > max_size:
            raise ValidationError(f"File exceeds maximum size of {max_size} bytes")
        
        # Check file type using magic bytes
        if self.file_magic:
            detected_type = self.file_magic.from_buffer(file_content)
            
            # Verify against allowed types
            allowed_mimes = [ft.value for ft in allowed_types]
            if detected_type not in allowed_mimes:
                raise ValidationError(f"File type {detected_type} not allowed")
        
        # Scan for malware
        scan_result = self.scan_file_for_malware(file_content)
        if not scan_result['safe']:
            raise ValidationError(f"File contains malware: {scan_result['threat']}")
        
        # Check with YARA rules
        if self.yara_rules:
            matches = self.yara_rules.match(data=file_content)
            if matches:
                raise ValidationError(f"File matches suspicious pattern: {matches[0].rule}")
        
        return {
            'filename': clean_filename,
            'content_type': detected_type if self.file_magic else 'application/octet-stream',
            'size': len(file_content),
            'scan_result': scan_result
        }
    
    def scan_file_for_malware(self, file_content: bytes) -> Dict[str, Any]:
        """Scan file for malware using ClamAV"""
        if self.clam_daemon:
            try:
                result = self.clam_daemon.instream(file_content)
                if result['stream'][0] == 'OK':
                    return {'safe': True, 'threat': None}
                else:
                    return {'safe': False, 'threat': result['stream'][1]}
            except:
                # If scan fails, be conservative
                return {'safe': False, 'threat': 'Scan failed'}
        
        # If no scanner available, return uncertain
        return {'safe': True, 'threat': None, 'warning': 'No malware scanner available'}
    
    # ========== General Sanitization ==========
    
    def sanitize_string(self, value: str) -> str:
        """General string sanitization"""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit consecutive whitespace
        value = re.sub(r'\s+', ' ', value)
        
        return value.strip()
    
    def sanitize_json(self, json_str: str) -> str:
        """Sanitize JSON string"""
        # Remove comments (not valid JSON)
        json_str = re.sub(r'//.*?\n', '', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def sanitize_for_logging(self, value: str) -> str:
        """Sanitize value for safe logging"""
        # Remove line breaks to prevent log injection
        value = value.replace('\n', '\\n').replace('\r', '\\r')
        
        # Limit length
        if len(value) > 1000:
            value = value[:997] + '...'
        
        return value
    
    # ========== Validation Decorators ==========
    
    @staticmethod
    def validate(**validators):
        """Decorator for input validation"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                validator = InputValidator()
                
                # Validate each parameter
                for param, rules in validators.items():
                    if param in kwargs:
                        try:
                            kwargs[param] = validator.validate_input(
                                kwargs[param],
                                rules.get('type', InputType.SQL),
                                rules.get('max_length'),
                                rules.get('min_length'),
                                rules.get('required', True)
                            )
                        except ValidationError as e:
                            raise ValidationError(f"Parameter '{param}': {str(e)}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


# ========== Utility Functions ==========

def validate_credit_card(number: str) -> bool:
    """Validate credit card number using Luhn algorithm"""
    number = re.sub(r'\D', '', number)
    
    if not number or len(number) < 13 or len(number) > 19:
        return False
    
    # Luhn algorithm
    total = 0
    for i, digit in enumerate(reversed(number)):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    
    return total % 10 == 0

def validate_ssn(ssn: str) -> bool:
    """Validate US Social Security Number format"""
    # Remove formatting
    ssn = re.sub(r'\D', '', ssn)
    
    if len(ssn) != 9:
        return False
    
    # Check for invalid patterns
    if ssn[:3] == '000' or ssn[:3] == '666' or ssn[:3] >= '900':
        return False
    if ssn[3:5] == '00':
        return False
    if ssn[5:] == '0000':
        return False
    
    return True

def mask_sensitive_data(value: str, data_type: str = 'generic') -> str:
    """Mask sensitive data for display"""
    if data_type == 'email':
        parts = value.split('@')
        if len(parts) == 2:
            name = parts[0]
            if len(name) > 2:
                masked = name[0] + '*' * (len(name) - 2) + name[-1]
            else:
                masked = '*' * len(name)
            return f"{masked}@{parts[1]}"
    
    elif data_type == 'phone':
        # Keep last 4 digits
        if len(value) >= 4:
            return '*' * (len(value) - 4) + value[-4:]
    
    elif data_type == 'ssn':
        # Show last 4 digits
        if len(value) == 9:
            return f"***-**-{value[-4:]}"
    
    elif data_type == 'credit_card':
        # Show last 4 digits
        if len(value) >= 12:
            return '*' * (len(value) - 4) + value[-4:]
    
    # Generic masking
    if len(value) > 4:
        return value[:2] + '*' * (len(value) - 4) + value[-2:]
    else:
        return '*' * len(value)