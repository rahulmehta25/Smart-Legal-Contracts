"""
Document Security Scanner and Sanitizer
Implements malware scanning, content sanitization, and injection prevention
OWASP Input Validation Cheat Sheet compliant
"""

import re
import hashlib
import magic
import mimetypes
import tempfile
import subprocess
import os
import json
from typing import Optional, Dict, Any, List, Tuple, BinaryIO
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import html
import urllib.parse
import base64
import zipfile
import tarfile
from pathlib import Path

import yara
import clamav
from PIL import Image
import PyPDF2
from lxml import etree
from lxml.html.clean import Cleaner
import python_magic
import requests


class ThreatType(str, Enum):
    """Types of security threats"""
    MALWARE = "malware"
    VIRUS = "virus"
    TROJAN = "trojan"
    RANSOMWARE = "ransomware"
    SPYWARE = "spyware"
    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    XXE = "xxe"  # XML External Entity
    SSRF = "ssrf"  # Server-Side Request Forgery
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    UNSAFE_CONTENT = "unsafe_content"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class ScanResult(str, Enum):
    """Scan result statuses"""
    CLEAN = "clean"
    INFECTED = "infected"
    SUSPICIOUS = "suspicious"
    QUARANTINED = "quarantined"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"


@dataclass
class ScanReport:
    """Security scan report"""
    file_name: str
    file_hash: str
    file_type: str
    file_size: int
    scan_timestamp: datetime
    result: ScanResult
    threats_found: List[Dict[str, Any]]
    sanitization_applied: List[str]
    metadata: Dict[str, Any]
    risk_score: float  # 0.0 to 1.0


class SecurityScanner:
    """
    Comprehensive security scanner for documents and content
    Features:
    - Virus/malware scanning with multiple engines
    - Content sanitization (XSS, injection prevention)
    - File type validation and verification
    - Archive inspection
    - Image metadata stripping
    - PDF security analysis
    """
    
    def __init__(self,
                 clamav_socket: Optional[str] = "/var/run/clamav/clamd.ctl",
                 yara_rules_path: Optional[str] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 quarantine_path: str = "/var/quarantine"):
        
        # Initialize ClamAV
        self.clamav_enabled = self._init_clamav(clamav_socket)
        
        # Initialize YARA rules
        self.yara_rules = self._load_yara_rules(yara_rules_path)
        
        # File size limit
        self.max_file_size = max_file_size
        
        # Quarantine directory
        self.quarantine_path = Path(quarantine_path)
        self.quarantine_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sanitizers
        self.html_cleaner = self._init_html_cleaner()
        
        # Dangerous file extensions
        self.dangerous_extensions = {
            '.exe', '.dll', '.scr', '.bat', '.cmd', '.com', '.pif',
            '.vbs', '.js', '.jar', '.zip', '.rar', '.7z',
            '.ps1', '.psm1', '.ps1xml', '.psc1', '.psd1', '.pssc',
            '.cdxml', '.ws', '.wsf', '.wsc', '.wsh', '.msh', '.msh1',
            '.msh2', '.mshxml', '.msh1xml', '.msh2xml'
        }
        
        # Safe content types
        self.safe_content_types = {
            'text/plain', 'text/csv', 'application/pdf',
            'image/jpeg', 'image/png', 'image/gif',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        
        # Initialize injection patterns
        self.injection_patterns = self._load_injection_patterns()
    
    # ========== Initialization ==========
    
    def _init_clamav(self, socket_path: str) -> bool:
        """Initialize ClamAV antivirus engine"""
        try:
            # Test ClamAV connection
            if os.path.exists(socket_path):
                # In production, use python-clamav or pyclamd
                return True
            return False
        except:
            return False
    
    def _load_yara_rules(self, rules_path: Optional[str]) -> Optional[yara.Rules]:
        """Load YARA rules for pattern-based detection"""
        
        if not rules_path:
            # Use default rules
            rules_content = '''
            rule SuspiciousStrings {
                strings:
                    $a = "cmd.exe"
                    $b = "powershell"
                    $c = "/etc/passwd"
                    $d = "DROP TABLE"
                    $e = "<script"
                    $f = "eval("
                    $g = "base64_decode"
                condition:
                    any of them
            }
            
            rule PotentialWebshell {
                strings:
                    $a = "system("
                    $b = "exec("
                    $c = "shell_exec("
                    $d = "passthru("
                    $e = "eval($_"
                condition:
                    any of them
            }
            '''
            
            return yara.compile(source=rules_content)
        
        try:
            return yara.compile(filepath=rules_path)
        except:
            return None
    
    def _init_html_cleaner(self) -> Cleaner:
        """Initialize HTML sanitizer"""
        
        return Cleaner(
            scripts=True,           # Remove <script> tags
            javascript=True,        # Remove JavaScript
            comments=True,          # Remove comments
            style=True,            # Remove style tags and attributes
            inline_style=True,     # Remove inline styles
            links=True,            # Remove links
            meta=True,             # Remove meta tags
            page_structure=False,  # Keep page structure
            processing_instructions=True,  # Remove processing instructions
            embedded=True,         # Remove embedded objects
            frames=True,           # Remove frames
            forms=True,            # Remove forms
            annoying_tags=True,    # Remove blink, marquee, etc.
            remove_unknown_tags=True,
            safe_attrs_only=True,
            add_nofollow=True
        )
    
    def _load_injection_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load patterns for detecting injection attacks"""
        
        return {
            "sql_injection": [
                re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)", re.I),
                re.compile(r"(--|#|\/\*|\*\/)", re.I),
                re.compile(r"(\bOR\b\s+\d+\s*=\s*\d+)", re.I),
                re.compile(r"(\bAND\b\s+\d+\s*=\s*\d+)", re.I),
                re.compile(r"(';|\";\s*(SELECT|INSERT|UPDATE|DELETE|DROP))", re.I),
                re.compile(r"(\bEXEC\b|\bEXECUTE\b|\bDECLARE\b)", re.I)
            ],
            "xss": [
                re.compile(r"<script[^>]*>.*?</script>", re.I | re.S),
                re.compile(r"javascript:\s*", re.I),
                re.compile(r"on\w+\s*=\s*[\"']", re.I),
                re.compile(r"<iframe[^>]*>", re.I),
                re.compile(r"<embed[^>]*>", re.I),
                re.compile(r"<object[^>]*>", re.I),
                re.compile(r"eval\s*\(", re.I),
                re.compile(r"expression\s*\(", re.I)
            ],
            "xxe": [
                re.compile(r"<!DOCTYPE[^>]*\[", re.I),
                re.compile(r"<!ENTITY", re.I),
                re.compile(r"SYSTEM\s+[\"']file:", re.I),
                re.compile(r"SYSTEM\s+[\"']http:", re.I)
            ],
            "path_traversal": [
                re.compile(r"\.\.\/|\.\.\\"),
                re.compile(r"%2e%2e%2f|%2e%2e\/", re.I),
                re.compile(r"\.\.%c0%af|\.\.%c1%9c", re.I)
            ],
            "command_injection": [
                re.compile(r"[;&|]\s*\w+"),
                re.compile(r"`[^`]*`"),
                re.compile(r"\$\([^)]*\)"),
                re.compile(r"\bsystem\b|\bexec\b|\beval\b", re.I)
            ],
            "ldap_injection": [
                re.compile(r"\(\s*\|\s*\(", re.I),
                re.compile(r"\(\s*&\s*\(", re.I),
                re.compile(r"\*\s*\)", re.I)
            ]
        }
    
    # ========== Main Scanning Functions ==========
    
    def scan_file(self, file_path: str) -> ScanReport:
        """
        Comprehensive file security scan
        """
        
        file_path = Path(file_path)
        
        # Basic file validation
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            return self._create_report(
                file_path.name,
                "",
                "unknown",
                file_size,
                ScanResult.BLOCKED,
                [{"type": "size_limit", "message": "File exceeds maximum size limit"}]
            )
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Detect file type
        file_type = self._detect_file_type(file_path)
        
        threats = []
        sanitizations = []
        
        # Check file extension
        if file_path.suffix.lower() in self.dangerous_extensions:
            threats.append({
                "type": ThreatType.UNSAFE_CONTENT,
                "severity": "high",
                "message": f"Dangerous file extension: {file_path.suffix}"
            })
        
        # Virus/malware scan
        if self.clamav_enabled:
            malware_result = self._scan_with_clamav(file_path)
            if malware_result:
                threats.extend(malware_result)
        
        # YARA pattern matching
        if self.yara_rules:
            yara_result = self._scan_with_yara(file_path)
            if yara_result:
                threats.extend(yara_result)
        
        # Content-specific scanning
        if file_type.startswith('text/'):
            content_threats = self._scan_text_content(file_path)
            threats.extend(content_threats)
        elif file_type == 'application/pdf':
            pdf_threats = self._scan_pdf(file_path)
            threats.extend(pdf_threats)
        elif file_type.startswith('image/'):
            image_result = self._scan_image(file_path)
            if image_result:
                sanitizations.append("image_metadata_stripped")
        elif file_type in ['application/zip', 'application/x-tar', 'application/x-gzip']:
            archive_threats = self._scan_archive(file_path)
            threats.extend(archive_threats)
        
        # Determine scan result
        if threats:
            if any(t.get('severity') == 'critical' for t in threats):
                result = ScanResult.INFECTED
            else:
                result = ScanResult.SUSPICIOUS
        else:
            result = ScanResult.CLEAN
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(threats)
        
        return ScanReport(
            file_name=file_path.name,
            file_hash=file_hash,
            file_type=file_type,
            file_size=file_size,
            scan_timestamp=datetime.utcnow(),
            result=result,
            threats_found=threats,
            sanitization_applied=sanitizations,
            metadata={"scanner_version": "1.0.0"},
            risk_score=risk_score
        )
    
    # ========== Malware Scanning ==========
    
    def _scan_with_clamav(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan file with ClamAV antivirus"""
        
        threats = []
        
        try:
            # In production, use python-clamav or pyclamd
            # This is a simulation
            result = subprocess.run(
                ['clamscan', '--no-summary', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if "FOUND" in result.stdout:
                threats.append({
                    "type": ThreatType.MALWARE,
                    "severity": "critical",
                    "engine": "clamav",
                    "message": result.stdout.strip()
                })
        except:
            pass
        
        return threats
    
    def _scan_with_yara(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan file with YARA rules"""
        
        threats = []
        
        try:
            matches = self.yara_rules.match(str(file_path))
            
            for match in matches:
                threats.append({
                    "type": ThreatType.SUSPICIOUS_PATTERN,
                    "severity": "medium",
                    "engine": "yara",
                    "rule": match.rule,
                    "strings": [str(s) for s in match.strings[:5]]  # Limit strings
                })
        except:
            pass
        
        return threats
    
    # ========== Content Scanning ==========
    
    def _scan_text_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan text content for injection attacks"""
        
        threats = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024 * 1024)  # Read first 1MB
            
            # Check for injection patterns
            for attack_type, patterns in self.injection_patterns.items():
                for pattern in patterns:
                    if pattern.search(content):
                        threats.append({
                            "type": getattr(ThreatType, attack_type.upper()),
                            "severity": "high",
                            "message": f"Potential {attack_type.replace('_', ' ')} detected"
                        })
                        break
        except:
            pass
        
        return threats
    
    def _scan_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan PDF for malicious content"""
        
        threats = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                
                # Check for JavaScript
                if '/JavaScript' in str(pdf.trailer):
                    threats.append({
                        "type": ThreatType.SUSPICIOUS_PATTERN,
                        "severity": "high",
                        "message": "PDF contains JavaScript"
                    })
                
                # Check for embedded files
                if '/EmbeddedFiles' in str(pdf.trailer):
                    threats.append({
                        "type": ThreatType.SUSPICIOUS_PATTERN,
                        "severity": "medium",
                        "message": "PDF contains embedded files"
                    })
                
                # Check for forms
                if '/AcroForm' in str(pdf.trailer):
                    threats.append({
                        "type": ThreatType.SUSPICIOUS_PATTERN,
                        "severity": "low",
                        "message": "PDF contains forms"
                    })
        except:
            pass
        
        return threats
    
    def _scan_image(self, file_path: Path) -> bool:
        """Scan and sanitize image files"""
        
        try:
            # Open and re-save image to strip metadata
            with Image.open(file_path) as img:
                # Remove EXIF data
                data = list(img.getdata())
                image_without_exif = Image.new(img.mode, img.size)
                image_without_exif.putdata(data)
                
                # Save sanitized version
                temp_path = file_path.with_suffix('.tmp')
                image_without_exif.save(temp_path, format=img.format)
                
                # Replace original with sanitized
                temp_path.replace(file_path)
                
                return True
        except:
            return False
    
    def _scan_archive(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan archive files for malicious content"""
        
        threats = []
        
        try:
            # Check for zip bombs
            if file_path.suffix in ['.zip', '.jar']:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    total_size = sum(info.file_size for info in zf.infolist())
                    compressed_size = file_path.stat().st_size
                    
                    if total_size / compressed_size > 100:
                        threats.append({
                            "type": ThreatType.SUSPICIOUS_PATTERN,
                            "severity": "high",
                            "message": "Potential zip bomb detected"
                        })
                    
                    # Check for dangerous files in archive
                    for info in zf.infolist():
                        if Path(info.filename).suffix.lower() in self.dangerous_extensions:
                            threats.append({
                                "type": ThreatType.UNSAFE_CONTENT,
                                "severity": "medium",
                                "message": f"Archive contains dangerous file: {info.filename}"
                            })
        except:
            pass
        
        return threats
    
    # ========== Content Sanitization ==========
    
    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content to prevent XSS"""
        
        # Parse and clean HTML
        try:
            tree = etree.HTML(html_content)
            cleaned = self.html_cleaner.clean_html(tree)
            return etree.tostring(cleaned, encoding='unicode')
        except:
            # If parsing fails, escape everything
            return html.escape(html_content)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove special characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        return name + ext
    
    def sanitize_sql_input(self, input_string: str) -> str:
        """Sanitize input to prevent SQL injection"""
        
        # Use parameterized queries instead of this in production
        # This is for additional defense in depth
        
        # Remove SQL meta-characters
        sanitized = re.sub(r'[;\'\"\\-]', '', input_string)
        
        # Remove SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 
                       'UNION', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC']
        
        for keyword in sql_keywords:
            sanitized = re.sub(rf'\b{keyword}\b', '', sanitized, flags=re.I)
        
        return sanitized
    
    def sanitize_json(self, json_string: str) -> Optional[str]:
        """Sanitize JSON to prevent injection"""
        
        try:
            # Parse and re-serialize to remove any injected code
            data = json.loads(json_string)
            
            # Recursively sanitize strings in the data
            def sanitize_value(value):
                if isinstance(value, str):
                    # Remove potential script tags and escape special characters
                    return html.escape(value)
                elif isinstance(value, dict):
                    return {k: sanitize_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [sanitize_value(item) for item in value]
                else:
                    return value
            
            sanitized_data = sanitize_value(data)
            return json.dumps(sanitized_data)
        except:
            return None
    
    def sanitize_xml(self, xml_content: str) -> Optional[str]:
        """Sanitize XML to prevent XXE attacks"""
        
        try:
            # Parse with secure settings
            parser = etree.XMLParser(
                no_network=True,
                dtd_validation=False,
                load_dtd=False,
                resolve_entities=False,
                huge_tree=False
            )
            
            tree = etree.fromstring(xml_content.encode(), parser)
            
            # Remove dangerous elements
            for element in tree.xpath('//*[local-name()="ENTITY"]'):
                element.getparent().remove(element)
            
            return etree.tostring(tree, encoding='unicode')
        except:
            return None
    
    # ========== Content Security Policy ==========
    
    @staticmethod
    def get_csp_headers() -> Dict[str, str]:
        """Get Content Security Policy headers"""
        
        csp = {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline' 'unsafe-eval'",  # Restrict in production
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' data:",
            "connect-src": "'self'",
            "media-src": "'self'",
            "object-src": "'none'",
            "frame-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "frame-ancestors": "'none'",
            "upgrade-insecure-requests": "",
            "block-all-mixed-content": ""
        }
        
        csp_string = "; ".join(f"{key} {value}" for key, value in csp.items() if value)
        
        return {
            "Content-Security-Policy": csp_string,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    # ========== Quarantine Management ==========
    
    def quarantine_file(self, file_path: Path, threat_info: Dict[str, Any]) -> str:
        """Move infected file to quarantine"""
        
        # Generate quarantine filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_hash = self._calculate_file_hash(file_path)
        quarantine_name = f"{timestamp}_{file_hash}_{file_path.name}"
        quarantine_path = self.quarantine_path / quarantine_name
        
        # Move file to quarantine
        file_path.rename(quarantine_path)
        
        # Create metadata file
        metadata = {
            "original_path": str(file_path),
            "quarantine_date": datetime.utcnow().isoformat(),
            "threat_info": threat_info,
            "file_hash": file_hash
        }
        
        metadata_path = quarantine_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(quarantine_path)
    
    # ========== Utility Functions ==========
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect actual file type using magic bytes"""
        
        try:
            mime = magic.from_file(str(file_path), mime=True)
            return mime
        except:
            # Fallback to extension-based detection
            return mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
    
    def _calculate_risk_score(self, threats: List[Dict[str, Any]]) -> float:
        """Calculate risk score based on threats found"""
        
        if not threats:
            return 0.0
        
        severity_scores = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }
        
        max_score = 0.0
        for threat in threats:
            score = severity_scores.get(threat.get('severity', 'low'), 0.2)
            max_score = max(max_score, score)
        
        return max_score
    
    def _create_report(self,
                      file_name: str,
                      file_hash: str,
                      file_type: str,
                      file_size: int,
                      result: ScanResult,
                      threats: List[Dict[str, Any]]) -> ScanReport:
        """Create scan report"""
        
        return ScanReport(
            file_name=file_name,
            file_hash=file_hash,
            file_type=file_type,
            file_size=file_size,
            scan_timestamp=datetime.utcnow(),
            result=result,
            threats_found=threats,
            sanitization_applied=[],
            metadata={},
            risk_score=self._calculate_risk_score(threats)
        )


# ========== FastAPI Integration ==========

from fastapi import UploadFile, HTTPException, status

async def scan_uploaded_file(file: UploadFile, scanner: SecurityScanner) -> ScanReport:
    """Scan uploaded file for security threats"""
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        # Scan file
        report = scanner.scan_file(str(tmp_path))
        
        # Handle based on result
        if report.result == ScanResult.INFECTED:
            # Quarantine file
            scanner.quarantine_file(tmp_path, report.threats_found)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File infected: {report.threats_found}"
            )
        elif report.result == ScanResult.SUSPICIOUS:
            # Log and potentially allow with warning
            pass
        
        return report
        
    finally:
        # Clean up temp file if not quarantined
        if tmp_path.exists():
            tmp_path.unlink()