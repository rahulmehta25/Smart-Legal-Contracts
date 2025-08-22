"""
Enterprise Audit Logging and Compliance System
Implements comprehensive audit trails, SIEM integration, and compliance reporting
SOC2, GDPR, HIPAA compliant
"""

import json
import hashlib
import uuid
import time
import traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import queue
import gzip
import re
from functools import wraps

import redis
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import requests


Base = declarative_base()


class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILED = "auth.login.failed"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"
    SESSION_CREATED = "auth.session.created"
    SESSION_EXPIRED = "auth.session.expired"
    
    # Authorization events
    PERMISSION_GRANTED = "authz.permission.granted"
    PERMISSION_DENIED = "authz.permission.denied"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REVOKED = "authz.role.revoked"
    
    # Data access events
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    DATA_IMPORTED = "data.imported"
    
    # Document events
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_DOWNLOADED = "document.downloaded"
    DOCUMENT_ANALYZED = "document.analyzed"
    DOCUMENT_ENCRYPTED = "document.encrypted"
    DOCUMENT_DECRYPTED = "document.decrypted"
    
    # API events
    API_KEY_CREATED = "api.key.created"
    API_KEY_REVOKED = "api.key.revoked"
    API_CALL = "api.call"
    API_RATE_LIMITED = "api.rate_limited"
    
    # Security events
    SECURITY_ALERT = "security.alert"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BRUTE_FORCE_DETECTED = "security.brute_force"
    SQL_INJECTION_ATTEMPT = "security.sql_injection"
    XSS_ATTEMPT = "security.xss"
    MALWARE_DETECTED = "security.malware"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGED = "system.config.changed"
    BACKUP_CREATED = "system.backup.created"
    BACKUP_RESTORED = "system.backup.restored"
    
    # Compliance events
    COMPLIANCE_VIOLATION = "compliance.violation"
    DATA_RETENTION_EXPIRED = "compliance.retention.expired"
    GDPR_REQUEST = "compliance.gdpr.request"
    AUDIT_EXPORT = "compliance.audit.export"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    CRITICAL = "critical"  # Security breaches, data loss
    HIGH = "high"         # Failed auth, permission denied
    MEDIUM = "medium"     # Configuration changes
    LOW = "low"          # Normal operations
    INFO = "info"        # Informational events


class ComplianceStandard(str, Enum):
    """Compliance standards"""
    SOC2 = "SOC2"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    ISO_27001 = "ISO_27001"
    CCPA = "CCPA"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]  # e.g., "document", "user", "analysis"
    resource_id: Optional[str]
    action: str
    result: str  # "success" or "failure"
    details: Dict[str, Any]
    session_id: Optional[str]
    correlation_id: Optional[str]  # For tracking related events
    metadata: Dict[str, Any]


class AuditLog(Base):
    """Database model for audit logs"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    user_id = Column(String(64), index=True)
    username = Column(String(100), index=True)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    resource_type = Column(String(50), index=True)
    resource_id = Column(String(100), index=True)
    action = Column(String(100))
    result = Column(String(20))
    details = Column(Text)  # JSON
    session_id = Column(String(64), index=True)
    correlation_id = Column(String(64), index=True)
    metadata = Column(Text)  # JSON
    checksum = Column(String(64))  # For tamper detection
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_resource', 'resource_type', 'resource_id'),
        Index('idx_severity_timestamp', 'severity', 'timestamp'),
    )


class AuditManager:
    """
    Comprehensive audit logging manager with:
    - Tamper-proof audit trails
    - Real-time SIEM integration
    - Suspicious activity detection
    - Compliance reporting
    - Log retention and archival
    """
    
    def __init__(self,
                 database_url: str,
                 redis_client: Optional[redis.Redis] = None,
                 siem_endpoints: Optional[List[str]] = None,
                 retention_days: int = 2555):  # 7 years default
        
        # Database setup
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Redis for real-time processing
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # SIEM integration endpoints
        self.siem_endpoints = siem_endpoints or []
        
        # Retention policy
        self.retention_days = retention_days
        
        # Async queue for batch processing
        self.event_queue = queue.Queue(maxsize=10000)
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Suspicious activity patterns
        self.suspicious_patterns = self._load_suspicious_patterns()
        
        # Metrics for monitoring
        self.metrics = {
            "events_logged": 0,
            "events_dropped": 0,
            "siem_failures": 0,
            "suspicious_detected": 0
        }
    
    # ========== Core Logging ==========
    
    def log_event(self, event: AuditEvent) -> str:
        """Log an audit event"""
        
        # Generate event ID if not provided
        if not event.event_id:
            event.event_id = str(uuid.uuid4())
        
        # Add timestamp if not provided
        if not event.timestamp:
            event.timestamp = datetime.utcnow()
        
        # Calculate checksum for tamper detection
        checksum = self._calculate_checksum(event)
        
        # Queue for async processing
        try:
            self.event_queue.put_nowait({
                "event": event,
                "checksum": checksum
            })
            self.metrics["events_logged"] += 1
        except queue.Full:
            # Log to emergency overflow file if queue is full
            self._emergency_log(event)
            self.metrics["events_dropped"] += 1
        
        # Check for suspicious activity
        if self._is_suspicious(event):
            self._handle_suspicious_activity(event)
        
        return event.event_id
    
    def _process_events(self):
        """Background thread for processing audit events"""
        
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Get event from queue with timeout
                item = self.event_queue.get(timeout=1)
                batch.append(item)
                
                # Flush batch if size or time threshold reached
                if len(batch) >= 100 or (time.time() - last_flush) > 5:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except queue.Empty:
                # Flush any remaining events
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
    
    def _flush_batch(self, batch: List[Dict]):
        """Flush batch of events to storage and SIEM"""
        
        session = self.SessionLocal()
        try:
            for item in batch:
                event = item["event"]
                checksum = item["checksum"]
                
                # Store in database
                audit_log = AuditLog(
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    user_id=event.user_id,
                    username=event.username,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    action=event.action,
                    result=event.result,
                    details=json.dumps(event.details),
                    session_id=event.session_id,
                    correlation_id=event.correlation_id,
                    metadata=json.dumps(event.metadata),
                    checksum=checksum
                )
                session.add(audit_log)
            
            session.commit()
            
            # Send to SIEM
            self._send_to_siem(batch)
            
        except Exception as e:
            session.rollback()
            self._emergency_log_batch(batch, str(e))
        finally:
            session.close()
    
    # ========== Tamper Detection ==========
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate cryptographic checksum for tamper detection"""
        
        # Create canonical representation
        canonical = json.dumps({
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "resource_id": event.resource_id,
            "action": event.action,
            "result": event.result
        }, sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def verify_integrity(self, event_id: str) -> bool:
        """Verify audit log integrity"""
        
        session = self.SessionLocal()
        try:
            log = session.query(AuditLog).filter_by(event_id=event_id).first()
            if not log:
                return False
            
            # Reconstruct event
            event = AuditEvent(
                event_id=log.event_id,
                timestamp=log.timestamp,
                event_type=AuditEventType(log.event_type),
                severity=AuditSeverity(log.severity),
                user_id=log.user_id,
                username=log.username,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                action=log.action,
                result=log.result,
                details=json.loads(log.details) if log.details else {},
                session_id=log.session_id,
                correlation_id=log.correlation_id,
                metadata=json.loads(log.metadata) if log.metadata else {}
            )
            
            # Verify checksum
            calculated = self._calculate_checksum(event)
            return calculated == log.checksum
            
        finally:
            session.close()
    
    # ========== Suspicious Activity Detection ==========
    
    def _load_suspicious_patterns(self) -> Dict[str, Any]:
        """Load patterns for detecting suspicious activity"""
        
        return {
            "failed_login_threshold": 5,
            "failed_login_window": 300,  # 5 minutes
            "rapid_api_calls": 100,      # per minute
            "unusual_hours": (0, 6),     # 12 AM - 6 AM
            "sql_patterns": [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
                r"(--|#|\/\*|\*\/)",
                r"(\bOR\b.*=.*)",
                r"(\bAND\b.*=.*)"
            ],
            "xss_patterns": [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>"
            ],
            "path_traversal_patterns": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e/",
                r"%252e%252e"
            ]
        }
    
    def _is_suspicious(self, event: AuditEvent) -> bool:
        """Check if event indicates suspicious activity"""
        
        # Check for security-related events
        if event.event_type in [
            AuditEventType.LOGIN_FAILED,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.API_RATE_LIMITED
        ]:
            return self._check_threshold_breach(event)
        
        # Check for SQL injection attempts
        if event.details:
            details_str = json.dumps(event.details)
            for pattern in self.suspicious_patterns["sql_patterns"]:
                if re.search(pattern, details_str, re.IGNORECASE):
                    return True
            
            # Check for XSS attempts
            for pattern in self.suspicious_patterns["xss_patterns"]:
                if re.search(pattern, details_str, re.IGNORECASE):
                    return True
        
        # Check unusual access times
        if event.timestamp.hour >= self.suspicious_patterns["unusual_hours"][0] and \
           event.timestamp.hour < self.suspicious_patterns["unusual_hours"][1]:
            return True
        
        return False
    
    def _check_threshold_breach(self, event: AuditEvent) -> bool:
        """Check if event breaches threshold limits"""
        
        if event.event_type == AuditEventType.LOGIN_FAILED:
            # Count failed logins in window
            key = f"failed_login:{event.user_id or event.ip_address}"
            count = self.redis_client.incr(key)
            self.redis_client.expire(key, self.suspicious_patterns["failed_login_window"])
            
            if count >= self.suspicious_patterns["failed_login_threshold"]:
                return True
        
        return False
    
    def _handle_suspicious_activity(self, event: AuditEvent):
        """Handle detected suspicious activity"""
        
        self.metrics["suspicious_detected"] += 1
        
        # Create security alert
        alert = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.HIGH,
            user_id=event.user_id,
            username=event.username,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            resource_type="security",
            resource_id=event.event_id,
            action="suspicious_activity_detected",
            result="alert_raised",
            details={
                "original_event": event.event_type.value,
                "detection_reason": "threshold_breach",
                "recommended_action": "investigate"
            },
            session_id=event.session_id,
            correlation_id=event.correlation_id,
            metadata={"alert_priority": "high"}
        )
        
        # Log the alert
        self.log_event(alert)
        
        # Send immediate notification to SIEM
        self._send_immediate_alert(alert)
    
    # ========== SIEM Integration ==========
    
    def _send_to_siem(self, events: List[Dict]):
        """Send events to SIEM systems"""
        
        if not self.siem_endpoints:
            return
        
        # Convert to CEF (Common Event Format) or other SIEM format
        cef_events = [self._to_cef_format(e["event"]) for e in events]
        
        for endpoint in self.siem_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json={"events": cef_events},
                    timeout=5,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
            except Exception as e:
                self.metrics["siem_failures"] += 1
                # Log locally if SIEM fails
                self._emergency_log_batch(events, f"SIEM failure: {e}")
    
    def _to_cef_format(self, event: AuditEvent) -> str:
        """Convert event to Common Event Format for SIEM"""
        
        cef = (
            f"CEF:0|ArbitrationSystem|AuditLog|1.0|{event.event_type.value}|"
            f"{event.action}|{self._severity_to_cef(event.severity)}|"
            f"rt={int(event.timestamp.timestamp() * 1000)} "
            f"src={event.ip_address or 'unknown'} "
            f"suser={event.username or 'unknown'} "
            f"cs1Label=ResourceType cs1={event.resource_type} "
            f"cs2Label=ResourceID cs2={event.resource_id} "
            f"cs3Label=Result cs3={event.result} "
            f"cs4Label=SessionID cs4={event.session_id}"
        )
        
        return cef
    
    def _severity_to_cef(self, severity: AuditSeverity) -> int:
        """Convert severity to CEF numeric value"""
        mapping = {
            AuditSeverity.CRITICAL: 10,
            AuditSeverity.HIGH: 7,
            AuditSeverity.MEDIUM: 4,
            AuditSeverity.LOW: 2,
            AuditSeverity.INFO: 0
        }
        return mapping.get(severity, 0)
    
    def _send_immediate_alert(self, alert: AuditEvent):
        """Send immediate alert for critical events"""
        
        # Send to all SIEM endpoints immediately
        for endpoint in self.siem_endpoints:
            try:
                requests.post(
                    f"{endpoint}/alert",
                    json=asdict(alert),
                    timeout=2
                )
            except:
                pass  # Best effort for alerts
    
    # ========== Compliance Reporting ==========
    
    def generate_compliance_report(self,
                                  standard: ComplianceStandard,
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific standard"""
        
        session = self.SessionLocal()
        try:
            # Define required events for each standard
            required_events = self._get_required_events(standard)
            
            report = {
                "standard": standard.value,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {},
                "details": [],
                "compliance_status": "COMPLIANT",
                "findings": []
            }
            
            # Analyze events for compliance
            for event_type in required_events:
                count = session.query(AuditLog).filter(
                    AuditLog.event_type == event_type,
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date
                ).count()
                
                report["summary"][event_type] = count
                
                # Check for missing required events
                if count == 0 and self._is_required_event(standard, event_type):
                    report["compliance_status"] = "NON_COMPLIANT"
                    report["findings"].append({
                        "severity": "HIGH",
                        "issue": f"Missing required audit events: {event_type}",
                        "recommendation": "Ensure all required events are being logged"
                    })
            
            # Additional compliance checks
            report["checks"] = self._perform_compliance_checks(session, standard, start_date, end_date)
            
            return report
            
        finally:
            session.close()
    
    def _get_required_events(self, standard: ComplianceStandard) -> List[str]:
        """Get required audit events for compliance standard"""
        
        requirements = {
            ComplianceStandard.SOC2: [
                AuditEventType.LOGIN_SUCCESS.value,
                AuditEventType.LOGIN_FAILED.value,
                AuditEventType.DATA_READ.value,
                AuditEventType.DATA_UPDATED.value,
                AuditEventType.DATA_DELETED.value,
                AuditEventType.PERMISSION_DENIED.value,
                AuditEventType.CONFIG_CHANGED.value
            ],
            ComplianceStandard.GDPR: [
                AuditEventType.DATA_READ.value,
                AuditEventType.DATA_EXPORTED.value,
                AuditEventType.DATA_DELETED.value,
                AuditEventType.GDPR_REQUEST.value
            ],
            ComplianceStandard.HIPAA: [
                AuditEventType.LOGIN_SUCCESS.value,
                AuditEventType.DATA_READ.value,
                AuditEventType.DATA_UPDATED.value,
                AuditEventType.DATA_EXPORTED.value,
                AuditEventType.DOCUMENT_ENCRYPTED.value,
                AuditEventType.DOCUMENT_DECRYPTED.value
            ],
            ComplianceStandard.PCI_DSS: [
                AuditEventType.LOGIN_SUCCESS.value,
                AuditEventType.LOGIN_FAILED.value,
                AuditEventType.DATA_READ.value,
                AuditEventType.API_KEY_CREATED.value,
                AuditEventType.API_KEY_REVOKED.value
            ]
        }
        
        return requirements.get(standard, [])
    
    def _is_required_event(self, standard: ComplianceStandard, event_type: str) -> bool:
        """Check if event is required for compliance"""
        required = self._get_required_events(standard)
        return event_type in required
    
    def _perform_compliance_checks(self,
                                  session: Session,
                                  standard: ComplianceStandard,
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Perform specific compliance checks"""
        
        checks = {
            "audit_integrity": self._check_audit_integrity(session, start_date, end_date),
            "retention_compliance": self._check_retention_compliance(session),
            "access_controls": self._check_access_controls(session, start_date, end_date),
            "encryption_status": self._check_encryption_status(session, start_date, end_date)
        }
        
        if standard == ComplianceStandard.GDPR:
            checks["gdpr_requests"] = self._check_gdpr_compliance(session, start_date, end_date)
        
        return checks
    
    def _check_audit_integrity(self, session: Session, start_date: datetime, end_date: datetime) -> Dict:
        """Check audit log integrity"""
        
        # Sample logs for integrity check
        logs = session.query(AuditLog).filter(
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).limit(100).all()
        
        failed_checks = []
        for log in logs:
            if not self.verify_integrity(log.event_id):
                failed_checks.append(log.event_id)
        
        return {
            "checked": len(logs),
            "passed": len(logs) - len(failed_checks),
            "failed": failed_checks,
            "status": "PASS" if not failed_checks else "FAIL"
        }
    
    def _check_retention_compliance(self, session: Session) -> Dict:
        """Check if retention policies are being followed"""
        
        oldest_log = session.query(AuditLog).order_by(AuditLog.timestamp).first()
        
        if oldest_log:
            age_days = (datetime.utcnow() - oldest_log.timestamp).days
            return {
                "oldest_log_days": age_days,
                "retention_policy_days": self.retention_days,
                "status": "PASS" if age_days <= self.retention_days else "FAIL"
            }
        
        return {"status": "NO_DATA"}
    
    def _check_access_controls(self, session: Session, start_date: datetime, end_date: datetime) -> Dict:
        """Check access control enforcement"""
        
        denied_count = session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.PERMISSION_DENIED.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        granted_count = session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.PERMISSION_GRANTED.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        return {
            "permission_checks": granted_count + denied_count,
            "denied": denied_count,
            "granted": granted_count,
            "enforcement_rate": denied_count / (granted_count + denied_count) if (granted_count + denied_count) > 0 else 0
        }
    
    def _check_encryption_status(self, session: Session, start_date: datetime, end_date: datetime) -> Dict:
        """Check encryption usage"""
        
        encrypted_count = session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.DOCUMENT_ENCRYPTED.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        decrypted_count = session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.DOCUMENT_DECRYPTED.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).count()
        
        return {
            "documents_encrypted": encrypted_count,
            "documents_decrypted": decrypted_count,
            "encryption_ratio": encrypted_count / (encrypted_count + decrypted_count) if (encrypted_count + decrypted_count) > 0 else 0
        }
    
    def _check_gdpr_compliance(self, session: Session, start_date: datetime, end_date: datetime) -> Dict:
        """Check GDPR-specific compliance"""
        
        gdpr_requests = session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.GDPR_REQUEST.value,
            AuditLog.timestamp >= start_date,
            AuditLog.timestamp <= end_date
        ).all()
        
        return {
            "total_requests": len(gdpr_requests),
            "completed": len([r for r in gdpr_requests if r.result == "success"]),
            "pending": len([r for r in gdpr_requests if r.result == "pending"]),
            "average_response_time": "N/A"  # Would calculate from metadata
        }
    
    # ========== Emergency Logging ==========
    
    def _emergency_log(self, event: AuditEvent):
        """Emergency logging when primary systems fail"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"/var/log/audit_emergency_{timestamp}.log"
        
        try:
            with open(filename, 'a') as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except:
            # Last resort - print to stderr
            import sys
            print(f"EMERGENCY AUDIT: {event}", file=sys.stderr)
    
    def _emergency_log_batch(self, batch: List[Dict], error: str):
        """Emergency logging for batch failures"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"/var/log/audit_batch_emergency_{timestamp}.log"
        
        try:
            with gzip.open(filename + '.gz', 'wt') as f:
                f.write(f"Error: {error}\n")
                for item in batch:
                    f.write(json.dumps(asdict(item["event"]), default=str) + "\n")
        except:
            pass
    
    # ========== Query and Export ==========
    
    def query_logs(self,
                  filters: Dict[str, Any],
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  limit: int = 1000) -> List[AuditLog]:
        """Query audit logs with filters"""
        
        session = self.SessionLocal()
        try:
            query = session.query(AuditLog)
            
            # Apply filters
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            for key, value in filters.items():
                if hasattr(AuditLog, key):
                    query = query.filter(getattr(AuditLog, key) == value)
            
            return query.limit(limit).all()
            
        finally:
            session.close()
    
    def export_logs(self,
                   format: str = "json",
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> str:
        """Export audit logs for archival or analysis"""
        
        logs = self.query_logs({}, start_date, end_date, limit=None)
        
        if format == "json":
            return json.dumps([{
                "event_id": log.event_id,
                "timestamp": log.timestamp.isoformat(),
                "event_type": log.event_type,
                "severity": log.severity,
                "user_id": log.user_id,
                "username": log.username,
                "ip_address": log.ip_address,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "action": log.action,
                "result": log.result,
                "details": json.loads(log.details) if log.details else {}
            } for log in logs], indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "event_id", "timestamp", "event_type", "severity",
                "user_id", "username", "ip_address", "resource_type",
                "resource_id", "action", "result"
            ])
            
            # Data
            for log in logs:
                writer.writerow([
                    log.event_id, log.timestamp, log.event_type, log.severity,
                    log.user_id, log.username, log.ip_address, log.resource_type,
                    log.resource_id, log.action, log.result
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# ========== Decorators for Automatic Audit Logging ==========

def audit_action(
    event_type: AuditEventType,
    resource_type: str,
    severity: AuditSeverity = AuditSeverity.LOW
):
    """Decorator to automatically audit function calls"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context (would get from request context in real app)
            context = kwargs.get('audit_context', {})
            
            # Create audit event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                user_id=context.get('user_id'),
                username=context.get('username'),
                ip_address=context.get('ip_address'),
                user_agent=context.get('user_agent'),
                resource_type=resource_type,
                resource_id=kwargs.get('resource_id'),
                action=func.__name__,
                result="pending",
                details={"function": func.__name__, "args": str(args)[:100]},
                session_id=context.get('session_id'),
                correlation_id=context.get('correlation_id'),
                metadata={}
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                event.result = "success"
                event.details["return_type"] = type(result).__name__
                
                # Log audit event
                if audit_manager:
                    audit_manager.log_event(event)
                
                return result
                
            except Exception as e:
                event.result = "failure"
                event.details["error"] = str(e)
                event.severity = AuditSeverity.HIGH
                
                # Log audit event
                if audit_manager:
                    audit_manager.log_event(event)
                
                raise
        
        return wrapper
    return decorator


# Global audit manager instance
audit_manager = None

def get_audit_manager() -> AuditManager:
    """Get audit manager instance"""
    global audit_manager
    if not audit_manager:
        from app.core.config import settings
        audit_manager = AuditManager(
            database_url=settings.database_url,
            retention_days=2555  # 7 years
        )
    return audit_manager