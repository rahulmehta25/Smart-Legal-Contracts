"""
Comprehensive Audit Logging System for Legal Compliance
Implements structured logging for security events and compliance requirements
GDPR, SOC 2, and HIPAA compliant
"""

import json
import hashlib
import hmac
import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import gzip
import base64

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import redis
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
from loguru import logger

Base = declarative_base()

class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILED = "auth.login.failed"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"
    MFA_SUCCESS = "auth.mfa.success"
    MFA_FAILED = "auth.mfa.failed"
    
    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGED = "authz.permission.changed"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REMOVED = "authz.role.removed"
    
    # Document events
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_VIEWED = "document.viewed"
    DOCUMENT_MODIFIED = "document.modified"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_SHARED = "document.shared"
    DOCUMENT_DOWNLOADED = "document.downloaded"
    DOCUMENT_ENCRYPTED = "document.encrypted"
    DOCUMENT_SIGNED = "document.signed"
    
    # Data events
    DATA_EXPORTED = "data.exported"
    DATA_IMPORTED = "data.imported"
    DATA_ANONYMIZED = "data.anonymized"
    DATA_RETENTION_APPLIED = "data.retention.applied"
    
    # Security events
    SECURITY_ALERT = "security.alert"
    INTRUSION_DETECTED = "security.intrusion"
    MALWARE_DETECTED = "security.malware"
    RATE_LIMIT_EXCEEDED = "security.ratelimit"
    IP_BLOCKED = "security.ip.blocked"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    
    # Compliance events
    GDPR_REQUEST = "compliance.gdpr.request"
    GDPR_DELETION = "compliance.gdpr.deletion"
    GDPR_EXPORT = "compliance.gdpr.export"
    CONSENT_GIVEN = "compliance.consent.given"
    CONSENT_WITHDRAWN = "compliance.consent.withdrawn"
    AUDIT_VIEWED = "compliance.audit.viewed"
    AUDIT_EXPORTED = "compliance.audit.exported"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGED = "system.config.changed"
    BACKUP_CREATED = "system.backup.created"
    ERROR_OCCURRED = "system.error"
    
    # API events
    API_KEY_CREATED = "api.key.created"
    API_KEY_REVOKED = "api.key.revoked"
    API_CALL = "api.call"
    API_ERROR = "api.error"

class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    CRITICAL = "critical"  # Security breaches, data loss
    HIGH = "high"         # Failed authentication, unauthorized access
    MEDIUM = "medium"     # Configuration changes, permission changes
    LOW = "low"          # Normal operations, successful logins
    INFO = "info"        # Informational events

@dataclass
class AuditEvent:
    """Structured audit event"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    user_email: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    resource_type: Optional[str]  # e.g., "document", "user", "api"
    resource_id: Optional[str]
    action: str  # e.g., "create", "read", "update", "delete"
    result: str  # "success" or "failure"
    details: Dict[str, Any]  # Additional context
    metadata: Dict[str, Any]  # System metadata
    signature: Optional[str]  # For tamper detection

class AuditLog(Base):
    """Database model for audit logs"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    user_id = Column(String(64), index=True)
    user_email = Column(String(255), index=True)
    ip_address = Column(String(45), index=True)
    user_agent = Column(Text)
    session_id = Column(String(64), index=True)
    resource_type = Column(String(32), index=True)
    resource_id = Column(String(64), index=True)
    action = Column(String(32))
    result = Column(String(16))
    details = Column(JSON)
    metadata = Column(JSON)
    signature = Column(String(128))
    archived = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_event_type', 'event_type'),
        Index('idx_user_resource', 'user_id', 'resource_type', 'resource_id'),
    )

class AuditLogger:
    """
    Comprehensive audit logging system with:
    - Structured logging for compliance
    - Tamper detection via signatures
    - Log retention and archival
    - Real-time alerting
    - Export capabilities for compliance reporting
    """
    
    def __init__(self,
                 db_session: Session,
                 redis_client: Optional[redis.Redis] = None,
                 signing_key: Optional[str] = None,
                 retention_days: int = 2555):  # 7 years default
        
        self.db = db_session
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Signing key for tamper detection
        if signing_key:
            self.signing_key = signing_key.encode()
        else:
            self.signing_key = self._generate_signing_key()
        
        self.retention_days = retention_days
        
        # Queue for async logging
        self.log_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Alert rules
        self.alert_rules = self._initialize_alert_rules()
        
    def _generate_signing_key(self) -> bytes:
        """Generate signing key for log integrity"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'audit_log_salt',
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(b'default_audit_key')
    
    def _initialize_alert_rules(self) -> List[Dict[str, Any]]:
        """Initialize real-time alert rules"""
        return [
            {
                'event_type': AuditEventType.LOGIN_FAILED,
                'threshold': 5,
                'window': 300,  # 5 minutes
                'action': 'alert_security_team'
            },
            {
                'event_type': AuditEventType.ACCESS_DENIED,
                'threshold': 10,
                'window': 600,
                'action': 'alert_security_team'
            },
            {
                'event_type': AuditEventType.MALWARE_DETECTED,
                'threshold': 1,
                'window': 0,
                'action': 'immediate_alert'
            },
            {
                'event_type': AuditEventType.DATA_EXPORTED,
                'threshold': 100,
                'window': 3600,
                'action': 'alert_compliance_team'
            }
        ]
    
    # ========== Core Logging Methods ==========
    
    def log_event(self,
                  event_type: AuditEventType,
                  user_id: Optional[str] = None,
                  user_email: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  session_id: Optional[str] = None,
                  resource_type: Optional[str] = None,
                  resource_id: Optional[str] = None,
                  action: str = "",
                  result: str = "success",
                  details: Optional[Dict[str, Any]] = None,
                  severity: Optional[AuditSeverity] = None) -> str:
        """
        Log an audit event
        Returns: event_id
        """
        
        # Auto-determine severity if not provided
        if not severity:
            severity = self._determine_severity(event_type, result)
        
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            details=details or {},
            metadata=self._get_metadata(),
            signature=None
        )
        
        # Sign the event
        event.signature = self._sign_event(event)
        
        # Queue for processing
        self.log_queue.put(event)
        
        # Check alert rules
        self._check_alerts(event)
        
        # Log to file system as backup
        self._log_to_file(event)
        
        return event.event_id
    
    def _determine_severity(self, event_type: AuditEventType, result: str) -> AuditSeverity:
        """Auto-determine event severity"""
        
        critical_events = [
            AuditEventType.INTRUSION_DETECTED,
            AuditEventType.MALWARE_DETECTED,
            AuditEventType.DATA_EXPORTED
        ]
        
        high_events = [
            AuditEventType.LOGIN_FAILED,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.MFA_FAILED,
            AuditEventType.SECURITY_ALERT
        ]
        
        if event_type in critical_events:
            return AuditSeverity.CRITICAL
        elif event_type in high_events and result == "failure":
            return AuditSeverity.HIGH
        elif "change" in event_type.value or "delete" in event_type.value:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get system metadata for audit event"""
        import platform
        import socket
        
        return {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'process_id': threading.current_thread().ident,
            'app_version': '1.0.0'  # Get from config
        }
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Sign event for tamper detection"""
        
        # Create signing payload
        payload = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'user_id': event.user_id,
            'resource_id': event.resource_id,
            'action': event.action,
            'result': event.result
        }
        
        payload_str = json.dumps(payload, sort_keys=True)
        
        # Create HMAC signature
        signature = hmac.new(
            self.signing_key,
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_event(self, event: AuditEvent) -> bool:
        """Verify event signature for tamper detection"""
        
        expected_signature = self._sign_event(event)
        return hmac.compare_digest(event.signature, expected_signature)
    
    # ========== Queue Processing ==========
    
    def _process_queue(self):
        """Background thread to process audit events"""
        
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Get event from queue with timeout
                event = self.log_queue.get(timeout=1)
                batch.append(event)
                
                # Flush batch if size or time threshold reached
                if len(batch) >= 100 or (time.time() - last_flush) > 10:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except:
                # Timeout - flush any pending events
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
    
    def _flush_batch(self, events: List[AuditEvent]):
        """Flush batch of events to database"""
        
        try:
            for event in events:
                log_entry = AuditLog(
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    user_id=event.user_id,
                    user_email=event.user_email,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    session_id=event.session_id,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    action=event.action,
                    result=event.result,
                    details=event.details,
                    metadata=event.metadata,
                    signature=event.signature
                )
                self.db.add(log_entry)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to flush audit batch: {e}")
            self.db.rollback()
            
            # Save to emergency file
            self._save_emergency_batch(events)
    
    def _save_emergency_batch(self, events: List[AuditEvent]):
        """Save events to file when database fails"""
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"/var/log/audit_emergency_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                events_dict = [asdict(e) for e in events]
                json.dump(events_dict, f, default=str)
        except:
            pass
    
    def _log_to_file(self, event: AuditEvent):
        """Log to file system as backup"""
        
        try:
            log_line = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'severity': event.severity.value,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'resource': f"{event.resource_type}:{event.resource_id}",
                'action': event.action,
                'result': event.result
            }
            
            logger.bind(audit=True).info(json.dumps(log_line))
            
        except:
            pass
    
    # ========== Alert System ==========
    
    def _check_alerts(self, event: AuditEvent):
        """Check if event triggers alerts"""
        
        for rule in self.alert_rules:
            if event.event_type == rule['event_type']:
                # Check threshold
                if rule['window'] > 0:
                    count = self._count_recent_events(
                        event.event_type,
                        rule['window']
                    )
                    if count >= rule['threshold']:
                        self._trigger_alert(rule, event, count)
                else:
                    # Immediate alert
                    self._trigger_alert(rule, event, 1)
    
    def _count_recent_events(self, event_type: AuditEventType, window: int) -> int:
        """Count recent events of type within window"""
        
        key = f"audit:count:{event_type.value}"
        
        # Increment counter
        count = self.redis_client.incr(key)
        
        # Set expiry on first increment
        if count == 1:
            self.redis_client.expire(key, window)
        
        return count
    
    def _trigger_alert(self, rule: Dict[str, Any], event: AuditEvent, count: int):
        """Trigger security alert"""
        
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'rule': rule,
            'event': asdict(event),
            'count': count,
            'message': f"Alert: {count} {event.event_type.value} events in {rule['window']} seconds"
        }
        
        # Send to alert queue
        self.redis_client.lpush('security_alerts', json.dumps(alert, default=str))
        
        # Log alert
        logger.warning(f"Security alert triggered: {alert['message']}")
    
    # ========== Query and Export ==========
    
    def query_logs(self,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   user_id: Optional[str] = None,
                   event_type: Optional[AuditEventType] = None,
                   resource_type: Optional[str] = None,
                   resource_id: Optional[str] = None,
                   limit: int = 1000) -> List[AuditLog]:
        """Query audit logs"""
        
        query = self.db.query(AuditLog)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if event_type:
            query = query.filter(AuditLog.event_type == event_type.value)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        if resource_id:
            query = query.filter(AuditLog.resource_id == resource_id)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    def export_logs(self,
                   format: str = "json",
                   compress: bool = True,
                   **query_params) -> bytes:
        """Export audit logs for compliance reporting"""
        
        # Query logs
        logs = self.query_logs(**query_params)
        
        # Convert to dict
        logs_data = []
        for log in logs:
            log_dict = {
                'event_id': log.event_id,
                'timestamp': log.timestamp.isoformat(),
                'event_type': log.event_type,
                'severity': log.severity,
                'user_id': log.user_id,
                'user_email': log.user_email,
                'ip_address': log.ip_address,
                'resource': f"{log.resource_type}:{log.resource_id}",
                'action': log.action,
                'result': log.result,
                'details': log.details,
                'verified': self.verify_event(log) if log.signature else False
            }
            logs_data.append(log_dict)
        
        # Format output
        if format == "json":
            output = json.dumps(logs_data, indent=2, default=str).encode()
        elif format == "csv":
            import csv
            import io
            
            buffer = io.StringIO()
            if logs_data:
                writer = csv.DictWriter(buffer, fieldnames=logs_data[0].keys())
                writer.writeheader()
                writer.writerows(logs_data)
            output = buffer.getvalue().encode()
        else:
            output = str(logs_data).encode()
        
        # Compress if requested
        if compress:
            output = gzip.compress(output)
        
        # Log the export
        self.log_event(
            event_type=AuditEventType.AUDIT_EXPORTED,
            details={'format': format, 'count': len(logs), 'compressed': compress}
        )
        
        return output
    
    # ========== Retention and Archival ==========
    
    def apply_retention_policy(self):
        """Apply retention policy to old logs"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Archive old logs
        old_logs = self.db.query(AuditLog).filter(
            AuditLog.timestamp < cutoff_date,
            AuditLog.archived == False
        ).all()
        
        if old_logs:
            # Archive to cold storage
            self._archive_logs(old_logs)
            
            # Mark as archived
            for log in old_logs:
                log.archived = True
            
            self.db.commit()
            
            # Log retention action
            self.log_event(
                event_type=AuditEventType.DATA_RETENTION_APPLIED,
                details={'logs_archived': len(old_logs), 'cutoff_date': cutoff_date.isoformat()}
            )
    
    def _archive_logs(self, logs: List[AuditLog]):
        """Archive logs to cold storage"""
        
        # Create archive file
        timestamp = datetime.utcnow().strftime('%Y%m%d')
        archive_file = f"/archive/audit_logs_{timestamp}.json.gz"
        
        # Convert to JSON
        logs_data = [
            {
                'event_id': log.event_id,
                'timestamp': log.timestamp.isoformat(),
                'event_type': log.event_type,
                'user_id': log.user_id,
                'details': log.details,
                'signature': log.signature
            }
            for log in logs
        ]
        
        # Compress and save
        compressed = gzip.compress(json.dumps(logs_data, default=str).encode())
        
        try:
            with open(archive_file, 'wb') as f:
                f.write(compressed)
        except Exception as e:
            logger.error(f"Failed to archive logs: {e}")
    
    # ========== GDPR Compliance ==========
    
    def get_user_audit_data(self, user_id: str) -> Dict[str, Any]:
        """Get all audit data for a user (GDPR export)"""
        
        logs = self.query_logs(user_id=user_id)
        
        user_data = {
            'user_id': user_id,
            'audit_logs': [
                {
                    'timestamp': log.timestamp.isoformat(),
                    'event_type': log.event_type,
                    'action': log.action,
                    'resource': f"{log.resource_type}:{log.resource_id}",
                    'ip_address': log.ip_address
                }
                for log in logs
            ],
            'export_date': datetime.utcnow().isoformat()
        }
        
        # Log GDPR export
        self.log_event(
            event_type=AuditEventType.GDPR_EXPORT,
            user_id=user_id,
            details={'logs_exported': len(logs)}
        )
        
        return user_data
    
    def anonymize_user_logs(self, user_id: str):
        """Anonymize user audit logs (GDPR compliance)"""
        
        logs = self.db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
        
        for log in logs:
            # Anonymize personal data
            log.user_id = f"ANON_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
            log.user_email = None
            log.ip_address = "0.0.0.0"
            log.user_agent = "REDACTED"
            
            # Update details to remove PII
            if log.details:
                log.details = self._redact_pii(log.details)
        
        self.db.commit()
        
        # Log anonymization
        self.log_event(
            event_type=AuditEventType.DATA_ANONYMIZED,
            details={'original_user_id': user_id, 'logs_anonymized': len(logs)}
        )
    
    def _redact_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from data"""
        
        pii_fields = ['email', 'phone', 'ssn', 'address', 'name']
        
        redacted = {}
        for key, value in data.items():
            if any(field in key.lower() for field in pii_fields):
                redacted[key] = "REDACTED"
            elif isinstance(value, dict):
                redacted[key] = self._redact_pii(value)
            else:
                redacted[key] = value
        
        return redacted


# ========== Decorators for Automatic Logging ==========

def audit_log(event_type: AuditEventType, 
             resource_type: Optional[str] = None,
             extract_resource_id: Optional[str] = None):
    """Decorator for automatic audit logging"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get audit logger from context
            audit_logger = kwargs.get('audit_logger')
            if not audit_logger:
                return func(*args, **kwargs)
            
            # Extract resource ID if specified
            resource_id = None
            if extract_resource_id:
                resource_id = kwargs.get(extract_resource_id)
            
            # Log start of action
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                audit_logger.log_event(
                    event_type=event_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action=func.__name__,
                    result="success",
                    details={
                        'duration': time.time() - start_time,
                        'args': str(args)[:100]  # Truncate for safety
                    }
                )
                
                return result
                
            except Exception as e:
                # Log failure
                audit_logger.log_event(
                    event_type=event_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action=func.__name__,
                    result="failure",
                    details={
                        'duration': time.time() - start_time,
                        'error': str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator