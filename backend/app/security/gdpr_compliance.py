"""
GDPR Compliance Module
Implements data privacy features for GDPR compliance
Handles data subject rights, consent management, and privacy controls
"""

import json
import hashlib
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
from queue import Queue

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
import redis

Base = declarative_base()

class ConsentPurpose(str, Enum):
    """Purposes for data processing consent"""
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PROFILING = "profiling"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AUTOMATED_DECISIONS = "automated_decisions"
    LEGAL_PROCESSING = "legal_processing"
    CONTRACT_PROCESSING = "contract_processing"
    ESSENTIAL_SERVICES = "essential_services"

class ConsentStatus(str, Enum):
    """Consent status"""
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"

class DataSubjectRequestType(str, Enum):
    """Types of GDPR data subject requests"""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Data portability
    RESTRICTION = "restriction"  # Restrict processing
    OBJECTION = "objection"  # Object to processing

class RequestStatus(str, Enum):
    """Status of data subject request"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    PARTIALLY_COMPLETED = "partially_completed"

@dataclass
class ConsentRecord:
    """Consent record structure"""
    consent_id: str
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    granted_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    expires_at: Optional[datetime]
    ip_address: str
    user_agent: str
    version: str
    details: Dict[str, Any]

@dataclass
class DataSubjectRequest:
    """Data subject request structure"""
    request_id: str
    user_id: str
    request_type: DataSubjectRequestType
    status: RequestStatus
    submitted_at: datetime
    completed_at: Optional[datetime]
    details: Dict[str, Any]
    verification_token: str
    response_data: Optional[Dict[str, Any]]

# Database Models
class UserConsent(Base):
    """Database model for user consent records"""
    __tablename__ = 'user_consents'
    
    id = Column(Integer, primary_key=True)
    consent_id = Column(String(64), unique=True, nullable=False)
    user_id = Column(String(64), nullable=False, index=True)
    purpose = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False)
    granted_at = Column(DateTime)
    withdrawn_at = Column(DateTime)
    expires_at = Column(DateTime)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    version = Column(String(16))
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DataRequest(Base):
    """Database model for data subject requests"""
    __tablename__ = 'data_subject_requests'
    
    id = Column(Integer, primary_key=True)
    request_id = Column(String(64), unique=True, nullable=False)
    user_id = Column(String(64), nullable=False, index=True)
    request_type = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False)
    submitted_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    processed_by = Column(String(64))
    details = Column(JSON)
    verification_token = Column(String(128))
    response_data = Column(JSON)
    rejection_reason = Column(Text)

class PrivacyPreference(Base):
    """Database model for privacy preferences"""
    __tablename__ = 'privacy_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), unique=True, nullable=False)
    data_retention_days = Column(Integer, default=365)
    allow_profiling = Column(Boolean, default=False)
    allow_automated_decisions = Column(Boolean, default=False)
    communication_preferences = Column(JSON)
    data_sharing_preferences = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GDPRComplianceManager:
    """
    GDPR Compliance Manager implementing:
    - Consent management
    - Data subject rights
    - Privacy by design
    - Data minimization
    - Right to be forgotten
    - Data portability
    - Privacy preferences
    """
    
    def __init__(self,
                 db_session: Session,
                 audit_logger,
                 encryption_manager,
                 notification_service=None):
        
        self.db = db_session
        self.audit_logger = audit_logger
        self.encryption_manager = encryption_manager
        self.notification_service = notification_service
        
        # Redis for caching
        self.cache = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Request processing queue
        self.request_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_requests)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Privacy policy version
        self.privacy_policy_version = "2.0"
        
        # Data retention policies
        self.retention_policies = {
            "user_data": 365 * 3,  # 3 years
            "logs": 365 * 2,  # 2 years
            "analytics": 365,  # 1 year
            "marketing": 180,  # 6 months
            "sessions": 30  # 30 days
        }
    
    # ========== Consent Management ==========
    
    def record_consent(self,
                      user_id: str,
                      purpose: ConsentPurpose,
                      granted: bool,
                      ip_address: str,
                      user_agent: str,
                      details: Optional[Dict[str, Any]] = None) -> ConsentRecord:
        """Record user consent"""
        
        consent = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            user_id=user_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED if granted else ConsentStatus.WITHDRAWN,
            granted_at=datetime.utcnow() if granted else None,
            withdrawn_at=None if granted else datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365) if granted else None,
            ip_address=ip_address,
            user_agent=user_agent,
            version=self.privacy_policy_version,
            details=details or {}
        )
        
        # Save to database
        db_consent = UserConsent(
            consent_id=consent.consent_id,
            user_id=consent.user_id,
            purpose=consent.purpose.value,
            status=consent.status.value,
            granted_at=consent.granted_at,
            withdrawn_at=consent.withdrawn_at,
            expires_at=consent.expires_at,
            ip_address=consent.ip_address,
            user_agent=consent.user_agent,
            version=consent.version,
            details=consent.details
        )
        
        self.db.add(db_consent)
        self.db.commit()
        
        # Clear cache
        self._clear_consent_cache(user_id)
        
        # Audit log
        self.audit_logger.log_event(
            event_type="compliance.consent.given" if granted else "compliance.consent.withdrawn",
            user_id=user_id,
            ip_address=ip_address,
            details={
                'purpose': purpose.value,
                'version': self.privacy_policy_version
            }
        )
        
        return consent
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for user"""
        
        # Check cache
        cache_key = f"gdpr:consent:{user_id}"
        cached = self.cache.get(cache_key)
        
        if cached:
            return [ConsentRecord(**c) for c in json.loads(cached)]
        
        # Query database
        consents = self.db.query(UserConsent).filter(
            UserConsent.user_id == user_id,
            UserConsent.status == ConsentStatus.GRANTED.value
        ).all()
        
        records = []
        for c in consents:
            record = ConsentRecord(
                consent_id=c.consent_id,
                user_id=c.user_id,
                purpose=ConsentPurpose(c.purpose),
                status=ConsentStatus(c.status),
                granted_at=c.granted_at,
                withdrawn_at=c.withdrawn_at,
                expires_at=c.expires_at,
                ip_address=c.ip_address,
                user_agent=c.user_agent,
                version=c.version,
                details=c.details or {}
            )
            records.append(record)
        
        # Cache results
        self.cache.setex(
            cache_key,
            300,
            json.dumps([asdict(r) for r in records], default=str)
        )
        
        return records
    
    def has_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """Check if user has given consent for purpose"""
        
        consents = self.get_user_consents(user_id)
        
        for consent in consents:
            if consent.purpose == purpose and consent.status == ConsentStatus.GRANTED:
                # Check expiration
                if consent.expires_at and consent.expires_at < datetime.utcnow():
                    return False
                return True
        
        return False
    
    def withdraw_consent(self,
                        user_id: str,
                        purpose: ConsentPurpose,
                        ip_address: str) -> bool:
        """Withdraw consent for specific purpose"""
        
        # Find active consent
        consent = self.db.query(UserConsent).filter(
            UserConsent.user_id == user_id,
            UserConsent.purpose == purpose.value,
            UserConsent.status == ConsentStatus.GRANTED.value
        ).first()
        
        if consent:
            consent.status = ConsentStatus.WITHDRAWN.value
            consent.withdrawn_at = datetime.utcnow()
            self.db.commit()
            
            # Clear cache
            self._clear_consent_cache(user_id)
            
            # Stop related processing
            self._stop_processing_for_purpose(user_id, purpose)
            
            # Audit log
            self.audit_logger.log_event(
                event_type="compliance.consent.withdrawn",
                user_id=user_id,
                ip_address=ip_address,
                details={'purpose': purpose.value}
            )
            
            return True
        
        return False
    
    def _clear_consent_cache(self, user_id: str):
        """Clear consent cache for user"""
        self.cache.delete(f"gdpr:consent:{user_id}")
    
    def _stop_processing_for_purpose(self, user_id: str, purpose: ConsentPurpose):
        """Stop data processing for withdrawn consent"""
        
        # Implement based on purpose
        if purpose == ConsentPurpose.MARKETING:
            # Unsubscribe from marketing
            pass
        elif purpose == ConsentPurpose.ANALYTICS:
            # Exclude from analytics
            pass
        elif purpose == ConsentPurpose.PROFILING:
            # Delete profiling data
            pass
    
    # ========== Data Subject Rights ==========
    
    def submit_data_request(self,
                           user_id: str,
                           request_type: DataSubjectRequestType,
                           details: Optional[Dict[str, Any]] = None) -> DataSubjectRequest:
        """Submit data subject request"""
        
        # Create request
        request = DataSubjectRequest(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            submitted_at=datetime.utcnow(),
            completed_at=None,
            details=details or {},
            verification_token=self._generate_verification_token(),
            response_data=None
        )
        
        # Save to database
        db_request = DataRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            request_type=request.request_type.value,
            status=request.status.value,
            submitted_at=request.submitted_at,
            details=request.details,
            verification_token=request.verification_token
        )
        
        self.db.add(db_request)
        self.db.commit()
        
        # Queue for processing
        self.request_queue.put(request)
        
        # Send verification email
        if self.notification_service:
            self.notification_service.send_verification(user_id, request.verification_token)
        
        # Audit log
        self.audit_logger.log_event(
            event_type="compliance.gdpr.request",
            user_id=user_id,
            details={
                'request_id': request.request_id,
                'request_type': request.request_type.value
            }
        )
        
        return request
    
    def _generate_verification_token(self) -> str:
        """Generate verification token for request"""
        return hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    
    def verify_request(self, request_id: str, token: str) -> bool:
        """Verify data subject request"""
        
        request = self.db.query(DataRequest).filter(
            DataRequest.request_id == request_id
        ).first()
        
        if request and request.verification_token == token:
            request.status = RequestStatus.IN_PROGRESS.value
            self.db.commit()
            return True
        
        return False
    
    def _process_requests(self):
        """Background thread to process data subject requests"""
        
        while True:
            try:
                request = self.request_queue.get(timeout=5)
                self._process_single_request(request)
            except:
                continue
    
    def _process_single_request(self, request: DataSubjectRequest):
        """Process a single data subject request"""
        
        try:
            if request.request_type == DataSubjectRequestType.ACCESS:
                self._process_access_request(request)
            elif request.request_type == DataSubjectRequestType.ERASURE:
                self._process_erasure_request(request)
            elif request.request_type == DataSubjectRequestType.PORTABILITY:
                self._process_portability_request(request)
            elif request.request_type == DataSubjectRequestType.RECTIFICATION:
                self._process_rectification_request(request)
            elif request.request_type == DataSubjectRequestType.RESTRICTION:
                self._process_restriction_request(request)
            elif request.request_type == DataSubjectRequestType.OBJECTION:
                self._process_objection_request(request)
            
            # Update status
            db_request = self.db.query(DataRequest).filter(
                DataRequest.request_id == request.request_id
            ).first()
            
            if db_request:
                db_request.status = RequestStatus.COMPLETED.value
                db_request.completed_at = datetime.utcnow()
                self.db.commit()
            
        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type="system.error",
                details={
                    'request_id': request.request_id,
                    'error': str(e)
                }
            )
    
    def _process_access_request(self, request: DataSubjectRequest):
        """Process right to access request"""
        
        user_data = self._collect_user_data(request.user_id)
        
        # Encrypt data for secure delivery
        encrypted_data = self.encryption_manager.encrypt_field(
            json.dumps(user_data, default=str),
            f"access_request_{request.request_id}"
        )
        
        # Store response
        db_request = self.db.query(DataRequest).filter(
            DataRequest.request_id == request.request_id
        ).first()
        
        if db_request:
            db_request.response_data = {
                'data_collected': True,
                'encrypted_data': encrypted_data,
                'categories': list(user_data.keys())
            }
            self.db.commit()
        
        # Notify user
        if self.notification_service:
            self.notification_service.send_access_request_complete(
                request.user_id,
                request.request_id
            )
    
    def _process_erasure_request(self, request: DataSubjectRequest):
        """Process right to be forgotten request"""
        
        # Check for legal obligations to retain data
        if self._has_legal_obligation(request.user_id):
            # Reject request
            db_request = self.db.query(DataRequest).filter(
                DataRequest.request_id == request.request_id
            ).first()
            
            if db_request:
                db_request.status = RequestStatus.REJECTED.value
                db_request.rejection_reason = "Legal obligation to retain data"
                self.db.commit()
            return
        
        # Anonymize/delete user data
        self._anonymize_user_data(request.user_id)
        
        # Log deletion
        self.audit_logger.log_event(
            event_type="compliance.gdpr.deletion",
            user_id=request.user_id,
            details={'request_id': request.request_id}
        )
    
    def _process_portability_request(self, request: DataSubjectRequest):
        """Process data portability request"""
        
        # Collect user data in portable format
        user_data = self._collect_user_data(request.user_id)
        
        # Convert to standard format (JSON)
        portable_data = {
            'export_date': datetime.utcnow().isoformat(),
            'user_id': request.user_id,
            'data': user_data,
            'format': 'JSON',
            'version': '1.0'
        }
        
        # Store for download
        db_request = self.db.query(DataRequest).filter(
            DataRequest.request_id == request.request_id
        ).first()
        
        if db_request:
            db_request.response_data = {
                'download_url': f"/api/gdpr/download/{request.request_id}",
                'format': 'JSON',
                'size': len(json.dumps(portable_data))
            }
            self.db.commit()
    
    def _process_rectification_request(self, request: DataSubjectRequest):
        """Process rectification request"""
        
        # Update incorrect data
        corrections = request.details.get('corrections', {})
        
        for field, value in corrections.items():
            # Update user data
            # Implementation depends on your data model
            pass
        
        # Log rectification
        self.audit_logger.log_event(
            event_type="compliance.gdpr.rectification",
            user_id=request.user_id,
            details={
                'request_id': request.request_id,
                'fields_updated': list(corrections.keys())
            }
        )
    
    def _process_restriction_request(self, request: DataSubjectRequest):
        """Process restriction of processing request"""
        
        # Mark user data as restricted
        self._restrict_user_processing(request.user_id)
        
        # Update preferences
        prefs = self.db.query(PrivacyPreference).filter(
            PrivacyPreference.user_id == request.user_id
        ).first()
        
        if not prefs:
            prefs = PrivacyPreference(user_id=request.user_id)
            self.db.add(prefs)
        
        prefs.allow_profiling = False
        prefs.allow_automated_decisions = False
        self.db.commit()
    
    def _process_objection_request(self, request: DataSubjectRequest):
        """Process objection to processing request"""
        
        # Stop specific processing
        objection_type = request.details.get('objection_type', 'all')
        
        if objection_type == 'marketing':
            self.withdraw_consent(request.user_id, ConsentPurpose.MARKETING, "system")
        elif objection_type == 'profiling':
            self.withdraw_consent(request.user_id, ConsentPurpose.PROFILING, "system")
        elif objection_type == 'all':
            # Stop all non-essential processing
            for purpose in [ConsentPurpose.MARKETING, ConsentPurpose.ANALYTICS, 
                          ConsentPurpose.PROFILING, ConsentPurpose.THIRD_PARTY_SHARING]:
                self.withdraw_consent(request.user_id, purpose, "system")
    
    def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all user data for access/portability requests"""
        
        user_data = {
            'profile': self._get_user_profile(user_id),
            'documents': self._get_user_documents(user_id),
            'activity': self._get_user_activity(user_id),
            'consents': [asdict(c) for c in self.get_user_consents(user_id)],
            'preferences': self._get_user_preferences(user_id),
            'processing_purposes': self._get_processing_purposes(user_id)
        }
        
        return user_data
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data"""
        # Implementation depends on your user model
        return {}
    
    def _get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user documents metadata"""
        # Implementation depends on your document model
        return []
    
    def _get_user_activity(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user activity logs"""
        # Query audit logs
        return []
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user privacy preferences"""
        
        prefs = self.db.query(PrivacyPreference).filter(
            PrivacyPreference.user_id == user_id
        ).first()
        
        if prefs:
            return {
                'data_retention_days': prefs.data_retention_days,
                'allow_profiling': prefs.allow_profiling,
                'allow_automated_decisions': prefs.allow_automated_decisions,
                'communication_preferences': prefs.communication_preferences,
                'data_sharing_preferences': prefs.data_sharing_preferences
            }
        
        return {}
    
    def _get_processing_purposes(self, user_id: str) -> List[str]:
        """Get purposes for which user data is processed"""
        
        purposes = []
        
        # Check consents
        consents = self.get_user_consents(user_id)
        for consent in consents:
            purposes.append(consent.purpose.value)
        
        # Add legal bases
        purposes.extend([
            "contract_fulfillment",
            "legal_compliance",
            "legitimate_interests"
        ])
        
        return purposes
    
    def _has_legal_obligation(self, user_id: str) -> bool:
        """Check if there's legal obligation to retain user data"""
        
        # Check for legal holds on documents
        # Check for ongoing litigation
        # Check regulatory requirements
        
        return False  # Placeholder
    
    def _anonymize_user_data(self, user_id: str):
        """Anonymize user data for deletion requests"""
        
        # Generate anonymous ID
        anon_id = f"DELETED_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
        
        # Anonymize in different tables
        # Implementation depends on your data model
        
        # Anonymize audit logs
        self.audit_logger.anonymize_user_logs(user_id)
    
    def _restrict_user_processing(self, user_id: str):
        """Restrict processing of user data"""
        
        # Add restriction flag to user account
        # Stop automated processing
        # Notify relevant systems
        
        pass
    
    # ========== Privacy Preferences ==========
    
    def update_privacy_preferences(self,
                                  user_id: str,
                                  preferences: Dict[str, Any]) -> bool:
        """Update user privacy preferences"""
        
        prefs = self.db.query(PrivacyPreference).filter(
            PrivacyPreference.user_id == user_id
        ).first()
        
        if not prefs:
            prefs = PrivacyPreference(user_id=user_id)
            self.db.add(prefs)
        
        # Update preferences
        for key, value in preferences.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        
        prefs.updated_at = datetime.utcnow()
        self.db.commit()
        
        # Apply preferences
        self._apply_privacy_preferences(user_id, preferences)
        
        return True
    
    def _apply_privacy_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Apply privacy preferences to user data"""
        
        # Apply data retention
        if 'data_retention_days' in preferences:
            self._apply_data_retention(user_id, preferences['data_retention_days'])
        
        # Apply profiling preference
        if 'allow_profiling' in preferences and not preferences['allow_profiling']:
            self._disable_profiling(user_id)
    
    def _apply_data_retention(self, user_id: str, retention_days: int):
        """Apply data retention policy for user"""
        
        # Schedule deletion of old data
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Delete old activity logs, analytics data, etc.
        # Implementation depends on your data model
        
        pass
    
    def _disable_profiling(self, user_id: str):
        """Disable profiling for user"""
        
        # Delete existing profiles
        # Exclude from future profiling
        # Update ML models to exclude user
        
        pass
    
    # ========== Data Protection Impact Assessment ==========
    
    def conduct_dpia(self, processing_activity: str) -> Dict[str, Any]:
        """Conduct Data Protection Impact Assessment"""
        
        assessment = {
            'activity': processing_activity,
            'date': datetime.utcnow().isoformat(),
            'risks': self._identify_privacy_risks(processing_activity),
            'mitigation': self._identify_mitigation_measures(processing_activity),
            'residual_risk': 'low',
            'approval_required': False
        }
        
        # Determine if high risk
        high_risk_indicators = [
            'automated_decision_making',
            'large_scale_processing',
            'systematic_monitoring',
            'sensitive_data'
        ]
        
        if any(indicator in processing_activity.lower() for indicator in high_risk_indicators):
            assessment['residual_risk'] = 'high'
            assessment['approval_required'] = True
        
        return assessment
    
    def _identify_privacy_risks(self, activity: str) -> List[Dict[str, str]]:
        """Identify privacy risks for activity"""
        
        risks = []
        
        if 'collection' in activity:
            risks.append({
                'risk': 'Excessive data collection',
                'severity': 'medium',
                'likelihood': 'low'
            })
        
        if 'sharing' in activity:
            risks.append({
                'risk': 'Unauthorized data sharing',
                'severity': 'high',
                'likelihood': 'low'
            })
        
        if 'automated' in activity:
            risks.append({
                'risk': 'Biased automated decisions',
                'severity': 'high',
                'likelihood': 'medium'
            })
        
        return risks
    
    def _identify_mitigation_measures(self, activity: str) -> List[str]:
        """Identify mitigation measures for activity"""
        
        measures = [
            "Data minimization",
            "Purpose limitation",
            "Encryption at rest and in transit",
            "Access controls",
            "Regular audits",
            "User consent",
            "Transparency measures"
        ]
        
        if 'automated' in activity:
            measures.extend([
                "Human review option",
                "Algorithm transparency",
                "Regular bias testing"
            ])
        
        return measures