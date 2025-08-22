"""
Vendor onboarding and management system.

This module provides comprehensive vendor onboarding including:
- Vendor registration and verification
- Document upload and validation
- Status management
- API key generation
"""

import logging
import secrets
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..core.config import get_settings
from .models import (
    Vendor, VendorStatus, VendorCreate, VendorResponse,
    COMMISSION_TIERS, get_commission_tier
)

logger = logging.getLogger(__name__)
settings = get_settings()


class VendorOnboardingManager:
    """Comprehensive vendor onboarding and management"""
    
    def __init__(self, db: Session):
        self.db = db
        self.required_documents = [
            "business_license",
            "tax_id_verification",
            "bank_verification",
            "identity_verification"
        ]
    
    async def register_vendor(
        self,
        user_id: int,
        vendor_data: VendorCreate,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Vendor:
        """Register a new vendor"""
        try:
            # Check if user already has a vendor account
            existing_vendor = self.db.query(Vendor).filter(
                Vendor.user_id == user_id
            ).first()
            
            if existing_vendor:
                raise ValueError("User already has a vendor account")
            
            # Generate unique vendor ID
            vendor_id = await self._generate_vendor_id(vendor_data.company_name)
            
            # Generate API key
            api_key = self._generate_api_key(vendor_id)
            
            # Create vendor record
            vendor = Vendor(
                vendor_id=vendor_id,
                user_id=user_id,
                company_name=vendor_data.company_name,
                company_website=vendor_data.company_website,
                company_description=vendor_data.company_description,
                contact_name=vendor_data.contact_name,
                contact_email=vendor_data.contact_email,
                contact_phone=vendor_data.contact_phone,
                business_type=vendor_data.business_type,
                tax_id=vendor_data.tax_id,
                business_address=vendor_data.business_address,
                status=VendorStatus.PENDING,
                api_key=api_key,
                commission_rate=COMMISSION_TIERS["standard"]["rate"],
                minimum_payout=COMMISSION_TIERS["standard"]["minimum_payout"],
                metadata=metadata or {}
            )
            
            self.db.add(vendor)
            self.db.commit()
            
            # Send welcome email and onboarding instructions
            await self._send_onboarding_email(vendor)
            
            logger.info(f"Registered new vendor: {vendor_id}")
            return vendor
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to register vendor: {e}")
            raise
    
    async def _generate_vendor_id(self, company_name: str) -> str:
        """Generate unique vendor ID"""
        # Create base ID from company name
        base_id = company_name.lower().replace(" ", "_")[:20]
        
        # Remove non-alphanumeric characters
        base_id = "".join(c for c in base_id if c.isalnum() or c == "_")
        
        # Ensure uniqueness
        counter = 1
        vendor_id = base_id
        
        while self.db.query(Vendor).filter(Vendor.vendor_id == vendor_id).first():
            vendor_id = f"{base_id}_{counter}"
            counter += 1
        
        return vendor_id
    
    def _generate_api_key(self, vendor_id: str) -> str:
        """Generate secure API key for vendor"""
        # Generate random bytes
        random_bytes = secrets.token_bytes(32)
        
        # Create hash with vendor ID
        hash_input = f"{vendor_id}:{random_bytes.hex()}:{datetime.utcnow().isoformat()}"
        api_key = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return f"vk_{api_key[:32]}"  # Vendor key prefix
    
    async def submit_verification_documents(
        self,
        vendor_id: int,
        documents: Dict[str, str]  # document_type -> file_url
    ) -> bool:
        """Submit verification documents"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            # Validate document types
            for doc_type in documents.keys():
                if doc_type not in self.required_documents:
                    raise ValueError(f"Invalid document type: {doc_type}")
            
            # Update vendor documents
            existing_docs = vendor.verification_documents or {}
            existing_docs.update({
                doc_type: {
                    "url": url,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "status": "pending_review"
                }
                for doc_type, url in documents.items()
            })
            
            vendor.verification_documents = existing_docs
            
            # Update status if all required documents are submitted
            if self._all_documents_submitted(existing_docs):
                vendor.status = VendorStatus.UNDER_REVIEW
                
                # Trigger review process
                await self._trigger_document_review(vendor)
            
            self.db.commit()
            
            logger.info(f"Documents submitted for vendor {vendor.vendor_id}: {list(documents.keys())}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to submit documents for vendor {vendor_id}: {e}")
            raise
    
    def _all_documents_submitted(self, documents: Dict[str, Any]) -> bool:
        """Check if all required documents are submitted"""
        submitted_types = set(documents.keys())
        required_types = set(self.required_documents)
        return required_types.issubset(submitted_types)
    
    async def _trigger_document_review(self, vendor: Vendor):
        """Trigger document review process"""
        # In production, this would integrate with document review service
        # For now, we'll just log and could auto-approve for demo
        logger.info(f"Document review triggered for vendor {vendor.vendor_id}")
        
        # Auto-approve for demo (remove in production)
        if settings.auto_approve_vendors:
            await self.approve_vendor(vendor.id, "Auto-approved for demo")
    
    async def approve_vendor(
        self,
        vendor_id: int,
        approval_notes: Optional[str] = None
    ) -> bool:
        """Approve vendor after verification"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            # Update status
            vendor.status = VendorStatus.APPROVED
            vendor.verified = True
            vendor.approved_at = datetime.utcnow()
            
            # Update commission rate based on verification
            vendor.commission_rate = COMMISSION_TIERS["verified"]["rate"]
            vendor.minimum_payout = COMMISSION_TIERS["verified"]["minimum_payout"]
            
            # Add approval notes to metadata
            metadata = vendor.metadata or {}
            metadata["approval"] = {
                "approved_at": datetime.utcnow().isoformat(),
                "notes": approval_notes,
                "approved_by": "system"  # In production, track who approved
            }
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send approval notification
            await self._send_approval_notification(vendor)
            
            logger.info(f"Approved vendor {vendor.vendor_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to approve vendor {vendor_id}: {e}")
            raise
    
    async def reject_vendor(
        self,
        vendor_id: int,
        rejection_reason: str
    ) -> bool:
        """Reject vendor application"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            # Update status
            vendor.status = VendorStatus.REJECTED
            
            # Add rejection details to metadata
            metadata = vendor.metadata or {}
            metadata["rejection"] = {
                "rejected_at": datetime.utcnow().isoformat(),
                "reason": rejection_reason,
                "rejected_by": "system"
            }
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send rejection notification
            await self._send_rejection_notification(vendor, rejection_reason)
            
            logger.info(f"Rejected vendor {vendor.vendor_id}: {rejection_reason}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to reject vendor {vendor_id}: {e}")
            raise
    
    async def activate_vendor(self, vendor_id: int) -> bool:
        """Activate approved vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if vendor.status != VendorStatus.APPROVED:
                raise ValueError("Vendor must be approved before activation")
            
            vendor.status = VendorStatus.ACTIVE
            
            # Add activation to metadata
            metadata = vendor.metadata or {}
            metadata["activation"] = {
                "activated_at": datetime.utcnow().isoformat(),
                "activated_by": "system"
            }
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send activation notification
            await self._send_activation_notification(vendor)
            
            logger.info(f"Activated vendor {vendor.vendor_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to activate vendor {vendor_id}: {e}")
            raise
    
    async def suspend_vendor(
        self,
        vendor_id: int,
        suspension_reason: str,
        temporary: bool = True
    ) -> bool:
        """Suspend vendor account"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            # Store previous status for potential reinstatement
            metadata = vendor.metadata or {}
            metadata["suspension"] = {
                "suspended_at": datetime.utcnow().isoformat(),
                "reason": suspension_reason,
                "temporary": temporary,
                "previous_status": vendor.status.value,
                "suspended_by": "system"
            }
            vendor.metadata = metadata
            
            vendor.status = VendorStatus.SUSPENDED
            
            self.db.commit()
            
            # Send suspension notification
            await self._send_suspension_notification(vendor, suspension_reason, temporary)
            
            logger.info(f"Suspended vendor {vendor.vendor_id}: {suspension_reason}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to suspend vendor {vendor_id}: {e}")
            raise
    
    async def reinstate_vendor(self, vendor_id: int) -> bool:
        """Reinstate suspended vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if vendor.status != VendorStatus.SUSPENDED:
                raise ValueError("Vendor is not suspended")
            
            # Restore previous status
            metadata = vendor.metadata or {}
            suspension_info = metadata.get("suspension", {})
            previous_status = suspension_info.get("previous_status", VendorStatus.ACTIVE.value)
            
            vendor.status = VendorStatus(previous_status)
            
            # Add reinstatement to metadata
            metadata["reinstatement"] = {
                "reinstated_at": datetime.utcnow().isoformat(),
                "reinstated_by": "system"
            }
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send reinstatement notification
            await self._send_reinstatement_notification(vendor)
            
            logger.info(f"Reinstated vendor {vendor.vendor_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to reinstate vendor {vendor_id}: {e}")
            raise
    
    async def update_commission_rate(
        self,
        vendor_id: int,
        new_rate: float,
        reason: Optional[str] = None
    ) -> bool:
        """Update vendor commission rate"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if not 0.0 <= new_rate <= 1.0:
                raise ValueError("Commission rate must be between 0.0 and 1.0")
            
            old_rate = vendor.commission_rate
            vendor.commission_rate = new_rate
            
            # Track rate change in metadata
            metadata = vendor.metadata or {}
            rate_changes = metadata.get("rate_changes", [])
            rate_changes.append({
                "changed_at": datetime.utcnow().isoformat(),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "reason": reason,
                "changed_by": "system"
            })
            metadata["rate_changes"] = rate_changes
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send rate change notification
            await self._send_rate_change_notification(vendor, old_rate, new_rate, reason)
            
            logger.info(f"Updated commission rate for vendor {vendor.vendor_id}: {old_rate} -> {new_rate}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update commission rate for vendor {vendor_id}: {e}")
            raise
    
    async def regenerate_api_key(self, vendor_id: int) -> str:
        """Regenerate API key for vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            old_key = vendor.api_key
            new_key = self._generate_api_key(vendor.vendor_id)
            
            vendor.api_key = new_key
            
            # Track key regeneration in metadata
            metadata = vendor.metadata or {}
            key_changes = metadata.get("api_key_changes", [])
            key_changes.append({
                "changed_at": datetime.utcnow().isoformat(),
                "old_key_prefix": old_key[:10] + "..." if old_key else None,
                "new_key_prefix": new_key[:10] + "...",
                "reason": "regenerated"
            })
            metadata["api_key_changes"] = key_changes
            vendor.metadata = metadata
            
            self.db.commit()
            
            # Send API key notification
            await self._send_api_key_notification(vendor, new_key)
            
            logger.info(f"Regenerated API key for vendor {vendor.vendor_id}")
            return new_key
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to regenerate API key for vendor {vendor_id}: {e}")
            raise
    
    async def get_vendor_analytics(
        self,
        vendor_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics for a vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get revenue shares for the period
            from .models import RevenueShare
            revenue_shares = self.db.query(RevenueShare).filter(
                RevenueShare.vendor_id == vendor_id,
                RevenueShare.created_at >= start_date,
                RevenueShare.created_at <= end_date
            ).all()
            
            # Calculate metrics
            total_revenue = sum(rs.gross_revenue for rs in revenue_shares)
            total_commission = sum(rs.platform_commission for rs in revenue_shares)
            total_vendor_share = sum(rs.vendor_share for rs in revenue_shares)
            
            # Get API endpoint metrics
            from .models import APIEndpoint, APIUsageRecord
            endpoints = self.db.query(APIEndpoint).filter(
                APIEndpoint.vendor_id == vendor_id
            ).all()
            
            endpoint_metrics = {}
            for endpoint in endpoints:
                usage_records = self.db.query(APIUsageRecord).filter(
                    APIUsageRecord.endpoint_id == endpoint.id,
                    APIUsageRecord.created_at >= start_date,
                    APIUsageRecord.created_at <= end_date
                ).all()
                
                endpoint_metrics[endpoint.endpoint_id] = {
                    "total_requests": len(usage_records),
                    "successful_requests": len([r for r in usage_records if r.success]),
                    "total_revenue": sum(r.total_amount for r in usage_records),
                    "average_response_time": sum(r.response_time_ms for r in usage_records if r.response_time_ms) / len(usage_records) if usage_records else 0
                }
            
            return {
                "vendor": {
                    "id": vendor.id,
                    "vendor_id": vendor.vendor_id,
                    "company_name": vendor.company_name,
                    "status": vendor.status.value,
                    "commission_rate": vendor.commission_rate
                },
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "revenue_metrics": {
                    "total_revenue": total_revenue,
                    "platform_commission": total_commission,
                    "vendor_share": total_vendor_share,
                    "transaction_count": len(revenue_shares)
                },
                "endpoint_metrics": endpoint_metrics,
                "performance": {
                    "active_endpoints": len([e for e in endpoints if e.enabled]),
                    "total_endpoints": len(endpoints)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get vendor analytics: {e}")
            return {"error": str(e)}
    
    async def get_vendor_list(
        self,
        status: Optional[VendorStatus] = None,
        verified_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[Vendor]:
        """Get list of vendors with filters"""
        query = self.db.query(Vendor)
        
        if status:
            query = query.filter(Vendor.status == status)
        
        if verified_only:
            query = query.filter(Vendor.verified == True)
        
        return query.order_by(Vendor.created_at.desc()).offset(offset).limit(limit).all()
    
    # Email notification methods (mock implementations)
    async def _send_onboarding_email(self, vendor: Vendor):
        """Send onboarding welcome email"""
        logger.info(f"Would send onboarding email to {vendor.contact_email}")
    
    async def _send_approval_notification(self, vendor: Vendor):
        """Send vendor approval notification"""
        logger.info(f"Would send approval notification to {vendor.contact_email}")
    
    async def _send_rejection_notification(self, vendor: Vendor, reason: str):
        """Send vendor rejection notification"""
        logger.info(f"Would send rejection notification to {vendor.contact_email}: {reason}")
    
    async def _send_activation_notification(self, vendor: Vendor):
        """Send vendor activation notification"""
        logger.info(f"Would send activation notification to {vendor.contact_email}")
    
    async def _send_suspension_notification(self, vendor: Vendor, reason: str, temporary: bool):
        """Send vendor suspension notification"""
        logger.info(f"Would send suspension notification to {vendor.contact_email}: {reason}")
    
    async def _send_reinstatement_notification(self, vendor: Vendor):
        """Send vendor reinstatement notification"""
        logger.info(f"Would send reinstatement notification to {vendor.contact_email}")
    
    async def _send_rate_change_notification(self, vendor: Vendor, old_rate: float, new_rate: float, reason: Optional[str]):
        """Send commission rate change notification"""
        logger.info(f"Would send rate change notification to {vendor.contact_email}: {old_rate} -> {new_rate}")
    
    async def _send_api_key_notification(self, vendor: Vendor, new_key: str):
        """Send new API key notification"""
        logger.info(f"Would send API key notification to {vendor.contact_email}")