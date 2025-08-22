"""
Revenue sharing and payout management system.

This module provides comprehensive revenue sharing including:
- Commission calculation and tracking
- Automated payout generation
- Multi-payment method support
- Reconciliation and reporting
"""

import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..core.config import get_settings
from .models import (
    Vendor, RevenueShare, VendorPayout, APIUsageRecord,
    VendorStatus, PayoutStatus, calculate_vendor_share,
    get_commission_tier, COMMISSION_TIERS
)

logger = logging.getLogger(__name__)
settings = get_settings()


class RevenueShareManager:
    """Manage revenue sharing between platform and vendors"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_revenue_share(
        self,
        vendor_id: int,
        user_id: int,
        gross_revenue: float,
        source_type: str,
        source_id: str,
        payment_id: Optional[int] = None,
        api_usage_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RevenueShare:
        """Create a revenue share record"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if vendor.status not in [VendorStatus.ACTIVE, VendorStatus.APPROVED]:
                raise ValueError(f"Vendor {vendor_id} is not active")
            
            # Calculate commission and vendor share
            commission_rate = vendor.commission_rate
            platform_commission, vendor_share = calculate_vendor_share(gross_revenue, commission_rate)
            
            # Create revenue share record
            revenue_share = RevenueShare(
                vendor_id=vendor_id,
                user_id=user_id,
                gross_revenue=gross_revenue,
                platform_commission=platform_commission,
                vendor_share=vendor_share,
                commission_rate=commission_rate,
                source_type=source_type,
                source_id=source_id,
                payment_id=payment_id,
                api_usage_id=api_usage_id,
                metadata=metadata or {}
            )
            
            self.db.add(revenue_share)
            
            # Update vendor totals
            vendor.total_revenue += gross_revenue
            vendor.total_commission += platform_commission
            
            self.db.commit()
            
            logger.info(
                f"Created revenue share: vendor={vendor_id}, revenue=${gross_revenue:.2f}, "
                f"commission=${platform_commission:.2f}, vendor_share=${vendor_share:.2f}"
            )
            
            # Check if vendor qualifies for commission tier upgrade
            await self._check_commission_tier_upgrade(vendor)
            
            return revenue_share
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create revenue share: {e}")
            raise
    
    async def process_api_usage_revenue(
        self,
        api_usage_record: APIUsageRecord
    ) -> Optional[RevenueShare]:
        """Process revenue sharing for API usage"""
        try:
            # Get the vendor from the API endpoint
            from .models import APIEndpoint
            endpoint = self.db.query(APIEndpoint).filter(
                APIEndpoint.id == api_usage_record.endpoint_id
            ).first()
            
            if not endpoint:
                logger.warning(f"API endpoint {api_usage_record.endpoint_id} not found")
                return None
            
            vendor = endpoint.vendor
            if vendor.status not in [VendorStatus.ACTIVE, VendorStatus.APPROVED]:
                logger.warning(f"Vendor {vendor.id} is not active for API usage {api_usage_record.id}")
                return None
            
            # Create revenue share
            revenue_share = await self.create_revenue_share(
                vendor_id=vendor.id,
                user_id=api_usage_record.user_id,
                gross_revenue=api_usage_record.total_amount,
                source_type="api_usage",
                source_id=str(api_usage_record.id),
                api_usage_id=api_usage_record.id,
                metadata={
                    "endpoint_id": endpoint.endpoint_id,
                    "endpoint_name": endpoint.name,
                    "billable_units": api_usage_record.billable_units,
                    "unit_price": api_usage_record.unit_price
                }
            )
            
            # Update endpoint revenue
            endpoint.total_revenue += api_usage_record.total_amount
            self.db.commit()
            
            return revenue_share
            
        except Exception as e:
            logger.error(f"Failed to process API usage revenue: {e}")
            return None
    
    async def process_subscription_revenue(
        self,
        vendor_id: int,
        subscription_id: int,
        payment_id: int,
        amount: float
    ) -> Optional[RevenueShare]:
        """Process revenue sharing for subscription payments"""
        try:
            # Get payment information
            from ..payments.models import Payment
            payment = self.db.query(Payment).filter(Payment.id == payment_id).first()
            
            if not payment:
                raise ValueError(f"Payment {payment_id} not found")
            
            revenue_share = await self.create_revenue_share(
                vendor_id=vendor_id,
                user_id=payment.user_id,
                gross_revenue=amount,
                source_type="subscription",
                source_id=str(subscription_id),
                payment_id=payment_id,
                metadata={
                    "subscription_id": subscription_id,
                    "payment_method": payment.payment_method.value,
                    "payment_provider": payment.provider.value
                }
            )
            
            return revenue_share
            
        except Exception as e:
            logger.error(f"Failed to process subscription revenue: {e}")
            return None
    
    async def _check_commission_tier_upgrade(self, vendor: Vendor):
        """Check if vendor qualifies for commission tier upgrade"""
        try:
            # Calculate monthly revenue for the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            monthly_revenue = self.db.query(func.sum(RevenueShare.gross_revenue)).filter(
                RevenueShare.vendor_id == vendor.id,
                RevenueShare.created_at >= thirty_days_ago
            ).scalar() or 0.0
            
            # Determine appropriate tier
            new_tier = get_commission_tier(monthly_revenue)
            new_rate = COMMISSION_TIERS[new_tier]["rate"]
            
            # Check if upgrade is needed
            if new_rate < vendor.commission_rate:
                from .vendor_onboarding import VendorOnboardingManager
                onboarding_manager = VendorOnboardingManager(self.db)
                
                await onboarding_manager.update_commission_rate(
                    vendor.id,
                    new_rate,
                    reason=f"Automatic tier upgrade to {new_tier} based on monthly revenue of ${monthly_revenue:.2f}"
                )
                
                logger.info(f"Upgraded vendor {vendor.vendor_id} to {new_tier} tier")
            
        except Exception as e:
            logger.error(f"Failed to check commission tier upgrade: {e}")
    
    async def get_vendor_revenue_summary(
        self,
        vendor_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get revenue summary for a vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get revenue shares for the period
            revenue_shares = self.db.query(RevenueShare).filter(
                RevenueShare.vendor_id == vendor_id,
                RevenueShare.created_at >= start_date,
                RevenueShare.created_at <= end_date
            ).all()
            
            # Calculate totals
            total_gross_revenue = sum(rs.gross_revenue for rs in revenue_shares)
            total_platform_commission = sum(rs.platform_commission for rs in revenue_shares)
            total_vendor_share = sum(rs.vendor_share for rs in revenue_shares)
            
            # Break down by source type
            source_breakdown = {}
            for rs in revenue_shares:
                source_type = rs.source_type
                if source_type not in source_breakdown:
                    source_breakdown[source_type] = {
                        "count": 0,
                        "gross_revenue": 0.0,
                        "platform_commission": 0.0,
                        "vendor_share": 0.0
                    }
                
                source_breakdown[source_type]["count"] += 1
                source_breakdown[source_type]["gross_revenue"] += rs.gross_revenue
                source_breakdown[source_type]["platform_commission"] += rs.platform_commission
                source_breakdown[source_type]["vendor_share"] += rs.vendor_share
            
            # Calculate pending payout amount
            pending_revenue_shares = self.db.query(RevenueShare).filter(
                RevenueShare.vendor_id == vendor_id,
                RevenueShare.processed == False
            ).all()
            
            pending_payout = sum(rs.vendor_share for rs in pending_revenue_shares)
            
            return {
                "vendor": {
                    "id": vendor.id,
                    "vendor_id": vendor.vendor_id,
                    "company_name": vendor.company_name,
                    "commission_rate": vendor.commission_rate,
                    "minimum_payout": vendor.minimum_payout
                },
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "gross_revenue": total_gross_revenue,
                    "platform_commission": total_platform_commission,
                    "vendor_share": total_vendor_share,
                    "transaction_count": len(revenue_shares)
                },
                "source_breakdown": source_breakdown,
                "pending_payout": pending_payout,
                "next_payout_eligible": pending_payout >= vendor.minimum_payout
            }
            
        except Exception as e:
            logger.error(f"Failed to get vendor revenue summary: {e}")
            return {"error": str(e)}
    
    async def get_revenue_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get platform-wide revenue analytics"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get all revenue shares for the period
            revenue_shares = self.db.query(RevenueShare).filter(
                RevenueShare.created_at >= start_date,
                RevenueShare.created_at <= end_date
            ).all()
            
            # Calculate totals
            total_gross_revenue = sum(rs.gross_revenue for rs in revenue_shares)
            total_platform_commission = sum(rs.platform_commission for rs in revenue_shares)
            total_vendor_share = sum(rs.vendor_share for rs in revenue_shares)
            
            # Vendor breakdown
            vendor_revenue = {}
            for rs in revenue_shares:
                vendor_id = rs.vendor_id
                if vendor_id not in vendor_revenue:
                    vendor_revenue[vendor_id] = {
                        "gross_revenue": 0.0,
                        "platform_commission": 0.0,
                        "vendor_share": 0.0,
                        "transaction_count": 0
                    }
                
                vendor_revenue[vendor_id]["gross_revenue"] += rs.gross_revenue
                vendor_revenue[vendor_id]["platform_commission"] += rs.platform_commission
                vendor_revenue[vendor_id]["vendor_share"] += rs.vendor_share
                vendor_revenue[vendor_id]["transaction_count"] += 1
            
            # Source type breakdown
            source_breakdown = {}
            for rs in revenue_shares:
                source_type = rs.source_type
                if source_type not in source_breakdown:
                    source_breakdown[source_type] = {
                        "count": 0,
                        "gross_revenue": 0.0,
                        "platform_commission": 0.0,
                        "vendor_share": 0.0
                    }
                
                source_breakdown[source_type]["count"] += 1
                source_breakdown[source_type]["gross_revenue"] += rs.gross_revenue
                source_breakdown[source_type]["platform_commission"] += rs.platform_commission
                source_breakdown[source_type]["vendor_share"] += rs.vendor_share
            
            # Top vendors by revenue
            top_vendors = sorted(
                vendor_revenue.items(),
                key=lambda x: x[1]["gross_revenue"],
                reverse=True
            )[:10]
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "gross_revenue": total_gross_revenue,
                    "platform_commission": total_platform_commission,
                    "vendor_share": total_vendor_share,
                    "transaction_count": len(revenue_shares),
                    "unique_vendors": len(vendor_revenue)
                },
                "source_breakdown": source_breakdown,
                "top_vendors": [
                    {
                        "vendor_id": vendor_id,
                        "metrics": metrics
                    }
                    for vendor_id, metrics in top_vendors
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get revenue analytics: {e}")
            return {"error": str(e)}


class PayoutManager:
    """Manage vendor payouts"""
    
    def __init__(self, db: Session):
        self.db = db
        self.payout_processors = {
            "bank_transfer": self._process_bank_transfer,
            "paypal": self._process_paypal_payout,
            "stripe_connect": self._process_stripe_connect,
            "check": self._process_check_payout
        }
    
    async def generate_vendor_payouts(
        self,
        vendor_ids: Optional[List[int]] = None,
        force_minimum: bool = False
    ) -> Dict[str, Any]:
        """Generate payouts for eligible vendors"""
        try:
            # Get eligible vendors
            if vendor_ids:
                vendors = self.db.query(Vendor).filter(
                    Vendor.id.in_(vendor_ids),
                    Vendor.status == VendorStatus.ACTIVE
                ).all()
            else:
                vendors = self.db.query(Vendor).filter(
                    Vendor.status == VendorStatus.ACTIVE
                ).all()
            
            generated_payouts = []
            total_payout_amount = 0.0
            
            for vendor in vendors:
                try:
                    payout = await self._generate_vendor_payout(vendor, force_minimum)
                    if payout:
                        generated_payouts.append(payout)
                        total_payout_amount += payout.amount
                        
                except Exception as e:
                    logger.error(f"Failed to generate payout for vendor {vendor.id}: {e}")
                    continue
            
            logger.info(
                f"Generated {len(generated_payouts)} payouts totaling ${total_payout_amount:.2f}"
            )
            
            return {
                "generated_count": len(generated_payouts),
                "total_amount": total_payout_amount,
                "payout_ids": [payout.payout_id for payout in generated_payouts]
            }
            
        except Exception as e:
            logger.error(f"Failed to generate vendor payouts: {e}")
            return {"error": str(e)}
    
    async def _generate_vendor_payout(
        self,
        vendor: Vendor,
        force_minimum: bool = False
    ) -> Optional[VendorPayout]:
        """Generate payout for a single vendor"""
        try:
            # Get unprocessed revenue shares
            unprocessed_shares = self.db.query(RevenueShare).filter(
                RevenueShare.vendor_id == vendor.id,
                RevenueShare.processed == False
            ).all()
            
            if not unprocessed_shares:
                return None
            
            # Calculate total payout amount
            total_amount = sum(rs.vendor_share for rs in unprocessed_shares)
            
            # Check minimum payout threshold
            if not force_minimum and total_amount < vendor.minimum_payout:
                logger.debug(
                    f"Vendor {vendor.vendor_id} payout amount ${total_amount:.2f} "
                    f"below minimum ${vendor.minimum_payout:.2f}"
                )
                return None
            
            # Determine payout period
            earliest_share = min(unprocessed_shares, key=lambda rs: rs.created_at)
            latest_share = max(unprocessed_shares, key=lambda rs: rs.created_at)
            
            # Calculate breakdown
            gross_revenue = sum(rs.gross_revenue for rs in unprocessed_shares)
            platform_commission = sum(rs.platform_commission for rs in unprocessed_shares)
            
            # Calculate processing fee (e.g., 2% for bank transfers)
            processing_fee = total_amount * 0.02
            final_amount = total_amount - processing_fee
            
            # Generate payout ID
            payout_id = f"PO_{datetime.utcnow().strftime('%Y%m%d')}_{vendor.vendor_id}_{uuid.uuid4().hex[:8]}"
            
            # Create payout record
            payout = VendorPayout(
                payout_id=payout_id,
                vendor_id=vendor.id,
                amount=final_amount,
                payout_method=self._get_vendor_payout_method(vendor),
                period_start=earliest_share.created_at,
                period_end=latest_share.created_at,
                total_revenue_shares=len(unprocessed_shares),
                gross_revenue=gross_revenue,
                platform_commission=platform_commission,
                processing_fee=processing_fee,
                metadata={
                    "original_amount": total_amount,
                    "fee_rate": 0.02,
                    "revenue_share_ids": [rs.id for rs in unprocessed_shares]
                }
            )
            
            self.db.add(payout)
            self.db.flush()
            
            # Mark revenue shares as processed
            for rs in unprocessed_shares:
                rs.processed = True
                rs.processed_at = datetime.utcnow()
                rs.payout_id = payout.id
            
            # Update vendor last payout date
            vendor.last_payout_at = datetime.utcnow()
            vendor.total_payouts += final_amount
            
            self.db.commit()
            
            logger.info(
                f"Generated payout {payout_id} for vendor {vendor.vendor_id}: ${final_amount:.2f}"
            )
            
            return payout
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to generate payout for vendor {vendor.id}: {e}")
            raise
    
    def _get_vendor_payout_method(self, vendor: Vendor) -> str:
        """Get preferred payout method for vendor"""
        bank_info = vendor.bank_account_info or {}
        
        if bank_info.get("type") == "paypal":
            return "paypal"
        elif bank_info.get("type") == "stripe_connect":
            return "stripe_connect"
        elif bank_info.get("routing_number") and bank_info.get("account_number"):
            return "bank_transfer"
        else:
            return "check"  # Default fallback
    
    async def process_payouts(
        self,
        payout_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process pending payouts"""
        try:
            # Get pending payouts
            query = self.db.query(VendorPayout).filter(
                VendorPayout.status == PayoutStatus.PENDING
            )
            
            if payout_ids:
                query = query.filter(VendorPayout.payout_id.in_(payout_ids))
            
            pending_payouts = query.all()
            
            processed_count = 0
            failed_count = 0
            total_processed_amount = 0.0
            
            for payout in pending_payouts:
                try:
                    success = await self._process_single_payout(payout)
                    if success:
                        processed_count += 1
                        total_processed_amount += payout.amount
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process payout {payout.payout_id}: {e}")
                    failed_count += 1
                    continue
            
            logger.info(
                f"Processed {processed_count} payouts (${total_processed_amount:.2f}), "
                f"{failed_count} failed"
            )
            
            return {
                "processed": processed_count,
                "failed": failed_count,
                "total_amount": total_processed_amount
            }
            
        except Exception as e:
            logger.error(f"Failed to process payouts: {e}")
            return {"error": str(e)}
    
    async def _process_single_payout(self, payout: VendorPayout) -> bool:
        """Process a single payout"""
        try:
            payout.status = PayoutStatus.PROCESSING
            self.db.commit()
            
            # Get processor for payout method
            processor = self.payout_processors.get(payout.payout_method)
            if not processor:
                raise ValueError(f"No processor for payout method: {payout.payout_method}")
            
            # Process payout
            result = await processor(payout)
            
            if result["success"]:
                payout.status = PayoutStatus.COMPLETED
                payout.processed_at = datetime.utcnow()
                payout.external_payout_id = result.get("external_id")
                
                # Send confirmation notification
                await self._send_payout_confirmation(payout)
                
            else:
                payout.status = PayoutStatus.FAILED
                payout.failure_reason = result.get("error", "Unknown error")
                payout.failure_code = result.get("code")
                payout.retry_count += 1
                
                # Send failure notification
                await self._send_payout_failure_notification(payout)
            
            self.db.commit()
            return result["success"]
            
        except Exception as e:
            payout.status = PayoutStatus.FAILED
            payout.failure_reason = str(e)
            payout.retry_count += 1
            self.db.commit()
            
            logger.error(f"Failed to process payout {payout.payout_id}: {e}")
            return False
    
    # Payout processors (mock implementations)
    async def _process_bank_transfer(self, payout: VendorPayout) -> Dict[str, Any]:
        """Process bank transfer payout"""
        try:
            vendor = payout.vendor
            bank_info = vendor.bank_account_info or {}
            
            # Validate bank information
            if not bank_info.get("routing_number") or not bank_info.get("account_number"):
                return {"success": False, "error": "Invalid bank account information"}
            
            # In production, integrate with banking API (e.g., Stripe Connect, Dwolla, etc.)
            # Mock successful transfer
            external_id = f"ACH_{uuid.uuid4().hex[:12]}"
            
            logger.info(
                f"Mock bank transfer: ${payout.amount:.2f} to account "
                f"****{bank_info.get('account_number', '')[-4:]}"
            )
            
            return {"success": True, "external_id": external_id, "method": "bank_transfer"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_paypal_payout(self, payout: VendorPayout) -> Dict[str, Any]:
        """Process PayPal payout"""
        try:
            vendor = payout.vendor
            bank_info = vendor.bank_account_info or {}
            
            paypal_email = bank_info.get("paypal_email")
            if not paypal_email:
                return {"success": False, "error": "PayPal email not configured"}
            
            # In production, integrate with PayPal Payouts API
            external_id = f"PP_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Mock PayPal payout: ${payout.amount:.2f} to {paypal_email}")
            
            return {"success": True, "external_id": external_id, "method": "paypal"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_stripe_connect(self, payout: VendorPayout) -> Dict[str, Any]:
        """Process Stripe Connect payout"""
        try:
            vendor = payout.vendor
            bank_info = vendor.bank_account_info or {}
            
            stripe_account_id = bank_info.get("stripe_account_id")
            if not stripe_account_id:
                return {"success": False, "error": "Stripe Connect account not configured"}
            
            # In production, use Stripe Connect API
            external_id = f"TR_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Mock Stripe Connect payout: ${payout.amount:.2f} to {stripe_account_id}")
            
            return {"success": True, "external_id": external_id, "method": "stripe_connect"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_check_payout(self, payout: VendorPayout) -> Dict[str, Any]:
        """Process check payout"""
        try:
            vendor = payout.vendor
            
            # In production, integrate with check printing service
            check_number = f"CHK_{datetime.utcnow().strftime('%Y%m%d')}_{payout.id:06d}"
            
            logger.info(f"Mock check payout: ${payout.amount:.2f} check #{check_number}")
            
            return {"success": True, "external_id": check_number, "method": "check"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_payout_confirmation(self, payout: VendorPayout):
        """Send payout confirmation notification"""
        vendor = payout.vendor
        logger.info(
            f"Would send payout confirmation to {vendor.contact_email}: "
            f"${payout.amount:.2f} via {payout.payout_method}"
        )
    
    async def _send_payout_failure_notification(self, payout: VendorPayout):
        """Send payout failure notification"""
        vendor = payout.vendor
        logger.info(
            f"Would send payout failure notification to {vendor.contact_email}: "
            f"${payout.amount:.2f} failed - {payout.failure_reason}"
        )
    
    async def get_payout_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get payout analytics"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get payouts for the period
            payouts = self.db.query(VendorPayout).filter(
                VendorPayout.created_at >= start_date,
                VendorPayout.created_at <= end_date
            ).all()
            
            # Calculate totals by status
            status_breakdown = {}
            for payout in payouts:
                status = payout.status.value
                if status not in status_breakdown:
                    status_breakdown[status] = {
                        "count": 0,
                        "total_amount": 0.0
                    }
                
                status_breakdown[status]["count"] += 1
                status_breakdown[status]["total_amount"] += payout.amount
            
            # Method breakdown
            method_breakdown = {}
            for payout in payouts:
                method = payout.payout_method
                if method not in method_breakdown:
                    method_breakdown[method] = {
                        "count": 0,
                        "total_amount": 0.0
                    }
                
                method_breakdown[method]["count"] += 1
                method_breakdown[method]["total_amount"] += payout.amount
            
            total_amount = sum(payout.amount for payout in payouts)
            total_fees = sum(payout.processing_fee for payout in payouts)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "payout_count": len(payouts),
                    "total_amount": total_amount,
                    "total_fees": total_fees,
                    "net_amount": total_amount - total_fees
                },
                "status_breakdown": status_breakdown,
                "method_breakdown": method_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to get payout analytics: {e}")
            return {"error": str(e)}