"""
Enterprise invoicing system for the arbitration RAG API.

This module provides comprehensive invoicing capabilities including:
- Invoice generation and management
- PDF generation
- Payment tracking
- Tax calculations
- Enterprise billing workflows
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid

from ..core.config import get_settings
from .models import (
    Invoice, Payment, Subscription, UsageRecord, User,
    PaymentStatus, SubscriptionTier, InvoiceLineItem,
    InvoiceCreate, InvoiceResponse
)
from .tax_calculation import TaxCalculator
from .pdf_generator import PDFInvoiceGenerator

logger = logging.getLogger(__name__)
settings = get_settings()


class InvoiceManager:
    """Comprehensive invoice management system"""
    
    def __init__(self, db: Session):
        self.db = db
        self.tax_calculator = TaxCalculator()
        self.pdf_generator = PDFInvoiceGenerator()
    
    async def create_invoice(
        self,
        user_id: int,
        line_items: List[InvoiceLineItem],
        due_date: Optional[datetime] = None,
        subscription_id: Optional[int] = None,
        billing_period_start: Optional[datetime] = None,
        billing_period_end: Optional[datetime] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Invoice:
        """Create a new invoice"""
        try:
            # Get user information
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Calculate totals
            subtotal = sum(item.amount for item in line_items)
            
            # Calculate taxes
            tax_amount = await self.tax_calculator.calculate_tax(
                subtotal, 
                user.organization or "US"  # Default to US if no organization
            )
            
            # Apply any discounts (future enhancement)
            discount_amount = 0.0
            
            total_amount = subtotal + tax_amount - discount_amount
            
            # Generate invoice number
            invoice_number = await self._generate_invoice_number()
            
            # Set due date (default 30 days)
            if not due_date:
                due_date = datetime.utcnow() + timedelta(days=30)
            
            # Capture customer data snapshot
            customer_data = {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "organization": user.organization,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
            
            # Create invoice
            invoice = Invoice(
                invoice_number=invoice_number,
                user_id=user_id,
                subscription_id=subscription_id,
                subtotal=subtotal,
                tax_amount=tax_amount,
                discount_amount=discount_amount,
                total_amount=total_amount,
                status="draft",
                due_date=due_date,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                customer_data=customer_data,
                line_items=[item.dict() for item in line_items],
                notes=notes,
                metadata=metadata or {}
            )
            
            self.db.add(invoice)
            self.db.flush()
            
            # Generate PDF
            await self._generate_invoice_pdf(invoice)
            
            self.db.commit()
            
            logger.info(f"Created invoice {invoice_number} for user {user_id}")
            return invoice
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Invoice creation failed: {e}")
            raise
    
    async def create_subscription_invoice(
        self,
        subscription_id: int,
        billing_period_start: datetime,
        billing_period_end: datetime
    ) -> Invoice:
        """Create invoice for subscription billing period"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            # Create base subscription line item
            line_items = [
                InvoiceLineItem(
                    description=f"{subscription.tier.value.title()} Subscription",
                    quantity=1,
                    unit_price=subscription.amount,
                    amount=subscription.amount
                )
            ]
            
            # Add usage-based charges if applicable
            usage_charges = await self._calculate_usage_charges(
                subscription_id, billing_period_start, billing_period_end
            )
            line_items.extend(usage_charges)
            
            # Calculate proration if needed
            proration_charges = await self._calculate_proration_charges(
                subscription_id, billing_period_start, billing_period_end
            )
            line_items.extend(proration_charges)
            
            return await self.create_invoice(
                user_id=subscription.user_id,
                line_items=line_items,
                subscription_id=subscription_id,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                notes=f"Billing period: {billing_period_start.date()} to {billing_period_end.date()}",
                metadata={
                    "subscription_tier": subscription.tier.value,
                    "billing_interval": subscription.billing_interval
                }
            )
            
        except Exception as e:
            logger.error(f"Subscription invoice creation failed: {e}")
            raise
    
    async def _calculate_usage_charges(
        self,
        subscription_id: int,
        period_start: datetime,
        period_end: datetime
    ) -> List[InvoiceLineItem]:
        """Calculate usage-based charges for billing period"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription or subscription.tier == SubscriptionTier.BUSINESS:
                # Business tier has unlimited usage
                return []
            
            # Get usage records for the period
            usage_records = self.db.query(UsageRecord).filter(
                UsageRecord.subscription_id == subscription_id,
                UsageRecord.timestamp >= period_start,
                UsageRecord.timestamp < period_end
            ).all()
            
            # Group by resource type
            usage_by_resource = {}
            for record in usage_records:
                resource_type = record.resource_type
                if resource_type not in usage_by_resource:
                    usage_by_resource[resource_type] = 0
                usage_by_resource[resource_type] += record.quantity
            
            line_items = []
            
            # Calculate overage charges
            for resource_type, usage_count in usage_by_resource.items():
                overage_charge = await self._calculate_overage_charge(
                    subscription.tier, resource_type, usage_count
                )
                
                if overage_charge > 0:
                    line_items.append(
                        InvoiceLineItem(
                            description=f"{resource_type.title()} Overage",
                            quantity=usage_count - self._get_included_limit(subscription.tier, resource_type),
                            unit_price=self._get_overage_rate(resource_type),
                            amount=overage_charge
                        )
                    )
            
            return line_items
            
        except Exception as e:
            logger.error(f"Usage charge calculation failed: {e}")
            return []
    
    async def _calculate_proration_charges(
        self,
        subscription_id: int,
        period_start: datetime,
        period_end: datetime
    ) -> List[InvoiceLineItem]:
        """Calculate proration charges for subscription changes"""
        # For now, return empty list - proration is handled by payment providers
        return []
    
    def _get_included_limit(self, tier: SubscriptionTier, resource_type: str) -> int:
        """Get included usage limit for subscription tier"""
        limits = {
            SubscriptionTier.FREE: {"documents": 10, "api_calls": 100},
            SubscriptionTier.PROFESSIONAL: {"documents": 500, "api_calls": 5000},
            SubscriptionTier.BUSINESS: {"documents": float('inf'), "api_calls": 50000},
            SubscriptionTier.ENTERPRISE: {"documents": float('inf'), "api_calls": float('inf')}
        }
        return limits.get(tier, {}).get(resource_type, 0)
    
    def _get_overage_rate(self, resource_type: str) -> float:
        """Get overage rate per unit"""
        rates = {
            "documents": 0.10,  # $0.10 per document
            "api_calls": 0.001  # $0.001 per API call
        }
        return rates.get(resource_type, 0)
    
    async def _calculate_overage_charge(
        self,
        tier: SubscriptionTier,
        resource_type: str,
        usage_count: int
    ) -> float:
        """Calculate overage charge for resource type"""
        included_limit = self._get_included_limit(tier, resource_type)
        
        if usage_count <= included_limit:
            return 0.0
        
        overage_units = usage_count - included_limit
        overage_rate = self._get_overage_rate(resource_type)
        
        return overage_units * overage_rate
    
    async def _generate_invoice_number(self) -> str:
        """Generate unique invoice number"""
        # Get current year and month
        now = datetime.utcnow()
        year_month = now.strftime("%Y%m")
        
        # Get next sequence number for this month
        latest_invoice = self.db.query(Invoice).filter(
            Invoice.invoice_number.like(f"INV-{year_month}-%")
        ).order_by(Invoice.id.desc()).first()
        
        if latest_invoice:
            # Extract sequence number and increment
            parts = latest_invoice.invoice_number.split("-")
            sequence = int(parts[-1]) + 1
        else:
            sequence = 1
        
        return f"INV-{year_month}-{sequence:04d}"
    
    async def _generate_invoice_pdf(self, invoice: Invoice):
        """Generate PDF for invoice"""
        try:
            pdf_content = await self.pdf_generator.generate_invoice_pdf(invoice)
            
            # Save PDF to storage (local file system for now)
            pdf_filename = f"invoice_{invoice.invoice_number}.pdf"
            pdf_path = os.path.join(settings.upload_directory, "invoices", pdf_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            
            # Write PDF
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)
            
            # Update invoice with PDF URL
            invoice.pdf_url = f"/api/invoices/{invoice.id}/pdf"
            
            logger.info(f"Generated PDF for invoice {invoice.invoice_number}")
            
        except Exception as e:
            logger.error(f"PDF generation failed for invoice {invoice.invoice_number}: {e}")
    
    async def send_invoice(self, invoice_id: int) -> bool:
        """Send invoice to customer"""
        try:
            invoice = self.db.query(Invoice).filter(Invoice.id == invoice_id).first()
            if not invoice:
                raise ValueError(f"Invoice {invoice_id} not found")
            
            # Update status
            invoice.status = "sent"
            self.db.commit()
            
            # Send email (implement email service)
            await self._send_invoice_email(invoice)
            
            logger.info(f"Sent invoice {invoice.invoice_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send invoice {invoice_id}: {e}")
            return False
    
    async def _send_invoice_email(self, invoice: Invoice):
        """Send invoice via email (mock implementation)"""
        # In production, integrate with email service like:
        # - SendGrid
        # - AWS SES
        # - Mailgun
        logger.info(f"Would send invoice {invoice.invoice_number} to {invoice.customer_data['email']}")
    
    async def mark_invoice_paid(
        self,
        invoice_id: int,
        payment_id: Optional[int] = None,
        paid_amount: Optional[float] = None
    ) -> bool:
        """Mark invoice as paid"""
        try:
            invoice = self.db.query(Invoice).filter(Invoice.id == invoice_id).first()
            if not invoice:
                raise ValueError(f"Invoice {invoice_id} not found")
            
            # Update status
            invoice.status = "paid"
            invoice.paid_date = datetime.utcnow()
            
            # If partial payment, update metadata
            if paid_amount and paid_amount < invoice.total_amount:
                invoice.status = "partially_paid"
                invoice.metadata = {
                    **(invoice.metadata or {}),
                    "paid_amount": paid_amount,
                    "remaining_amount": invoice.total_amount - paid_amount
                }
            
            self.db.commit()
            
            logger.info(f"Marked invoice {invoice.invoice_number} as paid")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark invoice {invoice_id} as paid: {e}")
            return False
    
    async def get_user_invoices(
        self,
        user_id: int,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Invoice]:
        """Get user's invoices"""
        query = self.db.query(Invoice).filter(Invoice.user_id == user_id)
        
        if status:
            query = query.filter(Invoice.status == status)
        
        return query.order_by(Invoice.created_at.desc()).offset(offset).limit(limit).all()
    
    async def get_overdue_invoices(self, days_overdue: int = 7) -> List[Invoice]:
        """Get overdue invoices"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_overdue)
        
        return self.db.query(Invoice).filter(
            Invoice.status.in_(["sent", "overdue"]),
            Invoice.due_date < cutoff_date
        ).all()
    
    async def mark_overdue_invoices(self):
        """Mark invoices as overdue"""
        try:
            overdue_invoices = await self.get_overdue_invoices()
            
            for invoice in overdue_invoices:
                if invoice.status != "overdue":
                    invoice.status = "overdue"
            
            self.db.commit()
            
            logger.info(f"Marked {len(overdue_invoices)} invoices as overdue")
            
        except Exception as e:
            logger.error(f"Failed to mark overdue invoices: {e}")
    
    async def get_invoice_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get invoice metrics for reporting"""
        try:
            if not start_date:
                start_date = datetime.utcnow().replace(day=1)  # First day of month
            if not end_date:
                end_date = datetime.utcnow()
            
            # Total invoices
            total_invoices = self.db.query(Invoice).filter(
                Invoice.created_at >= start_date,
                Invoice.created_at <= end_date
            ).count()
            
            # Total revenue
            total_revenue = self.db.query(func.sum(Invoice.total_amount)).filter(
                Invoice.created_at >= start_date,
                Invoice.created_at <= end_date,
                Invoice.status == "paid"
            ).scalar() or 0
            
            # Outstanding amount
            outstanding_amount = self.db.query(func.sum(Invoice.total_amount)).filter(
                Invoice.status.in_(["sent", "overdue"])
            ).scalar() or 0
            
            # Overdue amount
            overdue_amount = self.db.query(func.sum(Invoice.total_amount)).filter(
                Invoice.status == "overdue"
            ).scalar() or 0
            
            # Invoice status breakdown
            status_breakdown = {}
            status_results = self.db.query(
                Invoice.status,
                func.count(Invoice.id),
                func.sum(Invoice.total_amount)
            ).filter(
                Invoice.created_at >= start_date,
                Invoice.created_at <= end_date
            ).group_by(Invoice.status).all()
            
            for status, count, amount in status_results:
                status_breakdown[status] = {
                    "count": count,
                    "amount": float(amount or 0)
                }
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "invoices": total_invoices,
                    "revenue": float(total_revenue),
                    "outstanding": float(outstanding_amount),
                    "overdue": float(overdue_amount)
                },
                "status_breakdown": status_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to get invoice metrics: {e}")
            return {}


class InvoiceTemplateManager:
    """Manage invoice templates for different customer types"""
    
    def __init__(self):
        self.templates = {
            "standard": "Standard invoice template",
            "enterprise": "Enterprise invoice template with PO numbers",
            "government": "Government invoice template with special requirements"
        }
    
    def get_template(self, customer_type: str = "standard") -> str:
        """Get invoice template for customer type"""
        return self.templates.get(customer_type, self.templates["standard"])
    
    def customize_template(
        self,
        template_name: str,
        customizations: Dict[str, Any]
    ) -> str:
        """Customize invoice template"""
        # In production, implement template customization logic
        return self.templates.get(template_name, self.templates["standard"])


class InvoiceAutomation:
    """Automate invoice generation and management"""
    
    def __init__(self, invoice_manager: InvoiceManager):
        self.invoice_manager = invoice_manager
    
    async def generate_monthly_invoices(self):
        """Generate monthly invoices for all active subscriptions"""
        try:
            from .billing_cycles import BillingCycleManager
            
            billing_manager = BillingCycleManager(self.invoice_manager.db)
            
            # Get subscriptions due for billing
            due_subscriptions = await billing_manager.get_subscriptions_due_for_billing()
            
            generated_count = 0
            for subscription in due_subscriptions:
                try:
                    invoice = await self.invoice_manager.create_subscription_invoice(
                        subscription.id,
                        subscription.current_period_start,
                        subscription.current_period_end
                    )
                    
                    # Send invoice
                    await self.invoice_manager.send_invoice(invoice.id)
                    generated_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to generate invoice for subscription {subscription.id}: {e}")
                    continue
            
            logger.info(f"Generated {generated_count} monthly invoices")
            return generated_count
            
        except Exception as e:
            logger.error(f"Monthly invoice generation failed: {e}")
            return 0
    
    async def process_overdue_invoices(self):
        """Process overdue invoices"""
        try:
            # Mark overdue invoices
            await self.invoice_manager.mark_overdue_invoices()
            
            # Get overdue invoices
            overdue_invoices = await self.invoice_manager.get_overdue_invoices()
            
            # Send reminders (implement reminder service)
            reminder_count = 0
            for invoice in overdue_invoices:
                try:
                    await self._send_overdue_reminder(invoice)
                    reminder_count += 1
                except Exception as e:
                    logger.error(f"Failed to send reminder for invoice {invoice.id}: {e}")
                    continue
            
            logger.info(f"Sent {reminder_count} overdue reminders")
            return reminder_count
            
        except Exception as e:
            logger.error(f"Overdue invoice processing failed: {e}")
            return 0
    
    async def _send_overdue_reminder(self, invoice: Invoice):
        """Send overdue payment reminder"""
        # In production, implement reminder email service
        logger.info(f"Would send overdue reminder for invoice {invoice.invoice_number}")