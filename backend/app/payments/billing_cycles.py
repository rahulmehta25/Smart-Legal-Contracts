"""
Billing cycles and proration management for the arbitration RAG API.

This module provides comprehensive billing cycle management including:
- Monthly and annual billing cycles
- Proration calculations
- Subscription changes
- Billing cycle synchronization
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import calendar

from ..core.config import get_settings
from .models import (
    Subscription, Payment, UsageRecord, Invoice,
    SubscriptionStatus, SubscriptionTier, PaymentStatus,
    calculate_proration, SUBSCRIPTION_TIERS
)

logger = logging.getLogger(__name__)
settings = get_settings()


class BillingCycleManager:
    """Manage billing cycles and subscription billing"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def process_billing_cycles(self) -> Dict[str, Any]:
        """Process all due billing cycles"""
        try:
            # Get subscriptions due for billing
            due_subscriptions = await self.get_subscriptions_due_for_billing()
            
            processed_count = 0
            failed_count = 0
            total_revenue = 0.0
            
            for subscription in due_subscriptions:
                try:
                    result = await self.process_subscription_billing(subscription)
                    if result["success"]:
                        processed_count += 1
                        total_revenue += result.get("amount", 0)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process billing for subscription {subscription.id}: {e}")
                    failed_count += 1
                    continue
            
            logger.info(
                f"Billing cycle processing complete: "
                f"processed={processed_count}, failed={failed_count}, revenue=${total_revenue:.2f}"
            )
            
            return {
                "processed": processed_count,
                "failed": failed_count,
                "total_revenue": total_revenue,
                "subscriptions_processed": [sub.id for sub in due_subscriptions[:processed_count]]
            }
            
        except Exception as e:
            logger.error(f"Billing cycle processing failed: {e}")
            return {"error": str(e)}
    
    async def get_subscriptions_due_for_billing(self) -> List[Subscription]:
        """Get subscriptions that are due for billing"""
        try:
            now = datetime.utcnow()
            
            # Get active subscriptions where current_period_end has passed
            due_subscriptions = self.db.query(Subscription).filter(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.current_period_end <= now,
                ~Subscription.cancel_at_period_end  # Don't bill cancelled subscriptions
            ).all()
            
            return due_subscriptions
            
        except Exception as e:
            logger.error(f"Failed to get due subscriptions: {e}")
            return []
    
    async def process_subscription_billing(self, subscription: Subscription) -> Dict[str, Any]:
        """Process billing for a single subscription"""
        try:
            logger.info(f"Processing billing for subscription {subscription.id}")
            
            # Calculate next billing period
            next_period_start = subscription.current_period_end
            next_period_end = self._calculate_next_period_end(
                next_period_start, subscription.billing_interval
            )
            
            # Create invoice for this billing period
            from .invoicing import InvoiceManager
            invoice_manager = InvoiceManager(self.db)
            
            invoice = await invoice_manager.create_subscription_invoice(
                subscription.id,
                subscription.current_period_start,
                subscription.current_period_end
            )
            
            # Update subscription billing period
            subscription.current_period_start = next_period_start
            subscription.current_period_end = next_period_end
            
            # Send invoice
            await invoice_manager.send_invoice(invoice.id)
            
            # If subscription has automatic payment method, attempt to charge
            payment_result = await self._attempt_automatic_payment(subscription, invoice)
            
            self.db.commit()
            
            result = {
                "success": True,
                "subscription_id": subscription.id,
                "invoice_id": invoice.id,
                "amount": invoice.total_amount,
                "next_period_start": next_period_start.isoformat(),
                "next_period_end": next_period_end.isoformat(),
                "payment_result": payment_result
            }
            
            logger.info(f"Successfully processed billing for subscription {subscription.id}")
            return result
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to process billing for subscription {subscription.id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_next_period_end(self, period_start: datetime, billing_interval: str) -> datetime:
        """Calculate the end of the next billing period"""
        if billing_interval == "yearly":
            # Add one year
            try:
                return period_start.replace(year=period_start.year + 1)
            except ValueError:
                # Handle leap year edge case (Feb 29)
                return period_start.replace(year=period_start.year + 1, day=28)
        else:
            # Monthly billing (default)
            if period_start.month == 12:
                next_month = period_start.replace(year=period_start.year + 1, month=1)
            else:
                next_month = period_start.replace(month=period_start.month + 1)
            
            # Handle day adjustment for months with different lengths
            try:
                return next_month
            except ValueError:
                # Handle case where original day doesn't exist in next month (e.g., Jan 31 -> Feb 31)
                last_day = calendar.monthrange(next_month.year, next_month.month)[1]
                return next_month.replace(day=min(period_start.day, last_day))
    
    async def _attempt_automatic_payment(
        self,
        subscription: Subscription,
        invoice: Invoice
    ) -> Dict[str, Any]:
        """Attempt automatic payment for subscription"""
        try:
            # Check if subscription has automatic payment enabled
            provider_data = subscription.provider_data or {}
            if not provider_data.get("auto_payment_enabled", False):
                return {"auto_payment": False, "reason": "Auto payment not enabled"}
            
            # Get payment method from provider data
            payment_method_id = provider_data.get("payment_method_id")
            if not payment_method_id:
                return {"auto_payment": False, "reason": "No payment method on file"}
            
            # Attempt payment based on provider
            if subscription.provider.value == "stripe":
                result = await self._process_stripe_automatic_payment(subscription, invoice)
            elif subscription.provider.value == "paypal":
                result = await self._process_paypal_automatic_payment(subscription, invoice)
            else:
                return {"auto_payment": False, "reason": "Provider not supported for auto payment"}
            
            return result
            
        except Exception as e:
            logger.error(f"Automatic payment failed for subscription {subscription.id}: {e}")
            return {"auto_payment": False, "error": str(e)}
    
    async def _process_stripe_automatic_payment(
        self,
        subscription: Subscription,
        invoice: Invoice
    ) -> Dict[str, Any]:
        """Process automatic payment via Stripe"""
        try:
            from .stripe import StripePaymentProcessor
            
            # Initialize Stripe processor (you'd get these from config)
            stripe_processor = StripePaymentProcessor(
                api_key=settings.stripe_secret_key,
                webhook_secret=settings.stripe_webhook_secret
            )
            
            # Create payment intent
            payment_intent = await stripe_processor.create_payment_intent(
                amount=invoice.total_amount,
                customer_id=subscription.provider_data.get("customer_id"),
                payment_method=subscription.provider_data.get("payment_method_id"),
                metadata={
                    "subscription_id": subscription.id,
                    "invoice_id": invoice.id
                }
            )
            
            # Confirm payment
            if payment_intent.get("status") == "requires_confirmation":
                confirm_result = await stripe_processor.confirm_payment(payment_intent["id"])
                
                if confirm_result.get("status") == "succeeded":
                    # Mark invoice as paid
                    await self._mark_invoice_paid(invoice, confirm_result)
                    return {"auto_payment": True, "payment_id": confirm_result["id"]}
                else:
                    await self._handle_failed_payment(subscription, invoice, confirm_result)
                    return {"auto_payment": False, "reason": "Payment failed"}
            
            return {"auto_payment": True, "payment_id": payment_intent["id"]}
            
        except Exception as e:
            logger.error(f"Stripe automatic payment failed: {e}")
            return {"auto_payment": False, "error": str(e)}
    
    async def _process_paypal_automatic_payment(
        self,
        subscription: Subscription,
        invoice: Invoice
    ) -> Dict[str, Any]:
        """Process automatic payment via PayPal"""
        # PayPal subscriptions handle billing automatically
        # This would typically be handled by PayPal's subscription system
        return {"auto_payment": True, "note": "PayPal handles subscription billing automatically"}
    
    async def _mark_invoice_paid(self, invoice: Invoice, payment_data: Dict[str, Any]):
        """Mark invoice as paid and create payment record"""
        try:
            from .models import Payment, PaymentMethod, PaymentProvider
            
            # Create payment record
            payment = Payment(
                external_id=payment_data["id"],
                user_id=invoice.user_id,
                amount=invoice.total_amount,
                payment_method=PaymentMethod.CREDIT_CARD,  # Assume credit card for auto payments
                provider=PaymentProvider.STRIPE,  # This would be dynamic based on processor
                status=PaymentStatus.COMPLETED,
                invoice_id=invoice.id,
                processed_at=datetime.utcnow(),
                provider_data=payment_data
            )
            
            self.db.add(payment)
            
            # Update invoice
            invoice.status = "paid"
            invoice.paid_date = datetime.utcnow()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to mark invoice {invoice.id} as paid: {e}")
            raise
    
    async def _handle_failed_payment(
        self,
        subscription: Subscription,
        invoice: Invoice,
        payment_data: Dict[str, Any]
    ):
        """Handle failed automatic payment"""
        try:
            # Mark subscription as past due
            subscription.status = SubscriptionStatus.PAST_DUE
            
            # Create failed payment record
            from .models import Payment, PaymentMethod, PaymentProvider
            
            payment = Payment(
                external_id=payment_data.get("id", "failed"),
                user_id=subscription.user_id,
                amount=invoice.total_amount,
                payment_method=PaymentMethod.CREDIT_CARD,
                provider=PaymentProvider.STRIPE,
                status=PaymentStatus.FAILED,
                invoice_id=invoice.id,
                failure_reason=payment_data.get("failure_reason", "Payment failed"),
                provider_data=payment_data
            )
            
            self.db.add(payment)
            
            # Schedule dunning management
            await self._schedule_dunning_process(subscription, invoice)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to handle payment failure for subscription {subscription.id}: {e}")
            raise
    
    async def _schedule_dunning_process(self, subscription: Subscription, invoice: Invoice):
        """Schedule dunning process for failed payment"""
        # This would integrate with the dunning management system
        # For now, just log that dunning should be triggered
        logger.info(f"Dunning process should be triggered for subscription {subscription.id}")
    
    async def change_subscription_tier(
        self,
        subscription_id: int,
        new_tier: SubscriptionTier,
        effective_date: Optional[datetime] = None,
        proration_behavior: str = "immediate"
    ) -> Dict[str, Any]:
        """Change subscription tier with proration"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            if not effective_date:
                effective_date = datetime.utcnow()
            
            # Get tier configurations
            old_tier_config = SUBSCRIPTION_TIERS[subscription.tier]
            new_tier_config = SUBSCRIPTION_TIERS[new_tier]
            
            old_amount = subscription.amount
            new_amount = new_tier_config["price"] or 0
            
            # Calculate proration
            proration_amount = 0.0
            if proration_behavior == "immediate" and old_amount != new_amount:
                proration_amount = calculate_proration(
                    old_amount,
                    new_amount,
                    subscription.current_period_start,
                    subscription.current_period_end,
                    effective_date
                )
            
            # Update subscription
            old_tier = subscription.tier
            subscription.tier = new_tier
            subscription.amount = new_amount
            subscription.document_limit = new_tier_config.get("document_limit")
            subscription.api_limit = new_tier_config.get("api_limit")
            
            # Create invoice for proration if applicable
            invoice_id = None
            if proration_amount != 0:
                from .invoicing import InvoiceManager
                from .models import InvoiceLineItem
                
                invoice_manager = InvoiceManager(self.db)
                
                line_items = [
                    InvoiceLineItem(
                        description=f"Subscription tier change: {old_tier.value} to {new_tier.value}",
                        quantity=1,
                        unit_price=proration_amount,
                        amount=proration_amount
                    )
                ]
                
                invoice = await invoice_manager.create_invoice(
                    subscription.user_id,
                    line_items,
                    subscription_id=subscription.id,
                    notes=f"Proration for tier change on {effective_date.date()}"
                )
                
                invoice_id = invoice.id
            
            # Update provider subscription if needed
            await self._update_provider_subscription(subscription, new_tier)
            
            self.db.commit()
            
            logger.info(
                f"Changed subscription {subscription_id} from {old_tier.value} to {new_tier.value}, "
                f"proration: ${proration_amount:.2f}"
            )
            
            return {
                "success": True,
                "subscription_id": subscription_id,
                "old_tier": old_tier.value,
                "new_tier": new_tier.value,
                "proration_amount": proration_amount,
                "invoice_id": invoice_id,
                "effective_date": effective_date.isoformat()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to change subscription tier: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_provider_subscription(self, subscription: Subscription, new_tier: SubscriptionTier):
        """Update subscription with payment provider"""
        try:
            if subscription.provider.value == "stripe":
                await self._update_stripe_subscription(subscription, new_tier)
            elif subscription.provider.value == "paypal":
                await self._update_paypal_subscription(subscription, new_tier)
            
        except Exception as e:
            logger.error(f"Failed to update provider subscription: {e}")
            # Don't fail the entire operation if provider update fails
    
    async def _update_stripe_subscription(self, subscription: Subscription, new_tier: SubscriptionTier):
        """Update Stripe subscription"""
        try:
            from .stripe import StripePaymentProcessor
            
            stripe_processor = StripePaymentProcessor(
                api_key=settings.stripe_secret_key,
                webhook_secret=settings.stripe_webhook_secret
            )
            
            # Get new price ID for the tier
            price_id = f"{new_tier.value}_{subscription.billing_interval}"
            
            await stripe_processor.update_subscription(
                subscription.external_id,
                price_id=price_id,
                metadata={
                    "tier": new_tier.value,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to update Stripe subscription: {e}")
            raise
    
    async def _update_paypal_subscription(self, subscription: Subscription, new_tier: SubscriptionTier):
        """Update PayPal subscription"""
        # PayPal subscription updates are more complex and may require cancellation/recreation
        logger.info(f"PayPal subscription update needed for subscription {subscription.id}")
    
    async def cancel_subscription(
        self,
        subscription_id: int,
        at_period_end: bool = True,
        cancellation_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel a subscription"""
        try:
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            if not subscription:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            if at_period_end:
                subscription.cancel_at_period_end = True
                cancellation_date = subscription.current_period_end
            else:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.utcnow()
                cancellation_date = datetime.utcnow()
            
            # Update provider subscription
            await self._cancel_provider_subscription(subscription, at_period_end)
            
            # Add cancellation metadata
            subscription.metadata = {
                **(subscription.metadata or {}),
                "cancellation_reason": cancellation_reason,
                "cancelled_by": "user",  # Could be "admin", "system", etc.
                "cancellation_date": cancellation_date.isoformat()
            }
            
            self.db.commit()
            
            logger.info(f"Cancelled subscription {subscription_id}, at_period_end={at_period_end}")
            
            return {
                "success": True,
                "subscription_id": subscription_id,
                "cancelled_at_period_end": at_period_end,
                "cancellation_date": cancellation_date.isoformat(),
                "status": subscription.status.value
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to cancel subscription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cancel_provider_subscription(self, subscription: Subscription, at_period_end: bool):
        """Cancel subscription with payment provider"""
        try:
            if subscription.provider.value == "stripe":
                from .stripe import StripePaymentProcessor
                
                stripe_processor = StripePaymentProcessor(
                    api_key=settings.stripe_secret_key,
                    webhook_secret=settings.stripe_webhook_secret
                )
                
                await stripe_processor.cancel_subscription(
                    subscription.external_id,
                    at_period_end=at_period_end
                )
                
            elif subscription.provider.value == "paypal":
                from .paypal import PayPalPaymentProcessor
                
                paypal_processor = PayPalPaymentProcessor(
                    client_id=settings.paypal_client_id,
                    client_secret=settings.paypal_client_secret,
                    environment=settings.paypal_environment
                )
                
                await paypal_processor.cancel_subscription(
                    subscription.external_id,
                    at_period_end=at_period_end
                )
            
        except Exception as e:
            logger.error(f"Failed to cancel provider subscription: {e}")
            # Don't fail the entire operation if provider cancellation fails
    
    async def get_billing_calendar(
        self,
        year: int,
        month: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get billing calendar for a period"""
        try:
            if month:
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1)
                else:
                    end_date = datetime(year, month + 1, 1)
            else:
                start_date = datetime(year, 1, 1)
                end_date = datetime(year + 1, 1, 1)
            
            # Get subscriptions with billing dates in this period
            subscriptions = self.db.query(Subscription).filter(
                Subscription.status == SubscriptionStatus.ACTIVE,
                Subscription.current_period_end >= start_date,
                Subscription.current_period_end < end_date
            ).all()
            
            # Group by billing date
            billing_calendar = {}
            total_revenue = 0.0
            
            for subscription in subscriptions:
                billing_date = subscription.current_period_end.date().isoformat()
                
                if billing_date not in billing_calendar:
                    billing_calendar[billing_date] = {
                        "subscriptions": [],
                        "total_amount": 0.0,
                        "count": 0
                    }
                
                billing_calendar[billing_date]["subscriptions"].append({
                    "id": subscription.id,
                    "user_id": subscription.user_id,
                    "tier": subscription.tier.value,
                    "amount": subscription.amount
                })
                
                billing_calendar[billing_date]["total_amount"] += subscription.amount
                billing_calendar[billing_date]["count"] += 1
                total_revenue += subscription.amount
            
            return {
                "period": {
                    "year": year,
                    "month": month,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_revenue": total_revenue,
                "total_subscriptions": len(subscriptions),
                "billing_calendar": billing_calendar
            }
            
        except Exception as e:
            logger.error(f"Failed to get billing calendar: {e}")
            return {"error": str(e)}