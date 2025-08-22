"""
Stripe payment integration for the arbitration RAG API.

This module provides comprehensive Stripe integration including:
- Payment processing
- Subscription management
- Webhook handling
- Error handling and retry logic
- PCI compliance
"""

import stripe
import json
import logging
import hmac
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ..core.config import get_settings
from .models import (
    Payment, Subscription, PaymentWebhook, UsageRecord,
    PaymentStatus, SubscriptionStatus, SubscriptionTier,
    PaymentMethod, PaymentProvider, SUBSCRIPTION_TIERS
)
from .base import BasePaymentProcessor, PaymentError, WebhookError

logger = logging.getLogger(__name__)
settings = get_settings()


class StripePaymentProcessor(BasePaymentProcessor):
    """Stripe payment processor with comprehensive error handling"""
    
    def __init__(self, api_key: str, webhook_secret: str):
        """Initialize Stripe processor"""
        self.api_key = api_key
        self.webhook_secret = webhook_secret
        stripe.api_key = api_key
        
    async def create_payment_intent(
        self,
        amount: float,
        currency: str = "usd",
        payment_method: Optional[str] = None,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Stripe PaymentIntent"""
        try:
            # Convert amount to cents (Stripe uses smallest currency unit)
            amount_cents = int(amount * 100)
            
            intent_data = {
                "amount": amount_cents,
                "currency": currency.lower(),
                "automatic_payment_methods": {"enabled": True},
                "metadata": metadata or {}
            }
            
            if customer_id:
                intent_data["customer"] = customer_id
                
            if payment_method:
                intent_data["payment_method"] = payment_method
                intent_data["confirmation_method"] = "manual"
                intent_data["confirm"] = True
                
            payment_intent = stripe.PaymentIntent.create(**intent_data)
            
            return {
                "id": payment_intent.id,
                "client_secret": payment_intent.client_secret,
                "status": payment_intent.status,
                "amount": payment_intent.amount / 100,  # Convert back to dollars
                "currency": payment_intent.currency.upper(),
                "metadata": payment_intent.metadata
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment intent creation failed: {e}")
            raise PaymentError(f"Payment creation failed: {str(e)}")
    
    async def confirm_payment(
        self,
        payment_intent_id: str,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Confirm a payment intent"""
        try:
            confirm_data = {}
            if payment_method:
                confirm_data["payment_method"] = payment_method
                
            payment_intent = stripe.PaymentIntent.confirm(
                payment_intent_id,
                **confirm_data
            )
            
            return {
                "id": payment_intent.id,
                "status": payment_intent.status,
                "amount": payment_intent.amount / 100,
                "currency": payment_intent.currency.upper(),
                "charges": [
                    {
                        "id": charge.id,
                        "amount": charge.amount / 100,
                        "status": charge.status,
                        "receipt_url": charge.receipt_url
                    } for charge in payment_intent.charges.data
                ]
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe payment confirmation failed: {e}")
            raise PaymentError(f"Payment confirmation failed: {str(e)}")
    
    async def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a Stripe customer"""
        try:
            customer_data = {
                "email": email,
                "metadata": metadata or {}
            }
            
            if name:
                customer_data["name"] = name
                
            customer = stripe.Customer.create(**customer_data)
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe customer creation failed: {e}")
            raise PaymentError(f"Customer creation failed: {str(e)}")
    
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Stripe subscription"""
        try:
            subscription_data = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "payment_settings": {"save_default_payment_method": "on_subscription"},
                "expand": ["latest_invoice.payment_intent"],
                "metadata": metadata or {}
            }
            
            if trial_period_days:
                subscription_data["trial_period_days"] = trial_period_days
                
            subscription = stripe.Subscription.create(**subscription_data)
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret if subscription.latest_invoice.payment_intent else None,
                "metadata": subscription.metadata
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription creation failed: {e}")
            raise PaymentError(f"Subscription creation failed: {str(e)}")
    
    async def update_subscription(
        self,
        subscription_id: str,
        price_id: Optional[str] = None,
        proration_behavior: str = "create_prorations",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update a Stripe subscription"""
        try:
            update_data = {
                "proration_behavior": proration_behavior,
                "metadata": metadata or {}
            }
            
            if price_id:
                # Get current subscription to update items
                current_sub = stripe.Subscription.retrieve(subscription_id)
                update_data["items"] = [{
                    "id": current_sub.items.data[0].id,
                    "price": price_id
                }]
                
            subscription = stripe.Subscription.modify(subscription_id, **update_data)
            
            return {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "metadata": subscription.metadata
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription update failed: {e}")
            raise PaymentError(f"Subscription update failed: {str(e)}")
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Dict[str, Any]:
        """Cancel a Stripe subscription"""
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
                
            return {
                "id": subscription.id,
                "status": subscription.status,
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "canceled_at": datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe subscription cancellation failed: {e}")
            raise PaymentError(f"Subscription cancellation failed: {str(e)}")
    
    async def create_price(
        self,
        product_id: str,
        amount: float,
        currency: str = "usd",
        interval: str = "month",
        interval_count: int = 1
    ) -> str:
        """Create a Stripe price"""
        try:
            price = stripe.Price.create(
                product=product_id,
                unit_amount=int(amount * 100),  # Convert to cents
                currency=currency.lower(),
                recurring={"interval": interval, "interval_count": interval_count}
            )
            return price.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe price creation failed: {e}")
            raise PaymentError(f"Price creation failed: {str(e)}")
    
    async def create_refund(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a refund for a payment"""
        try:
            refund_data = {"payment_intent": payment_intent_id}
            
            if amount:
                refund_data["amount"] = int(amount * 100)  # Convert to cents
                
            if reason:
                refund_data["reason"] = reason
                
            refund = stripe.Refund.create(**refund_data)
            
            return {
                "id": refund.id,
                "amount": refund.amount / 100,  # Convert back to dollars
                "currency": refund.currency.upper(),
                "status": refund.status,
                "reason": refund.reason
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe refund creation failed: {e}")
            raise PaymentError(f"Refund creation failed: {str(e)}")
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Stripe webhook signature"""
        try:
            stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return True
        except ValueError:
            logger.error("Invalid payload in webhook")
            return False
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid signature in webhook")
            return False
    
    async def handle_webhook(
        self,
        request: Request,
        db: Session
    ) -> JSONResponse:
        """Handle Stripe webhook events"""
        try:
            payload = await request.body()
            signature = request.headers.get("stripe-signature")
            
            if not signature:
                raise WebhookError("Missing Stripe signature")
            
            if not self.verify_webhook_signature(payload, signature):
                raise WebhookError("Invalid webhook signature")
            
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            
            # Store webhook event
            webhook_record = PaymentWebhook(
                provider=PaymentProvider.STRIPE,
                event_id=event["id"],
                event_type=event["type"],
                data=event["data"]
            )
            db.add(webhook_record)
            db.flush()
            
            # Process event
            await self._process_webhook_event(event, db)
            
            # Mark as processed
            webhook_record.processed = True
            webhook_record.processed_at = datetime.utcnow()
            db.commit()
            
            return JSONResponse({"status": "success"})
            
        except WebhookError as e:
            logger.error(f"Webhook processing error: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=400
            )
        except Exception as e:
            logger.error(f"Unexpected webhook error: {e}")
            if 'webhook_record' in locals():
                webhook_record.last_error = str(e)
                webhook_record.processing_attempts += 1
                db.commit()
            return JSONResponse(
                {"error": "Internal server error"}, 
                status_code=500
            )
    
    async def _process_webhook_event(self, event: Dict[str, Any], db: Session):
        """Process specific webhook events"""
        event_type = event["type"]
        data = event["data"]["object"]
        
        if event_type == "payment_intent.succeeded":
            await self._handle_payment_succeeded(data, db)
        elif event_type == "payment_intent.payment_failed":
            await self._handle_payment_failed(data, db)
        elif event_type == "customer.subscription.created":
            await self._handle_subscription_created(data, db)
        elif event_type == "customer.subscription.updated":
            await self._handle_subscription_updated(data, db)
        elif event_type == "customer.subscription.deleted":
            await self._handle_subscription_deleted(data, db)
        elif event_type == "invoice.payment_succeeded":
            await self._handle_invoice_payment_succeeded(data, db)
        elif event_type == "invoice.payment_failed":
            await self._handle_invoice_payment_failed(data, db)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")
    
    async def _handle_payment_succeeded(self, data: Dict[str, Any], db: Session):
        """Handle successful payment"""
        payment_intent_id = data["id"]
        
        payment = db.query(Payment).filter(
            Payment.external_id == payment_intent_id
        ).first()
        
        if payment:
            payment.status = PaymentStatus.COMPLETED
            payment.processed_at = datetime.utcnow()
            payment.provider_data = data
            db.commit()
            
            logger.info(f"Payment {payment_intent_id} marked as completed")
    
    async def _handle_payment_failed(self, data: Dict[str, Any], db: Session):
        """Handle failed payment"""
        payment_intent_id = data["id"]
        
        payment = db.query(Payment).filter(
            Payment.external_id == payment_intent_id
        ).first()
        
        if payment:
            payment.status = PaymentStatus.FAILED
            payment.failure_reason = data.get("last_payment_error", {}).get("message")
            payment.failure_code = data.get("last_payment_error", {}).get("code")
            payment.provider_data = data
            db.commit()
            
            logger.info(f"Payment {payment_intent_id} marked as failed")
    
    async def _handle_subscription_created(self, data: Dict[str, Any], db: Session):
        """Handle subscription creation"""
        subscription_id = data["id"]
        
        subscription = db.query(Subscription).filter(
            Subscription.external_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = self._map_stripe_subscription_status(data["status"])
            subscription.current_period_start = datetime.fromtimestamp(data["current_period_start"])
            subscription.current_period_end = datetime.fromtimestamp(data["current_period_end"])
            subscription.provider_data = data
            db.commit()
            
            logger.info(f"Subscription {subscription_id} created")
    
    async def _handle_subscription_updated(self, data: Dict[str, Any], db: Session):
        """Handle subscription update"""
        subscription_id = data["id"]
        
        subscription = db.query(Subscription).filter(
            Subscription.external_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = self._map_stripe_subscription_status(data["status"])
            subscription.current_period_start = datetime.fromtimestamp(data["current_period_start"])
            subscription.current_period_end = datetime.fromtimestamp(data["current_period_end"])
            subscription.cancel_at_period_end = data.get("cancel_at_period_end", False)
            subscription.provider_data = data
            db.commit()
            
            logger.info(f"Subscription {subscription_id} updated")
    
    async def _handle_subscription_deleted(self, data: Dict[str, Any], db: Session):
        """Handle subscription deletion"""
        subscription_id = data["id"]
        
        subscription = db.query(Subscription).filter(
            Subscription.external_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.utcnow()
            subscription.provider_data = data
            db.commit()
            
            logger.info(f"Subscription {subscription_id} cancelled")
    
    async def _handle_invoice_payment_succeeded(self, data: Dict[str, Any], db: Session):
        """Handle successful invoice payment"""
        subscription_id = data.get("subscription")
        
        if subscription_id:
            subscription = db.query(Subscription).filter(
                Subscription.external_id == subscription_id
            ).first()
            
            if subscription and subscription.status == SubscriptionStatus.PAST_DUE:
                subscription.status = SubscriptionStatus.ACTIVE
                db.commit()
                
                logger.info(f"Subscription {subscription_id} reactivated after payment")
    
    async def _handle_invoice_payment_failed(self, data: Dict[str, Any], db: Session):
        """Handle failed invoice payment"""
        subscription_id = data.get("subscription")
        
        if subscription_id:
            subscription = db.query(Subscription).filter(
                Subscription.external_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = SubscriptionStatus.PAST_DUE
                db.commit()
                
                logger.info(f"Subscription {subscription_id} marked as past due")
    
    def _map_stripe_subscription_status(self, stripe_status: str) -> SubscriptionStatus:
        """Map Stripe subscription status to internal status"""
        status_mapping = {
            "active": SubscriptionStatus.ACTIVE,
            "canceled": SubscriptionStatus.CANCELLED,
            "incomplete": SubscriptionStatus.INCOMPLETE,
            "incomplete_expired": SubscriptionStatus.INCOMPLETE_EXPIRED,
            "past_due": SubscriptionStatus.PAST_DUE,
            "trialing": SubscriptionStatus.TRIALING,
            "unpaid": SubscriptionStatus.UNPAID
        }
        return status_mapping.get(stripe_status, SubscriptionStatus.ACTIVE)


class StripeSubscriptionManager:
    """Manage Stripe subscriptions and pricing"""
    
    def __init__(self, processor: StripePaymentProcessor):
        self.processor = processor
    
    async def setup_products_and_prices(self) -> Dict[str, str]:
        """Set up Stripe products and prices for subscription tiers"""
        try:
            price_ids = {}
            
            for tier, config in SUBSCRIPTION_TIERS.items():
                if tier == SubscriptionTier.FREE or config["price"] is None:
                    continue
                
                # Create or get product
                product = stripe.Product.create(
                    name=config["name"],
                    description=f"{config['name']} subscription tier",
                    metadata={"tier": tier.value}
                )
                
                # Create monthly price
                monthly_price = await self.processor.create_price(
                    product.id,
                    config["price"],
                    interval="month"
                )
                
                # Create yearly price (20% discount)
                yearly_amount = config["price"] * 12 * 0.8
                yearly_price = await self.processor.create_price(
                    product.id,
                    yearly_amount,
                    interval="year"
                )
                
                price_ids[f"{tier.value}_monthly"] = monthly_price
                price_ids[f"{tier.value}_yearly"] = yearly_price
            
            return price_ids
            
        except Exception as e:
            logger.error(f"Failed to setup Stripe products and prices: {e}")
            raise PaymentError(f"Product setup failed: {str(e)}")
    
    async def create_subscription_with_trial(
        self,
        customer_id: str,
        tier: SubscriptionTier,
        billing_interval: str = "monthly",
        trial_days: int = 14
    ) -> Dict[str, Any]:
        """Create subscription with trial period"""
        price_id = f"{tier.value}_{billing_interval}"
        
        return await self.processor.create_subscription(
            customer_id=customer_id,
            price_id=price_id,
            trial_period_days=trial_days,
            metadata={
                "tier": tier.value,
                "billing_interval": billing_interval
            }
        )
    
    async def upgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        billing_interval: str = "monthly"
    ) -> Dict[str, Any]:
        """Upgrade subscription with prorated billing"""
        new_price_id = f"{new_tier.value}_{billing_interval}"
        
        return await self.processor.update_subscription(
            subscription_id=subscription_id,
            price_id=new_price_id,
            proration_behavior="create_prorations",
            metadata={
                "tier": new_tier.value,
                "billing_interval": billing_interval
            }
        )
    
    async def downgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        billing_interval: str = "monthly"
    ) -> Dict[str, Any]:
        """Downgrade subscription at period end"""
        new_price_id = f"{new_tier.value}_{billing_interval}"
        
        return await self.processor.update_subscription(
            subscription_id=subscription_id,
            price_id=new_price_id,
            proration_behavior="none",  # No immediate charge for downgrades
            metadata={
                "tier": new_tier.value,
                "billing_interval": billing_interval,
                "scheduled_downgrade": "true"
            }
        )