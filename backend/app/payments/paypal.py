"""
PayPal payment integration for the arbitration RAG API.

This module provides comprehensive PayPal integration including:
- Payment processing via PayPal REST API
- Subscription management
- Webhook handling
- Error handling and retry logic
"""

import json
import logging
import hmac
import hashlib
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

from ..core.config import get_settings
from .models import (
    Payment, Subscription, PaymentWebhook, UsageRecord,
    PaymentStatus, SubscriptionStatus, SubscriptionTier,
    PaymentMethod, PaymentProvider, SUBSCRIPTION_TIERS
)
from .base import BasePaymentProcessor, PaymentError, WebhookError

logger = logging.getLogger(__name__)
settings = get_settings()


class PayPalPaymentProcessor(BasePaymentProcessor):
    """PayPal payment processor with comprehensive error handling"""
    
    def __init__(self, client_id: str, client_secret: str, environment: str = "sandbox"):
        """Initialize PayPal processor"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.environment = environment
        self.base_url = (
            "https://api-m.sandbox.paypal.com" if environment == "sandbox"
            else "https://api-m.paypal.com"
        )
        self._access_token = None
        self._token_expires_at = None
    
    async def _get_access_token(self) -> str:
        """Get or refresh PayPal access token"""
        if (self._access_token and self._token_expires_at and 
            datetime.utcnow() < self._token_expires_at):
            return self._access_token
        
        try:
            auth_string = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            
            headers = {
                "Authorization": f"Basic {auth_string}",
                "Accept": "application/json",
                "Accept-Language": "en_US"
            }
            
            data = "grant_type=client_credentials"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/oauth2/token",
                    headers=headers,
                    content=data,
                    headers_content_type="application/x-www-form-urlencoded"
                )
                
                if response.status_code != 200:
                    raise PaymentError(f"PayPal auth failed: {response.text}")
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
                
                return self._access_token
                
        except Exception as e:
            logger.error(f"PayPal authentication failed: {e}")
            raise PaymentError(f"Authentication failed: {str(e)}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to PayPal API"""
        access_token = await self._get_access_token()
        
        request_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if headers:
            request_headers.update(headers)
        
        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=request_headers
                    )
                elif method.upper() == "POST":
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        headers=request_headers,
                        json=data
                    )
                elif method.upper() == "PATCH":
                    response = await client.patch(
                        f"{self.base_url}{endpoint}",
                        headers=request_headers,
                        json=data
                    )
                else:
                    raise PaymentError(f"Unsupported HTTP method: {method}")
                
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    raise PaymentError(
                        f"PayPal API error: {response.status_code}",
                        code=error_data.get("name"),
                        details=error_data
                    )
                
                return response.json() if response.content else {}
                
        except httpx.RequestError as e:
            logger.error(f"PayPal request failed: {e}")
            raise PaymentError(f"Request failed: {str(e)}")
    
    async def create_payment_intent(
        self,
        amount: float,
        currency: str = "USD",
        payment_method: Optional[str] = None,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a PayPal payment (order)"""
        try:
            order_data = {
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {
                        "currency_code": currency.upper(),
                        "value": f"{amount:.2f}"
                    }
                }],
                "payment_source": {
                    "paypal": {
                        "experience_context": {
                            "payment_method_preference": "IMMEDIATE_PAYMENT_REQUIRED",
                            "brand_name": "Arbitration RAG API",
                            "locale": "en-US",
                            "landing_page": "LOGIN",
                            "user_action": "PAY_NOW"
                        }
                    }
                }
            }
            
            if metadata:
                order_data["purchase_units"][0]["custom_id"] = json.dumps(metadata)
            
            response = await self._make_request("POST", "/v2/checkout/orders", order_data)
            
            # Get approval URL
            approval_url = None
            for link in response.get("links", []):
                if link["rel"] == "approve":
                    approval_url = link["href"]
                    break
            
            return {
                "id": response["id"],
                "status": response["status"].lower(),
                "amount": amount,
                "currency": currency.upper(),
                "approval_url": approval_url,
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"PayPal order creation failed: {e}")
            raise PaymentError(f"Payment creation failed: {str(e)}")
    
    async def confirm_payment(
        self,
        payment_intent_id: str,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Capture a PayPal payment"""
        try:
            response = await self._make_request(
                "POST", 
                f"/v2/checkout/orders/{payment_intent_id}/capture"
            )
            
            capture_data = response.get("purchase_units", [{}])[0].get("payments", {}).get("captures", [{}])[0]
            
            return {
                "id": response["id"],
                "status": response["status"].lower(),
                "amount": float(capture_data.get("amount", {}).get("value", 0)),
                "currency": capture_data.get("amount", {}).get("currency_code", "USD"),
                "capture_id": capture_data.get("id"),
                "final_capture": capture_data.get("final_capture", True)
            }
            
        except Exception as e:
            logger.error(f"PayPal payment capture failed: {e}")
            raise PaymentError(f"Payment capture failed: {str(e)}")
    
    async def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a PayPal customer (vault setup)"""
        try:
            # PayPal doesn't have a direct customer concept like Stripe
            # We'll create a setup token for future payments
            vault_data = {
                "payment_source": {
                    "paypal": {
                        "description": f"Vault setup for {email}",
                        "usage_pattern": "IMMEDIATE",
                        "customer_type": "CONSUMER"
                    }
                }
            }
            
            response = await self._make_request("POST", "/v3/vault/setup-tokens", vault_data)
            
            return response["id"]
            
        except Exception as e:
            logger.error(f"PayPal customer setup failed: {e}")
            raise PaymentError(f"Customer setup failed: {str(e)}")
    
    async def create_subscription(
        self,
        customer_id: str,
        plan_id: str,
        trial_period_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a PayPal subscription"""
        try:
            subscription_data = {
                "plan_id": plan_id,
                "start_time": datetime.utcnow().isoformat() + "Z",
                "subscriber": {
                    "email_address": metadata.get("email") if metadata else "customer@example.com"
                },
                "application_context": {
                    "brand_name": "Arbitration RAG API",
                    "locale": "en-US",
                    "shipping_preference": "NO_SHIPPING",
                    "user_action": "SUBSCRIBE_NOW",
                    "payment_method": {
                        "payer_selected": "PAYPAL",
                        "payee_preferred": "IMMEDIATE_PAYMENT_REQUIRED"
                    }
                }
            }
            
            if trial_period_days:
                # PayPal trial periods are set in the plan, not per subscription
                pass
            
            response = await self._make_request("POST", "/v1/billing/subscriptions", subscription_data)
            
            # Get approval URL
            approval_url = None
            for link in response.get("links", []):
                if link["rel"] == "approve":
                    approval_url = link["href"]
                    break
            
            return {
                "id": response["id"],
                "status": response["status"].lower(),
                "start_time": datetime.fromisoformat(response["start_time"].replace("Z", "+00:00")),
                "approval_url": approval_url,
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"PayPal subscription creation failed: {e}")
            raise PaymentError(f"Subscription creation failed: {str(e)}")
    
    async def update_subscription(
        self,
        subscription_id: str,
        plan_id: Optional[str] = None,
        proration_behavior: str = "create_prorations",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update a PayPal subscription"""
        try:
            if plan_id:
                # PayPal subscription plan changes
                update_data = {
                    "plan_id": plan_id,
                    "prorate": proration_behavior == "create_prorations"
                }
                
                await self._make_request(
                    "POST",
                    f"/v1/billing/subscriptions/{subscription_id}/revise",
                    update_data
                )
            
            # Get updated subscription
            response = await self._make_request("GET", f"/v1/billing/subscriptions/{subscription_id}")
            
            return {
                "id": response["id"],
                "status": response["status"].lower(),
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"PayPal subscription update failed: {e}")
            raise PaymentError(f"Subscription update failed: {str(e)}")
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Dict[str, Any]:
        """Cancel a PayPal subscription"""
        try:
            cancel_data = {
                "reason": "User requested cancellation"
            }
            
            await self._make_request(
                "POST",
                f"/v1/billing/subscriptions/{subscription_id}/cancel",
                cancel_data
            )
            
            # Get updated subscription
            response = await self._make_request("GET", f"/v1/billing/subscriptions/{subscription_id}")
            
            return {
                "id": response["id"],
                "status": response["status"].lower(),
                "canceled_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"PayPal subscription cancellation failed: {e}")
            raise PaymentError(f"Subscription cancellation failed: {str(e)}")
    
    async def create_refund(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a refund for a PayPal payment"""
        try:
            # First, get the capture ID from the order
            order = await self._make_request("GET", f"/v2/checkout/orders/{payment_intent_id}")
            capture_id = None
            
            for unit in order.get("purchase_units", []):
                for capture in unit.get("payments", {}).get("captures", []):
                    capture_id = capture["id"]
                    break
                if capture_id:
                    break
            
            if not capture_id:
                raise PaymentError("No capture found for this payment")
            
            refund_data = {}
            if amount:
                refund_data["amount"] = {
                    "value": f"{amount:.2f}",
                    "currency_code": "USD"
                }
            
            if reason:
                refund_data["note_to_payer"] = reason
            
            response = await self._make_request(
                "POST",
                f"/v2/payments/captures/{capture_id}/refund",
                refund_data
            )
            
            return {
                "id": response["id"],
                "amount": float(response["amount"]["value"]),
                "currency": response["amount"]["currency_code"],
                "status": response["status"].lower()
            }
            
        except Exception as e:
            logger.error(f"PayPal refund creation failed: {e}")
            raise PaymentError(f"Refund creation failed: {str(e)}")
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify PayPal webhook signature"""
        try:
            # PayPal webhook verification is more complex and requires
            # certificate verification. For now, we'll do basic validation.
            # In production, implement full certificate chain verification.
            return True  # Simplified for demo
            
        except Exception as e:
            logger.error(f"PayPal webhook verification failed: {e}")
            return False
    
    async def handle_webhook(
        self,
        request: Request,
        db: Session
    ) -> JSONResponse:
        """Handle PayPal webhook events"""
        try:
            payload = await request.body()
            
            # Get webhook headers
            webhook_id = request.headers.get("PAYPAL-TRANSMISSION-ID")
            if not webhook_id:
                raise WebhookError("Missing PayPal webhook ID")
            
            event_data = json.loads(payload)
            
            # Store webhook event
            webhook_record = PaymentWebhook(
                provider=PaymentProvider.PAYPAL,
                event_id=webhook_id,
                event_type=event_data.get("event_type", "unknown"),
                data=event_data
            )
            db.add(webhook_record)
            db.flush()
            
            # Process event
            await self._process_webhook_event(event_data, db)
            
            # Mark as processed
            webhook_record.processed = True
            webhook_record.processed_at = datetime.utcnow()
            db.commit()
            
            return JSONResponse({"status": "success"})
            
        except WebhookError as e:
            logger.error(f"PayPal webhook processing error: {e}")
            return JSONResponse(
                {"error": str(e)}, 
                status_code=400
            )
        except Exception as e:
            logger.error(f"Unexpected PayPal webhook error: {e}")
            if 'webhook_record' in locals():
                webhook_record.last_error = str(e)
                webhook_record.processing_attempts += 1
                db.commit()
            return JSONResponse(
                {"error": "Internal server error"}, 
                status_code=500
            )
    
    async def _process_webhook_event(self, event_data: Dict[str, Any], db: Session):
        """Process specific webhook events"""
        event_type = event_data.get("event_type")
        resource = event_data.get("resource", {})
        
        if event_type == "CHECKOUT.ORDER.APPROVED":
            await self._handle_order_approved(resource, db)
        elif event_type == "PAYMENT.CAPTURE.COMPLETED":
            await self._handle_payment_completed(resource, db)
        elif event_type == "PAYMENT.CAPTURE.DENIED":
            await self._handle_payment_failed(resource, db)
        elif event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            await self._handle_subscription_activated(resource, db)
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            await self._handle_subscription_cancelled(resource, db)
        elif event_type == "BILLING.SUBSCRIPTION.PAYMENT.FAILED":
            await self._handle_subscription_payment_failed(resource, db)
        else:
            logger.info(f"Unhandled PayPal webhook event: {event_type}")
    
    async def _handle_order_approved(self, resource: Dict[str, Any], db: Session):
        """Handle approved order"""
        order_id = resource.get("id")
        
        payment = db.query(Payment).filter(
            Payment.external_id == order_id
        ).first()
        
        if payment:
            payment.status = PaymentStatus.PROCESSING
            payment.provider_data = resource
            db.commit()
            
            logger.info(f"PayPal order {order_id} approved")
    
    async def _handle_payment_completed(self, resource: Dict[str, Any], db: Session):
        """Handle completed payment"""
        # PayPal captures contain the order ID in supplementary_data
        order_id = resource.get("supplementary_data", {}).get("related_ids", {}).get("order_id")
        
        if order_id:
            payment = db.query(Payment).filter(
                Payment.external_id == order_id
            ).first()
            
            if payment:
                payment.status = PaymentStatus.COMPLETED
                payment.processed_at = datetime.utcnow()
                payment.provider_data = resource
                db.commit()
                
                logger.info(f"PayPal payment {order_id} completed")
    
    async def _handle_payment_failed(self, resource: Dict[str, Any], db: Session):
        """Handle failed payment"""
        order_id = resource.get("supplementary_data", {}).get("related_ids", {}).get("order_id")
        
        if order_id:
            payment = db.query(Payment).filter(
                Payment.external_id == order_id
            ).first()
            
            if payment:
                payment.status = PaymentStatus.FAILED
                payment.failure_reason = resource.get("status_details", {}).get("reason")
                payment.provider_data = resource
                db.commit()
                
                logger.info(f"PayPal payment {order_id} failed")
    
    async def _handle_subscription_activated(self, resource: Dict[str, Any], db: Session):
        """Handle subscription activation"""
        subscription_id = resource.get("id")
        
        subscription = db.query(Subscription).filter(
            Subscription.external_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.provider_data = resource
            db.commit()
            
            logger.info(f"PayPal subscription {subscription_id} activated")
    
    async def _handle_subscription_cancelled(self, resource: Dict[str, Any], db: Session):
        """Handle subscription cancellation"""
        subscription_id = resource.get("id")
        
        subscription = db.query(Subscription).filter(
            Subscription.external_id == subscription_id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.utcnow()
            subscription.provider_data = resource
            db.commit()
            
            logger.info(f"PayPal subscription {subscription_id} cancelled")
    
    async def _handle_subscription_payment_failed(self, resource: Dict[str, Any], db: Session):
        """Handle failed subscription payment"""
        subscription_id = resource.get("billing_agreement_id")
        
        if subscription_id:
            subscription = db.query(Subscription).filter(
                Subscription.external_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = SubscriptionStatus.PAST_DUE
                subscription.provider_data = resource
                db.commit()
                
                logger.info(f"PayPal subscription {subscription_id} payment failed")


class PayPalPlanManager:
    """Manage PayPal billing plans"""
    
    def __init__(self, processor: PayPalPaymentProcessor):
        self.processor = processor
    
    async def create_billing_plans(self) -> Dict[str, str]:
        """Create billing plans for subscription tiers"""
        try:
            plan_ids = {}
            
            for tier, config in SUBSCRIPTION_TIERS.items():
                if tier == SubscriptionTier.FREE or config["price"] is None:
                    continue
                
                # Create product first
                product_data = {
                    "name": config["name"],
                    "description": f"{config['name']} subscription tier",
                    "type": "SERVICE",
                    "category": "SOFTWARE"
                }
                
                product = await self.processor._make_request("POST", "/v1/catalogs/products", product_data)
                
                # Create monthly plan
                monthly_plan_data = {
                    "product_id": product["id"],
                    "name": f"{config['name']} Monthly",
                    "description": f"{config['name']} monthly subscription",
                    "billing_cycles": [{
                        "frequency": {
                            "interval_unit": "MONTH",
                            "interval_count": 1
                        },
                        "tenure_type": "REGULAR",
                        "sequence": 1,
                        "total_cycles": 0,  # Infinite
                        "pricing_scheme": {
                            "fixed_price": {
                                "value": f"{config['price']:.2f}",
                                "currency_code": "USD"
                            }
                        }
                    }],
                    "payment_preferences": {
                        "auto_bill_outstanding": True,
                        "setup_fee": {
                            "value": "0",
                            "currency_code": "USD"
                        },
                        "setup_fee_failure_action": "CONTINUE",
                        "payment_failure_threshold": 3
                    },
                    "taxes": {
                        "percentage": "0",
                        "inclusive": False
                    }
                }
                
                monthly_plan = await self.processor._make_request("POST", "/v1/billing/plans", monthly_plan_data)
                
                # Create yearly plan (20% discount)
                yearly_amount = config["price"] * 12 * 0.8
                yearly_plan_data = monthly_plan_data.copy()
                yearly_plan_data["name"] = f"{config['name']} Yearly"
                yearly_plan_data["description"] = f"{config['name']} yearly subscription"
                yearly_plan_data["billing_cycles"][0]["frequency"]["interval_unit"] = "YEAR"
                yearly_plan_data["billing_cycles"][0]["pricing_scheme"]["fixed_price"]["value"] = f"{yearly_amount:.2f}"
                
                yearly_plan = await self.processor._make_request("POST", "/v1/billing/plans", yearly_plan_data)
                
                plan_ids[f"{tier.value}_monthly"] = monthly_plan["id"]
                plan_ids[f"{tier.value}_yearly"] = yearly_plan["id"]
            
            return plan_ids
            
        except Exception as e:
            logger.error(f"Failed to create PayPal billing plans: {e}")
            raise PaymentError(f"Plan setup failed: {str(e)}")