"""
Dunning management system for failed payment recovery.

This module provides comprehensive dunning capabilities including:
- Failed payment retry logic
- Customer communication workflows
- Grace periods and account suspension
- Recovery tracking and analytics
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import asyncio

from ..core.config import get_settings
from .models import (
    Subscription, Payment, Invoice, User,
    SubscriptionStatus, PaymentStatus, PaymentProvider
)

logger = logging.getLogger(__name__)
settings = get_settings()


class DunningStage(str, Enum):
    """Dunning process stages"""
    INITIAL_FAILURE = "initial_failure"
    FIRST_RETRY = "first_retry"
    SECOND_RETRY = "second_retry"
    THIRD_RETRY = "third_retry"
    FINAL_NOTICE = "final_notice"
    ACCOUNT_SUSPENDED = "account_suspended"
    COLLECTION_AGENCY = "collection_agency"
    WRITTEN_OFF = "written_off"


class DunningAction(str, Enum):
    """Dunning actions"""
    SEND_EMAIL = "send_email"
    RETRY_PAYMENT = "retry_payment"
    SUSPEND_ACCOUNT = "suspend_account"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    SEND_TO_COLLECTIONS = "send_to_collections"
    WRITE_OFF = "write_off"


class DunningManager:
    """Comprehensive dunning management system"""
    
    def __init__(self, db: Session):
        self.db = db
        self.dunning_config = self._get_dunning_configuration()
    
    def _get_dunning_configuration(self) -> Dict[str, Dict[str, Any]]:
        """Get dunning configuration for each stage"""
        return {
            DunningStage.INITIAL_FAILURE: {
                "delay_days": 1,
                "actions": [DunningAction.SEND_EMAIL, DunningAction.RETRY_PAYMENT],
                "email_template": "payment_failed_initial",
                "retry_count": 1
            },
            DunningStage.FIRST_RETRY: {
                "delay_days": 3,
                "actions": [DunningAction.SEND_EMAIL, DunningAction.RETRY_PAYMENT],
                "email_template": "payment_failed_retry_1",
                "retry_count": 1
            },
            DunningStage.SECOND_RETRY: {
                "delay_days": 7,
                "actions": [DunningAction.SEND_EMAIL, DunningAction.RETRY_PAYMENT],
                "email_template": "payment_failed_retry_2",
                "retry_count": 1
            },
            DunningStage.THIRD_RETRY: {
                "delay_days": 14,
                "actions": [DunningAction.SEND_EMAIL, DunningAction.RETRY_PAYMENT],
                "email_template": "payment_failed_retry_3",
                "retry_count": 1
            },
            DunningStage.FINAL_NOTICE: {
                "delay_days": 21,
                "actions": [DunningAction.SEND_EMAIL],
                "email_template": "payment_failed_final_notice",
                "retry_count": 0
            },
            DunningStage.ACCOUNT_SUSPENDED: {
                "delay_days": 30,
                "actions": [DunningAction.SUSPEND_ACCOUNT, DunningAction.SEND_EMAIL],
                "email_template": "account_suspended",
                "retry_count": 0
            },
            DunningStage.COLLECTION_AGENCY: {
                "delay_days": 60,
                "actions": [DunningAction.SEND_TO_COLLECTIONS],
                "email_template": "sent_to_collections",
                "retry_count": 0
            },
            DunningStage.WRITTEN_OFF: {
                "delay_days": 120,
                "actions": [DunningAction.WRITE_OFF],
                "email_template": "debt_written_off",
                "retry_count": 0
            }
        }
    
    async def process_dunning_workflows(self) -> Dict[str, Any]:
        """Process all active dunning workflows"""
        try:
            # Get subscriptions in dunning process
            dunning_subscriptions = await self.get_subscriptions_in_dunning()
            
            processed_count = 0
            escalated_count = 0
            recovered_count = 0
            
            for subscription in dunning_subscriptions:
                try:
                    result = await self.process_subscription_dunning(subscription)
                    
                    if result["action_taken"]:
                        processed_count += 1
                        
                    if result.get("escalated"):
                        escalated_count += 1
                        
                    if result.get("recovered"):
                        recovered_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process dunning for subscription {subscription.id}: {e}")
                    continue
            
            logger.info(
                f"Dunning processing complete: processed={processed_count}, "
                f"escalated={escalated_count}, recovered={recovered_count}"
            )
            
            return {
                "processed": processed_count,
                "escalated": escalated_count,
                "recovered": recovered_count,
                "total_in_dunning": len(dunning_subscriptions)
            }
            
        except Exception as e:
            logger.error(f"Dunning workflow processing failed: {e}")
            return {"error": str(e)}
    
    async def get_subscriptions_in_dunning(self) -> List[Subscription]:
        """Get subscriptions that are in the dunning process"""
        try:
            # Get subscriptions that are past due or have failed payments
            past_due_subscriptions = self.db.query(Subscription).filter(
                Subscription.status == SubscriptionStatus.PAST_DUE
            ).all()
            
            # Also check for recently failed payments that might not have updated subscription status
            failed_payment_subscriptions = self.db.query(Subscription).join(Payment).filter(
                Payment.status == PaymentStatus.FAILED,
                Payment.created_at >= datetime.utcnow() - timedelta(days=30),
                Subscription.status == SubscriptionStatus.ACTIVE
            ).distinct().all()
            
            # Combine and deduplicate
            all_subscriptions = past_due_subscriptions + failed_payment_subscriptions
            unique_subscriptions = {sub.id: sub for sub in all_subscriptions}.values()
            
            return list(unique_subscriptions)
            
        except Exception as e:
            logger.error(f"Failed to get subscriptions in dunning: {e}")
            return []
    
    async def process_subscription_dunning(self, subscription: Subscription) -> Dict[str, Any]:
        """Process dunning for a single subscription"""
        try:
            logger.info(f"Processing dunning for subscription {subscription.id}")
            
            # Get dunning record or create new one
            dunning_record = await self.get_or_create_dunning_record(subscription.id)
            
            # Check if it's time for next action
            if not await self._is_time_for_next_action(dunning_record):
                return {"action_taken": False, "reason": "Not time for next action"}
            
            # Determine current dunning stage
            current_stage = self._determine_dunning_stage(dunning_record)
            stage_config = self.dunning_config[current_stage]
            
            result = {
                "subscription_id": subscription.id,
                "current_stage": current_stage.value,
                "action_taken": False,
                "actions_performed": [],
                "escalated": False,
                "recovered": False
            }
            
            # Check if payment has been recovered before taking action
            if await self._check_payment_recovery(subscription, dunning_record):
                await self._handle_payment_recovery(subscription, dunning_record)
                result["recovered"] = True
                result["action_taken"] = True
                return result
            
            # Execute stage actions
            for action in stage_config["actions"]:
                try:
                    action_result = await self._execute_dunning_action(
                        action, subscription, dunning_record, stage_config
                    )
                    
                    if action_result["success"]:
                        result["actions_performed"].append({
                            "action": action.value,
                            "result": action_result
                        })
                        result["action_taken"] = True
                        
                except Exception as e:
                    logger.error(f"Failed to execute dunning action {action}: {e}")
                    continue
            
            # Update dunning record
            await self._update_dunning_record(dunning_record, current_stage, result["actions_performed"])
            
            # Check if we need to escalate to next stage
            if await self._should_escalate_stage(dunning_record, current_stage):
                next_stage = self._get_next_dunning_stage(current_stage)
                if next_stage:
                    dunning_record["next_stage"] = next_stage.value
                    dunning_record["next_action_date"] = self._calculate_next_action_date(next_stage)
                    result["escalated"] = True
            
            self.db.commit()
            
            logger.info(f"Processed dunning for subscription {subscription.id}: {result}")
            return result
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to process dunning for subscription {subscription.id}: {e}")
            return {"action_taken": False, "error": str(e)}
    
    async def get_or_create_dunning_record(self, subscription_id: int) -> Dict[str, Any]:
        """Get existing dunning record or create new one"""
        try:
            # Check if subscription already has a dunning record in metadata
            subscription = self.db.query(Subscription).filter(
                Subscription.id == subscription_id
            ).first()
            
            metadata = subscription.metadata or {}
            dunning_record = metadata.get("dunning_record")
            
            if not dunning_record:
                # Create new dunning record
                dunning_record = {
                    "started_at": datetime.utcnow().isoformat(),
                    "current_stage": DunningStage.INITIAL_FAILURE.value,
                    "retry_count": 0,
                    "total_attempts": 0,
                    "last_action_date": None,
                    "next_action_date": self._calculate_next_action_date(DunningStage.INITIAL_FAILURE).isoformat(),
                    "recovery_attempts": [],
                    "communications_sent": [],
                    "total_amount_owed": 0.0
                }
                
                # Update subscription metadata
                metadata["dunning_record"] = dunning_record
                subscription.metadata = metadata
                self.db.commit()
            
            return dunning_record
            
        except Exception as e:
            logger.error(f"Failed to get/create dunning record: {e}")
            raise
    
    def _determine_dunning_stage(self, dunning_record: Dict[str, Any]) -> DunningStage:
        """Determine current dunning stage based on record"""
        current_stage = dunning_record.get("current_stage", DunningStage.INITIAL_FAILURE.value)
        return DunningStage(current_stage)
    
    async def _is_time_for_next_action(self, dunning_record: Dict[str, Any]) -> bool:
        """Check if it's time for the next dunning action"""
        next_action_date = dunning_record.get("next_action_date")
        if not next_action_date:
            return True
        
        next_action_datetime = datetime.fromisoformat(next_action_date.replace("Z", "+00:00"))
        return datetime.utcnow() >= next_action_datetime
    
    def _calculate_next_action_date(self, stage: DunningStage) -> datetime:
        """Calculate next action date for a dunning stage"""
        stage_config = self.dunning_config[stage]
        delay_days = stage_config["delay_days"]
        return datetime.utcnow() + timedelta(days=delay_days)
    
    async def _check_payment_recovery(
        self,
        subscription: Subscription,
        dunning_record: Dict[str, Any]
    ) -> bool:
        """Check if payment has been recovered"""
        try:
            # Check for recent successful payments
            recent_payments = self.db.query(Payment).filter(
                Payment.subscription_id == subscription.id,
                Payment.status == PaymentStatus.COMPLETED,
                Payment.processed_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            return len(recent_payments) > 0
            
        except Exception as e:
            logger.error(f"Failed to check payment recovery: {e}")
            return False
    
    async def _handle_payment_recovery(
        self,
        subscription: Subscription,
        dunning_record: Dict[str, Any]
    ):
        """Handle successful payment recovery"""
        try:
            # Update subscription status
            subscription.status = SubscriptionStatus.ACTIVE
            
            # Clear dunning record
            metadata = subscription.metadata or {}
            if "dunning_record" in metadata:
                # Keep history but mark as recovered
                metadata["dunning_record"]["recovered_at"] = datetime.utcnow().isoformat()
                metadata["dunning_record"]["status"] = "recovered"
                metadata["dunning_history"] = metadata.get("dunning_history", [])
                metadata["dunning_history"].append(metadata["dunning_record"])
                del metadata["dunning_record"]
            
            subscription.metadata = metadata
            
            # Send recovery notification
            await self._send_recovery_notification(subscription)
            
            logger.info(f"Payment recovered for subscription {subscription.id}")
            
        except Exception as e:
            logger.error(f"Failed to handle payment recovery: {e}")
            raise
    
    async def _execute_dunning_action(
        self,
        action: DunningAction,
        subscription: Subscription,
        dunning_record: Dict[str, Any],
        stage_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific dunning action"""
        try:
            if action == DunningAction.SEND_EMAIL:
                return await self._send_dunning_email(subscription, stage_config)
            
            elif action == DunningAction.RETRY_PAYMENT:
                return await self._retry_payment(subscription, dunning_record)
            
            elif action == DunningAction.SUSPEND_ACCOUNT:
                return await self._suspend_account(subscription)
            
            elif action == DunningAction.CANCEL_SUBSCRIPTION:
                return await self._cancel_subscription(subscription)
            
            elif action == DunningAction.SEND_TO_COLLECTIONS:
                return await self._send_to_collections(subscription)
            
            elif action == DunningAction.WRITE_OFF:
                return await self._write_off_debt(subscription)
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Failed to execute dunning action {action}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_dunning_email(
        self,
        subscription: Subscription,
        stage_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send dunning email to customer"""
        try:
            template = stage_config.get("email_template", "payment_failed_generic")
            
            # Get user information
            user = self.db.query(User).filter(User.id == subscription.user_id).first()
            if not user:
                return {"success": False, "error": "User not found"}
            
            # Get outstanding invoices
            outstanding_invoices = self.db.query(Invoice).filter(
                Invoice.user_id == subscription.user_id,
                Invoice.status.in_(["sent", "overdue"])
            ).all()
            
            total_amount_owed = sum(invoice.total_amount for invoice in outstanding_invoices)
            
            # Email content
            email_data = {
                "to": user.email,
                "template": template,
                "data": {
                    "user_name": user.full_name or user.username,
                    "subscription_tier": subscription.tier.value,
                    "amount_owed": total_amount_owed,
                    "invoice_count": len(outstanding_invoices),
                    "subscription_id": subscription.id
                }
            }
            
            # Send email (mock implementation)
            success = await self._send_email(email_data)
            
            if success:
                # Track email sent
                dunning_record = subscription.metadata.get("dunning_record", {})
                communications = dunning_record.get("communications_sent", [])
                communications.append({
                    "type": "email",
                    "template": template,
                    "sent_at": datetime.utcnow().isoformat(),
                    "recipient": user.email
                })
                dunning_record["communications_sent"] = communications
                
                metadata = subscription.metadata or {}
                metadata["dunning_record"] = dunning_record
                subscription.metadata = metadata
            
            return {"success": success, "email_sent": True, "template": template}
            
        except Exception as e:
            logger.error(f"Failed to send dunning email: {e}")
            return {"success": False, "error": str(e)}
    
    async def _retry_payment(
        self,
        subscription: Subscription,
        dunning_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retry failed payment"""
        try:
            # Get the most recent failed payment
            failed_payment = self.db.query(Payment).filter(
                Payment.subscription_id == subscription.id,
                Payment.status == PaymentStatus.FAILED
            ).order_by(Payment.created_at.desc()).first()
            
            if not failed_payment:
                return {"success": False, "error": "No failed payment found"}
            
            # Attempt payment retry based on provider
            if subscription.provider == PaymentProvider.STRIPE:
                result = await self._retry_stripe_payment(subscription, failed_payment)
            elif subscription.provider == PaymentProvider.PAYPAL:
                result = await self._retry_paypal_payment(subscription, failed_payment)
            else:
                return {"success": False, "error": "Payment retry not supported for this provider"}
            
            # Update retry count
            dunning_record["retry_count"] = dunning_record.get("retry_count", 0) + 1
            dunning_record["total_attempts"] = dunning_record.get("total_attempts", 0) + 1
            
            recovery_attempts = dunning_record.get("recovery_attempts", [])
            recovery_attempts.append({
                "attempted_at": datetime.utcnow().isoformat(),
                "method": "automatic_retry",
                "result": result
            })
            dunning_record["recovery_attempts"] = recovery_attempts
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retry payment: {e}")
            return {"success": False, "error": str(e)}
    
    async def _retry_stripe_payment(
        self,
        subscription: Subscription,
        failed_payment: Payment
    ) -> Dict[str, Any]:
        """Retry payment via Stripe"""
        try:
            from .stripe import StripePaymentProcessor
            
            stripe_processor = StripePaymentProcessor(
                api_key=settings.stripe_secret_key,
                webhook_secret=settings.stripe_webhook_secret
            )
            
            # Attempt to confirm the failed payment intent
            result = await stripe_processor.confirm_payment(failed_payment.external_id)
            
            if result.get("status") == "succeeded":
                # Update payment status
                failed_payment.status = PaymentStatus.COMPLETED
                failed_payment.processed_at = datetime.utcnow()
                failed_payment.provider_data = result
                
                # Update subscription status
                subscription.status = SubscriptionStatus.ACTIVE
                
                return {"success": True, "payment_id": result["id"]}
            else:
                return {"success": False, "reason": result.get("status", "Unknown")}
                
        except Exception as e:
            logger.error(f"Stripe payment retry failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _retry_paypal_payment(
        self,
        subscription: Subscription,
        failed_payment: Payment
    ) -> Dict[str, Any]:
        """Retry payment via PayPal"""
        # PayPal retries are typically handled by their subscription system
        return {"success": False, "reason": "PayPal retries handled automatically"}
    
    async def _suspend_account(self, subscription: Subscription) -> Dict[str, Any]:
        """Suspend user account"""
        try:
            # Update subscription status
            subscription.status = SubscriptionStatus.UNPAID
            
            # Add suspension metadata
            metadata = subscription.metadata or {}
            metadata["suspended_at"] = datetime.utcnow().isoformat()
            metadata["suspension_reason"] = "non_payment"
            subscription.metadata = metadata
            
            # Suspend user account
            user = self.db.query(User).filter(User.id == subscription.user_id).first()
            if user:
                user.is_active = False
                
                # Add suspension to user metadata
                user_metadata = getattr(user, 'metadata', None) or {}
                user_metadata["account_suspended"] = True
                user_metadata["suspended_at"] = datetime.utcnow().isoformat()
                user_metadata["suspension_reason"] = "payment_failure"
                
                # Store metadata (if User model supports it)
                if hasattr(user, 'metadata'):
                    user.metadata = user_metadata
            
            logger.info(f"Suspended account for subscription {subscription.id}")
            return {"success": True, "account_suspended": True}
            
        except Exception as e:
            logger.error(f"Failed to suspend account: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cancel_subscription(self, subscription: Subscription) -> Dict[str, Any]:
        """Cancel subscription due to non-payment"""
        try:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.utcnow()
            
            # Add cancellation metadata
            metadata = subscription.metadata or {}
            metadata["cancelled_reason"] = "non_payment"
            metadata["cancelled_by"] = "dunning_process"
            subscription.metadata = metadata
            
            logger.info(f"Cancelled subscription {subscription.id} due to non-payment")
            return {"success": True, "subscription_cancelled": True}
            
        except Exception as e:
            logger.error(f"Failed to cancel subscription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_to_collections(self, subscription: Subscription) -> Dict[str, Any]:
        """Send debt to collections agency"""
        try:
            # In production, integrate with collections agency API
            
            # Calculate total debt
            outstanding_invoices = self.db.query(Invoice).filter(
                Invoice.user_id == subscription.user_id,
                Invoice.status.in_(["sent", "overdue"])
            ).all()
            
            total_debt = sum(invoice.total_amount for invoice in outstanding_invoices)
            
            # Create collections record
            collections_data = {
                "subscription_id": subscription.id,
                "user_id": subscription.user_id,
                "total_debt": total_debt,
                "sent_to_collections_at": datetime.utcnow().isoformat(),
                "agency": "mock_collections_agency"
            }
            
            # Add to subscription metadata
            metadata = subscription.metadata or {}
            metadata["collections"] = collections_data
            subscription.metadata = metadata
            
            logger.info(f"Sent subscription {subscription.id} to collections: ${total_debt:.2f}")
            return {"success": True, "sent_to_collections": True, "amount": total_debt}
            
        except Exception as e:
            logger.error(f"Failed to send to collections: {e}")
            return {"success": False, "error": str(e)}
    
    async def _write_off_debt(self, subscription: Subscription) -> Dict[str, Any]:
        """Write off uncollectable debt"""
        try:
            # Calculate total debt
            outstanding_invoices = self.db.query(Invoice).filter(
                Invoice.user_id == subscription.user_id,
                Invoice.status.in_(["sent", "overdue"])
            ).all()
            
            total_debt = sum(invoice.total_amount for invoice in outstanding_invoices)
            
            # Mark invoices as written off
            for invoice in outstanding_invoices:
                invoice.status = "written_off"
                invoice.metadata = {
                    **(invoice.metadata or {}),
                    "written_off_at": datetime.utcnow().isoformat(),
                    "write_off_reason": "uncollectable"
                }
            
            # Update subscription
            subscription.status = SubscriptionStatus.CANCELLED
            metadata = subscription.metadata or {}
            metadata["debt_written_off"] = {
                "amount": total_debt,
                "written_off_at": datetime.utcnow().isoformat()
            }
            subscription.metadata = metadata
            
            logger.info(f"Wrote off debt for subscription {subscription.id}: ${total_debt:.2f}")
            return {"success": True, "debt_written_off": True, "amount": total_debt}
            
        except Exception as e:
            logger.error(f"Failed to write off debt: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_email(self, email_data: Dict[str, Any]) -> bool:
        """Send email (mock implementation)"""
        # In production, integrate with email service
        logger.info(f"Would send email: {email_data['template']} to {email_data['to']}")
        return True
    
    async def _send_recovery_notification(self, subscription: Subscription):
        """Send payment recovery notification"""
        user = self.db.query(User).filter(User.id == subscription.user_id).first()
        if user:
            email_data = {
                "to": user.email,
                "template": "payment_recovered",
                "data": {
                    "user_name": user.full_name or user.username,
                    "subscription_tier": subscription.tier.value
                }
            }
            await self._send_email(email_data)
    
    def _get_next_dunning_stage(self, current_stage: DunningStage) -> Optional[DunningStage]:
        """Get the next dunning stage"""
        stage_order = list(DunningStage)
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    async def _should_escalate_stage(
        self,
        dunning_record: Dict[str, Any],
        current_stage: DunningStage
    ) -> bool:
        """Determine if dunning should escalate to next stage"""
        stage_config = self.dunning_config[current_stage]
        retry_count = dunning_record.get("retry_count", 0)
        max_retries = stage_config.get("retry_count", 0)
        
        return retry_count >= max_retries
    
    async def _update_dunning_record(
        self,
        dunning_record: Dict[str, Any],
        current_stage: DunningStage,
        actions_performed: List[Dict[str, Any]]
    ):
        """Update dunning record with latest actions"""
        dunning_record["last_action_date"] = datetime.utcnow().isoformat()
        dunning_record["current_stage"] = current_stage.value
        
        if actions_performed:
            dunning_record["last_actions"] = actions_performed
    
    async def get_dunning_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get dunning process analytics"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get subscriptions with dunning records
            subscriptions_with_dunning = self.db.query(Subscription).filter(
                Subscription.metadata.isnot(None)
            ).all()
            
            # Filter subscriptions with actual dunning records
            dunning_subscriptions = []
            for sub in subscriptions_with_dunning:
                metadata = sub.metadata or {}
                if "dunning_record" in metadata or "dunning_history" in metadata:
                    dunning_subscriptions.append(sub)
            
            # Calculate metrics
            total_in_dunning = len(dunning_subscriptions)
            stage_breakdown = {}
            recovery_rate = 0
            total_recovered_amount = 0.0
            
            for subscription in dunning_subscriptions:
                metadata = subscription.metadata or {}
                
                # Current dunning stage
                if "dunning_record" in metadata:
                    stage = metadata["dunning_record"].get("current_stage", "unknown")
                    stage_breakdown[stage] = stage_breakdown.get(stage, 0) + 1
                
                # Recovery metrics
                if "dunning_history" in metadata:
                    for history_item in metadata["dunning_history"]:
                        if history_item.get("status") == "recovered":
                            recovery_rate += 1
                            # Estimate recovered amount (would need more detailed tracking in production)
                            total_recovered_amount += subscription.amount
            
            if total_in_dunning > 0:
                recovery_percentage = (recovery_rate / total_in_dunning) * 100
            else:
                recovery_percentage = 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "subscriptions_in_dunning": total_in_dunning,
                    "recovery_rate": recovery_rate,
                    "recovery_percentage": recovery_percentage,
                    "total_recovered_amount": total_recovered_amount
                },
                "stage_breakdown": stage_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to get dunning analytics: {e}")
            return {"error": str(e)}