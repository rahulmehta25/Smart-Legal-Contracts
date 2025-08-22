"""
Usage tracking and metering system for the arbitration RAG API.

This module provides comprehensive usage tracking including:
- Document processing usage
- API call metering
- Usage analytics
- Quota enforcement
- Billing integration
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import asyncio
from contextlib import asynccontextmanager

from ..core.config import get_settings
from .models import (
    UsageRecord, Subscription, User, SubscriptionTier,
    SUBSCRIPTION_TIERS, get_tier_config
)

logger = logging.getLogger(__name__)
settings = get_settings()


class UsageTracker:
    """Track and meter usage for billing purposes"""
    
    def __init__(self, db: Session):
        self.db = db
        self.resource_types = {
            "documents": "Document processing",
            "api_calls": "API requests",
            "storage": "Storage usage (MB)",
            "analysis_time": "Analysis time (seconds)"
        }
    
    async def record_usage(
        self,
        user_id: int,
        resource_type: str,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> UsageRecord:
        """Record a usage event"""
        try:
            if not timestamp:
                timestamp = datetime.utcnow()
            
            # Get user's active subscription
            subscription = await self._get_active_subscription(user_id)
            if not subscription:
                # Create free tier usage record
                subscription = await self._create_free_tier_subscription(user_id)
            
            # Get billing period for the timestamp
            billing_period_start, billing_period_end = self._get_billing_period(
                subscription, timestamp
            )
            
            # Create usage record
            usage_record = UsageRecord(
                user_id=user_id,
                subscription_id=subscription.id,
                resource_type=resource_type,
                quantity=quantity,
                timestamp=timestamp,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                metadata=metadata or {}
            )
            
            self.db.add(usage_record)
            self.db.commit()
            
            logger.debug(f"Recorded usage: user={user_id}, resource={resource_type}, quantity={quantity}")
            return usage_record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to record usage: {e}")
            raise
    
    async def check_usage_limits(
        self,
        user_id: int,
        resource_type: str,
        requested_quantity: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if user can perform the requested action within limits"""
        try:
            subscription = await self._get_active_subscription(user_id)
            if not subscription:
                # Use free tier limits
                tier_config = get_tier_config(SubscriptionTier.FREE)
            else:
                tier_config = get_tier_config(subscription.tier)
            
            # Get current usage for this billing period
            current_usage = await self.get_current_usage(user_id, resource_type)
            
            # Get limit for this resource type
            limit = self._get_resource_limit(tier_config, resource_type)
            
            # Check if request would exceed limit
            total_after_request = current_usage + requested_quantity
            within_limits = limit is None or total_after_request <= limit
            
            usage_info = {
                "current_usage": current_usage,
                "requested_quantity": requested_quantity,
                "limit": limit,
                "remaining": limit - current_usage if limit is not None else None,
                "within_limits": within_limits,
                "tier": subscription.tier.value if subscription else SubscriptionTier.FREE.value
            }
            
            if not within_limits:
                logger.warning(
                    f"Usage limit exceeded: user={user_id}, resource={resource_type}, "
                    f"current={current_usage}, requested={requested_quantity}, limit={limit}"
                )
            
            return within_limits, usage_info
            
        except Exception as e:
            logger.error(f"Failed to check usage limits: {e}")
            # Default to allowing the request in case of error
            return True, {"error": str(e)}
    
    async def get_current_usage(
        self,
        user_id: int,
        resource_type: str,
        billing_period_start: Optional[datetime] = None,
        billing_period_end: Optional[datetime] = None
    ) -> int:
        """Get current usage for a resource type in the billing period"""
        try:
            if not billing_period_start or not billing_period_end:
                subscription = await self._get_active_subscription(user_id)
                if subscription:
                    billing_period_start = subscription.current_period_start
                    billing_period_end = subscription.current_period_end
                else:
                    # Use current month for free tier
                    now = datetime.utcnow()
                    billing_period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    if now.month == 12:
                        billing_period_end = billing_period_start.replace(year=now.year + 1, month=1)
                    else:
                        billing_period_end = billing_period_start.replace(month=now.month + 1)
            
            # Sum usage for the billing period
            total_usage = self.db.query(func.sum(UsageRecord.quantity)).filter(
                UsageRecord.user_id == user_id,
                UsageRecord.resource_type == resource_type,
                UsageRecord.billing_period_start == billing_period_start,
                UsageRecord.billing_period_end == billing_period_end
            ).scalar()
            
            return int(total_usage or 0)
            
        except Exception as e:
            logger.error(f"Failed to get current usage: {e}")
            return 0
    
    def _get_resource_limit(self, tier_config: Dict[str, Any], resource_type: str) -> Optional[int]:
        """Get resource limit from tier configuration"""
        if resource_type == "documents":
            return tier_config.get("document_limit")
        elif resource_type == "api_calls":
            return tier_config.get("api_limit")
        else:
            # No limit for other resource types by default
            return None
    
    async def _get_active_subscription(self, user_id: int) -> Optional[Subscription]:
        """Get user's active subscription"""
        return self.db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active"
        ).first()
    
    async def _create_free_tier_subscription(self, user_id: int) -> Subscription:
        """Create a free tier subscription for new users"""
        from .models import SubscriptionStatus, PaymentProvider
        
        now = datetime.utcnow()
        
        subscription = Subscription(
            user_id=user_id,
            tier=SubscriptionTier.FREE,
            status=SubscriptionStatus.ACTIVE,
            provider=PaymentProvider.INTERNAL,
            amount=0.0,
            billing_interval="monthly",
            document_limit=get_tier_config(SubscriptionTier.FREE)["document_limit"],
            api_limit=get_tier_config(SubscriptionTier.FREE)["api_limit"],
            current_period_start=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            current_period_end=self._get_next_month_start(now)
        )
        
        self.db.add(subscription)
        self.db.commit()
        
        logger.info(f"Created free tier subscription for user {user_id}")
        return subscription
    
    def _get_next_month_start(self, date: datetime) -> datetime:
        """Get the start of the next month"""
        if date.month == 12:
            return date.replace(year=date.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return date.replace(month=date.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _get_billing_period(
        self,
        subscription: Subscription,
        timestamp: datetime
    ) -> Tuple[datetime, datetime]:
        """Get billing period for a timestamp"""
        if subscription.billing_interval == "yearly":
            # Annual billing period
            year = timestamp.year
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
        else:
            # Monthly billing period (default)
            start = subscription.current_period_start
            end = subscription.current_period_end
            
            # If timestamp is outside current period, calculate correct period
            while timestamp >= end:
                start = end
                if subscription.billing_interval == "yearly":
                    end = start.replace(year=start.year + 1)
                else:
                    end = self._get_next_month_start(start)
        
        return start, end
    
    async def get_usage_analytics(
        self,
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage analytics"""
        try:
            if not start_date:
                start_date = datetime.utcnow().replace(day=1)  # Start of current month
            if not end_date:
                end_date = datetime.utcnow()
            
            query = self.db.query(UsageRecord).filter(
                UsageRecord.timestamp >= start_date,
                UsageRecord.timestamp <= end_date
            )
            
            if user_id:
                query = query.filter(UsageRecord.user_id == user_id)
            
            if resource_type:
                query = query.filter(UsageRecord.resource_type == resource_type)
            
            usage_records = query.all()
            
            # Aggregate by resource type
            resource_totals = {}
            for record in usage_records:
                if record.resource_type not in resource_totals:
                    resource_totals[record.resource_type] = 0
                resource_totals[record.resource_type] += record.quantity
            
            # Aggregate by user (if not filtering by user)
            user_totals = {}
            if not user_id:
                for record in usage_records:
                    if record.user_id not in user_totals:
                        user_totals[record.user_id] = {}
                    if record.resource_type not in user_totals[record.user_id]:
                        user_totals[record.user_id][record.resource_type] = 0
                    user_totals[record.user_id][record.resource_type] += record.quantity
            
            # Daily usage patterns
            daily_usage = {}
            for record in usage_records:
                date_key = record.timestamp.date().isoformat()
                if date_key not in daily_usage:
                    daily_usage[date_key] = {}
                if record.resource_type not in daily_usage[date_key]:
                    daily_usage[date_key][record.resource_type] = 0
                daily_usage[date_key][record.resource_type] += record.quantity
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_records": len(usage_records),
                "resource_totals": resource_totals,
                "user_totals": user_totals,
                "daily_usage": daily_usage
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage analytics: {e}")
            return {}
    
    async def get_user_usage_summary(
        self,
        user_id: int,
        billing_period_start: Optional[datetime] = None,
        billing_period_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage summary for a user"""
        try:
            subscription = await self._get_active_subscription(user_id)
            
            if not billing_period_start or not billing_period_end:
                if subscription:
                    billing_period_start = subscription.current_period_start
                    billing_period_end = subscription.current_period_end
                else:
                    # Use current month for free tier
                    now = datetime.utcnow()
                    billing_period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    billing_period_end = self._get_next_month_start(now)
            
            # Get usage for each resource type
            usage_by_resource = {}
            for resource_type in self.resource_types.keys():
                current_usage = await self.get_current_usage(
                    user_id, resource_type, billing_period_start, billing_period_end
                )
                
                tier_config = get_tier_config(subscription.tier if subscription else SubscriptionTier.FREE)
                limit = self._get_resource_limit(tier_config, resource_type)
                
                usage_by_resource[resource_type] = {
                    "current_usage": current_usage,
                    "limit": limit,
                    "remaining": limit - current_usage if limit is not None else None,
                    "percentage_used": (current_usage / limit * 100) if limit else 0,
                    "description": self.resource_types[resource_type]
                }
            
            return {
                "user_id": user_id,
                "subscription_tier": subscription.tier.value if subscription else SubscriptionTier.FREE.value,
                "billing_period": {
                    "start": billing_period_start.isoformat(),
                    "end": billing_period_end.isoformat()
                },
                "usage": usage_by_resource
            }
            
        except Exception as e:
            logger.error(f"Failed to get user usage summary: {e}")
            return {}


class UsageEnforcer:
    """Enforce usage limits and quotas"""
    
    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker
    
    @asynccontextmanager
    async def enforce_usage_limit(
        self,
        user_id: int,
        resource_type: str,
        quantity: int = 1
    ):
        """Context manager to enforce usage limits"""
        # Check limits before allowing the operation
        allowed, usage_info = await self.usage_tracker.check_usage_limits(
            user_id, resource_type, quantity
        )
        
        if not allowed:
            raise UsageLimitExceededError(
                f"Usage limit exceeded for {resource_type}",
                usage_info
            )
        
        try:
            # Allow the operation
            yield usage_info
            
            # Record the usage after successful operation
            await self.usage_tracker.record_usage(
                user_id, resource_type, quantity
            )
            
        except Exception as e:
            # Don't record usage if operation failed
            logger.warning(f"Operation failed, not recording usage: {e}")
            raise
    
    async def get_rate_limit_info(
        self,
        user_id: int,
        resource_type: str = "api_calls"
    ) -> Dict[str, Any]:
        """Get rate limiting information"""
        subscription = await self.usage_tracker._get_active_subscription(user_id)
        tier_config = get_tier_config(subscription.tier if subscription else SubscriptionTier.FREE)
        
        # Calculate rate limits based on monthly allowance
        monthly_limit = self._get_resource_limit(tier_config, resource_type)
        if monthly_limit is None:
            return {"unlimited": True}
        
        # Calculate per-minute limit (assuming 30 days, 24 hours, 60 minutes)
        per_minute_limit = monthly_limit / (30 * 24 * 60)
        per_hour_limit = monthly_limit / (30 * 24)
        per_day_limit = monthly_limit / 30
        
        return {
            "monthly_limit": monthly_limit,
            "daily_limit": int(per_day_limit),
            "hourly_limit": int(per_hour_limit),
            "per_minute_limit": int(per_minute_limit)
        }


class UsageLimitExceededError(Exception):
    """Exception raised when usage limits are exceeded"""
    
    def __init__(self, message: str, usage_info: Dict[str, Any]):
        self.message = message
        self.usage_info = usage_info
        super().__init__(self.message)


class UsageAggregator:
    """Aggregate usage data for billing and analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def aggregate_monthly_usage(
        self,
        year: int,
        month: int
    ) -> Dict[str, Any]:
        """Aggregate usage for a specific month"""
        try:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            # Get all usage records for the month
            usage_records = self.db.query(UsageRecord).filter(
                UsageRecord.timestamp >= start_date,
                UsageRecord.timestamp < end_date
            ).all()
            
            # Aggregate by user and resource type
            user_usage = {}
            for record in usage_records:
                user_id = record.user_id
                resource_type = record.resource_type
                
                if user_id not in user_usage:
                    user_usage[user_id] = {}
                
                if resource_type not in user_usage[user_id]:
                    user_usage[user_id][resource_type] = 0
                
                user_usage[user_id][resource_type] += record.quantity
            
            # Calculate billing amounts
            billing_data = {}
            for user_id, usage in user_usage.items():
                billing_data[user_id] = await self._calculate_usage_billing(
                    user_id, usage, start_date, end_date
                )
            
            return {
                "period": {
                    "year": year,
                    "month": month,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_users": len(user_usage),
                "total_records": len(usage_records),
                "user_usage": user_usage,
                "billing_data": billing_data
            }
            
        except Exception as e:
            logger.error(f"Failed to aggregate monthly usage: {e}")
            return {}
    
    async def _calculate_usage_billing(
        self,
        user_id: int,
        usage: Dict[str, int],
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Calculate billing amounts for user usage"""
        try:
            from .usage_tracking import UsageTracker
            
            tracker = UsageTracker(self.db)
            subscription = await tracker._get_active_subscription(user_id)
            
            if not subscription:
                return {"tier": "free", "overage_charges": 0.0}
            
            tier_config = get_tier_config(subscription.tier)
            overage_charges = 0.0
            overage_details = {}
            
            # Calculate overage charges
            for resource_type, used_quantity in usage.items():
                limit = tracker._get_resource_limit(tier_config, resource_type)
                
                if limit is not None and used_quantity > limit:
                    overage_quantity = used_quantity - limit
                    overage_rate = self._get_overage_rate(resource_type)
                    overage_amount = overage_quantity * overage_rate
                    
                    overage_charges += overage_amount
                    overage_details[resource_type] = {
                        "limit": limit,
                        "used": used_quantity,
                        "overage": overage_quantity,
                        "rate": overage_rate,
                        "amount": overage_amount
                    }
            
            return {
                "tier": subscription.tier.value,
                "subscription_amount": subscription.amount,
                "overage_charges": overage_charges,
                "overage_details": overage_details,
                "total_amount": subscription.amount + overage_charges
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate usage billing: {e}")
            return {"error": str(e)}
    
    def _get_overage_rate(self, resource_type: str) -> float:
        """Get overage rate for resource type"""
        rates = {
            "documents": 0.10,  # $0.10 per document
            "api_calls": 0.001,  # $0.001 per API call
            "storage": 0.05,     # $0.05 per MB
            "analysis_time": 0.01  # $0.01 per second
        }
        return rates.get(resource_type, 0.0)