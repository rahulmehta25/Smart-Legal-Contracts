"""
Payment analytics engine for admin dashboard.

This module provides comprehensive payment analytics including:
- Revenue tracking and forecasting
- Customer lifetime value analysis
- Subscription metrics (MRR, churn, etc.)
- Payment success rates
- Geographic and demographic insights
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, case, extract
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PaymentAnalytics:
    """Comprehensive payment analytics engine"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_payment_overview(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get high-level payment overview"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, PaymentStatus
            
            # Get payments in the period
            payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date
            ).all()
            
            # Calculate metrics
            total_payments = len(payments)
            successful_payments = len([p for p in payments if p.status == PaymentStatus.COMPLETED])
            failed_payments = len([p for p in payments if p.status == PaymentStatus.FAILED])
            pending_payments = len([p for p in payments if p.status == PaymentStatus.PENDING])
            
            total_volume = sum(p.amount for p in payments if p.status == PaymentStatus.COMPLETED)
            average_transaction = total_volume / successful_payments if successful_payments > 0 else 0
            
            success_rate = (successful_payments / total_payments * 100) if total_payments > 0 else 0
            
            # Previous period comparison
            period_length = (end_date - start_date).days
            prev_start = start_date - timedelta(days=period_length)
            prev_end = start_date
            
            prev_payments = self.db.query(Payment).filter(
                Payment.created_at >= prev_start,
                Payment.created_at < prev_end
            ).all()
            
            prev_successful = len([p for p in prev_payments if p.status == PaymentStatus.COMPLETED])
            prev_volume = sum(p.amount for p in prev_payments if p.status == PaymentStatus.COMPLETED)
            
            volume_growth = ((total_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
            transaction_growth = ((successful_payments - prev_successful) / prev_successful * 100) if prev_successful > 0 else 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "totals": {
                    "total_payments": total_payments,
                    "successful_payments": successful_payments,
                    "failed_payments": failed_payments,
                    "pending_payments": pending_payments,
                    "total_volume": total_volume,
                    "average_transaction": average_transaction
                },
                "rates": {
                    "success_rate": success_rate,
                    "failure_rate": (failed_payments / total_payments * 100) if total_payments > 0 else 0
                },
                "growth": {
                    "volume_growth_percent": volume_growth,
                    "transaction_growth_percent": transaction_growth
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get payment overview: {e}")
            return {"error": str(e)}
    
    async def get_payment_trends(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: str = "daily"
    ) -> Dict[str, Any]:
        """Get payment trends over time"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, PaymentStatus
            
            # Generate date range
            if granularity == "daily":
                date_format = "%Y-%m-%d"
                delta = timedelta(days=1)
            elif granularity == "weekly":
                date_format = "%Y-W%U"
                delta = timedelta(weeks=1)
            elif granularity == "monthly":
                date_format = "%Y-%m"
                delta = timedelta(days=30)
            else:
                date_format = "%Y-%m-%d"
                delta = timedelta(days=1)
            
            # Get payments
            payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date
            ).all()
            
            # Group by date
            trends = {}
            current_date = start_date
            
            while current_date <= end_date:
                date_key = current_date.strftime(date_format)
                trends[date_key] = {
                    "date": date_key,
                    "total_payments": 0,
                    "successful_payments": 0,
                    "failed_payments": 0,
                    "total_volume": 0.0,
                    "average_amount": 0.0
                }
                current_date += delta
            
            # Populate with actual data
            for payment in payments:
                date_key = payment.created_at.strftime(date_format)
                
                if date_key in trends:
                    trends[date_key]["total_payments"] += 1
                    
                    if payment.status == PaymentStatus.COMPLETED:
                        trends[date_key]["successful_payments"] += 1
                        trends[date_key]["total_volume"] += payment.amount
                    elif payment.status == PaymentStatus.FAILED:
                        trends[date_key]["failed_payments"] += 1
            
            # Calculate averages
            for trend in trends.values():
                if trend["successful_payments"] > 0:
                    trend["average_amount"] = trend["total_volume"] / trend["successful_payments"]
                trend["success_rate"] = (trend["successful_payments"] / trend["total_payments"] * 100) if trend["total_payments"] > 0 else 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "granularity": granularity
                },
                "trends": list(trends.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to get payment trends: {e}")
            return {"error": str(e)}
    
    async def get_payment_method_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze payments by payment method"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, PaymentStatus
            
            payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date
            ).all()
            
            method_stats = {}
            
            for payment in payments:
                method = payment.payment_method.value
                
                if method not in method_stats:
                    method_stats[method] = {
                        "method": method,
                        "total_payments": 0,
                        "successful_payments": 0,
                        "failed_payments": 0,
                        "total_volume": 0.0,
                        "average_amount": 0.0
                    }
                
                method_stats[method]["total_payments"] += 1
                
                if payment.status == PaymentStatus.COMPLETED:
                    method_stats[method]["successful_payments"] += 1
                    method_stats[method]["total_volume"] += payment.amount
                elif payment.status == PaymentStatus.FAILED:
                    method_stats[method]["failed_payments"] += 1
            
            # Calculate rates and averages
            for stats in method_stats.values():
                if stats["successful_payments"] > 0:
                    stats["average_amount"] = stats["total_volume"] / stats["successful_payments"]
                
                stats["success_rate"] = (stats["successful_payments"] / stats["total_payments"] * 100) if stats["total_payments"] > 0 else 0
                stats["volume_share"] = 0  # Will calculate after getting total
            
            # Calculate volume share
            total_volume = sum(stats["total_volume"] for stats in method_stats.values())
            for stats in method_stats.values():
                stats["volume_share"] = (stats["total_volume"] / total_volume * 100) if total_volume > 0 else 0
            
            # Sort by volume
            sorted_methods = sorted(method_stats.values(), key=lambda x: x["total_volume"], reverse=True)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "payment_methods": sorted_methods,
                "summary": {
                    "total_methods": len(method_stats),
                    "total_volume": total_volume,
                    "top_method": sorted_methods[0]["method"] if sorted_methods else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get payment method analytics: {e}")
            return {"error": str(e)}
    
    async def get_geographic_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze payments by geographic location"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, PaymentStatus, User
            
            # Get payments with user information
            payments_with_users = self.db.query(Payment, User).join(User).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date
            ).all()
            
            country_stats = {}
            
            for payment, user in payments_with_users:
                # Extract country from user organization or default to "Unknown"
                country = "Unknown"
                if user.organization:
                    # In production, parse country from organization/address
                    # For now, use a simple heuristic
                    if "US" in user.organization.upper():
                        country = "US"
                    elif "UK" in user.organization.upper() or "BRITAIN" in user.organization.upper():
                        country = "GB"
                    elif "CANADA" in user.organization.upper():
                        country = "CA"
                    # Add more country detection logic as needed
                
                if country not in country_stats:
                    country_stats[country] = {
                        "country": country,
                        "total_payments": 0,
                        "successful_payments": 0,
                        "total_volume": 0.0,
                        "unique_customers": set()
                    }
                
                country_stats[country]["total_payments"] += 1
                country_stats[country]["unique_customers"].add(user.id)
                
                if payment.status == PaymentStatus.COMPLETED:
                    country_stats[country]["successful_payments"] += 1
                    country_stats[country]["total_volume"] += payment.amount
            
            # Convert sets to counts and calculate metrics
            for stats in country_stats.values():
                stats["unique_customers"] = len(stats["unique_customers"])
                stats["average_amount"] = (stats["total_volume"] / stats["successful_payments"]) if stats["successful_payments"] > 0 else 0
                stats["success_rate"] = (stats["successful_payments"] / stats["total_payments"] * 100) if stats["total_payments"] > 0 else 0
            
            # Sort by volume
            sorted_countries = sorted(country_stats.values(), key=lambda x: x["total_volume"], reverse=True)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "countries": sorted_countries,
                "summary": {
                    "total_countries": len(country_stats),
                    "top_country": sorted_countries[0]["country"] if sorted_countries else None,
                    "total_international_volume": sum(c["total_volume"] for c in sorted_countries if c["country"] != "US")
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get geographic analytics: {e}")
            return {"error": str(e)}
    
    async def get_failure_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze payment failures"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, PaymentStatus
            
            failed_payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date,
                Payment.status == PaymentStatus.FAILED
            ).all()
            
            # Analyze failure reasons
            failure_reasons = {}
            failure_codes = {}
            method_failures = {}
            
            for payment in failed_payments:
                # Group by failure reason
                reason = payment.failure_reason or "Unknown"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
                # Group by failure code
                code = payment.failure_code or "Unknown"
                failure_codes[code] = failure_codes.get(code, 0) + 1
                
                # Group by payment method
                method = payment.payment_method.value
                method_failures[method] = method_failures.get(method, 0) + 1
            
            # Convert to sorted lists
            top_reasons = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            top_codes = sorted(failure_codes.items(), key=lambda x: x[1], reverse=True)[:10]
            method_failure_rates = sorted(method_failures.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_failures": len(failed_payments),
                    "unique_failure_reasons": len(failure_reasons),
                    "unique_failure_codes": len(failure_codes)
                },
                "top_failure_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in top_reasons
                ],
                "top_failure_codes": [
                    {"code": code, "count": count}
                    for code, count in top_codes
                ],
                "failures_by_method": [
                    {"method": method, "failures": count}
                    for method, count in method_failure_rates
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get failure analysis: {e}")
            return {"error": str(e)}


class RevenueAnalytics:
    """Revenue-focused analytics for business intelligence"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_revenue_overview(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive revenue overview"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, Subscription, PaymentStatus, SubscriptionStatus
            
            # One-time payments revenue
            one_time_payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date,
                Payment.status == PaymentStatus.COMPLETED,
                Payment.subscription_id.is_(None)
            ).all()
            
            one_time_revenue = sum(p.amount for p in one_time_payments)
            
            # Subscription revenue
            subscription_payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date,
                Payment.status == PaymentStatus.COMPLETED,
                Payment.subscription_id.isnot(None)
            ).all()
            
            subscription_revenue = sum(p.amount for p in subscription_payments)
            
            # Active subscriptions
            active_subscriptions = self.db.query(Subscription).filter(
                Subscription.status == SubscriptionStatus.ACTIVE
            ).all()
            
            # Calculate MRR (Monthly Recurring Revenue)
            mrr = 0
            arr = 0  # Annual Recurring Revenue
            
            for subscription in active_subscriptions:
                if subscription.billing_interval == "monthly":
                    mrr += subscription.amount
                elif subscription.billing_interval == "yearly":
                    annual_amount = subscription.amount
                    mrr += annual_amount / 12
                    arr += annual_amount
            
            arr += mrr * 12  # Add monthly subscriptions to ARR
            
            # Previous period comparison
            period_length = (end_date - start_date).days
            prev_start = start_date - timedelta(days=period_length)
            prev_end = start_date
            
            prev_payments = self.db.query(Payment).filter(
                Payment.created_at >= prev_start,
                Payment.created_at < prev_end,
                Payment.status == PaymentStatus.COMPLETED
            ).all()
            
            prev_revenue = sum(p.amount for p in prev_payments)
            total_revenue = one_time_revenue + subscription_revenue
            
            revenue_growth = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "revenue": {
                    "total_revenue": total_revenue,
                    "one_time_revenue": one_time_revenue,
                    "subscription_revenue": subscription_revenue,
                    "recurring_percentage": (subscription_revenue / total_revenue * 100) if total_revenue > 0 else 0
                },
                "recurring_metrics": {
                    "mrr": mrr,
                    "arr": arr,
                    "active_subscriptions": len(active_subscriptions)
                },
                "growth": {
                    "revenue_growth_percent": revenue_growth,
                    "previous_period_revenue": prev_revenue
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get revenue overview: {e}")
            return {"error": str(e)}
    
    async def get_subscription_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed subscription metrics"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Subscription, SubscriptionStatus, SubscriptionTier
            
            # Get all subscriptions
            all_subscriptions = self.db.query(Subscription).all()
            
            # Filter by period for new subscriptions
            new_subscriptions = [
                s for s in all_subscriptions
                if start_date <= s.created_at <= end_date
            ]
            
            # Filter by period for cancelled subscriptions
            cancelled_subscriptions = [
                s for s in all_subscriptions
                if (s.cancelled_at and start_date <= s.cancelled_at <= end_date)
            ]
            
            # Active subscriptions
            active_subscriptions = [
                s for s in all_subscriptions
                if s.status == SubscriptionStatus.ACTIVE
            ]
            
            # Calculate churn rate
            period_start_active = len([
                s for s in all_subscriptions
                if s.created_at < start_date and 
                (not s.cancelled_at or s.cancelled_at > start_date)
            ])
            
            churn_rate = (len(cancelled_subscriptions) / period_start_active * 100) if period_start_active > 0 else 0
            
            # Subscription tiers breakdown
            tier_breakdown = {}
            for subscription in active_subscriptions:
                tier = subscription.tier.value
                if tier not in tier_breakdown:
                    tier_breakdown[tier] = {
                        "tier": tier,
                        "count": 0,
                        "mrr": 0,
                        "average_amount": 0
                    }
                
                tier_breakdown[tier]["count"] += 1
                
                # Calculate MRR contribution
                if subscription.billing_interval == "monthly":
                    tier_breakdown[tier]["mrr"] += subscription.amount
                elif subscription.billing_interval == "yearly":
                    tier_breakdown[tier]["mrr"] += subscription.amount / 12
            
            # Calculate average amounts
            for tier_data in tier_breakdown.values():
                if tier_data["count"] > 0:
                    tier_data["average_amount"] = tier_data["mrr"] / tier_data["count"]
            
            # Customer lifetime value (simplified)
            total_revenue = sum(s.amount for s in all_subscriptions)
            total_customers = len(set(s.user_id for s in all_subscriptions))
            avg_customer_value = total_revenue / total_customers if total_customers > 0 else 0
            
            # Average subscription duration
            completed_subscriptions = [s for s in all_subscriptions if s.cancelled_at]
            if completed_subscriptions:
                durations = [(s.cancelled_at - s.created_at).days for s in completed_subscriptions]
                avg_duration_days = sum(durations) / len(durations)
            else:
                avg_duration_days = 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "subscription_counts": {
                    "active_subscriptions": len(active_subscriptions),
                    "new_subscriptions": len(new_subscriptions),
                    "cancelled_subscriptions": len(cancelled_subscriptions),
                    "total_subscriptions": len(all_subscriptions)
                },
                "metrics": {
                    "churn_rate_percent": churn_rate,
                    "average_customer_value": avg_customer_value,
                    "average_duration_days": avg_duration_days
                },
                "tier_breakdown": list(tier_breakdown.values()),
                "growth": {
                    "net_new_subscriptions": len(new_subscriptions) - len(cancelled_subscriptions),
                    "growth_rate_percent": (len(new_subscriptions) / period_start_active * 100) if period_start_active > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get subscription metrics: {e}")
            return {"error": str(e)}
    
    async def get_customer_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze customer behavior and value"""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            from ..models import Payment, User, Subscription, PaymentStatus
            
            # Get customer payment data
            customers_with_payments = self.db.query(User).join(Payment).filter(
                Payment.created_at >= start_date,
                Payment.created_at <= end_date,
                Payment.status == PaymentStatus.COMPLETED
            ).distinct().all()
            
            # Calculate customer segments
            customer_segments = {
                "new_customers": 0,
                "returning_customers": 0,
                "high_value_customers": 0,
                "at_risk_customers": 0
            }
            
            customer_values = []
            
            for customer in customers_with_payments:
                # Get all customer payments
                customer_payments = self.db.query(Payment).filter(
                    Payment.user_id == customer.id,
                    Payment.status == PaymentStatus.COMPLETED
                ).all()
                
                total_value = sum(p.amount for p in customer_payments)
                customer_values.append(total_value)
                
                # First payment in this period?
                first_payment = min(customer_payments, key=lambda p: p.created_at)
                if start_date <= first_payment.created_at <= end_date:
                    customer_segments["new_customers"] += 1
                else:
                    customer_segments["returning_customers"] += 1
                
                # High value customer? (top 20% by value)
                if total_value > 1000:  # Simplified threshold
                    customer_segments["high_value_customers"] += 1
                
                # At risk? (no payments in last 30 days)
                last_payment = max(customer_payments, key=lambda p: p.created_at)
                if (datetime.utcnow() - last_payment.created_at).days > 30:
                    customer_segments["at_risk_customers"] += 1
            
            # Customer value distribution
            if customer_values:
                avg_customer_value = sum(customer_values) / len(customer_values)
                median_customer_value = sorted(customer_values)[len(customer_values) // 2]
                top_10_percent_value = sum(sorted(customer_values, reverse=True)[:len(customer_values) // 10])
            else:
                avg_customer_value = median_customer_value = top_10_percent_value = 0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "customer_segments": customer_segments,
                "value_metrics": {
                    "total_customers": len(customers_with_payments),
                    "average_customer_value": avg_customer_value,
                    "median_customer_value": median_customer_value,
                    "top_10_percent_value": top_10_percent_value
                },
                "insights": {
                    "new_customer_rate": (customer_segments["new_customers"] / len(customers_with_payments) * 100) if customers_with_payments else 0,
                    "customer_concentration": (top_10_percent_value / sum(customer_values) * 100) if customer_values else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get customer analytics: {e}")
            return {"error": str(e)}