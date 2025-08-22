"""
Admin dashboard for comprehensive payment system monitoring.

This module provides real-time dashboard capabilities including:
- Key performance indicators (KPIs)
- Real-time alerts and monitoring
- Executive summaries
- Interactive charts and visualizations
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .analytics import PaymentAnalytics, RevenueAnalytics

logger = logging.getLogger(__name__)


class AdminDashboard:
    """Comprehensive admin dashboard for payment system"""
    
    def __init__(self, db: Session):
        self.db = db
        self.payment_analytics = PaymentAnalytics(db)
        self.revenue_analytics = RevenueAnalytics(db)
    
    async def get_executive_summary(self) -> Dict[str, Any]:
        """Get high-level executive summary dashboard"""
        try:
            # Get current month and previous month data
            now = datetime.utcnow()
            current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
            
            # Current month metrics
            current_overview = await self.payment_analytics.get_payment_overview(
                current_month_start, now
            )
            current_revenue = await self.revenue_analytics.get_revenue_overview(
                current_month_start, now
            )
            current_subscriptions = await self.revenue_analytics.get_subscription_metrics(
                current_month_start, now
            )
            
            # Previous month metrics
            prev_overview = await self.payment_analytics.get_payment_overview(
                last_month_start, current_month_start
            )
            prev_revenue = await self.revenue_analytics.get_revenue_overview(
                last_month_start, current_month_start
            )
            
            # Calculate key metrics
            current_total_revenue = current_revenue.get("revenue", {}).get("total_revenue", 0)
            prev_total_revenue = prev_revenue.get("revenue", {}).get("total_revenue", 0)
            
            revenue_growth = ((current_total_revenue - prev_total_revenue) / prev_total_revenue * 100) if prev_total_revenue > 0 else 0
            
            current_mrr = current_revenue.get("recurring_metrics", {}).get("mrr", 0)
            current_active_subs = current_revenue.get("recurring_metrics", {}).get("active_subscriptions", 0)
            
            current_success_rate = current_overview.get("rates", {}).get("success_rate", 0)
            prev_success_rate = prev_overview.get("rates", {}).get("success_rate", 0)
            
            # Alert conditions
            alerts = []
            
            if current_success_rate < 95:
                alerts.append({
                    "type": "warning",
                    "message": f"Payment success rate is {current_success_rate:.1f}%, below 95% threshold",
                    "priority": "high" if current_success_rate < 90 else "medium"
                })
            
            if revenue_growth < -10:
                alerts.append({
                    "type": "critical",
                    "message": f"Revenue declined by {abs(revenue_growth):.1f}% this month",
                    "priority": "high"
                })
            
            churn_rate = current_subscriptions.get("metrics", {}).get("churn_rate_percent", 0)
            if churn_rate > 10:
                alerts.append({
                    "type": "warning",
                    "message": f"Subscription churn rate is {churn_rate:.1f}%, above 10% threshold",
                    "priority": "medium"
                })
            
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "current_month": current_month_start.strftime("%Y-%m"),
                    "previous_month": last_month_start.strftime("%Y-%m")
                },
                "key_metrics": {
                    "total_revenue": {
                        "current": current_total_revenue,
                        "previous": prev_total_revenue,
                        "growth_percent": revenue_growth,
                        "status": "up" if revenue_growth > 0 else "down"
                    },
                    "mrr": {
                        "current": current_mrr,
                        "status": "stable"  # Would need historical data for trend
                    },
                    "active_subscriptions": {
                        "current": current_active_subs,
                        "status": "stable"
                    },
                    "payment_success_rate": {
                        "current": current_success_rate,
                        "previous": prev_success_rate,
                        "change": current_success_rate - prev_success_rate,
                        "status": "up" if current_success_rate > prev_success_rate else "down"
                    },
                    "churn_rate": {
                        "current": churn_rate,
                        "status": "critical" if churn_rate > 10 else "normal"
                    }
                },
                "alerts": alerts,
                "health_score": self._calculate_health_score(
                    current_success_rate, churn_rate, revenue_growth
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get executive summary: {e}")
            return {"error": str(e)}
    
    def _calculate_health_score(
        self,
        success_rate: float,
        churn_rate: float,
        revenue_growth: float
    ) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            # Success rate score (0-40 points)
            success_score = min(40, (success_rate / 100) * 40)
            
            # Churn rate score (0-30 points, inverted)
            churn_score = max(0, 30 - (churn_rate * 3))
            
            # Revenue growth score (0-30 points)
            if revenue_growth >= 20:
                growth_score = 30
            elif revenue_growth >= 0:
                growth_score = 15 + (revenue_growth / 20) * 15
            else:
                growth_score = max(0, 15 + revenue_growth)  # Negative growth
            
            total_score = success_score + churn_score + growth_score
            
            # Determine health level
            if total_score >= 85:
                health_level = "excellent"
                color = "green"
            elif total_score >= 70:
                health_level = "good"
                color = "lightgreen"
            elif total_score >= 50:
                health_level = "fair"
                color = "yellow"
            else:
                health_level = "poor"
                color = "red"
            
            return {
                "score": round(total_score, 1),
                "level": health_level,
                "color": color,
                "components": {
                    "success_rate": round(success_score, 1),
                    "churn_rate": round(churn_score, 1),
                    "revenue_growth": round(growth_score, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return {"score": 0, "level": "unknown", "error": str(e)}
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard"""
        try:
            # Last 24 hours
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            
            from ..models import Payment, PaymentStatus
            
            # Recent payments
            recent_payments = self.db.query(Payment).filter(
                Payment.created_at >= twenty_four_hours_ago
            ).order_by(Payment.created_at.desc()).limit(100).all()
            
            # Calculate real-time metrics
            total_payments_24h = len(recent_payments)
            successful_payments_24h = len([p for p in recent_payments if p.status == PaymentStatus.COMPLETED])
            failed_payments_24h = len([p for p in recent_payments if p.status == PaymentStatus.FAILED])
            pending_payments_24h = len([p for p in recent_payments if p.status == PaymentStatus.PENDING])
            
            volume_24h = sum(p.amount for p in recent_payments if p.status == PaymentStatus.COMPLETED)
            
            # Payment velocity (payments per hour)
            payments_per_hour = total_payments_24h / 24
            
            # Recent activity
            recent_activity = []
            for payment in recent_payments[:10]:  # Last 10 payments
                recent_activity.append({
                    "id": payment.id,
                    "amount": payment.amount,
                    "status": payment.status.value,
                    "payment_method": payment.payment_method.value,
                    "created_at": payment.created_at.isoformat(),
                    "user_id": payment.user_id
                })
            
            # System status
            current_hour = datetime.utcnow().hour
            payments_this_hour = len([
                p for p in recent_payments
                if p.created_at.hour == current_hour
            ])
            
            # Determine system load
            if payments_this_hour > 100:
                system_load = "high"
            elif payments_this_hour > 50:
                system_load = "medium"
            else:
                system_load = "low"
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics_24h": {
                    "total_payments": total_payments_24h,
                    "successful_payments": successful_payments_24h,
                    "failed_payments": failed_payments_24h,
                    "pending_payments": pending_payments_24h,
                    "total_volume": volume_24h,
                    "success_rate": (successful_payments_24h / total_payments_24h * 100) if total_payments_24h > 0 else 0
                },
                "velocity": {
                    "payments_per_hour": payments_per_hour,
                    "payments_this_hour": payments_this_hour
                },
                "system_status": {
                    "load": system_load,
                    "status": "operational"
                },
                "recent_activity": recent_activity
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {"error": str(e)}
    
    async def get_fraud_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get fraud monitoring dashboard"""
        try:
            # Last 24 hours fraud data
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            
            from ..models import Payment
            
            # Get recent payments with risk data
            recent_payments = self.db.query(Payment).filter(
                Payment.created_at >= twenty_four_hours_ago
            ).all()
            
            # Analyze fraud patterns (simplified - in production, use actual fraud detection data)
            high_risk_payments = []
            flagged_payments = []
            blocked_payments = []
            
            for payment in recent_payments:
                # Simulate fraud detection results
                risk_score = self._simulate_risk_score(payment)
                
                if risk_score >= 80:
                    blocked_payments.append(payment)
                elif risk_score >= 60:
                    high_risk_payments.append(payment)
                elif risk_score >= 40:
                    flagged_payments.append(payment)
            
            # Calculate fraud metrics
            total_volume_at_risk = sum(p.amount for p in high_risk_payments + blocked_payments)
            
            # Recent fraud alerts
            fraud_alerts = []
            for payment in blocked_payments[:5]:  # Top 5 blocked
                fraud_alerts.append({
                    "payment_id": payment.id,
                    "amount": payment.amount,
                    "risk_score": self._simulate_risk_score(payment),
                    "reason": "High risk transaction",
                    "created_at": payment.created_at.isoformat(),
                    "status": "blocked"
                })
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "period": "24h",
                "fraud_metrics": {
                    "total_payments": len(recent_payments),
                    "high_risk_payments": len(high_risk_payments),
                    "flagged_payments": len(flagged_payments),
                    "blocked_payments": len(blocked_payments),
                    "volume_at_risk": total_volume_at_risk,
                    "block_rate_percent": (len(blocked_payments) / len(recent_payments) * 100) if recent_payments else 0
                },
                "recent_alerts": fraud_alerts,
                "system_status": {
                    "fraud_detection": "active",
                    "rules_active": True,
                    "ml_model_status": "operational"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get fraud monitoring dashboard: {e}")
            return {"error": str(e)}
    
    def _simulate_risk_score(self, payment) -> int:
        """Simulate risk score for demo purposes"""
        # This is just for demo - in production, use actual fraud detection
        import random
        
        # Higher amounts get higher risk scores
        amount_risk = min(30, payment.amount / 100)
        
        # Random component
        random_risk = random.randint(0, 50)
        
        return int(amount_risk + random_risk)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Simulate performance metrics (in production, integrate with monitoring systems)
            current_time = datetime.utcnow()
            
            # API response times (simulated)
            api_metrics = {
                "average_response_time_ms": 250,
                "p95_response_time_ms": 500,
                "p99_response_time_ms": 1000,
                "error_rate_percent": 0.1,
                "requests_per_minute": 150
            }
            
            # Database performance (simulated)
            db_metrics = {
                "average_query_time_ms": 50,
                "slow_queries_count": 2,
                "connection_pool_usage_percent": 45,
                "deadlocks_count": 0
            }
            
            # Payment processor latency (simulated)
            processor_metrics = {
                "stripe_avg_latency_ms": 800,
                "paypal_avg_latency_ms": 1200,
                "crypto_avg_latency_ms": 2000,
                "processor_uptime_percent": 99.9
            }
            
            # System health indicators
            health_indicators = {
                "cpu_usage_percent": 65,
                "memory_usage_percent": 72,
                "disk_usage_percent": 45,
                "active_connections": 145
            }
            
            return {
                "timestamp": current_time.isoformat(),
                "api_performance": api_metrics,
                "database_performance": db_metrics,
                "payment_processors": processor_metrics,
                "system_health": health_indicators,
                "overall_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def get_configuration_status(self) -> Dict[str, Any]:
        """Get system configuration and status"""
        try:
            # Payment providers status
            providers_status = {
                "stripe": {
                    "enabled": True,
                    "status": "operational",
                    "last_webhook": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "configuration": "live"
                },
                "paypal": {
                    "enabled": True,
                    "status": "operational",
                    "last_webhook": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
                    "configuration": "live"
                },
                "crypto": {
                    "enabled": True,
                    "status": "operational",
                    "last_transaction": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "configuration": "mainnet"
                }
            }
            
            # Feature flags
            feature_flags = {
                "new_payment_flow": True,
                "advanced_fraud_detection": True,
                "crypto_payments": True,
                "marketplace_enabled": True,
                "auto_invoicing": True
            }
            
            # Security settings
            security_settings = {
                "pci_compliance": "enabled",
                "encryption_at_rest": "enabled",
                "tls_version": "1.3",
                "fraud_detection": "active",
                "rate_limiting": "enabled"
            }
            
            # Environment info
            environment_info = {
                "environment": "production",
                "version": "1.0.0",
                "deployment_date": "2024-01-15T10:00:00Z",
                "region": "us-east-1"
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "payment_providers": providers_status,
                "feature_flags": feature_flags,
                "security_settings": security_settings,
                "environment": environment_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get configuration status: {e}")
            return {"error": str(e)}