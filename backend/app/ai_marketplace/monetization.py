"""
Model Monetization and Billing System

Handles usage tracking, billing, revenue sharing, and marketplace economics.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import stripe
import boto3
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import redis
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import hmac
import logging
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

Base = declarative_base()
logger = logging.getLogger(__name__)


class PricingModel(Enum):
    """Pricing models for AI models"""
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    TIERED = "tiered"
    FREEMIUM = "freemium"
    USAGE_BASED = "usage_based"
    FLAT_RATE = "flat_rate"
    CUSTOM = "custom"


class BillingPeriod(Enum):
    """Billing periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class TransactionType(Enum):
    """Transaction types"""
    MODEL_USAGE = "model_usage"
    SUBSCRIPTION_PAYMENT = "subscription_payment"
    ONE_TIME_PURCHASE = "one_time_purchase"
    REFUND = "refund"
    CREDIT = "credit"
    REVENUE_SHARE = "revenue_share"
    PLATFORM_FEE = "platform_fee"


class RevenueShareTier(Enum):
    """Revenue sharing tiers"""
    BRONZE = "bronze"  # 70% to developer
    SILVER = "silver"  # 75% to developer
    GOLD = "gold"      # 80% to developer
    PLATINUM = "platinum"  # 85% to developer
    ENTERPRISE = "enterprise"  # Custom negotiated


@dataclass
class PricingConfig:
    """Pricing configuration for a model"""
    model_id: str
    pricing_model: PricingModel
    base_price: Decimal
    currency: str = "USD"
    
    # Pay-per-use pricing
    price_per_request: Optional[Decimal] = None
    price_per_token: Optional[Decimal] = None
    price_per_second: Optional[Decimal] = None
    
    # Subscription pricing
    subscription_tiers: Optional[Dict[str, Decimal]] = None
    
    # Tiered pricing
    tier_limits: Optional[List[Dict[str, Any]]] = None
    
    # Volume discounts
    volume_discounts: Optional[List[Dict[str, Any]]] = None
    
    # Free tier
    free_requests: int = 0
    free_tokens: int = 0
    
    # Revenue sharing
    revenue_share_tier: RevenueShareTier = RevenueShareTier.BRONZE
    custom_revenue_share: Optional[float] = None
    
    # Additional costs
    setup_fee: Decimal = Decimal("0")
    minimum_charge: Decimal = Decimal("0")


@dataclass
class UsageRecord:
    """Usage record for billing"""
    user_id: str
    model_id: str
    timestamp: datetime
    request_count: int
    token_count: int
    compute_seconds: float
    input_size_bytes: int
    output_size_bytes: int
    latency_ms: float
    success: bool
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Invoice:
    """Invoice for user"""
    invoice_id: str
    user_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    line_items: List[Dict[str, Any]]
    subtotal: Decimal
    tax: Decimal
    total: Decimal
    currency: str
    status: str
    due_date: datetime
    paid_date: Optional[datetime] = None
    payment_method: Optional[str] = None


# Database Models

class UserAccount(Base):
    """User account for billing"""
    __tablename__ = 'user_accounts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), nullable=False)
    stripe_customer_id = Column(String(255))
    balance = Column(Numeric(10, 2), default=0)
    credit_limit = Column(Numeric(10, 2), default=0)
    currency = Column(String(3), default='USD')
    billing_address = Column(JSON)
    payment_methods = Column(JSON)
    auto_recharge = Column(Boolean, default=False)
    auto_recharge_amount = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    usage_records = relationship("UsageRecordDB", back_populates="user")
    invoices = relationship("InvoiceDB", back_populates="user")
    subscriptions = relationship("SubscriptionDB", back_populates="user")


class ModelPricing(Base):
    """Model pricing configuration"""
    __tablename__ = 'model_pricing'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), unique=True, nullable=False)
    pricing_model = Column(String(50), nullable=False)
    base_price = Column(Numeric(10, 4), default=0)
    currency = Column(String(3), default='USD')
    price_per_request = Column(Numeric(10, 6))
    price_per_token = Column(Numeric(10, 8))
    price_per_second = Column(Numeric(10, 4))
    subscription_tiers = Column(JSON)
    tier_limits = Column(JSON)
    volume_discounts = Column(JSON)
    free_requests = Column(Integer, default=0)
    free_tokens = Column(Integer, default=0)
    revenue_share_tier = Column(String(50), default=RevenueShareTier.BRONZE.value)
    custom_revenue_share = Column(Float)
    setup_fee = Column(Numeric(10, 2), default=0)
    minimum_charge = Column(Numeric(10, 2), default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UsageRecordDB(Base):
    """Usage records for billing"""
    __tablename__ = 'usage_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user_accounts.id'))
    model_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_count = Column(Integer, default=1)
    token_count = Column(Integer, default=0)
    compute_seconds = Column(Float, default=0)
    input_size_bytes = Column(Integer, default=0)
    output_size_bytes = Column(Integer, default=0)
    latency_ms = Column(Float)
    success = Column(Boolean, default=True)
    error_code = Column(String(50))
    cost = Column(Numeric(10, 6))
    billed = Column(Boolean, default=False)
    metadata = Column(JSON)
    
    user = relationship("UserAccount", back_populates="usage_records")


class InvoiceDB(Base):
    """Invoices"""
    __tablename__ = 'invoices'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_number = Column(String(50), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user_accounts.id'))
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    line_items = Column(JSON)
    subtotal = Column(Numeric(10, 2))
    tax = Column(Numeric(10, 2))
    total = Column(Numeric(10, 2))
    currency = Column(String(3), default='USD')
    status = Column(String(50), default='pending')
    due_date = Column(DateTime)
    paid_date = Column(DateTime)
    payment_method = Column(String(50))
    stripe_invoice_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("UserAccount", back_populates="invoices")


class SubscriptionDB(Base):
    """User subscriptions"""
    __tablename__ = 'subscriptions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user_accounts.id'))
    model_id = Column(String(255), nullable=False)
    plan_id = Column(String(50), nullable=False)
    status = Column(String(50), default='active')
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    renewal_date = Column(DateTime)
    auto_renew = Column(Boolean, default=True)
    price = Column(Numeric(10, 2))
    billing_period = Column(String(50))
    stripe_subscription_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    cancelled_at = Column(DateTime)
    
    user = relationship("UserAccount", back_populates="subscriptions")


class RevenueShareDB(Base):
    """Revenue share tracking"""
    __tablename__ = 'revenue_shares'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False)
    developer_id = Column(String(255), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    total_revenue = Column(Numeric(10, 2))
    platform_fee = Column(Numeric(10, 2))
    developer_share = Column(Numeric(10, 2))
    status = Column(String(50), default='pending')
    paid_date = Column(DateTime)
    payout_method = Column(String(50))
    transaction_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


# Metrics
usage_counter = Counter('model_usage_total', 'Total model usage', ['model_id', 'user_id'])
revenue_gauge = Gauge('model_revenue_total', 'Total revenue', ['model_id'])
billing_errors = Counter('billing_errors_total', 'Total billing errors')


class MonetizationEngine:
    """
    Comprehensive monetization and billing system
    """
    
    def __init__(self,
                 db_url: str = "postgresql://localhost/ai_marketplace",
                 stripe_api_key: Optional[str] = None,
                 cache_enabled: bool = True):
        
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize Stripe
        if stripe_api_key:
            stripe.api_key = stripe_api_key
        
        # Initialize cache
        self.cache = redis.Redis(host='localhost', port=6379, db=3) if cache_enabled else None
        
        # Initialize encryption for sensitive data
        self.cipher_suite = Fernet(Fernet.generate_key())
        
        # Platform configuration
        self.platform_fee_percentage = 0.15  # 15% platform fee by default
        self.tax_rate = 0.0  # Set based on jurisdiction
    
    async def track_usage(self, usage: UsageRecord) -> str:
        """
        Track model usage for billing
        
        Args:
            usage: Usage record
        
        Returns:
            Usage record ID
        """
        try:
            # Get user account
            user_account = self.session.query(UserAccount).filter_by(
                user_id=usage.user_id
            ).first()
            
            if not user_account:
                raise ValueError(f"User account not found: {usage.user_id}")
            
            # Get model pricing
            pricing = self.session.query(ModelPricing).filter_by(
                model_id=usage.model_id
            ).first()
            
            if not pricing:
                raise ValueError(f"Pricing not configured for model: {usage.model_id}")
            
            # Calculate cost
            cost = await self._calculate_usage_cost(usage, pricing)
            
            # Check if user has sufficient balance or credit
            if not await self._check_credit(user_account, cost):
                raise ValueError("Insufficient balance or credit limit exceeded")
            
            # Create usage record
            usage_db = UsageRecordDB(
                user_id=user_account.id,
                model_id=usage.model_id,
                timestamp=usage.timestamp,
                request_count=usage.request_count,
                token_count=usage.token_count,
                compute_seconds=usage.compute_seconds,
                input_size_bytes=usage.input_size_bytes,
                output_size_bytes=usage.output_size_bytes,
                latency_ms=usage.latency_ms,
                success=usage.success,
                error_code=usage.error_code,
                cost=cost,
                metadata=usage.metadata
            )
            
            # Update user balance
            user_account.balance -= cost
            
            # Update metrics
            usage_counter.labels(model_id=usage.model_id, user_id=usage.user_id).inc()
            revenue_gauge.labels(model_id=usage.model_id).inc(float(cost))
            
            # Cache usage for real-time analytics
            if self.cache:
                cache_key = f"usage:{usage.model_id}:{usage.user_id}:{datetime.now().strftime('%Y%m%d')}"
                self.cache.hincrby(cache_key, "requests", usage.request_count)
                self.cache.hincrby(cache_key, "tokens", usage.token_count)
                self.cache.hincrbyfloat(cache_key, "cost", float(cost))
                self.cache.expire(cache_key, 86400 * 7)  # Keep for 7 days
            
            # Commit to database
            self.session.add(usage_db)
            self.session.commit()
            
            # Check for auto-recharge
            if user_account.auto_recharge and user_account.balance < user_account.auto_recharge_amount:
                await self._auto_recharge(user_account)
            
            logger.info(f"Usage tracked: {usage_db.id}, cost: {cost}")
            return str(usage_db.id)
            
        except Exception as e:
            self.session.rollback()
            billing_errors.inc()
            logger.error(f"Usage tracking failed: {e}")
            raise
    
    async def create_subscription(self,
                                user_id: str,
                                model_id: str,
                                plan_id: str,
                                payment_method: str) -> str:
        """
        Create subscription for user
        
        Args:
            user_id: User ID
            model_id: Model ID
            plan_id: Subscription plan ID
            payment_method: Payment method ID
        
        Returns:
            Subscription ID
        """
        try:
            # Get user account
            user_account = self.session.query(UserAccount).filter_by(
                user_id=user_id
            ).first()
            
            if not user_account:
                raise ValueError(f"User account not found: {user_id}")
            
            # Get pricing configuration
            pricing = self.session.query(ModelPricing).filter_by(
                model_id=model_id
            ).first()
            
            if not pricing or not pricing.subscription_tiers:
                raise ValueError(f"Subscription not available for model: {model_id}")
            
            # Get plan details
            plan_price = pricing.subscription_tiers.get(plan_id)
            if not plan_price:
                raise ValueError(f"Invalid plan: {plan_id}")
            
            # Create Stripe subscription if configured
            stripe_subscription_id = None
            if stripe.api_key and user_account.stripe_customer_id:
                stripe_subscription = stripe.Subscription.create(
                    customer=user_account.stripe_customer_id,
                    items=[{
                        'price_data': {
                            'currency': pricing.currency.lower(),
                            'product_data': {
                                'name': f'Model {model_id} - {plan_id}',
                            },
                            'unit_amount': int(Decimal(plan_price) * 100),
                            'recurring': {
                                'interval': 'month'
                            }
                        }
                    }],
                    payment_behavior='default_incomplete',
                    expand=['latest_invoice.payment_intent']
                )
                stripe_subscription_id = stripe_subscription.id
            
            # Calculate dates
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30)  # Monthly by default
            
            # Create subscription record
            subscription = SubscriptionDB(
                user_id=user_account.id,
                model_id=model_id,
                plan_id=plan_id,
                status='active',
                start_date=start_date,
                end_date=end_date,
                renewal_date=end_date,
                price=Decimal(plan_price),
                billing_period=BillingPeriod.MONTHLY.value,
                stripe_subscription_id=stripe_subscription_id
            )
            
            self.session.add(subscription)
            self.session.commit()
            
            logger.info(f"Subscription created: {subscription.id}")
            return str(subscription.id)
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Subscription creation failed: {e}")
            raise
    
    async def generate_invoice(self,
                              user_id: str,
                              billing_period_start: datetime,
                              billing_period_end: datetime) -> Invoice:
        """
        Generate invoice for user
        
        Args:
            user_id: User ID
            billing_period_start: Billing period start
            billing_period_end: Billing period end
        
        Returns:
            Invoice
        """
        try:
            # Get user account
            user_account = self.session.query(UserAccount).filter_by(
                user_id=user_id
            ).first()
            
            if not user_account:
                raise ValueError(f"User account not found: {user_id}")
            
            # Get unbilled usage records
            usage_records = self.session.query(UsageRecordDB).filter(
                UsageRecordDB.user_id == user_account.id,
                UsageRecordDB.timestamp >= billing_period_start,
                UsageRecordDB.timestamp < billing_period_end,
                UsageRecordDB.billed == False
            ).all()
            
            # Group by model and calculate totals
            line_items = []
            subtotal = Decimal("0")
            
            model_usage = {}
            for record in usage_records:
                if record.model_id not in model_usage:
                    model_usage[record.model_id] = {
                        'requests': 0,
                        'tokens': 0,
                        'compute_seconds': 0,
                        'cost': Decimal("0")
                    }
                
                model_usage[record.model_id]['requests'] += record.request_count
                model_usage[record.model_id]['tokens'] += record.token_count
                model_usage[record.model_id]['compute_seconds'] += record.compute_seconds
                model_usage[record.model_id]['cost'] += record.cost
            
            # Create line items
            for model_id, usage in model_usage.items():
                line_items.append({
                    'description': f'Model {model_id} Usage',
                    'quantity': usage['requests'],
                    'unit_price': float(usage['cost'] / usage['requests']) if usage['requests'] > 0 else 0,
                    'total': float(usage['cost']),
                    'details': {
                        'tokens': usage['tokens'],
                        'compute_seconds': usage['compute_seconds']
                    }
                })
                subtotal += usage['cost']
            
            # Add subscription charges
            subscriptions = self.session.query(SubscriptionDB).filter(
                SubscriptionDB.user_id == user_account.id,
                SubscriptionDB.status == 'active',
                SubscriptionDB.renewal_date >= billing_period_start,
                SubscriptionDB.renewal_date < billing_period_end
            ).all()
            
            for subscription in subscriptions:
                line_items.append({
                    'description': f'Subscription - Model {subscription.model_id} ({subscription.plan_id})',
                    'quantity': 1,
                    'unit_price': float(subscription.price),
                    'total': float(subscription.price)
                })
                subtotal += subscription.price
            
            # Calculate tax
            tax = subtotal * Decimal(str(self.tax_rate))
            total = subtotal + tax
            
            # Generate invoice number
            invoice_number = f"INV-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:8].upper()}"
            
            # Create invoice
            invoice = Invoice(
                invoice_id=str(uuid.uuid4()),
                user_id=user_id,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                line_items=line_items,
                subtotal=subtotal,
                tax=tax,
                total=total,
                currency=user_account.currency,
                status='pending',
                due_date=datetime.utcnow() + timedelta(days=30)
            )
            
            # Save to database
            invoice_db = InvoiceDB(
                invoice_number=invoice_number,
                user_id=user_account.id,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                line_items=line_items,
                subtotal=subtotal,
                tax=tax,
                total=total,
                currency=user_account.currency,
                status='pending',
                due_date=invoice.due_date
            )
            
            # Mark usage records as billed
            for record in usage_records:
                record.billed = True
            
            self.session.add(invoice_db)
            self.session.commit()
            
            # Send invoice email
            await self._send_invoice_email(user_account, invoice)
            
            logger.info(f"Invoice generated: {invoice_number}")
            return invoice
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Invoice generation failed: {e}")
            raise
    
    async def process_payment(self,
                            invoice_id: str,
                            payment_method: str,
                            amount: Optional[Decimal] = None) -> bool:
        """
        Process payment for invoice
        
        Args:
            invoice_id: Invoice ID
            payment_method: Payment method
            amount: Payment amount (optional, uses invoice total if not specified)
        
        Returns:
            Success status
        """
        try:
            # Get invoice
            invoice = self.session.query(InvoiceDB).filter_by(id=invoice_id).first()
            if not invoice:
                raise ValueError(f"Invoice not found: {invoice_id}")
            
            if invoice.status == 'paid':
                return True  # Already paid
            
            # Get user account
            user_account = self.session.query(UserAccount).filter_by(
                id=invoice.user_id
            ).first()
            
            payment_amount = amount or invoice.total
            
            # Process payment based on method
            if payment_method == 'balance':
                # Pay from account balance
                if user_account.balance < payment_amount:
                    raise ValueError("Insufficient balance")
                
                user_account.balance -= payment_amount
                
            elif payment_method == 'stripe':
                # Process Stripe payment
                if not stripe.api_key or not user_account.stripe_customer_id:
                    raise ValueError("Stripe not configured")
                
                payment_intent = stripe.PaymentIntent.create(
                    amount=int(payment_amount * 100),
                    currency=invoice.currency.lower(),
                    customer=user_account.stripe_customer_id,
                    description=f"Invoice {invoice.invoice_number}",
                    metadata={'invoice_id': str(invoice_id)}
                )
                
                # Wait for payment confirmation
                if payment_intent.status != 'succeeded':
                    raise ValueError("Payment failed")
            
            # Update invoice
            invoice.status = 'paid'
            invoice.paid_date = datetime.utcnow()
            invoice.payment_method = payment_method
            
            self.session.commit()
            
            # Process revenue sharing
            await self._process_revenue_sharing(invoice)
            
            logger.info(f"Payment processed for invoice: {invoice.invoice_number}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Payment processing failed: {e}")
            raise
    
    async def calculate_revenue_share(self,
                                    model_id: str,
                                    period_start: datetime,
                                    period_end: datetime) -> Dict[str, Decimal]:
        """
        Calculate revenue share for model
        
        Args:
            model_id: Model ID
            period_start: Period start date
            period_end: Period end date
        
        Returns:
            Revenue share breakdown
        """
        try:
            # Get model pricing
            pricing = self.session.query(ModelPricing).filter_by(
                model_id=model_id
            ).first()
            
            if not pricing:
                raise ValueError(f"Pricing not found for model: {model_id}")
            
            # Calculate total revenue
            total_revenue = self.session.query(
                func.sum(UsageRecordDB.cost)
            ).filter(
                UsageRecordDB.model_id == model_id,
                UsageRecordDB.timestamp >= period_start,
                UsageRecordDB.timestamp < period_end,
                UsageRecordDB.billed == True
            ).scalar() or Decimal("0")
            
            # Determine revenue share percentage
            if pricing.custom_revenue_share:
                developer_percentage = pricing.custom_revenue_share
            else:
                share_percentages = {
                    RevenueShareTier.BRONZE: 0.70,
                    RevenueShareTier.SILVER: 0.75,
                    RevenueShareTier.GOLD: 0.80,
                    RevenueShareTier.PLATINUM: 0.85,
                    RevenueShareTier.ENTERPRISE: 0.90
                }
                developer_percentage = share_percentages.get(
                    RevenueShareTier(pricing.revenue_share_tier),
                    0.70
                )
            
            # Calculate shares
            platform_fee = total_revenue * Decimal(str(1 - developer_percentage))
            developer_share = total_revenue - platform_fee
            
            return {
                'total_revenue': total_revenue,
                'platform_fee': platform_fee,
                'developer_share': developer_share,
                'developer_percentage': developer_percentage * 100
            }
            
        except Exception as e:
            logger.error(f"Revenue share calculation failed: {e}")
            raise
    
    async def process_payouts(self, cutoff_date: Optional[datetime] = None) -> List[str]:
        """
        Process developer payouts
        
        Args:
            cutoff_date: Process payouts up to this date
        
        Returns:
            List of processed payout IDs
        """
        try:
            cutoff = cutoff_date or datetime.utcnow()
            processed_payouts = []
            
            # Get pending revenue shares
            pending_shares = self.session.query(RevenueShareDB).filter(
                RevenueShareDB.status == 'pending',
                RevenueShareDB.period_end <= cutoff
            ).all()
            
            for share in pending_shares:
                try:
                    # Process payout based on method
                    if share.payout_method == 'stripe':
                        # Create Stripe transfer
                        transfer = stripe.Transfer.create(
                            amount=int(share.developer_share * 100),
                            currency='usd',
                            destination=share.developer_id,  # Stripe Connect account
                            description=f"Revenue share for model {share.model_id}"
                        )
                        share.transaction_id = transfer.id
                    
                    elif share.payout_method == 'bank_transfer':
                        # Process bank transfer (implementation depends on banking API)
                        pass
                    
                    # Update status
                    share.status = 'paid'
                    share.paid_date = datetime.utcnow()
                    
                    processed_payouts.append(str(share.id))
                    
                except Exception as e:
                    logger.error(f"Payout failed for {share.id}: {e}")
                    share.status = 'failed'
            
            self.session.commit()
            
            logger.info(f"Processed {len(processed_payouts)} payouts")
            return processed_payouts
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Payout processing failed: {e}")
            raise
    
    async def get_usage_analytics(self,
                                 model_id: str,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """
        Get usage analytics for model
        
        Args:
            model_id: Model ID
            start_date: Start date
            end_date: End date
        
        Returns:
            Usage analytics
        """
        try:
            # Query usage data
            usage_data = self.session.query(UsageRecordDB).filter(
                UsageRecordDB.model_id == model_id,
                UsageRecordDB.timestamp >= start_date,
                UsageRecordDB.timestamp < end_date
            ).all()
            
            if not usage_data:
                return {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'total_revenue': 0,
                    'unique_users': 0,
                    'avg_latency_ms': 0,
                    'success_rate': 0
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'user_id': r.user_id,
                'requests': r.request_count,
                'tokens': r.token_count,
                'compute_seconds': r.compute_seconds,
                'latency_ms': r.latency_ms,
                'cost': float(r.cost),
                'success': r.success
            } for r in usage_data])
            
            # Calculate metrics
            analytics = {
                'total_requests': int(df['requests'].sum()),
                'total_tokens': int(df['tokens'].sum()),
                'total_compute_hours': float(df['compute_seconds'].sum() / 3600),
                'total_revenue': float(df['cost'].sum()),
                'unique_users': int(df['user_id'].nunique()),
                'avg_latency_ms': float(df['latency_ms'].mean()),
                'p95_latency_ms': float(df['latency_ms'].quantile(0.95)),
                'success_rate': float(df['success'].mean() * 100),
                
                # Time series data
                'daily_usage': df.groupby(df['timestamp'].dt.date).agg({
                    'requests': 'sum',
                    'tokens': 'sum',
                    'cost': 'sum'
                }).to_dict('index'),
                
                # User distribution
                'top_users': df.groupby('user_id').agg({
                    'requests': 'sum',
                    'cost': 'sum'
                }).nlargest(10, 'requests').to_dict('index'),
                
                # Peak usage times
                'hourly_distribution': df.groupby(df['timestamp'].dt.hour)['requests'].sum().to_dict()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            raise
    
    async def set_pricing(self, pricing_config: PricingConfig) -> bool:
        """
        Set or update model pricing
        
        Args:
            pricing_config: Pricing configuration
        
        Returns:
            Success status
        """
        try:
            # Check if pricing exists
            existing = self.session.query(ModelPricing).filter_by(
                model_id=pricing_config.model_id
            ).first()
            
            if existing:
                # Update existing pricing
                existing.pricing_model = pricing_config.pricing_model.value
                existing.base_price = pricing_config.base_price
                existing.currency = pricing_config.currency
                existing.price_per_request = pricing_config.price_per_request
                existing.price_per_token = pricing_config.price_per_token
                existing.price_per_second = pricing_config.price_per_second
                existing.subscription_tiers = pricing_config.subscription_tiers
                existing.tier_limits = pricing_config.tier_limits
                existing.volume_discounts = pricing_config.volume_discounts
                existing.free_requests = pricing_config.free_requests
                existing.free_tokens = pricing_config.free_tokens
                existing.revenue_share_tier = pricing_config.revenue_share_tier.value
                existing.custom_revenue_share = pricing_config.custom_revenue_share
                existing.setup_fee = pricing_config.setup_fee
                existing.minimum_charge = pricing_config.minimum_charge
                existing.updated_at = datetime.utcnow()
            else:
                # Create new pricing
                new_pricing = ModelPricing(
                    model_id=pricing_config.model_id,
                    pricing_model=pricing_config.pricing_model.value,
                    base_price=pricing_config.base_price,
                    currency=pricing_config.currency,
                    price_per_request=pricing_config.price_per_request,
                    price_per_token=pricing_config.price_per_token,
                    price_per_second=pricing_config.price_per_second,
                    subscription_tiers=pricing_config.subscription_tiers,
                    tier_limits=pricing_config.tier_limits,
                    volume_discounts=pricing_config.volume_discounts,
                    free_requests=pricing_config.free_requests,
                    free_tokens=pricing_config.free_tokens,
                    revenue_share_tier=pricing_config.revenue_share_tier.value,
                    custom_revenue_share=pricing_config.custom_revenue_share,
                    setup_fee=pricing_config.setup_fee,
                    minimum_charge=pricing_config.minimum_charge
                )
                self.session.add(new_pricing)
            
            self.session.commit()
            
            # Clear cache
            if self.cache:
                self.cache.delete(f"pricing:{pricing_config.model_id}")
            
            logger.info(f"Pricing updated for model: {pricing_config.model_id}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Pricing update failed: {e}")
            raise
    
    async def _calculate_usage_cost(self, 
                                   usage: UsageRecord,
                                   pricing: ModelPricing) -> Decimal:
        """Calculate cost for usage"""
        cost = Decimal("0")
        
        # Check free tier
        free_requests_used = await self._get_free_tier_usage(usage.user_id, usage.model_id)
        
        if free_requests_used < pricing.free_requests:
            # Within free tier
            return Decimal("0")
        
        # Calculate based on pricing model
        if pricing.pricing_model == PricingModel.PAY_PER_USE.value:
            if pricing.price_per_request:
                cost += pricing.price_per_request * usage.request_count
            if pricing.price_per_token:
                cost += pricing.price_per_token * usage.token_count
            if pricing.price_per_second:
                cost += pricing.price_per_second * Decimal(str(usage.compute_seconds))
        
        elif pricing.pricing_model == PricingModel.TIERED.value:
            # Apply tiered pricing
            if pricing.tier_limits:
                monthly_usage = await self._get_monthly_usage(usage.user_id, usage.model_id)
                for tier in pricing.tier_limits:
                    if monthly_usage <= tier['limit']:
                        cost = Decimal(str(tier['price'])) * usage.request_count
                        break
        
        # Apply volume discounts
        if pricing.volume_discounts:
            monthly_volume = await self._get_monthly_volume(usage.user_id, usage.model_id)
            for discount in pricing.volume_discounts:
                if monthly_volume >= discount['threshold']:
                    cost *= Decimal(str(1 - discount['discount']))
        
        # Apply minimum charge
        if cost < pricing.minimum_charge:
            cost = pricing.minimum_charge
        
        return cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
    
    async def _check_credit(self, user_account: UserAccount, amount: Decimal) -> bool:
        """Check if user has sufficient credit"""
        available_credit = user_account.balance + user_account.credit_limit
        return available_credit >= amount
    
    async def _auto_recharge(self, user_account: UserAccount):
        """Auto-recharge user account"""
        try:
            if not user_account.stripe_customer_id:
                return
            
            # Create charge
            charge = stripe.Charge.create(
                amount=int(user_account.auto_recharge_amount * 100),
                currency=user_account.currency.lower(),
                customer=user_account.stripe_customer_id,
                description="Auto-recharge"
            )
            
            if charge.status == 'succeeded':
                user_account.balance += user_account.auto_recharge_amount
                self.session.commit()
                logger.info(f"Auto-recharge successful for user: {user_account.user_id}")
        
        except Exception as e:
            logger.error(f"Auto-recharge failed: {e}")
    
    async def _get_free_tier_usage(self, user_id: str, model_id: str) -> int:
        """Get free tier usage for current month"""
        if self.cache:
            cache_key = f"free_tier:{user_id}:{model_id}:{datetime.now().strftime('%Y%m')}"
            usage = self.cache.get(cache_key)
            if usage:
                return int(usage)
        
        # Query database
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        
        usage = self.session.query(
            func.sum(UsageRecordDB.request_count)
        ).filter(
            UsageRecordDB.user_id == user_id,
            UsageRecordDB.model_id == model_id,
            UsageRecordDB.timestamp >= start_of_month
        ).scalar() or 0
        
        return usage
    
    async def _get_monthly_usage(self, user_id: str, model_id: str) -> int:
        """Get monthly usage for user and model"""
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        
        usage = self.session.query(
            func.sum(UsageRecordDB.request_count)
        ).filter(
            UsageRecordDB.user_id == user_id,
            UsageRecordDB.model_id == model_id,
            UsageRecordDB.timestamp >= start_of_month
        ).scalar() or 0
        
        return usage
    
    async def _get_monthly_volume(self, user_id: str, model_id: str) -> Decimal:
        """Get monthly volume for user and model"""
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        
        volume = self.session.query(
            func.sum(UsageRecordDB.cost)
        ).filter(
            UsageRecordDB.user_id == user_id,
            UsageRecordDB.model_id == model_id,
            UsageRecordDB.timestamp >= start_of_month
        ).scalar() or Decimal("0")
        
        return volume
    
    async def _process_revenue_sharing(self, invoice: InvoiceDB):
        """Process revenue sharing for paid invoice"""
        try:
            # Group revenue by model
            model_revenues = {}
            
            for item in invoice.line_items:
                if 'Model' in item.get('description', ''):
                    model_id = item['description'].split(' ')[1]
                    if model_id not in model_revenues:
                        model_revenues[model_id] = Decimal("0")
                    model_revenues[model_id] += Decimal(str(item['total']))
            
            # Create revenue share records
            for model_id, revenue in model_revenues.items():
                pricing = self.session.query(ModelPricing).filter_by(
                    model_id=model_id
                ).first()
                
                if not pricing:
                    continue
                
                # Calculate shares
                share_info = await self.calculate_revenue_share(
                    model_id,
                    invoice.billing_period_start,
                    invoice.billing_period_end
                )
                
                # Create revenue share record
                revenue_share = RevenueShareDB(
                    model_id=model_id,
                    developer_id=model_id,  # Should be actual developer ID
                    period_start=invoice.billing_period_start,
                    period_end=invoice.billing_period_end,
                    total_revenue=revenue,
                    platform_fee=share_info['platform_fee'],
                    developer_share=share_info['developer_share'],
                    status='pending'
                )
                
                self.session.add(revenue_share)
            
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Revenue sharing processing failed: {e}")
    
    async def _send_invoice_email(self, user_account: UserAccount, invoice: Invoice):
        """Send invoice email to user"""
        try:
            # Create email content
            subject = f"Invoice {invoice.invoice_id} - AI Model Marketplace"
            
            html_content = f"""
            <html>
                <body>
                    <h2>Invoice</h2>
                    <p>Dear Customer,</p>
                    <p>Your invoice for the period {invoice.billing_period_start.strftime('%Y-%m-%d')} to {invoice.billing_period_end.strftime('%Y-%m-%d')} is ready.</p>
                    
                    <h3>Invoice Details</h3>
                    <table border="1">
                        <tr>
                            <th>Description</th>
                            <th>Quantity</th>
                            <th>Unit Price</th>
                            <th>Total</th>
                        </tr>
            """
            
            for item in invoice.line_items:
                html_content += f"""
                        <tr>
                            <td>{item['description']}</td>
                            <td>{item['quantity']}</td>
                            <td>{invoice.currency} {item['unit_price']:.2f}</td>
                            <td>{invoice.currency} {item['total']:.2f}</td>
                        </tr>
                """
            
            html_content += f"""
                    </table>
                    
                    <h3>Summary</h3>
                    <p>Subtotal: {invoice.currency} {invoice.subtotal:.2f}</p>
                    <p>Tax: {invoice.currency} {invoice.tax:.2f}</p>
                    <p><strong>Total: {invoice.currency} {invoice.total:.2f}</strong></p>
                    
                    <p>Due Date: {invoice.due_date.strftime('%Y-%m-%d')}</p>
                    
                    <p>Thank you for using AI Model Marketplace!</p>
                </body>
            </html>
            """
            
            # Send email (implementation depends on email service)
            # This is a placeholder - implement actual email sending
            logger.info(f"Invoice email sent to {user_account.email}")
            
        except Exception as e:
            logger.error(f"Failed to send invoice email: {e}")


from sqlalchemy import func  # Add this import at the top