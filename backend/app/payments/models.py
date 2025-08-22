"""
Payment and subscription models for the arbitration RAG API.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    ForeignKey, Enum as SQLEnum, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

Base = declarative_base()


class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class PaymentMethod(str, Enum):
    """Payment method enumeration"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    INVOICE = "invoice"


class SubscriptionTier(str, Enum):
    """Subscription tier enumeration"""
    FREE = "free"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration"""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"


class PaymentProvider(str, Enum):
    """Payment provider enumeration"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    CRYPTO = "crypto"
    INTERNAL = "internal"


# SQLAlchemy Models
class Payment(Base):
    """Payment record model"""
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True)  # Provider's payment ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Payment details
    amount = Column(Float, nullable=False)  # Amount in USD
    currency = Column(String(3), default="USD")
    payment_method = Column(SQLEnum(PaymentMethod), nullable=False)
    provider = Column(SQLEnum(PaymentProvider), nullable=False)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    
    # Metadata
    description = Column(Text)
    metadata = Column(JSON)  # Flexible metadata storage
    
    # Provider-specific data
    provider_data = Column(JSON)  # Store provider-specific response data
    
    # Failure information
    failure_reason = Column(String(500))
    failure_code = Column(String(100))
    
    # Relationships
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription", back_populates="payments")
    invoice = relationship("Invoice", back_populates="payments")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_payment_status_created', 'status', 'created_at'),
        Index('idx_payment_provider_external', 'provider', 'external_id'),
        Index('idx_payment_user_created', 'user_id', 'created_at'),
    )


class Subscription(Base):
    """Subscription model"""
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True)  # Provider's subscription ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Subscription details
    tier = Column(SQLEnum(SubscriptionTier), nullable=False)
    status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)
    provider = Column(SQLEnum(PaymentProvider), nullable=False)
    
    # Pricing
    amount = Column(Float, nullable=False)  # Monthly amount in USD
    currency = Column(String(3), default="USD")
    billing_interval = Column(String(20), default="monthly")  # monthly, yearly
    
    # Usage limits based on tier
    document_limit = Column(Integer, nullable=True)  # null = unlimited
    api_limit = Column(Integer, nullable=True)  # requests per month
    
    # Billing cycle
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    trial_end = Column(DateTime, nullable=True)
    
    # Cancellation
    cancel_at_period_end = Column(Boolean, default=False)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Provider-specific data
    provider_data = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription")
    usage_records = relationship("UsageRecord", back_populates="subscription")
    
    # Indexes
    __table_args__ = (
        Index('idx_subscription_user_status', 'user_id', 'status'),
        Index('idx_subscription_period_end', 'current_period_end'),
        Index('idx_subscription_tier_status', 'tier', 'status'),
    )


class UsageRecord(Base):
    """Usage tracking for metered billing"""
    __tablename__ = "usage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=False)
    
    # Usage details
    resource_type = Column(String(50), nullable=False)  # documents, api_calls, etc.
    quantity = Column(Integer, default=1)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Billing period association
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    
    # Metadata
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User")
    subscription = relationship("Subscription", back_populates="usage_records")
    
    # Indexes for aggregation queries
    __table_args__ = (
        Index('idx_usage_user_period', 'user_id', 'billing_period_start', 'billing_period_end'),
        Index('idx_usage_subscription_resource', 'subscription_id', 'resource_type'),
        Index('idx_usage_timestamp', 'timestamp'),
    )


class Invoice(Base):
    """Invoice model for enterprise billing"""
    __tablename__ = "invoices"
    
    id = Column(Integer, primary_key=True, index=True)
    invoice_number = Column(String(100), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=True)
    
    # Invoice details
    subtotal = Column(Float, nullable=False)
    tax_amount = Column(Float, default=0.0)
    discount_amount = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    
    # Status and dates
    status = Column(String(20), default="draft")  # draft, sent, paid, overdue, cancelled
    issue_date = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime, nullable=False)
    paid_date = Column(DateTime, nullable=True)
    
    # Billing details
    billing_period_start = Column(DateTime, nullable=True)
    billing_period_end = Column(DateTime, nullable=True)
    
    # Customer details (snapshot at invoice creation)
    customer_data = Column(JSON)
    
    # Line items and metadata
    line_items = Column(JSON)
    notes = Column(Text)
    metadata = Column(JSON)
    
    # PDF storage
    pdf_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    subscription = relationship("Subscription")
    payments = relationship("Payment", back_populates="invoice")
    
    # Indexes
    __table_args__ = (
        Index('idx_invoice_user_status', 'user_id', 'status'),
        Index('idx_invoice_due_date', 'due_date'),
        Index('idx_invoice_period', 'billing_period_start', 'billing_period_end'),
    )


class PaymentWebhook(Base):
    """Webhook events from payment providers"""
    __tablename__ = "payment_webhooks"
    
    id = Column(Integer, primary_key=True, index=True)
    provider = Column(SQLEnum(PaymentProvider), nullable=False)
    event_id = Column(String(255), unique=True, index=True)
    event_type = Column(String(100), nullable=False)
    
    # Event data
    data = Column(JSON, nullable=False)
    processed = Column(Boolean, default=False)
    processing_attempts = Column(Integer, default=0)
    
    # Error handling
    last_error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_webhook_provider_processed', 'provider', 'processed'),
        Index('idx_webhook_event_type', 'event_type'),
        Index('idx_webhook_created', 'created_at'),
    )


# Pydantic Models for API
class PaymentCreate(BaseModel):
    """Create payment request"""
    amount: float
    currency: str = "USD"
    payment_method: PaymentMethod
    provider: PaymentProvider
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PaymentResponse(BaseModel):
    """Payment response"""
    id: int
    external_id: Optional[str]
    amount: float
    currency: str
    payment_method: PaymentMethod
    provider: PaymentProvider
    status: PaymentStatus
    description: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class SubscriptionCreate(BaseModel):
    """Create subscription request"""
    tier: SubscriptionTier
    billing_interval: str = "monthly"
    payment_method: PaymentMethod
    provider: PaymentProvider
    trial_days: Optional[int] = None


class SubscriptionResponse(BaseModel):
    """Subscription response"""
    id: int
    tier: SubscriptionTier
    status: SubscriptionStatus
    amount: float
    currency: str
    billing_interval: str
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime]
    cancel_at_period_end: bool
    document_limit: Optional[int]
    api_limit: Optional[int]
    
    class Config:
        from_attributes = True


class UsageRecordCreate(BaseModel):
    """Create usage record"""
    resource_type: str
    quantity: int = 1
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageRecordResponse(BaseModel):
    """Usage record response"""
    id: int
    resource_type: str
    quantity: int
    timestamp: datetime
    billing_period_start: datetime
    billing_period_end: datetime
    
    class Config:
        from_attributes = True


class InvoiceLineItem(BaseModel):
    """Invoice line item"""
    description: str
    quantity: int
    unit_price: float
    amount: float


class InvoiceCreate(BaseModel):
    """Create invoice request"""
    line_items: List[InvoiceLineItem]
    due_date: datetime
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InvoiceResponse(BaseModel):
    """Invoice response"""
    id: int
    invoice_number: str
    subtotal: float
    tax_amount: float
    discount_amount: float
    total_amount: float
    currency: str
    status: str
    issue_date: datetime
    due_date: datetime
    paid_date: Optional[datetime]
    pdf_url: Optional[str]
    
    class Config:
        from_attributes = True


# Subscription tier configuration
SUBSCRIPTION_TIERS = {
    SubscriptionTier.FREE: {
        "name": "Free",
        "price": 0.0,
        "document_limit": 10,
        "api_limit": 100,
        "features": ["Basic document analysis", "Email support"]
    },
    SubscriptionTier.PROFESSIONAL: {
        "name": "Professional",
        "price": 99.0,
        "document_limit": 500,
        "api_limit": 5000,
        "features": [
            "Advanced document analysis",
            "Priority support",
            "API access",
            "Export capabilities"
        ]
    },
    SubscriptionTier.BUSINESS: {
        "name": "Business",
        "price": 499.0,
        "document_limit": None,  # Unlimited
        "api_limit": 50000,
        "features": [
            "Unlimited documents",
            "Advanced API features",
            "Custom integrations",
            "Priority support",
            "Analytics dashboard"
        ]
    },
    SubscriptionTier.ENTERPRISE: {
        "name": "Enterprise",
        "price": None,  # Custom pricing
        "document_limit": None,  # Unlimited
        "api_limit": None,  # Unlimited
        "features": [
            "Custom pricing",
            "SLA guarantees",
            "Dedicated support",
            "On-premise deployment",
            "Custom features",
            "Training and onboarding"
        ]
    }
}


def get_tier_config(tier: SubscriptionTier) -> Dict[str, Any]:
    """Get configuration for a subscription tier"""
    return SUBSCRIPTION_TIERS.get(tier, {})


def calculate_proration(
    old_amount: float,
    new_amount: float,
    period_start: datetime,
    period_end: datetime,
    change_date: datetime
) -> float:
    """Calculate prorated amount for subscription changes"""
    total_days = (period_end - period_start).days
    remaining_days = (period_end - change_date).days
    
    if total_days <= 0 or remaining_days <= 0:
        return 0.0
    
    # Calculate unused portion of old plan
    unused_old = (old_amount / total_days) * remaining_days
    
    # Calculate prorated amount for new plan
    prorated_new = (new_amount / total_days) * remaining_days
    
    return prorated_new - unused_old