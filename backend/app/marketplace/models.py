"""
Marketplace data models for vendor management and revenue sharing.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Float, Text, 
    ForeignKey, Enum as SQLEnum, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

Base = declarative_base()


class VendorStatus(str, Enum):
    """Vendor status enumeration"""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REJECTED = "rejected"
    TERMINATED = "terminated"


class PayoutStatus(str, Enum):
    """Payout status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class APIEndpointStatus(str, Enum):
    """API endpoint status enumeration"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


# SQLAlchemy Models
class Vendor(Base):
    """Vendor/Partner model"""
    __tablename__ = "vendors"
    
    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(String(100), unique=True, index=True)  # Public vendor ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Company information
    company_name = Column(String(255), nullable=False)
    company_website = Column(String(255))
    company_description = Column(Text)
    
    # Contact information
    contact_name = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=False)
    contact_phone = Column(String(50))
    
    # Business information
    business_type = Column(String(100))  # corporation, llc, individual, etc.
    tax_id = Column(String(50))  # EIN, SSN, etc.
    business_address = Column(JSON)  # Full address object
    
    # Status and verification
    status = Column(SQLEnum(VendorStatus), default=VendorStatus.PENDING)
    verified = Column(Boolean, default=False)
    verification_documents = Column(JSON)  # List of uploaded documents
    
    # Revenue sharing
    commission_rate = Column(Float, default=0.30)  # 30% default commission
    minimum_payout = Column(Float, default=50.0)  # Minimum payout threshold
    
    # API key for vendor's services
    api_key = Column(String(255), unique=True, index=True)
    webhook_url = Column(String(500))  # Vendor's webhook endpoint
    
    # Banking information (encrypted in production)
    bank_account_info = Column(JSON)  # Bank details for payouts
    
    # Metrics
    total_revenue = Column(Float, default=0.0)
    total_commission = Column(Float, default=0.0)
    total_payouts = Column(Float, default=0.0)
    
    # Metadata and settings
    settings = Column(JSON)  # Vendor-specific settings
    metadata = Column(JSON)  # Additional vendor data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    last_payout_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")
    api_endpoints = relationship("APIEndpoint", back_populates="vendor")
    payouts = relationship("VendorPayout", back_populates="vendor")
    revenue_shares = relationship("RevenueShare", back_populates="vendor")
    
    # Indexes
    __table_args__ = (
        Index('idx_vendor_status_created', 'status', 'created_at'),
        Index('idx_vendor_user_status', 'user_id', 'status'),
    )


class APIEndpoint(Base):
    """API endpoints provided by vendors"""
    __tablename__ = "api_endpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint_id = Column(String(100), unique=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    
    # Endpoint details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    endpoint_url = Column(String(500), nullable=False)
    method = Column(String(10), default="POST")  # HTTP method
    
    # Pricing
    price_per_request = Column(Float, default=0.01)  # Price per API call
    price_per_document = Column(Float, nullable=True)  # Price per document processed
    pricing_model = Column(String(50), default="per_request")  # per_request, per_document, subscription
    
    # Capabilities
    capabilities = Column(JSON)  # List of supported features
    input_formats = Column(JSON)  # Supported input formats
    output_formats = Column(JSON)  # Supported output formats
    
    # Documentation
    documentation_url = Column(String(500))
    api_schema = Column(JSON)  # OpenAPI/Swagger schema
    example_request = Column(JSON)
    example_response = Column(JSON)
    
    # Status and configuration
    status = Column(SQLEnum(APIEndpointStatus), default=APIEndpointStatus.DRAFT)
    enabled = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    
    # Quality metrics
    success_rate = Column(Float, default=0.0)  # Success rate percentage
    average_response_time = Column(Float, default=0.0)  # Average response time in ms
    uptime_percentage = Column(Float, default=0.0)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    total_revenue = Column(Float, default=0.0)
    
    # Metadata
    tags = Column(JSON)  # Searchable tags
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    vendor = relationship("Vendor", back_populates="api_endpoints")
    usage_records = relationship("APIUsageRecord", back_populates="endpoint")
    
    # Indexes
    __table_args__ = (
        Index('idx_endpoint_vendor_status', 'vendor_id', 'status'),
        Index('idx_endpoint_enabled_status', 'enabled', 'status'),
    )


class APIUsageRecord(Base):
    """Track API endpoint usage for billing"""
    __tablename__ = "api_usage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint_id = Column(Integer, ForeignKey("api_endpoints.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Request details
    request_id = Column(String(100), unique=True, index=True)
    method = Column(String(10))
    path = Column(String(500))
    
    # Billing information
    billable_units = Column(Integer, default=1)  # Number of billable units
    unit_price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    
    # Performance metrics
    response_time_ms = Column(Integer)
    status_code = Column(Integer)
    success = Column(Boolean, default=True)
    
    # Request/response metadata
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    endpoint = relationship("APIEndpoint", back_populates="usage_records")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_endpoint_created', 'endpoint_id', 'created_at'),
        Index('idx_usage_user_created', 'user_id', 'created_at'),
        Index('idx_usage_success_created', 'success', 'created_at'),
    )


class RevenueShare(Base):
    """Revenue sharing records between platform and vendors"""
    __tablename__ = "revenue_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Customer who made the payment
    
    # Revenue details
    gross_revenue = Column(Float, nullable=False)  # Total amount paid by customer
    platform_commission = Column(Float, nullable=False)  # Platform's share
    vendor_share = Column(Float, nullable=False)  # Vendor's share
    commission_rate = Column(Float, nullable=False)  # Commission rate at time of transaction
    
    # Source information
    source_type = Column(String(50), nullable=False)  # api_usage, subscription, one_time
    source_id = Column(String(100))  # ID of the source transaction
    
    # Related records
    payment_id = Column(Integer, ForeignKey("payments.id"), nullable=True)
    api_usage_id = Column(Integer, ForeignKey("api_usage_records.id"), nullable=True)
    
    # Status
    processed = Column(Boolean, default=False)
    payout_id = Column(Integer, ForeignKey("vendor_payouts.id"), nullable=True)
    
    # Metadata
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    vendor = relationship("Vendor", back_populates="revenue_shares")
    user = relationship("User")
    payment = relationship("Payment")
    payout = relationship("VendorPayout", back_populates="revenue_shares")
    
    # Indexes
    __table_args__ = (
        Index('idx_revenue_vendor_created', 'vendor_id', 'created_at'),
        Index('idx_revenue_processed', 'processed'),
        Index('idx_revenue_source', 'source_type', 'source_id'),
    )


class VendorPayout(Base):
    """Vendor payout records"""
    __tablename__ = "vendor_payouts"
    
    id = Column(Integer, primary_key=True, index=True)
    payout_id = Column(String(100), unique=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    
    # Payout details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    payout_method = Column(String(50), default="bank_transfer")  # bank_transfer, paypal, check
    
    # Period covered
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Status and processing
    status = Column(SQLEnum(PayoutStatus), default=PayoutStatus.PENDING)
    external_payout_id = Column(String(255))  # ID from payment processor
    
    # Banking information
    bank_account_last_four = Column(String(4))  # Last 4 digits for verification
    
    # Breakdown
    total_revenue_shares = Column(Integer, default=0)  # Number of revenue share records
    gross_revenue = Column(Float, default=0.0)  # Total gross revenue
    platform_commission = Column(Float, default=0.0)  # Total platform commission
    
    # Fees and deductions
    processing_fee = Column(Float, default=0.0)  # Payment processing fee
    adjustment_amount = Column(Float, default=0.0)  # Manual adjustments
    
    # Failure information
    failure_reason = Column(String(500))
    failure_code = Column(String(100))
    retry_count = Column(Integer, default=0)
    
    # Metadata
    metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    vendor = relationship("Vendor", back_populates="payouts")
    revenue_shares = relationship("RevenueShare", back_populates="payout")
    
    # Indexes
    __table_args__ = (
        Index('idx_payout_vendor_status', 'vendor_id', 'status'),
        Index('idx_payout_period', 'period_start', 'period_end'),
        Index('idx_payout_status_created', 'status', 'created_at'),
    )


# Pydantic Models for API
class VendorCreate(BaseModel):
    """Create vendor request"""
    company_name: str
    company_website: Optional[str] = None
    company_description: Optional[str] = None
    contact_name: str
    contact_email: str
    contact_phone: Optional[str] = None
    business_type: Optional[str] = None
    tax_id: Optional[str] = None
    business_address: Optional[Dict[str, Any]] = None


class VendorResponse(BaseModel):
    """Vendor response"""
    id: int
    vendor_id: str
    company_name: str
    contact_name: str
    contact_email: str
    status: VendorStatus
    verified: bool
    commission_rate: float
    total_revenue: float
    total_commission: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class APIEndpointCreate(BaseModel):
    """Create API endpoint request"""
    name: str
    description: Optional[str] = None
    endpoint_url: str
    method: str = "POST"
    price_per_request: float = 0.01
    price_per_document: Optional[float] = None
    pricing_model: str = "per_request"
    capabilities: Optional[List[str]] = None
    input_formats: Optional[List[str]] = None
    output_formats: Optional[List[str]] = None
    documentation_url: Optional[str] = None


class APIEndpointResponse(BaseModel):
    """API endpoint response"""
    id: int
    endpoint_id: str
    name: str
    description: Optional[str]
    endpoint_url: str
    method: str
    price_per_request: float
    pricing_model: str
    status: APIEndpointStatus
    enabled: bool
    success_rate: float
    total_requests: int
    total_revenue: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class RevenueShareResponse(BaseModel):
    """Revenue share response"""
    id: int
    vendor_id: int
    gross_revenue: float
    platform_commission: float
    vendor_share: float
    commission_rate: float
    source_type: str
    processed: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class VendorPayoutResponse(BaseModel):
    """Vendor payout response"""
    id: int
    payout_id: str
    amount: float
    currency: str
    status: PayoutStatus
    period_start: datetime
    period_end: datetime
    total_revenue_shares: int
    gross_revenue: float
    platform_commission: float
    created_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# Commission configuration
COMMISSION_TIERS = {
    "standard": {
        "rate": 0.30,  # 30%
        "minimum_payout": 50.0,
        "description": "Standard commission rate for new vendors"
    },
    "verified": {
        "rate": 0.25,  # 25%
        "minimum_payout": 100.0,
        "description": "Reduced rate for verified vendors"
    },
    "premium": {
        "rate": 0.20,  # 20%
        "minimum_payout": 250.0,
        "description": "Premium rate for high-volume vendors"
    },
    "enterprise": {
        "rate": 0.15,  # 15%
        "minimum_payout": 1000.0,
        "description": "Enterprise rate for strategic partners"
    }
}


def get_commission_tier(monthly_revenue: float) -> str:
    """Determine commission tier based on monthly revenue"""
    if monthly_revenue >= 10000:
        return "enterprise"
    elif monthly_revenue >= 5000:
        return "premium"
    elif monthly_revenue >= 1000:
        return "verified"
    else:
        return "standard"


def calculate_vendor_share(gross_revenue: float, commission_rate: float) -> Tuple[float, float]:
    """Calculate platform commission and vendor share"""
    platform_commission = gross_revenue * commission_rate
    vendor_share = gross_revenue - platform_commission
    return platform_commission, vendor_share