"""
Base payment processor interface and common utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PaymentError(Exception):
    """Base payment processing error"""
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class WebhookError(Exception):
    """Webhook processing error"""
    pass


class BasePaymentProcessor(ABC):
    """Abstract base class for payment processors"""
    
    @abstractmethod
    async def create_payment_intent(
        self,
        amount: float,
        currency: str = "usd",
        payment_method: Optional[str] = None,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a payment intent"""
        pass
    
    @abstractmethod
    async def confirm_payment(
        self,
        payment_intent_id: str,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Confirm a payment"""
        pass
    
    @abstractmethod
    async def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a customer"""
        pass
    
    @abstractmethod
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a subscription"""
        pass
    
    @abstractmethod
    async def create_refund(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a refund"""
        pass
    
    @abstractmethod
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        pass


class PaymentRetryHandler:
    """Handle payment retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def retry_payment(self, payment_func, *args, **kwargs):
        """Retry payment function with exponential backoff"""
        import asyncio
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await payment_func(*args, **kwargs)
            except PaymentError as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Payment attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Payment failed after {self.max_retries + 1} attempts: {e}")
                    
        raise last_exception


class PaymentSecurityValidator:
    """Validate payment requests for security"""
    
    @staticmethod
    def validate_amount(amount: float) -> bool:
        """Validate payment amount"""
        return 0.50 <= amount <= 50000.00  # $0.50 to $50,000
    
    @staticmethod
    def validate_currency(currency: str) -> bool:
        """Validate currency code"""
        supported_currencies = {"USD", "EUR", "GBP", "CAD", "AUD"}
        return currency.upper() in supported_currencies
    
    @staticmethod
    def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to prevent injection"""
        if not metadata:
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            # Only allow alphanumeric keys
            if isinstance(key, str) and key.replace("_", "").replace("-", "").isalnum():
                # Only allow simple string/number values
                if isinstance(value, (str, int, float, bool)):
                    sanitized[key] = str(value)[:100]  # Limit length
                    
        return sanitized
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) and len(email) <= 254


class PaymentAuditLogger:
    """Audit logging for payment operations"""
    
    def __init__(self):
        self.logger = logging.getLogger("payment_audit")
        
    def log_payment_attempt(
        self,
        user_id: int,
        amount: float,
        currency: str,
        payment_method: str,
        provider: str
    ):
        """Log payment attempt"""
        self.logger.info(
            f"PAYMENT_ATTEMPT: user_id={user_id}, amount={amount}, "
            f"currency={currency}, method={payment_method}, provider={provider}"
        )
    
    def log_payment_success(
        self,
        user_id: int,
        payment_id: str,
        amount: float,
        provider: str
    ):
        """Log successful payment"""
        self.logger.info(
            f"PAYMENT_SUCCESS: user_id={user_id}, payment_id={payment_id}, "
            f"amount={amount}, provider={provider}"
        )
    
    def log_payment_failure(
        self,
        user_id: int,
        amount: float,
        provider: str,
        error: str
    ):
        """Log payment failure"""
        self.logger.warning(
            f"PAYMENT_FAILURE: user_id={user_id}, amount={amount}, "
            f"provider={provider}, error={error}"
        )
    
    def log_subscription_change(
        self,
        user_id: int,
        subscription_id: str,
        old_tier: str,
        new_tier: str,
        action: str
    ):
        """Log subscription changes"""
        self.logger.info(
            f"SUBSCRIPTION_CHANGE: user_id={user_id}, subscription_id={subscription_id}, "
            f"old_tier={old_tier}, new_tier={new_tier}, action={action}"
        )
    
    def log_refund(
        self,
        user_id: int,
        payment_id: str,
        refund_amount: float,
        reason: str
    ):
        """Log refund"""
        self.logger.info(
            f"REFUND: user_id={user_id}, payment_id={payment_id}, "
            f"amount={refund_amount}, reason={reason}"
        )


class PaymentIdempotencyManager:
    """Manage payment idempotency to prevent duplicate charges"""
    
    def __init__(self):
        self._processed_requests = {}  # In production, use Redis
    
    def generate_idempotency_key(self, user_id: int, amount: float, timestamp: datetime) -> str:
        """Generate idempotency key"""
        import hashlib
        
        key_data = f"{user_id}:{amount}:{timestamp.isoformat()}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def is_duplicate_request(self, idempotency_key: str) -> bool:
        """Check if request is duplicate"""
        return idempotency_key in self._processed_requests
    
    def mark_request_processed(self, idempotency_key: str, result: Dict[str, Any]):
        """Mark request as processed"""
        self._processed_requests[idempotency_key] = {
            "result": result,
            "processed_at": datetime.utcnow()
        }
    
    def get_processed_result(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Get result of previously processed request"""
        if idempotency_key in self._processed_requests:
            return self._processed_requests[idempotency_key]["result"]
        return None