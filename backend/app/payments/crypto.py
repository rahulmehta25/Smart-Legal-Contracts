"""
Cryptocurrency payment integration for the arbitration RAG API.

This module provides cryptocurrency payment processing including:
- Bitcoin and Ethereum payment processing
- Wallet integration
- Transaction monitoring
- Price conversion and volatility handling
- Security and fraud prevention
"""

import json
import logging
import asyncio
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

from ..core.config import get_settings
from .models import (
    Payment, PaymentWebhook,
    PaymentStatus, PaymentMethod, PaymentProvider
)
from .base import BasePaymentProcessor, PaymentError, WebhookError

logger = logging.getLogger(__name__)
settings = get_settings()


class CryptoPaymentProcessor(BasePaymentProcessor):
    """Cryptocurrency payment processor"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize crypto processor"""
        self.config = config
        self.supported_currencies = ["BTC", "ETH"]
        self.confirmation_requirements = {
            "BTC": 6,  # Bitcoin confirmations
            "ETH": 12  # Ethereum confirmations
        }
        self.network_fees = {
            "BTC": 0.0001,  # Bitcoin network fee
            "ETH": 0.01     # Ethereum gas fee estimate
        }
    
    async def get_exchange_rate(self, crypto_currency: str, fiat_currency: str = "USD") -> float:
        """Get current exchange rate for cryptocurrency"""
        try:
            async with httpx.AsyncClient() as client:
                # Using CoinGecko API for real-time rates
                response = await client.get(
                    f"https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": self._get_coingecko_id(crypto_currency),
                        "vs_currencies": fiat_currency.lower()
                    }
                )
                
                if response.status_code != 200:
                    raise PaymentError(f"Failed to get exchange rate: {response.status_code}")
                
                data = response.json()
                coin_id = self._get_coingecko_id(crypto_currency)
                
                if coin_id not in data:
                    raise PaymentError(f"Exchange rate not found for {crypto_currency}")
                
                return float(data[coin_id][fiat_currency.lower()])
                
        except Exception as e:
            logger.error(f"Exchange rate lookup failed: {e}")
            raise PaymentError(f"Exchange rate lookup failed: {str(e)}")
    
    def _get_coingecko_id(self, currency: str) -> str:
        """Map currency symbols to CoinGecko IDs"""
        mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum"
        }
        return mapping.get(currency.upper(), currency.lower())
    
    async def generate_payment_address(
        self,
        currency: str,
        amount_usd: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a unique payment address for the transaction"""
        try:
            if currency.upper() not in self.supported_currencies:
                raise PaymentError(f"Unsupported cryptocurrency: {currency}")
            
            # Get current exchange rate
            exchange_rate = await self.get_exchange_rate(currency, "USD")
            
            # Calculate crypto amount with buffer for volatility (5%)
            crypto_amount = (amount_usd / exchange_rate) * 1.05
            
            # Generate unique address (in production, use proper wallet service)
            payment_id = self._generate_payment_id()
            address = await self._generate_address(currency, payment_id)
            
            # Calculate expiration (30 minutes for crypto payments)
            expires_at = datetime.utcnow() + timedelta(minutes=30)
            
            return {
                "payment_id": payment_id,
                "address": address,
                "currency": currency.upper(),
                "amount_crypto": crypto_amount,
                "amount_usd": amount_usd,
                "exchange_rate": exchange_rate,
                "network_fee": self.network_fees.get(currency.upper(), 0),
                "confirmations_required": self.confirmation_requirements.get(currency.upper(), 6),
                "expires_at": expires_at,
                "qr_code_url": self._generate_qr_code_url(address, crypto_amount, currency),
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Address generation failed: {e}")
            raise PaymentError(f"Address generation failed: {str(e)}")
    
    def _generate_payment_id(self) -> str:
        """Generate unique payment ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def _generate_address(self, currency: str, payment_id: str) -> str:
        """Generate cryptocurrency address (mock implementation)"""
        # In production, integrate with proper wallet service like:
        # - BitGo
        # - Coinbase Commerce
        # - Block.io
        # - Custom HD wallet implementation
        
        if currency.upper() == "BTC":
            # Mock Bitcoin address generation
            return f"bc1q{hashlib.sha256(payment_id.encode()).hexdigest()[:39]}"
        elif currency.upper() == "ETH":
            # Mock Ethereum address generation
            return f"0x{hashlib.sha256(payment_id.encode()).hexdigest()[:40]}"
        else:
            raise PaymentError(f"Unsupported currency: {currency}")
    
    def _generate_qr_code_url(self, address: str, amount: float, currency: str) -> str:
        """Generate QR code URL for payment"""
        # Use QR code service
        payment_uri = f"{currency.lower()}:{address}?amount={amount:.8f}"
        return f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={payment_uri}"
    
    async def create_payment_intent(
        self,
        amount: float,
        currency: str = "BTC",
        payment_method: Optional[str] = None,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create cryptocurrency payment intent"""
        try:
            payment_data = await self.generate_payment_address(
                currency, amount, metadata
            )
            
            return {
                "id": payment_data["payment_id"],
                "status": "pending",
                "amount": amount,
                "currency": "USD",
                "crypto_currency": currency.upper(),
                "crypto_amount": payment_data["amount_crypto"],
                "payment_address": payment_data["address"],
                "qr_code_url": payment_data["qr_code_url"],
                "expires_at": payment_data["expires_at"],
                "exchange_rate": payment_data["exchange_rate"],
                "confirmations_required": payment_data["confirmations_required"],
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Crypto payment intent creation failed: {e}")
            raise PaymentError(f"Payment creation failed: {str(e)}")
    
    async def confirm_payment(
        self,
        payment_intent_id: str,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check payment status on blockchain"""
        try:
            # In production, check blockchain for transaction
            transaction_status = await self._check_blockchain_transaction(payment_intent_id)
            
            return {
                "id": payment_intent_id,
                "status": transaction_status["status"],
                "transaction_hash": transaction_status.get("tx_hash"),
                "confirmations": transaction_status.get("confirmations", 0),
                "block_height": transaction_status.get("block_height"),
                "confirmed_amount": transaction_status.get("amount")
            }
            
        except Exception as e:
            logger.error(f"Crypto payment confirmation failed: {e}")
            raise PaymentError(f"Payment confirmation failed: {str(e)}")
    
    async def _check_blockchain_transaction(self, payment_id: str) -> Dict[str, Any]:
        """Check blockchain for transaction (mock implementation)"""
        # In production, integrate with blockchain APIs:
        # - Bitcoin: BlockCypher, Blockchain.info, or Bitcoin Core RPC
        # - Ethereum: Infura, Alchemy, or Ethereum node
        
        # Mock response
        return {
            "status": "pending",
            "confirmations": 0,
            "tx_hash": None,
            "amount": None
        }
    
    async def monitor_payments(self, db: Session):
        """Monitor pending cryptocurrency payments"""
        try:
            # Get pending crypto payments
            pending_payments = db.query(Payment).filter(
                Payment.provider == PaymentProvider.CRYPTO,
                Payment.status.in_([PaymentStatus.PENDING, PaymentStatus.PROCESSING])
            ).all()
            
            for payment in pending_payments:
                try:
                    # Check payment status
                    status = await self.confirm_payment(payment.external_id)
                    
                    if status["confirmations"] >= self._get_required_confirmations(payment):
                        # Payment confirmed
                        payment.status = PaymentStatus.COMPLETED
                        payment.processed_at = datetime.utcnow()
                        payment.provider_data = status
                        
                        logger.info(f"Crypto payment {payment.external_id} confirmed")
                        
                    elif status["confirmations"] > 0:
                        # Payment processing
                        payment.status = PaymentStatus.PROCESSING
                        payment.provider_data = status
                        
                    # Check for expiration
                    if payment.created_at < datetime.utcnow() - timedelta(hours=2):
                        payment.status = PaymentStatus.FAILED
                        payment.failure_reason = "Payment expired"
                        
                        logger.warning(f"Crypto payment {payment.external_id} expired")
                        
                except Exception as e:
                    logger.error(f"Failed to check payment {payment.external_id}: {e}")
                    continue
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Payment monitoring failed: {e}")
    
    def _get_required_confirmations(self, payment: Payment) -> int:
        """Get required confirmations for payment"""
        provider_data = payment.provider_data or {}
        currency = provider_data.get("crypto_currency", "BTC")
        return self.confirmation_requirements.get(currency, 6)
    
    async def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create customer for crypto payments"""
        # Crypto payments don't require customer creation
        import uuid
        return str(uuid.uuid4())
    
    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Crypto subscriptions not supported"""
        raise PaymentError("Cryptocurrency subscriptions are not supported")
    
    async def create_refund(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crypto refunds require manual processing"""
        raise PaymentError("Cryptocurrency refunds require manual processing")
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify blockchain webhook signature"""
        # Implementation depends on blockchain service provider
        return True
    
    async def handle_webhook(
        self,
        request: Request,
        db: Session
    ) -> JSONResponse:
        """Handle blockchain webhook events"""
        try:
            payload = await request.body()
            event_data = json.loads(payload)
            
            # Store webhook event
            webhook_record = PaymentWebhook(
                provider=PaymentProvider.CRYPTO,
                event_id=event_data.get("id", "unknown"),
                event_type=event_data.get("type", "transaction"),
                data=event_data
            )
            db.add(webhook_record)
            db.flush()
            
            # Process transaction event
            await self._process_blockchain_event(event_data, db)
            
            # Mark as processed
            webhook_record.processed = True
            webhook_record.processed_at = datetime.utcnow()
            db.commit()
            
            return JSONResponse({"status": "success"})
            
        except Exception as e:
            logger.error(f"Crypto webhook error: {e}")
            return JSONResponse(
                {"error": "Internal server error"}, 
                status_code=500
            )
    
    async def _process_blockchain_event(self, event_data: Dict[str, Any], db: Session):
        """Process blockchain transaction events"""
        transaction_hash = event_data.get("hash")
        address = event_data.get("address")
        amount = event_data.get("amount")
        confirmations = event_data.get("confirmations", 0)
        
        if not transaction_hash or not address:
            logger.warning("Invalid blockchain event data")
            return
        
        # Find matching payment by address
        # In production, maintain address-to-payment mapping
        payment = db.query(Payment).filter(
            Payment.provider == PaymentProvider.CRYPTO,
            Payment.provider_data.contains({"address": address})
        ).first()
        
        if payment:
            # Update payment status based on confirmations
            required_confirmations = self._get_required_confirmations(payment)
            
            if confirmations >= required_confirmations:
                payment.status = PaymentStatus.COMPLETED
                payment.processed_at = datetime.utcnow()
            elif confirmations > 0:
                payment.status = PaymentStatus.PROCESSING
            
            # Update provider data
            payment.provider_data = {
                **(payment.provider_data or {}),
                "transaction_hash": transaction_hash,
                "confirmations": confirmations,
                "confirmed_amount": amount
            }
            
            db.commit()
            logger.info(f"Updated crypto payment {payment.external_id}")


class CryptoSecurityValidator:
    """Security validation for cryptocurrency payments"""
    
    def __init__(self):
        self.suspicious_addresses = set()  # Blacklisted addresses
        self.max_payment_amount = 50000.0  # Maximum payment in USD
    
    def validate_address(self, address: str, currency: str) -> bool:
        """Validate cryptocurrency address format"""
        if currency.upper() == "BTC":
            return self._validate_bitcoin_address(address)
        elif currency.upper() == "ETH":
            return self._validate_ethereum_address(address)
        return False
    
    def _validate_bitcoin_address(self, address: str) -> bool:
        """Validate Bitcoin address format"""
        # Basic validation - in production use proper library
        if address.startswith(("1", "3", "bc1")):
            return 26 <= len(address) <= 62
        return False
    
    def _validate_ethereum_address(self, address: str) -> bool:
        """Validate Ethereum address format"""
        # Basic validation - in production use proper library
        if address.startswith("0x"):
            return len(address) == 42
        return False
    
    def is_suspicious_address(self, address: str) -> bool:
        """Check if address is on blacklist"""
        return address in self.suspicious_addresses
    
    def validate_payment_amount(self, amount_usd: float) -> bool:
        """Validate payment amount"""
        return 1.0 <= amount_usd <= self.max_payment_amount
    
    def check_rate_limiting(self, user_id: int, amount_usd: float) -> bool:
        """Check if user exceeds rate limits"""
        # In production, implement proper rate limiting
        # Check daily/monthly limits per user
        return True


class CryptoPriceOracle:
    """Price oracle for cryptocurrency conversions"""
    
    def __init__(self):
        self.cache_duration = 60  # Cache prices for 1 minute
        self.price_cache = {}
    
    async def get_reliable_price(self, currency: str, fiat: str = "USD") -> Tuple[float, float]:
        """Get price from multiple sources for reliability"""
        try:
            sources = [
                self._get_coingecko_price,
                self._get_coinbase_price,
                self._get_binance_price
            ]
            
            prices = []
            for source in sources:
                try:
                    price = await source(currency, fiat)
                    if price > 0:
                        prices.append(price)
                except Exception as e:
                    logger.warning(f"Price source failed: {e}")
                    continue
            
            if not prices:
                raise PaymentError("No price data available")
            
            # Use median price for stability
            prices.sort()
            median_price = prices[len(prices) // 2]
            
            # Calculate price deviation
            if len(prices) > 1:
                min_price = min(prices)
                max_price = max(prices)
                deviation = (max_price - min_price) / median_price
            else:
                deviation = 0.0
            
            return median_price, deviation
            
        except Exception as e:
            logger.error(f"Price oracle failed: {e}")
            raise PaymentError(f"Price lookup failed: {str(e)}")
    
    async def _get_coingecko_price(self, currency: str, fiat: str) -> float:
        """Get price from CoinGecko"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": self._get_coingecko_id(currency),
                    "vs_currencies": fiat.lower()
                }
            )
            data = response.json()
            coin_id = self._get_coingecko_id(currency)
            return float(data[coin_id][fiat.lower()])
    
    async def _get_coinbase_price(self, currency: str, fiat: str) -> float:
        """Get price from Coinbase"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.coinbase.com/v2/exchange-rates",
                params={"currency": currency}
            )
            data = response.json()
            return float(data["data"]["rates"][fiat])
    
    async def _get_binance_price(self, currency: str, fiat: str) -> float:
        """Get price from Binance"""
        async with httpx.AsyncClient() as client:
            symbol = f"{currency}{fiat}"
            response = await client.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol}
            )
            data = response.json()
            return float(data["price"])
    
    def _get_coingecko_id(self, currency: str) -> str:
        """Map currency to CoinGecko ID"""
        mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum"
        }
        return mapping.get(currency.upper(), currency.lower())