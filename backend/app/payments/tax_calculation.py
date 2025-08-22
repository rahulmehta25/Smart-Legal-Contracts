"""
Tax calculation system for payments and invoices.

This module provides comprehensive tax calculation including:
- Sales tax calculation for US states
- VAT calculation for EU countries
- Tax rate lookup and caching
- Tax exemption handling
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import asyncio

logger = logging.getLogger(__name__)


class TaxCalculator:
    """Comprehensive tax calculation system"""
    
    def __init__(self):
        self.tax_rates_cache = {}
        self.cache_expiry = {}
        
        # US state tax rates (simplified - in production, use TaxJar, Avalara, etc.)
        self.us_tax_rates = {
            "CA": 0.0725,  # California
            "NY": 0.08,    # New York
            "TX": 0.0625,  # Texas
            "FL": 0.06,    # Florida
            "WA": 0.065,   # Washington
            "OR": 0.0,     # Oregon (no sales tax)
            "NH": 0.0,     # New Hampshire (no sales tax)
            "MT": 0.0,     # Montana (no sales tax)
            "DE": 0.0,     # Delaware (no sales tax)
            "AK": 0.0,     # Alaska (no state sales tax)
        }
        
        # EU VAT rates (simplified)
        self.eu_vat_rates = {
            "DE": 0.19,    # Germany
            "FR": 0.20,    # France
            "GB": 0.20,    # United Kingdom
            "IT": 0.22,    # Italy
            "ES": 0.21,    # Spain
            "NL": 0.21,    # Netherlands
            "BE": 0.21,    # Belgium
            "AT": 0.20,    # Austria
            "SE": 0.25,    # Sweden
            "DK": 0.25,    # Denmark
        }
    
    async def calculate_tax(
        self,
        amount: float,
        customer_location: str,
        product_type: str = "digital_service",
        tax_exempt: bool = False
    ) -> float:
        """Calculate tax for a given amount and customer location"""
        try:
            if tax_exempt or amount <= 0:
                return 0.0
            
            # Parse location
            country, state = self._parse_location(customer_location)
            
            if tax_exempt:
                return 0.0
            
            # Get tax rate
            tax_rate = await self._get_tax_rate(country, state, product_type)
            
            # Calculate tax
            tax_amount = amount * tax_rate
            
            logger.debug(f"Tax calculation: amount=${amount:.2f}, rate={tax_rate:.4f}, tax=${tax_amount:.2f}")
            
            return round(tax_amount, 2)
            
        except Exception as e:
            logger.error(f"Tax calculation failed: {e}")
            return 0.0  # Default to no tax on error
    
    def _parse_location(self, location: str) -> Tuple[str, Optional[str]]:
        """Parse customer location into country and state/province"""
        try:
            # Handle various formats: "US-CA", "US", "CA", "Germany", "DE"
            if "-" in location:
                parts = location.split("-")
                country = parts[0].upper()
                state = parts[1].upper() if len(parts) > 1 else None
            elif len(location) == 2:
                # Two-letter code - could be country or state
                if location.upper() in self.us_tax_rates:
                    country = "US"
                    state = location.upper()
                else:
                    country = location.upper()
                    state = None
            else:
                # Full country name or other format
                country = self._normalize_country_name(location)
                state = None
            
            return country, state
            
        except Exception as e:
            logger.warning(f"Failed to parse location '{location}': {e}")
            return "US", None  # Default to US
    
    def _normalize_country_name(self, country_name: str) -> str:
        """Normalize country name to ISO country code"""
        country_mapping = {
            "united states": "US",
            "usa": "US",
            "america": "US",
            "germany": "DE",
            "france": "FR",
            "united kingdom": "GB",
            "uk": "GB",
            "britain": "GB",
            "italy": "IT",
            "spain": "ES",
            "netherlands": "NL",
            "holland": "NL",
            "belgium": "BE",
            "austria": "AT",
            "sweden": "SE",
            "denmark": "DK"
        }
        
        normalized = country_name.lower().strip()
        return country_mapping.get(normalized, country_name.upper()[:2])
    
    async def _get_tax_rate(
        self,
        country: str,
        state: Optional[str],
        product_type: str
    ) -> float:
        """Get tax rate for country/state and product type"""
        try:
            cache_key = f"{country}:{state}:{product_type}"
            
            # Check cache
            if cache_key in self.tax_rates_cache:
                cache_time = self.cache_expiry.get(cache_key, datetime.min)
                if datetime.utcnow() < cache_time:
                    return self.tax_rates_cache[cache_key]
            
            # Calculate tax rate
            tax_rate = 0.0
            
            if country == "US" and state:
                # US state sales tax
                tax_rate = self.us_tax_rates.get(state, 0.0)
                
                # Add local tax rates (simplified - in production, use detailed tax APIs)
                if state in ["CA", "NY", "IL"]:  # High-tax states
                    tax_rate += 0.01  # Add local tax
                    
            elif country in self.eu_vat_rates:
                # EU VAT
                if product_type in ["digital_service", "software"]:
                    tax_rate = self.eu_vat_rates[country]
                    
            elif country == "CA":  # Canada
                # Canadian GST/HST (simplified)
                tax_rate = 0.05  # Basic GST
                if state in ["ON", "NB", "NL", "NS", "PE"]:  # HST provinces
                    tax_rate = 0.13
                    
            # Cache the result
            self.tax_rates_cache[cache_key] = tax_rate
            self.cache_expiry[cache_key] = datetime.utcnow() + timedelta(hours=24)
            
            return tax_rate
            
        except Exception as e:
            logger.error(f"Failed to get tax rate for {country}:{state}: {e}")
            return 0.0
    
    async def get_tax_info(
        self,
        amount: float,
        customer_location: str,
        product_type: str = "digital_service"
    ) -> Dict[str, Any]:
        """Get detailed tax information"""
        try:
            country, state = self._parse_location(customer_location)
            tax_rate = await self._get_tax_rate(country, state, product_type)
            tax_amount = await self.calculate_tax(amount, customer_location, product_type)
            
            # Determine tax type
            tax_type = "none"
            tax_jurisdiction = "unknown"
            
            if country == "US" and state:
                tax_type = "sales_tax"
                tax_jurisdiction = f"US-{state}"
            elif country in self.eu_vat_rates:
                tax_type = "vat"
                tax_jurisdiction = country
            elif country == "CA":
                tax_type = "gst_hst"
                tax_jurisdiction = f"CA-{state}" if state else "CA"
            
            return {
                "tax_amount": tax_amount,
                "tax_rate": tax_rate,
                "tax_type": tax_type,
                "tax_jurisdiction": tax_jurisdiction,
                "taxable_amount": amount,
                "total_amount": amount + tax_amount,
                "breakdown": {
                    "subtotal": amount,
                    "tax": tax_amount,
                    "total": amount + tax_amount
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get tax info: {e}")
            return {
                "tax_amount": 0.0,
                "tax_rate": 0.0,
                "tax_type": "none",
                "tax_jurisdiction": "unknown",
                "taxable_amount": amount,
                "total_amount": amount,
                "error": str(e)
            }
    
    async def validate_tax_id(self, tax_id: str, country: str) -> Dict[str, Any]:
        """Validate tax ID format and potentially exempt status"""
        try:
            # Basic format validation (in production, use proper validation APIs)
            is_valid = False
            tax_id_type = "unknown"
            
            if country == "US":
                # US EIN format: XX-XXXXXXX
                if len(tax_id) == 10 and tax_id[2] == "-" and tax_id.replace("-", "").isdigit():
                    is_valid = True
                    tax_id_type = "EIN"
            elif country in self.eu_vat_rates:
                # EU VAT number format varies by country
                if tax_id.startswith(country) and len(tax_id) >= 8:
                    is_valid = True
                    tax_id_type = "VAT"
            
            # In production, also check if tax ID qualifies for exemption
            eligible_for_exemption = is_valid and tax_id_type in ["EIN", "VAT"]
            
            return {
                "valid": is_valid,
                "tax_id_type": tax_id_type,
                "eligible_for_exemption": eligible_for_exemption,
                "country": country
            }
            
        except Exception as e:
            logger.error(f"Tax ID validation failed: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def get_tax_report(
        self,
        transactions: list,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate tax report for a period"""
        try:
            report = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_transactions": len(transactions),
                    "total_revenue": 0.0,
                    "total_tax_collected": 0.0,
                    "tax_exempt_transactions": 0
                },
                "by_jurisdiction": {},
                "by_tax_type": {}
            }
            
            for transaction in transactions:
                # Extract transaction data
                amount = transaction.get("amount", 0.0)
                tax_amount = transaction.get("tax_amount", 0.0)
                jurisdiction = transaction.get("tax_jurisdiction", "unknown")
                tax_type = transaction.get("tax_type", "none")
                
                # Update summary
                report["summary"]["total_revenue"] += amount
                report["summary"]["total_tax_collected"] += tax_amount
                
                if tax_amount == 0:
                    report["summary"]["tax_exempt_transactions"] += 1
                
                # Group by jurisdiction
                if jurisdiction not in report["by_jurisdiction"]:
                    report["by_jurisdiction"][jurisdiction] = {
                        "transaction_count": 0,
                        "revenue": 0.0,
                        "tax_collected": 0.0
                    }
                
                report["by_jurisdiction"][jurisdiction]["transaction_count"] += 1
                report["by_jurisdiction"][jurisdiction]["revenue"] += amount
                report["by_jurisdiction"][jurisdiction]["tax_collected"] += tax_amount
                
                # Group by tax type
                if tax_type not in report["by_tax_type"]:
                    report["by_tax_type"][tax_type] = {
                        "transaction_count": 0,
                        "revenue": 0.0,
                        "tax_collected": 0.0
                    }
                
                report["by_tax_type"][tax_type]["transaction_count"] += 1
                report["by_tax_type"][tax_type]["revenue"] += amount
                report["by_tax_type"][tax_type]["tax_collected"] += tax_amount
            
            return report
            
        except Exception as e:
            logger.error(f"Tax report generation failed: {e}")
            return {"error": str(e)}


class TaxExemptionManager:
    """Manage tax exemptions for customers"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def check_exemption_status(self, user_id: int) -> Dict[str, Any]:
        """Check if user has valid tax exemption"""
        try:
            # In production, query exemption database
            # For now, return mock data
            
            return {
                "exempt": False,
                "exemption_type": None,
                "exemption_certificate": None,
                "valid_until": None,
                "jurisdictions": []
            }
            
        except Exception as e:
            logger.error(f"Failed to check exemption status: {e}")
            return {"exempt": False, "error": str(e)}
    
    async def submit_exemption_certificate(
        self,
        user_id: int,
        certificate_data: Dict[str, Any]
    ) -> bool:
        """Submit tax exemption certificate for review"""
        try:
            # Validate certificate data
            required_fields = ["certificate_number", "issuing_state", "exemption_type"]
            for field in required_fields:
                if field not in certificate_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # In production, store certificate and trigger review process
            logger.info(f"Tax exemption certificate submitted for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit exemption certificate: {e}")
            return False