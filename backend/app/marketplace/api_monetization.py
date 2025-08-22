"""
API monetization system for marketplace vendors.

This module provides comprehensive API monetization including:
- API endpoint registration and management
- Usage tracking and billing
- Rate limiting and quotas
- Performance monitoring
"""

import logging
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..core.config import get_settings
from .models import (
    Vendor, APIEndpoint, APIUsageRecord, APIEndpointStatus,
    APIEndpointCreate, APIEndpointResponse
)
from .revenue_sharing import RevenueShareManager

logger = logging.getLogger(__name__)
settings = get_settings()


class APIMonetizationManager:
    """Manage API endpoint monetization for vendors"""
    
    def __init__(self, db: Session):
        self.db = db
        self.revenue_manager = RevenueShareManager(db)
    
    async def register_api_endpoint(
        self,
        vendor_id: int,
        endpoint_data: APIEndpointCreate,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIEndpoint:
        """Register a new API endpoint for monetization"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if vendor.status != "active":
                raise ValueError("Vendor must be active to register API endpoints")
            
            # Generate unique endpoint ID
            endpoint_id = await self._generate_endpoint_id(endpoint_data.name, vendor.vendor_id)
            
            # Validate endpoint URL is accessible
            if not await self._validate_endpoint_url(endpoint_data.endpoint_url):
                raise ValueError("Endpoint URL is not accessible")
            
            # Create API endpoint
            api_endpoint = APIEndpoint(
                endpoint_id=endpoint_id,
                vendor_id=vendor_id,
                name=endpoint_data.name,
                description=endpoint_data.description,
                endpoint_url=endpoint_data.endpoint_url,
                method=endpoint_data.method.upper(),
                price_per_request=endpoint_data.price_per_request,
                price_per_document=endpoint_data.price_per_document,
                pricing_model=endpoint_data.pricing_model,
                capabilities=endpoint_data.capabilities or [],
                input_formats=endpoint_data.input_formats or [],
                output_formats=endpoint_data.output_formats or [],
                documentation_url=endpoint_data.documentation_url,
                status=APIEndpointStatus.DRAFT,
                metadata=metadata or {}
            )
            
            self.db.add(api_endpoint)
            self.db.commit()
            
            # Send for review
            await self._submit_for_review(api_endpoint)
            
            logger.info(f"Registered API endpoint {endpoint_id} for vendor {vendor.vendor_id}")
            return api_endpoint
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to register API endpoint: {e}")
            raise
    
    async def _generate_endpoint_id(self, name: str, vendor_id: str) -> str:
        """Generate unique endpoint ID"""
        # Create base ID from name and vendor
        base_name = name.lower().replace(" ", "_")[:20]
        base_name = "".join(c for c in base_name if c.isalnum() or c == "_")
        
        base_id = f"{vendor_id}_{base_name}"
        
        # Ensure uniqueness
        counter = 1
        endpoint_id = base_id
        
        while self.db.query(APIEndpoint).filter(APIEndpoint.endpoint_id == endpoint_id).first():
            endpoint_id = f"{base_id}_{counter}"
            counter += 1
        
        return endpoint_id
    
    async def _validate_endpoint_url(self, url: str) -> bool:
        """Validate that endpoint URL is accessible"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.head(url)
                return response.status_code < 500
                
        except Exception as e:
            logger.warning(f"Endpoint URL validation failed: {e}")
            return False
    
    async def _submit_for_review(self, endpoint: APIEndpoint):
        """Submit endpoint for review"""
        endpoint.status = APIEndpointStatus.PENDING_REVIEW
        
        # In production, trigger review workflow
        logger.info(f"Endpoint {endpoint.endpoint_id} submitted for review")
        
        # Auto-approve for demo (remove in production)
        if settings.auto_approve_endpoints:
            await self.approve_endpoint(endpoint.id, "Auto-approved for demo")
    
    async def approve_endpoint(
        self,
        endpoint_id: int,
        approval_notes: Optional[str] = None
    ) -> bool:
        """Approve API endpoint"""
        try:
            endpoint = self.db.query(APIEndpoint).filter(APIEndpoint.id == endpoint_id).first()
            if not endpoint:
                raise ValueError(f"API endpoint {endpoint_id} not found")
            
            endpoint.status = APIEndpointStatus.APPROVED
            endpoint.enabled = True
            endpoint.approved_at = datetime.utcnow()
            
            # Add approval metadata
            metadata = endpoint.metadata or {}
            metadata["approval"] = {
                "approved_at": datetime.utcnow().isoformat(),
                "notes": approval_notes,
                "approved_by": "system"
            }
            endpoint.metadata = metadata
            
            self.db.commit()
            
            # Notify vendor
            await self._send_approval_notification(endpoint)
            
            logger.info(f"Approved API endpoint {endpoint.endpoint_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to approve endpoint {endpoint_id}: {e}")
            raise
    
    async def track_api_usage(
        self,
        endpoint_id: str,
        user_id: int,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        billable_units: int = 1
    ) -> APIUsageRecord:
        """Track API usage for billing"""
        try:
            # Get endpoint
            endpoint = self.db.query(APIEndpoint).filter(
                APIEndpoint.endpoint_id == endpoint_id
            ).first()
            
            if not endpoint:
                raise ValueError(f"API endpoint {endpoint_id} not found")
            
            if not endpoint.enabled:
                raise ValueError(f"API endpoint {endpoint_id} is not enabled")
            
            # Calculate billing amount
            if endpoint.pricing_model == "per_request":
                unit_price = endpoint.price_per_request
                total_amount = unit_price * billable_units
            elif endpoint.pricing_model == "per_document":
                unit_price = endpoint.price_per_document or endpoint.price_per_request
                total_amount = unit_price * billable_units
            else:
                unit_price = endpoint.price_per_request
                total_amount = unit_price * billable_units
            
            # Generate unique request ID
            request_id = self._generate_request_id(endpoint_id, user_id)
            
            # Create usage record
            usage_record = APIUsageRecord(
                endpoint_id=endpoint.id,
                user_id=user_id,
                request_id=request_id,
                method=request_data.get("method", "POST"),
                path=request_data.get("path", "/"),
                billable_units=billable_units,
                unit_price=unit_price,
                total_amount=total_amount,
                response_time_ms=response_data.get("response_time_ms", 0),
                status_code=response_data.get("status_code", 200),
                success=response_data.get("status_code", 200) < 400,
                request_size_bytes=request_data.get("size_bytes", 0),
                response_size_bytes=response_data.get("size_bytes", 0),
                metadata={
                    "request_headers": request_data.get("headers", {}),
                    "response_headers": response_data.get("headers", {}),
                    "user_agent": request_data.get("user_agent"),
                    "ip_address": request_data.get("ip_address")
                }
            )
            
            self.db.add(usage_record)
            
            # Update endpoint metrics
            endpoint.total_requests += 1
            endpoint.total_revenue += total_amount
            endpoint.last_used_at = datetime.utcnow()
            
            # Update performance metrics
            await self._update_endpoint_metrics(endpoint, response_data)
            
            self.db.commit()
            
            # Process revenue sharing
            await self.revenue_manager.process_api_usage_revenue(usage_record)
            
            logger.debug(
                f"Tracked API usage: endpoint={endpoint_id}, user={user_id}, "
                f"amount=${total_amount:.2f}"
            )
            
            return usage_record
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to track API usage: {e}")
            raise
    
    def _generate_request_id(self, endpoint_id: str, user_id: int) -> str:
        """Generate unique request ID"""
        timestamp = str(int(time.time() * 1000))  # Milliseconds
        hash_input = f"{endpoint_id}:{user_id}:{timestamp}"
        request_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"req_{timestamp}_{request_hash}"
    
    async def _update_endpoint_metrics(
        self,
        endpoint: APIEndpoint,
        response_data: Dict[str, Any]
    ):
        """Update endpoint performance metrics"""
        try:
            response_time = response_data.get("response_time_ms", 0)
            success = response_data.get("status_code", 200) < 400
            
            # Calculate new averages (simple moving average)
            total_requests = endpoint.total_requests
            
            if total_requests > 0:
                # Update average response time
                current_avg = endpoint.average_response_time
                new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
                endpoint.average_response_time = new_avg
                
                # Update success rate
                if success:
                    successful_requests = endpoint.success_rate * (total_requests - 1) / 100 + 1
                else:
                    successful_requests = endpoint.success_rate * (total_requests - 1) / 100
                
                endpoint.success_rate = (successful_requests / total_requests) * 100
            
        except Exception as e:
            logger.error(f"Failed to update endpoint metrics: {e}")
    
    async def check_rate_limits(
        self,
        endpoint_id: str,
        user_id: int,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Check if user has exceeded rate limits for endpoint"""
        try:
            endpoint = self.db.query(APIEndpoint).filter(
                APIEndpoint.endpoint_id == endpoint_id
            ).first()
            
            if not endpoint:
                raise ValueError(f"API endpoint {endpoint_id} not found")
            
            # Get usage in the time window
            window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            usage_count = self.db.query(APIUsageRecord).filter(
                APIUsageRecord.endpoint_id == endpoint.id,
                APIUsageRecord.user_id == user_id,
                APIUsageRecord.created_at >= window_start
            ).count()
            
            # Calculate limit for the window
            rate_limit_per_hour = endpoint.rate_limit
            rate_limit_for_window = int(rate_limit_per_hour * (window_minutes / 60))
            
            within_limits = usage_count < rate_limit_for_window
            
            return {
                "within_limits": within_limits,
                "current_usage": usage_count,
                "limit": rate_limit_for_window,
                "window_minutes": window_minutes,
                "reset_time": (window_start + timedelta(minutes=window_minutes)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to check rate limits: {e}")
            return {"within_limits": True, "error": str(e)}
    
    async def get_endpoint_analytics(
        self,
        endpoint_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics for an API endpoint"""
        try:
            endpoint = self.db.query(APIEndpoint).filter(
                APIEndpoint.endpoint_id == endpoint_id
            ).first()
            
            if not endpoint:
                raise ValueError(f"API endpoint {endpoint_id} not found")
            
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get usage records for the period
            usage_records = self.db.query(APIUsageRecord).filter(
                APIUsageRecord.endpoint_id == endpoint.id,
                APIUsageRecord.created_at >= start_date,
                APIUsageRecord.created_at <= end_date
            ).all()
            
            # Calculate metrics
            total_requests = len(usage_records)
            successful_requests = len([r for r in usage_records if r.success])
            total_revenue = sum(r.total_amount for r in usage_records)
            
            # Response time statistics
            response_times = [r.response_time_ms for r in usage_records if r.response_time_ms]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Daily usage breakdown
            daily_usage = {}
            for record in usage_records:
                date_key = record.created_at.date().isoformat()
                if date_key not in daily_usage:
                    daily_usage[date_key] = {
                        "requests": 0,
                        "revenue": 0.0,
                        "successful_requests": 0
                    }
                
                daily_usage[date_key]["requests"] += 1
                daily_usage[date_key]["revenue"] += record.total_amount
                if record.success:
                    daily_usage[date_key]["successful_requests"] += 1
            
            # Top users
            user_usage = {}
            for record in usage_records:
                user_id = record.user_id
                if user_id not in user_usage:
                    user_usage[user_id] = {
                        "requests": 0,
                        "revenue": 0.0
                    }
                
                user_usage[user_id]["requests"] += 1
                user_usage[user_id]["revenue"] += record.total_amount
            
            top_users = sorted(
                user_usage.items(),
                key=lambda x: x[1]["requests"],
                reverse=True
            )[:10]
            
            return {
                "endpoint": {
                    "id": endpoint.id,
                    "endpoint_id": endpoint.endpoint_id,
                    "name": endpoint.name,
                    "vendor_id": endpoint.vendor_id,
                    "status": endpoint.status.value,
                    "enabled": endpoint.enabled
                },
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "metrics": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                    "total_revenue": total_revenue,
                    "average_response_time_ms": avg_response_time
                },
                "daily_usage": daily_usage,
                "top_users": [
                    {
                        "user_id": user_id,
                        "metrics": metrics
                    }
                    for user_id, metrics in top_users
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint analytics: {e}")
            return {"error": str(e)}
    
    async def get_vendor_api_summary(
        self,
        vendor_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get API summary for a vendor"""
        try:
            vendor = self.db.query(Vendor).filter(Vendor.id == vendor_id).first()
            if not vendor:
                raise ValueError(f"Vendor {vendor_id} not found")
            
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get vendor's endpoints
            endpoints = self.db.query(APIEndpoint).filter(
                APIEndpoint.vendor_id == vendor_id
            ).all()
            
            # Get usage records for all endpoints
            endpoint_ids = [ep.id for ep in endpoints]
            
            if endpoint_ids:
                usage_records = self.db.query(APIUsageRecord).filter(
                    APIUsageRecord.endpoint_id.in_(endpoint_ids),
                    APIUsageRecord.created_at >= start_date,
                    APIUsageRecord.created_at <= end_date
                ).all()
            else:
                usage_records = []
            
            # Calculate totals
            total_requests = len(usage_records)
            successful_requests = len([r for r in usage_records if r.success])
            total_revenue = sum(r.total_amount for r in usage_records)
            
            # Endpoint breakdown
            endpoint_metrics = {}
            for endpoint in endpoints:
                endpoint_usage = [r for r in usage_records if r.endpoint_id == endpoint.id]
                
                endpoint_metrics[endpoint.endpoint_id] = {
                    "name": endpoint.name,
                    "status": endpoint.status.value,
                    "enabled": endpoint.enabled,
                    "requests": len(endpoint_usage),
                    "revenue": sum(r.total_amount for r in endpoint_usage),
                    "success_rate": endpoint.success_rate,
                    "average_response_time": endpoint.average_response_time
                }
            
            return {
                "vendor": {
                    "id": vendor.id,
                    "vendor_id": vendor.vendor_id,
                    "company_name": vendor.company_name
                },
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_endpoints": len(endpoints),
                    "active_endpoints": len([ep for ep in endpoints if ep.enabled]),
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                    "total_revenue": total_revenue
                },
                "endpoints": endpoint_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get vendor API summary: {e}")
            return {"error": str(e)}
    
    async def suspend_endpoint(
        self,
        endpoint_id: str,
        reason: str
    ) -> bool:
        """Suspend an API endpoint"""
        try:
            endpoint = self.db.query(APIEndpoint).filter(
                APIEndpoint.endpoint_id == endpoint_id
            ).first()
            
            if not endpoint:
                raise ValueError(f"API endpoint {endpoint_id} not found")
            
            endpoint.status = APIEndpointStatus.SUSPENDED
            endpoint.enabled = False
            
            # Add suspension metadata
            metadata = endpoint.metadata or {}
            metadata["suspension"] = {
                "suspended_at": datetime.utcnow().isoformat(),
                "reason": reason,
                "suspended_by": "system"
            }
            endpoint.metadata = metadata
            
            self.db.commit()
            
            # Notify vendor
            await self._send_suspension_notification(endpoint, reason)
            
            logger.info(f"Suspended API endpoint {endpoint_id}: {reason}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to suspend endpoint {endpoint_id}: {e}")
            raise
    
    async def _send_approval_notification(self, endpoint: APIEndpoint):
        """Send endpoint approval notification"""
        vendor = endpoint.vendor
        logger.info(f"Would send approval notification for endpoint {endpoint.endpoint_id} to {vendor.contact_email}")
    
    async def _send_suspension_notification(self, endpoint: APIEndpoint, reason: str):
        """Send endpoint suspension notification"""
        vendor = endpoint.vendor
        logger.info(f"Would send suspension notification for endpoint {endpoint.endpoint_id} to {vendor.contact_email}: {reason}")
    
    async def get_marketplace_catalog(
        self,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get marketplace catalog of available API endpoints"""
        try:
            query = self.db.query(APIEndpoint).filter(
                APIEndpoint.status == APIEndpointStatus.APPROVED,
                APIEndpoint.enabled == True
            )
            
            # Apply filters
            if category:
                query = query.filter(
                    APIEndpoint.capabilities.contains([category])
                )
            
            if search_query:
                query = query.filter(
                    or_(
                        APIEndpoint.name.ilike(f"%{search_query}%"),
                        APIEndpoint.description.ilike(f"%{search_query}%")
                    )
                )
            
            endpoints = query.order_by(
                APIEndpoint.success_rate.desc(),
                APIEndpoint.total_requests.desc()
            ).offset(offset).limit(limit).all()
            
            catalog = []
            for endpoint in endpoints:
                vendor = endpoint.vendor
                
                catalog.append({
                    "endpoint_id": endpoint.endpoint_id,
                    "name": endpoint.name,
                    "description": endpoint.description,
                    "vendor": {
                        "vendor_id": vendor.vendor_id,
                        "company_name": vendor.company_name,
                        "verified": vendor.verified
                    },
                    "pricing": {
                        "model": endpoint.pricing_model,
                        "price_per_request": endpoint.price_per_request,
                        "price_per_document": endpoint.price_per_document
                    },
                    "capabilities": endpoint.capabilities,
                    "input_formats": endpoint.input_formats,
                    "output_formats": endpoint.output_formats,
                    "performance": {
                        "success_rate": endpoint.success_rate,
                        "average_response_time": endpoint.average_response_time,
                        "total_requests": endpoint.total_requests
                    },
                    "documentation_url": endpoint.documentation_url,
                    "rate_limit": endpoint.rate_limit
                })
            
            return catalog
            
        except Exception as e:
            logger.error(f"Failed to get marketplace catalog: {e}")
            return []