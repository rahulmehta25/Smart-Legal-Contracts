"""
IP Filtering and Geolocation System
Implements IP allowlist/blocklist with geolocation and threat intelligence
OWASP Security compliant
"""

import ipaddress
import time
import json
import hashlib
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

import redis
import requests
from fastapi import HTTPException, status, Request


class IPListType(str, Enum):
    """Types of IP lists"""
    ALLOWLIST = "allowlist"
    BLOCKLIST = "blocklist"
    GRAYLIST = "graylist"      # Temporary/conditional block
    VPN_LIST = "vpn_list"       # Known VPN/proxy IPs
    TOR_LIST = "tor_list"       # Tor exit nodes
    THREAT_LIST = "threat_list" # Known malicious IPs


class IPFilterAction(str, Enum):
    """Actions to take on IP match"""
    ALLOW = "allow"
    BLOCK = "block"
    CHALLENGE = "challenge"     # Require additional verification
    RATE_LIMIT = "rate_limit"   # Apply stricter rate limits
    LOG = "log"                 # Log but allow


@dataclass
class IPInfo:
    """IP address information"""
    ip: str
    country: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    asn: Optional[str] = None
    org: Optional[str] = None
    is_vpn: bool = False
    is_tor: bool = False
    is_proxy: bool = False
    is_hosting: bool = False
    threat_score: float = 0.0
    reputation: float = 100.0
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class IPFilterRule:
    """IP filtering rule"""
    pattern: str                    # IP, CIDR, or range
    list_type: IPListType
    action: IPFilterAction
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at: datetime = None
    priority: int = 0               # Higher priority rules override
    conditions: Dict[str, Any] = None  # Additional conditions


class IPFilter:
    """
    Advanced IP filtering system with:
    - IP allowlist/blocklist management
    - CIDR range support
    - Geolocation-based filtering
    - VPN/Tor/Proxy detection
    - Threat intelligence integration
    - Dynamic list updates
    - Temporary bans
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 enable_geolocation: bool = True,
                 enable_threat_intel: bool = True):
        
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        self.enable_geolocation = enable_geolocation
        self.enable_threat_intel = enable_threat_intel
        
        # In-memory cache for performance
        self.ip_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 300  # 5 minutes
        
        # IP lists
        self.lists = {
            IPListType.ALLOWLIST: set(),
            IPListType.BLOCKLIST: set(),
            IPListType.GRAYLIST: set(),
            IPListType.VPN_LIST: set(),
            IPListType.TOR_LIST: set(),
            IPListType.THREAT_LIST: set()
        }
        
        # CIDR ranges
        self.cidr_ranges = {
            IPListType.ALLOWLIST: [],
            IPListType.BLOCKLIST: [],
            IPListType.GRAYLIST: []
        }
        
        # Geolocation rules
        self.geo_rules = {
            "blocked_countries": set(),
            "allowed_countries": set(),
            "high_risk_countries": set()
        }
        
        # Metrics
        self.metrics = defaultdict(int)
        
        # Load initial lists
        self._load_lists()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_lists)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    # ========== Main Filtering Logic ==========
    
    def check_ip(self, ip: str, context: Optional[Dict[str, Any]] = None) -> Tuple[IPFilterAction, Optional[str]]:
        """
        Check IP against all filters
        Returns: (action, reason)
        """
        
        # Validate IP format
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return IPFilterAction.BLOCK, "Invalid IP address"
        
        # Check cache first
        cached = self._get_cached_result(ip)
        if cached:
            self.metrics["cache_hits"] += 1
            return cached
        
        self.metrics["cache_misses"] += 1
        
        # Check private/reserved IPs
        if ip_obj.is_private or ip_obj.is_reserved:
            return IPFilterAction.ALLOW, "Private/reserved IP"
        
        # Check explicit lists (highest priority)
        list_action, list_reason = self._check_lists(ip)
        if list_action != IPFilterAction.ALLOW:
            self._cache_result(ip, list_action, list_reason)
            return list_action, list_reason
        
        # Check CIDR ranges
        cidr_action, cidr_reason = self._check_cidr_ranges(ip)
        if cidr_action != IPFilterAction.ALLOW:
            self._cache_result(ip, cidr_action, cidr_reason)
            return cidr_action, cidr_reason
        
        # Get IP information
        ip_info = self._get_ip_info(ip)
        
        # Check geolocation rules
        if self.enable_geolocation and ip_info.country:
            geo_action, geo_reason = self._check_geolocation(ip_info)
            if geo_action != IPFilterAction.ALLOW:
                self._cache_result(ip, geo_action, geo_reason)
                return geo_action, geo_reason
        
        # Check VPN/Tor/Proxy
        vpn_action, vpn_reason = self._check_vpn_tor_proxy(ip_info)
        if vpn_action != IPFilterAction.ALLOW:
            self._cache_result(ip, vpn_action, vpn_reason)
            return vpn_action, vpn_reason
        
        # Check threat intelligence
        if self.enable_threat_intel:
            threat_action, threat_reason = self._check_threat_intel(ip_info)
            if threat_action != IPFilterAction.ALLOW:
                self._cache_result(ip, threat_action, threat_reason)
                return threat_action, threat_reason
        
        # Default allow
        self._cache_result(ip, IPFilterAction.ALLOW, None)
        return IPFilterAction.ALLOW, None
    
    # ========== List Management ==========
    
    def add_to_list(self, ip_or_cidr: str, list_type: IPListType, 
                    duration: Optional[int] = None, reason: Optional[str] = None):
        """Add IP or CIDR to list"""
        
        # Check if CIDR
        try:
            network = ipaddress.ip_network(ip_or_cidr, strict=False)
            if network.num_addresses > 1:
                # It's a CIDR range
                self._add_cidr_range(network, list_type, duration, reason)
            else:
                # Single IP
                self._add_single_ip(str(network.network_address), list_type, duration, reason)
        except ValueError:
            raise ValueError(f"Invalid IP or CIDR: {ip_or_cidr}")
    
    def remove_from_list(self, ip_or_cidr: str, list_type: IPListType):
        """Remove IP or CIDR from list"""
        
        # Remove from Redis
        self.redis_client.srem(f"ip_list:{list_type}", ip_or_cidr)
        
        # Remove from memory
        self.lists[list_type].discard(ip_or_cidr)
        
        # Clear cache
        self._clear_cache(ip_or_cidr)
    
    def _add_single_ip(self, ip: str, list_type: IPListType, 
                       duration: Optional[int], reason: Optional[str]):
        """Add single IP to list"""
        
        # Add to Redis
        self.redis_client.sadd(f"ip_list:{list_type}", ip)
        
        # Add metadata
        metadata = {
            "added_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "list_type": list_type
        }
        
        if duration:
            # Set expiration
            expire_at = datetime.utcnow() + timedelta(seconds=duration)
            metadata["expires_at"] = expire_at.isoformat()
            
            # Schedule removal
            self.redis_client.zadd(
                "ip_expirations",
                {f"{list_type}:{ip}": expire_at.timestamp()}
            )
        
        self.redis_client.hset(f"ip_metadata:{ip}", mapping=metadata)
        
        # Add to memory
        self.lists[list_type].add(ip)
        
        # Clear cache
        self._clear_cache(ip)
    
    def _add_cidr_range(self, network: ipaddress.IPv4Network, list_type: IPListType,
                        duration: Optional[int], reason: Optional[str]):
        """Add CIDR range to list"""
        
        cidr_str = str(network)
        
        # Add to Redis
        self.redis_client.sadd(f"cidr_list:{list_type}", cidr_str)
        
        # Add metadata
        metadata = {
            "added_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "list_type": list_type,
            "num_addresses": network.num_addresses
        }
        
        if duration:
            expire_at = datetime.utcnow() + timedelta(seconds=duration)
            metadata["expires_at"] = expire_at.isoformat()
            
            self.redis_client.zadd(
                "cidr_expirations",
                {f"{list_type}:{cidr_str}": expire_at.timestamp()}
            )
        
        self.redis_client.hset(f"cidr_metadata:{cidr_str}", mapping=metadata)
        
        # Add to memory
        if list_type in self.cidr_ranges:
            self.cidr_ranges[list_type].append(network)
        
        # Clear cache for all IPs in range
        # (In practice, might want to be more selective)
        with self.cache_lock:
            self.ip_cache.clear()
    
    # ========== IP Checking Methods ==========
    
    def _check_lists(self, ip: str) -> Tuple[IPFilterAction, Optional[str]]:
        """Check IP against explicit lists"""
        
        # Check allowlist first (highest priority)
        if ip in self.lists[IPListType.ALLOWLIST]:
            return IPFilterAction.ALLOW, "IP in allowlist"
        
        # Check blocklist
        if ip in self.lists[IPListType.BLOCKLIST]:
            self.metrics["blocked_ips"] += 1
            return IPFilterAction.BLOCK, "IP in blocklist"
        
        # Check graylist
        if ip in self.lists[IPListType.GRAYLIST]:
            return IPFilterAction.CHALLENGE, "IP in graylist"
        
        # Check VPN list
        if ip in self.lists[IPListType.VPN_LIST]:
            # Could be configurable action
            return IPFilterAction.CHALLENGE, "VPN/Proxy detected"
        
        # Check Tor list
        if ip in self.lists[IPListType.TOR_LIST]:
            return IPFilterAction.BLOCK, "Tor exit node"
        
        # Check threat list
        if ip in self.lists[IPListType.THREAT_LIST]:
            self.metrics["threat_blocked"] += 1
            return IPFilterAction.BLOCK, "IP in threat list"
        
        return IPFilterAction.ALLOW, None
    
    def _check_cidr_ranges(self, ip: str) -> Tuple[IPFilterAction, Optional[str]]:
        """Check IP against CIDR ranges"""
        
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return IPFilterAction.BLOCK, "Invalid IP"
        
        # Check allowlist CIDR ranges first
        for network in self.cidr_ranges[IPListType.ALLOWLIST]:
            if ip_obj in network:
                return IPFilterAction.ALLOW, f"IP in allowed range {network}"
        
        # Check blocklist CIDR ranges
        for network in self.cidr_ranges[IPListType.BLOCKLIST]:
            if ip_obj in network:
                self.metrics["blocked_cidrs"] += 1
                return IPFilterAction.BLOCK, f"IP in blocked range {network}"
        
        # Check graylist CIDR ranges
        for network in self.cidr_ranges[IPListType.GRAYLIST]:
            if ip_obj in network:
                return IPFilterAction.CHALLENGE, f"IP in graylisted range {network}"
        
        return IPFilterAction.ALLOW, None
    
    def _check_geolocation(self, ip_info: IPInfo) -> Tuple[IPFilterAction, Optional[str]]:
        """Check geolocation rules"""
        
        if not ip_info.country:
            return IPFilterAction.ALLOW, None
        
        # Check blocked countries
        if ip_info.country in self.geo_rules["blocked_countries"]:
            self.metrics["geo_blocked"] += 1
            return IPFilterAction.BLOCK, f"Country blocked: {ip_info.country}"
        
        # Check allowed countries (if specified, only these are allowed)
        if self.geo_rules["allowed_countries"]:
            if ip_info.country not in self.geo_rules["allowed_countries"]:
                return IPFilterAction.BLOCK, f"Country not allowed: {ip_info.country}"
        
        # Check high-risk countries
        if ip_info.country in self.geo_rules["high_risk_countries"]:
            return IPFilterAction.CHALLENGE, f"High-risk country: {ip_info.country}"
        
        return IPFilterAction.ALLOW, None
    
    def _check_vpn_tor_proxy(self, ip_info: IPInfo) -> Tuple[IPFilterAction, Optional[str]]:
        """Check for VPN/Tor/Proxy usage"""
        
        if ip_info.is_tor:
            self.metrics["tor_blocked"] += 1
            return IPFilterAction.BLOCK, "Tor exit node detected"
        
        if ip_info.is_vpn or ip_info.is_proxy:
            # Could be configurable
            return IPFilterAction.CHALLENGE, "VPN/Proxy detected"
        
        if ip_info.is_hosting:
            # Hosting providers might be suspicious
            return IPFilterAction.RATE_LIMIT, "Hosting provider IP"
        
        return IPFilterAction.ALLOW, None
    
    def _check_threat_intel(self, ip_info: IPInfo) -> Tuple[IPFilterAction, Optional[str]]:
        """Check threat intelligence"""
        
        if ip_info.threat_score > 80:
            self.metrics["threat_blocked"] += 1
            return IPFilterAction.BLOCK, f"High threat score: {ip_info.threat_score}"
        
        if ip_info.threat_score > 50:
            return IPFilterAction.CHALLENGE, f"Moderate threat score: {ip_info.threat_score}"
        
        if ip_info.reputation < 20:
            return IPFilterAction.BLOCK, f"Low reputation: {ip_info.reputation}"
        
        if ip_info.reputation < 50:
            return IPFilterAction.RATE_LIMIT, f"Poor reputation: {ip_info.reputation}"
        
        return IPFilterAction.ALLOW, None
    
    # ========== IP Information ==========
    
    def _get_ip_info(self, ip: str) -> IPInfo:
        """Get detailed IP information"""
        
        # Check cache
        cache_key = f"ip_info:{ip}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            return IPInfo(**data)
        
        # Fetch fresh data
        ip_info = IPInfo(ip=ip)
        
        # Get geolocation
        if self.enable_geolocation:
            geo_data = self._fetch_geolocation(ip)
            if geo_data:
                ip_info.country = geo_data.get("country")
                ip_info.city = geo_data.get("city")
                ip_info.region = geo_data.get("region")
                ip_info.asn = geo_data.get("asn")
                ip_info.org = geo_data.get("org")
        
        # Check VPN/Proxy
        vpn_data = self._check_vpn_proxy_api(ip)
        if vpn_data:
            ip_info.is_vpn = vpn_data.get("is_vpn", False)
            ip_info.is_proxy = vpn_data.get("is_proxy", False)
            ip_info.is_hosting = vpn_data.get("is_hosting", False)
        
        # Check Tor
        ip_info.is_tor = self._check_tor_exit(ip)
        
        # Get threat score
        if self.enable_threat_intel:
            threat_data = self._fetch_threat_intel(ip)
            if threat_data:
                ip_info.threat_score = threat_data.get("score", 0)
                ip_info.reputation = threat_data.get("reputation", 100)
        
        # Cache result
        self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(asdict(ip_info))
        )
        
        return ip_info
    
    def _fetch_geolocation(self, ip: str) -> Optional[Dict[str, Any]]:
        """Fetch geolocation data"""
        
        # Would integrate with service like MaxMind, IPInfo, etc.
        # For now, return mock data
        return {
            "country": "US",
            "city": "San Francisco",
            "region": "CA",
            "asn": "AS13335",
            "org": "Cloudflare"
        }
    
    def _check_vpn_proxy_api(self, ip: str) -> Optional[Dict[str, Any]]:
        """Check VPN/Proxy detection API"""
        
        # Would integrate with service like IPQualityScore, ProxyCheck, etc.
        return {
            "is_vpn": False,
            "is_proxy": False,
            "is_hosting": False
        }
    
    def _check_tor_exit(self, ip: str) -> bool:
        """Check if IP is Tor exit node"""
        
        # Check against Tor exit node list
        return ip in self.lists[IPListType.TOR_LIST]
    
    def _fetch_threat_intel(self, ip: str) -> Optional[Dict[str, Any]]:
        """Fetch threat intelligence data"""
        
        # Would integrate with threat intel feeds
        # AbuseIPDB, AlienVault OTX, etc.
        return {
            "score": 10,
            "reputation": 90
        }
    
    # ========== List Updates ==========
    
    def _load_lists(self):
        """Load IP lists from Redis"""
        
        for list_type in IPListType:
            # Load single IPs
            members = self.redis_client.smembers(f"ip_list:{list_type}")
            self.lists[list_type] = set(members)
            
            # Load CIDR ranges
            if list_type in [IPListType.ALLOWLIST, IPListType.BLOCKLIST, IPListType.GRAYLIST]:
                cidr_members = self.redis_client.smembers(f"cidr_list:{list_type}")
                self.cidr_ranges[list_type] = [
                    ipaddress.ip_network(cidr) for cidr in cidr_members
                ]
    
    def _update_lists(self):
        """Background thread to update lists"""
        
        while True:
            try:
                # Update Tor exit nodes
                self._update_tor_list()
                
                # Update threat lists
                self._update_threat_lists()
                
                # Process expirations
                self._process_expirations()
                
                # Reload lists
                self._load_lists()
                
                time.sleep(3600)  # Update hourly
            except Exception as e:
                print(f"List update error: {e}")
                time.sleep(3600)
    
    def _update_tor_list(self):
        """Update Tor exit node list"""
        
        try:
            # Fetch Tor exit node list
            response = requests.get(
                "https://check.torproject.org/exit-addresses",
                timeout=10
            )
            
            if response.status_code == 200:
                # Parse exit nodes
                tor_ips = set()
                for line in response.text.split('\n'):
                    if line.startswith('ExitAddress'):
                        ip = line.split()[1]
                        tor_ips.add(ip)
                
                # Update Redis
                if tor_ips:
                    self.redis_client.delete(f"ip_list:{IPListType.TOR_LIST}")
                    self.redis_client.sadd(f"ip_list:{IPListType.TOR_LIST}", *tor_ips)
        except:
            pass  # Fail silently
    
    def _update_threat_lists(self):
        """Update threat intelligence lists"""
        
        # Would fetch from various threat feeds
        # - Emerging Threats
        # - Spamhaus
        # - DShield
        # - etc.
        pass
    
    def _process_expirations(self):
        """Process expired IP entries"""
        
        now = time.time()
        
        # Process IP expirations
        expired = self.redis_client.zrangebyscore("ip_expirations", 0, now)
        for entry in expired:
            list_type, ip = entry.split(":", 1)
            self.remove_from_list(ip, IPListType(list_type))
        
        # Remove processed entries
        if expired:
            self.redis_client.zrem("ip_expirations", *expired)
        
        # Process CIDR expirations
        expired_cidrs = self.redis_client.zrangebyscore("cidr_expirations", 0, now)
        for entry in expired_cidrs:
            list_type, cidr = entry.split(":", 1)
            self.remove_from_list(cidr, IPListType(list_type))
        
        if expired_cidrs:
            self.redis_client.zrem("cidr_expirations", *expired_cidrs)
    
    # ========== Caching ==========
    
    def _get_cached_result(self, ip: str) -> Optional[Tuple[IPFilterAction, Optional[str]]]:
        """Get cached filter result"""
        
        with self.cache_lock:
            if ip in self.ip_cache:
                entry = self.ip_cache[ip]
                if time.time() - entry["timestamp"] < self.cache_ttl:
                    return entry["action"], entry["reason"]
                else:
                    del self.ip_cache[ip]
        
        return None
    
    def _cache_result(self, ip: str, action: IPFilterAction, reason: Optional[str]):
        """Cache filter result"""
        
        with self.cache_lock:
            self.ip_cache[ip] = {
                "action": action,
                "reason": reason,
                "timestamp": time.time()
            }
    
    def _clear_cache(self, ip: str):
        """Clear cached result for IP"""
        
        with self.cache_lock:
            self.ip_cache.pop(ip, None)
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics"""
        
        stats = {
            "metrics": dict(self.metrics),
            "list_sizes": {
                list_type: len(ips) for list_type, ips in self.lists.items()
            },
            "cidr_counts": {
                list_type: len(ranges) for list_type, ranges in self.cidr_ranges.items()
            },
            "cache_size": len(self.ip_cache),
            "geo_rules": {
                "blocked_countries": list(self.geo_rules["blocked_countries"]),
                "allowed_countries": list(self.geo_rules["allowed_countries"]),
                "high_risk_countries": list(self.geo_rules["high_risk_countries"])
            }
        }
        
        return stats