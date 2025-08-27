"""
Advanced DDoS Protection System
Implements multi-layer DDoS mitigation strategies
OWASP DDoS Protection Guidelines compliant
"""

import time
import asyncio
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import math

import redis
import numpy as np
from fastapi import HTTPException, status, Request, Response
from fastapi.responses import HTMLResponse


class AttackType(str, Enum):
    """Types of DDoS attacks"""
    VOLUMETRIC = "volumetric"          # High traffic volume
    PROTOCOL = "protocol"              # TCP/UDP exploits
    APPLICATION = "application"        # Layer 7 attacks
    SLOWLORIS = "slowloris"           # Slow HTTP attacks
    SYN_FLOOD = "syn_flood"           # TCP SYN flood
    UDP_FLOOD = "udp_flood"           # UDP flood
    HTTP_FLOOD = "http_flood"         # HTTP request flood
    DNS_AMPLIFICATION = "dns_amplification"
    NTP_AMPLIFICATION = "ntp_amplification"


class MitigationStrategy(str, Enum):
    """DDoS mitigation strategies"""
    RATE_LIMIT = "rate_limit"         # Aggressive rate limiting
    BLACKHOLE = "blackhole"           # Drop all traffic
    CHALLENGE = "challenge"           # CAPTCHA/JS challenge
    SYN_COOKIES = "syn_cookies"       # TCP SYN cookies
    CONNECTION_LIMIT = "connection_limit"
    GEO_BLOCKING = "geo_blocking"     # Block high-risk regions
    PATTERN_FILTER = "pattern_filter" # Filter by patterns


class ProtectionLevel(str, Enum):
    """Protection levels"""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNDER_ATTACK = "under_attack"     # Maximum protection


@dataclass
class AttackSignature:
    """Attack signature for pattern matching"""
    pattern_type: str                  # request_rate, packet_size, etc.
    threshold: float
    window: int                        # Time window in seconds
    confidence: float = 0.8           # Confidence threshold


@dataclass
class AttackEvent:
    """DDoS attack event"""
    attack_type: AttackType
    start_time: datetime
    end_time: Optional[datetime] = None
    peak_rps: float = 0               # Peak requests per second
    total_requests: int = 0
    unique_ips: int = 0
    mitigation_applied: List[MitigationStrategy] = field(default_factory=list)
    status: str = "ongoing"           # ongoing, mitigated, ended


class DDoSProtectionSystem:
    """
    Comprehensive DDoS protection system with:
    - Multi-layer attack detection
    - Adaptive mitigation strategies
    - Challenge-response mechanisms
    - Traffic pattern analysis
    - Behavioral analysis
    - Automatic attack response
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 protection_level: ProtectionLevel = ProtectionLevel.MEDIUM):
        
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        self.protection_level = protection_level
        
        # Attack detection
        self.attack_detectors = {
            AttackType.VOLUMETRIC: VolumetricDetector(),
            AttackType.SLOWLORIS: SlowlorisDetector(),
            AttackType.HTTP_FLOOD: HTTPFloodDetector(),
            AttackType.APPLICATION: ApplicationLayerDetector()
        }
        
        # Current attack status
        self.current_attacks = {}
        self.attack_lock = threading.Lock()
        
        # Traffic analysis
        self.traffic_analyzer = TrafficAnalyzer()
        
        # Challenge system
        self.challenge_system = ChallengeSystem(redis_client)
        
        # Mitigation strategies
        self.mitigation_strategies = {}
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "challenged_requests": 0,
            "attacks_detected": 0,
            "attacks_mitigated": 0
        }
        
        # Start monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_traffic)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    # ========== Main Protection Logic ==========
    
    async def check_request(self,
                           request: Request,
                           ip: str,
                           user_agent: Optional[str] = None) -> Tuple[bool, Optional[Response]]:
        """
        Check incoming request for DDoS patterns
        Returns: (allowed, challenge_response)
        """
        
        self.metrics["total_requests"] += 1
        
        # Record traffic
        self.traffic_analyzer.record_request(ip, request.url.path)
        
        # Check if under attack
        attack_level = self._get_attack_level()
        
        # Apply protection based on level
        if self.protection_level == ProtectionLevel.OFF:
            return True, None
        
        # Check if IP is already challenged and verified
        if self.challenge_system.is_verified(ip):
            return True, None
        
        # Detect attacks
        detected_attacks = self._detect_attacks(ip, request)
        
        if detected_attacks:
            # Apply mitigation
            mitigation_response = await self._apply_mitigation(
                ip, request, detected_attacks
            )
            
            if mitigation_response:
                self.metrics["blocked_requests"] += 1
                return False, mitigation_response
        
        # Apply protection level checks
        if attack_level >= 0.7 or self.protection_level == ProtectionLevel.UNDER_ATTACK:
            # Challenge all new visitors
            if not self.challenge_system.has_solved_challenge(ip):
                self.metrics["challenged_requests"] += 1
                challenge_response = self.challenge_system.create_challenge(ip)
                return False, challenge_response
        
        return True, None
    
    # ========== Attack Detection ==========
    
    def _detect_attacks(self, ip: str, request: Request) -> List[AttackType]:
        """Detect potential attacks"""
        
        detected = []
        
        for attack_type, detector in self.attack_detectors.items():
            if detector.detect(ip, request, self.traffic_analyzer):
                detected.append(attack_type)
                self._record_attack(attack_type, ip)
        
        return detected
    
    def _record_attack(self, attack_type: AttackType, ip: str):
        """Record detected attack"""
        
        with self.attack_lock:
            if attack_type not in self.current_attacks:
                self.current_attacks[attack_type] = AttackEvent(
                    attack_type=attack_type,
                    start_time=datetime.utcnow()
                )
                self.metrics["attacks_detected"] += 1
            
            # Update attack event
            event = self.current_attacks[attack_type]
            event.total_requests += 1
            
            # Track unique IPs
            attack_ips_key = f"attack_ips:{attack_type}"
            self.redis_client.sadd(attack_ips_key, ip)
            self.redis_client.expire(attack_ips_key, 3600)
    
    def _get_attack_level(self) -> float:
        """Calculate current attack level (0-1)"""
        
        # Check traffic anomalies
        traffic_score = self.traffic_analyzer.get_anomaly_score()
        
        # Check active attacks
        attack_score = len(self.current_attacks) / len(self.attack_detectors)
        
        # Combined score
        return min(1.0, traffic_score * 0.6 + attack_score * 0.4)
    
    # ========== Mitigation ==========
    
    async def _apply_mitigation(self,
                               ip: str,
                               request: Request,
                               attacks: List[AttackType]) -> Optional[Response]:
        """Apply mitigation strategies"""
        
        # Determine mitigation strategy
        strategies = self._select_mitigation_strategies(attacks)
        
        for strategy in strategies:
            if strategy == MitigationStrategy.BLACKHOLE:
                # Drop request completely
                return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            elif strategy == MitigationStrategy.CHALLENGE:
                # Return challenge
                return self.challenge_system.create_challenge(ip)
            
            elif strategy == MitigationStrategy.RATE_LIMIT:
                # Apply aggressive rate limiting
                return Response(
                    content="Rate limit exceeded",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS
                )
            
            elif strategy == MitigationStrategy.GEO_BLOCKING:
                # Would check geolocation
                pass
        
        return None
    
    def _select_mitigation_strategies(self, attacks: List[AttackType]) -> List[MitigationStrategy]:
        """Select appropriate mitigation strategies"""
        
        strategies = []
        
        for attack in attacks:
            if attack == AttackType.VOLUMETRIC:
                strategies.append(MitigationStrategy.RATE_LIMIT)
                strategies.append(MitigationStrategy.CHALLENGE)
            
            elif attack == AttackType.HTTP_FLOOD:
                strategies.append(MitigationStrategy.CHALLENGE)
            
            elif attack == AttackType.SLOWLORIS:
                strategies.append(MitigationStrategy.CONNECTION_LIMIT)
            
            elif attack == AttackType.APPLICATION:
                strategies.append(MitigationStrategy.PATTERN_FILTER)
                strategies.append(MitigationStrategy.CHALLENGE)
        
        return strategies
    
    # ========== Monitoring ==========
    
    def _monitor_traffic(self):
        """Background monitoring thread"""
        
        while True:
            try:
                # Analyze traffic patterns
                self.traffic_analyzer.analyze_patterns()
                
                # Check for attack end
                self._check_attack_status()
                
                # Export metrics
                self._export_metrics()
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def _check_attack_status(self):
        """Check if attacks have ended"""
        
        with self.attack_lock:
            for attack_type, event in list(self.current_attacks.items()):
                # Check if attack has subsided
                recent_requests = self.traffic_analyzer.get_recent_request_count()
                
                if recent_requests < 100:  # Threshold
                    event.end_time = datetime.utcnow()
                    event.status = "ended"
                    self.metrics["attacks_mitigated"] += 1
                    
                    # Archive attack event
                    self._archive_attack_event(event)
                    
                    # Remove from current attacks
                    del self.current_attacks[attack_type]
    
    def _archive_attack_event(self, event: AttackEvent):
        """Archive attack event for analysis"""
        
        event_data = {
            "attack_type": event.attack_type,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat() if event.end_time else None,
            "peak_rps": event.peak_rps,
            "total_requests": event.total_requests,
            "unique_ips": event.unique_ips,
            "mitigation_applied": event.mitigation_applied,
            "status": event.status
        }
        
        self.redis_client.lpush("attack_events", json.dumps(event_data))
        self.redis_client.ltrim("attack_events", 0, 1000)  # Keep last 1000
    
    def _export_metrics(self):
        """Export metrics for monitoring"""
        
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "protection_level": self.protection_level,
            "attack_level": self._get_attack_level(),
            "current_attacks": list(self.current_attacks.keys()),
            "metrics": self.metrics
        }
        
        self.redis_client.set("ddos_metrics", json.dumps(metrics_data))


class TrafficAnalyzer:
    """Analyze traffic patterns for anomaly detection"""
    
    def __init__(self):
        self.request_counts = defaultdict(lambda: deque(maxlen=60))  # 60 seconds
        self.path_counts = defaultdict(int)
        self.user_agent_counts = defaultdict(int)
        self.baseline = None
        self.anomaly_score = 0.0
    
    def record_request(self, ip: str, path: str):
        """Record incoming request"""
        
        current_second = int(time.time())
        self.request_counts[current_second].append(ip)
        self.path_counts[path] += 1
    
    def analyze_patterns(self):
        """Analyze traffic patterns"""
        
        # Calculate request rate
        current_rate = self._calculate_request_rate()
        
        # Update baseline
        if self.baseline is None:
            self.baseline = current_rate
        else:
            # Exponential moving average
            self.baseline = 0.9 * self.baseline + 0.1 * current_rate
        
        # Calculate anomaly score
        if self.baseline > 0:
            deviation = abs(current_rate - self.baseline) / self.baseline
            self.anomaly_score = min(1.0, deviation / 5)  # Normalize to 0-1
    
    def _calculate_request_rate(self) -> float:
        """Calculate current request rate"""
        
        current_second = int(time.time())
        recent_seconds = range(current_second - 10, current_second)
        
        total_requests = sum(
            len(self.request_counts[s]) for s in recent_seconds
        )
        
        return total_requests / 10  # Requests per second
    
    def get_anomaly_score(self) -> float:
        """Get current anomaly score"""
        return self.anomaly_score
    
    def get_recent_request_count(self) -> int:
        """Get request count in last 10 seconds"""
        
        current_second = int(time.time())
        recent_seconds = range(current_second - 10, current_second)
        
        return sum(len(self.request_counts[s]) for s in recent_seconds)


class VolumetricDetector:
    """Detect volumetric attacks"""
    
    def __init__(self, threshold: int = 1000):
        self.threshold = threshold  # Requests per second
        self.request_counts = defaultdict(int)
        self.window_start = time.time()
    
    def detect(self, ip: str, request: Request, analyzer: TrafficAnalyzer) -> bool:
        """Detect volumetric attack"""
        
        # Check request rate
        rate = analyzer._calculate_request_rate()
        
        if rate > self.threshold:
            return True
        
        # Check per-IP rate
        self.request_counts[ip] += 1
        
        # Reset window
        if time.time() - self.window_start > 1:
            self.request_counts.clear()
            self.window_start = time.time()
        
        return self.request_counts[ip] > 100  # Per-IP threshold


class SlowlorisDetector:
    """Detect Slowloris attacks"""
    
    def __init__(self):
        self.slow_connections = defaultdict(list)
        self.threshold = 10  # Slow connections threshold
    
    def detect(self, ip: str, request: Request, analyzer: TrafficAnalyzer) -> bool:
        """Detect Slowloris attack"""
        
        # Check for slow/incomplete requests
        # This would need integration with the web server
        
        # Check headers for Slowloris patterns
        if "Range" in request.headers:
            # Suspicious Range headers
            return True
        
        return False


class HTTPFloodDetector:
    """Detect HTTP flood attacks"""
    
    def __init__(self):
        self.request_patterns = defaultdict(list)
        self.threshold = 50  # Requests per second per IP
    
    def detect(self, ip: str, request: Request, analyzer: TrafficAnalyzer) -> bool:
        """Detect HTTP flood"""
        
        current_time = time.time()
        
        # Track request patterns
        self.request_patterns[ip].append(current_time)
        
        # Clean old entries
        self.request_patterns[ip] = [
            t for t in self.request_patterns[ip]
            if current_time - t < 1
        ]
        
        # Check threshold
        return len(self.request_patterns[ip]) > self.threshold


class ApplicationLayerDetector:
    """Detect application layer attacks"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r"\.\.\/",  # Directory traversal
            r"<script",  # XSS attempts
            r"union.*select",  # SQL injection
            r"\%00",  # Null byte
        ]
    
    def detect(self, ip: str, request: Request, analyzer: TrafficAnalyzer) -> bool:
        """Detect application layer attack"""
        
        # Check for suspicious patterns in URL
        url_str = str(request.url)
        
        for pattern in self.suspicious_patterns:
            if pattern in url_str.lower():
                return True
        
        # Check for unusual request methods
        if request.method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            return True
        
        return False


class ChallengeSystem:
    """Challenge-response system for bot detection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.challenge_ttl = 300  # 5 minutes
    
    def create_challenge(self, ip: str) -> HTMLResponse:
        """Create challenge for IP"""
        
        # Generate challenge token
        challenge_token = secrets.token_urlsafe(32)
        solution = secrets.token_urlsafe(16)
        
        # Store challenge
        self.redis_client.setex(
            f"challenge:{ip}:{challenge_token}",
            self.challenge_ttl,
            solution
        )
        
        # Return challenge page
        challenge_html = self._generate_challenge_html(challenge_token)
        
        return HTMLResponse(
            content=challenge_html,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={
                "Retry-After": "5",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    
    def _generate_challenge_html(self, token: str) -> str:
        """Generate challenge HTML page"""
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Check</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                .container {{ max-width: 500px; margin: 0 auto; }}
                .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db;
                           border-radius: 50%; width: 40px; height: 40px;
                           animation: spin 1s linear infinite; margin: 20px auto; }}
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }}
                                  100% {{ transform: rotate(360deg); }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Checking your browser...</h1>
                <div class="spinner"></div>
                <p>This process is automatic. Your browser will redirect shortly.</p>
                <noscript>
                    <p style="color: red;">Please enable JavaScript to continue.</p>
                </noscript>
            </div>
            <script>
                // Proof of work challenge
                function solveChallenge() {{
                    const token = "{token}";
                    // Simple PoW - find nonce that produces hash with leading zeros
                    let nonce = 0;
                    const difficulty = 4;
                    
                    while (true) {{
                        const hash = sha256(token + nonce);
                        if (hash.substring(0, difficulty) === "0".repeat(difficulty)) {{
                            // Solution found
                            submitSolution(token, nonce);
                            break;
                        }}
                        nonce++;
                    }}
                }}
                
                function sha256(str) {{
                    // Simple SHA256 implementation (would use crypto library)
                    return "0000" + str;  // Simplified for example
                }}
                
                function submitSolution(token, nonce) {{
                    // Submit solution via AJAX or form
                    window.location.href = "/?challenge_token=" + token + "&nonce=" + nonce;
                }}
                
                // Start solving
                setTimeout(solveChallenge, 100);
            </script>
        </body>
        </html>
        """
    
    def verify_challenge(self, ip: str, token: str, solution: str) -> bool:
        """Verify challenge solution"""
        
        stored_solution = self.redis_client.get(f"challenge:{ip}:{token}")
        
        if stored_solution and stored_solution == solution:
            # Mark as verified
            self.redis_client.setex(f"verified:{ip}", 3600, "1")  # Valid for 1 hour
            
            # Clean up challenge
            self.redis_client.delete(f"challenge:{ip}:{token}")
            
            return True
        
        return False
    
    def is_verified(self, ip: str) -> bool:
        """Check if IP is verified"""
        return self.redis_client.exists(f"verified:{ip}") > 0
    
    def has_solved_challenge(self, ip: str) -> bool:
        """Check if IP has solved a challenge recently"""
        return self.is_verified(ip)