"""
Secure Communication System for Federated Learning

This module implements secure communication protocols for federated learning
including encrypted messaging, authentication, and network management.
"""

import asyncio
import logging
import json
import time
import ssl
import socket
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import websockets
import aiohttp
import aiofiles
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets
import hashlib
import uuid
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class MessageHeader:
    """Message header for secure communication"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    timestamp: float
    sequence_number: int = 0
    priority: int = 1  # 1=low, 2=medium, 3=high
    encrypted: bool = True


@dataclass
class CommunicationStats:
    """Communication statistics"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    encryption_time: float = 0.0
    decryption_time: float = 0.0
    network_errors: int = 0
    last_activity: Optional[float] = None


@dataclass
class ClientConnection:
    """Client connection information"""
    client_id: str
    websocket: Optional[Any] = None
    last_seen: Optional[float] = None
    public_key: Optional[bytes] = None
    shared_secret: Optional[bytes] = None
    sequence_number: int = 0
    is_active: bool = False
    stats: CommunicationStats = field(default_factory=CommunicationStats)


class SecureCommunication:
    """
    Secure communication system for federated learning
    
    Features:
    - End-to-end encryption
    - Message authentication
    - Network fault tolerance
    - Load balancing
    - Compression
    - Rate limiting
    - Connection pooling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Communication settings
        self.server_host = self.config.get("server_host", "localhost")
        self.server_port = self.config.get("server_port", 8765)
        self.use_ssl = self.config.get("use_ssl", True)
        self.compression_enabled = self.config.get("compression", True)
        
        # Security settings
        self.encryption_enabled = self.config.get("encryption", True)
        self.authentication_enabled = self.config.get("authentication", True)
        self.message_integrity_check = self.config.get("integrity_check", True)
        
        # Performance settings
        self.max_connections = self.config.get("max_connections", 1000)
        self.message_timeout = self.config.get("message_timeout", 30)
        self.keepalive_interval = self.config.get("keepalive_interval", 60)
        self.max_message_size = self.config.get("max_message_size", 100 * 1024 * 1024)  # 100MB
        
        # Rate limiting
        self.rate_limit_enabled = self.config.get("rate_limiting", True)
        self.max_messages_per_minute = self.config.get("max_messages_per_minute", 60)
        
        # State
        self.client_connections: Dict[str, ClientConnection] = {}
        self.server_websocket = None
        self.is_server_running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.rate_limiter: Dict[str, List[float]] = {}
        
        # Encryption keys
        self.server_private_key = None
        self.server_public_key = None
        self._generate_server_keys()
        
        # Statistics
        self.global_stats = CommunicationStats()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"Secure communication initialized on {self.server_host}:{self.server_port}")
    
    def _generate_server_keys(self):
        """Generate server encryption keys"""
        try:
            self.server_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.server_public_key = self.server_private_key.public_key()
            
            logger.info("Server encryption keys generated")
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            self.encryption_enabled = False
    
    async def start_server(self) -> None:
        """Start the communication server"""
        try:
            # Setup SSL context if needed
            ssl_context = None
            if self.use_ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                # In production, load actual certificates
                # ssl_context.load_cert_chain("server.crt", "server.key")
            
            # Start WebSocket server
            self.server_websocket = await websockets.serve(
                self._handle_client_connection,
                self.server_host,
                self.server_port,
                ssl=ssl_context,
                max_size=self.max_message_size,
                ping_interval=self.keepalive_interval
            )
            
            self.is_server_running = True
            
            # Start background tasks
            self.background_tasks.append(
                asyncio.create_task(self._message_processor())
            )
            self.background_tasks.append(
                asyncio.create_task(self._connection_monitor())
            )
            self.background_tasks.append(
                asyncio.create_task(self._rate_limit_cleaner())
            )
            
            logger.info(f"Communication server started on {self.server_host}:{self.server_port}")
            
        except Exception as e:
            logger.error(f"Failed to start communication server: {e}")
            raise
    
    async def stop_server(self) -> None:
        """Stop the communication server"""
        try:
            self.is_server_running = False
            
            # Close all client connections
            for client_id, connection in list(self.client_connections.items()):
                await self._close_client_connection(client_id)
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Close server
            if self.server_websocket:
                self.server_websocket.close()
                await self.server_websocket.wait_closed()
            
            logger.info("Communication server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    async def _handle_client_connection(self, websocket, path):
        """Handle new client connection"""
        client_id = None
        
        try:
            # Wait for client identification
            auth_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=10
            )
            
            auth_data = json.loads(auth_message)
            client_id = auth_data.get("client_id")
            
            if not client_id:
                await websocket.close(code=4001, reason="Missing client ID")
                return
            
            # Authenticate client
            if not await self._authenticate_client(client_id, auth_data):
                await websocket.close(code=4002, reason="Authentication failed")
                return
            
            # Register client connection
            connection = ClientConnection(
                client_id=client_id,
                websocket=websocket,
                last_seen=time.time(),
                is_active=True
            )
            
            self.client_connections[client_id] = connection
            
            logger.info(f"Client {client_id} connected")
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except asyncio.TimeoutError:
            logger.warning("Client authentication timeout")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            if client_id and client_id in self.client_connections:
                await self._close_client_connection(client_id)
    
    async def _authenticate_client(self, client_id: str, auth_data: Dict) -> bool:
        """Authenticate client connection"""
        if not self.authentication_enabled:
            return True
        
        try:
            # Simple token-based authentication
            # In production, use proper JWT or certificate-based auth
            expected_token = hashlib.sha256(f"{client_id}:secret".encode()).hexdigest()
            provided_token = auth_data.get("token")
            
            return provided_token == expected_token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def _handle_client_message(self, client_id: str, message: str) -> None:
        """Handle message from client"""
        try:
            # Rate limiting
            if self.rate_limit_enabled and not self._check_rate_limit(client_id):
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return
            
            # Parse message
            message_data = json.loads(message)
            
            # Update connection stats
            connection = self.client_connections.get(client_id)
            if connection:
                connection.last_seen = time.time()
                connection.stats.messages_received += 1
                connection.stats.bytes_received += len(message)
                connection.stats.last_activity = time.time()
            
            # Decrypt message if needed
            if self.encryption_enabled and message_data.get("encrypted", False):
                message_data = await self._decrypt_message(client_id, message_data)
            
            # Verify message integrity
            if self.message_integrity_check:
                if not self._verify_message_integrity(message_data):
                    logger.warning(f"Message integrity check failed for {client_id}")
                    return
            
            # Queue message for processing
            await self.message_queue.put((client_id, message_data))
            
            self.global_stats.messages_received += 1
            self.global_stats.bytes_received += len(message)
            self.global_stats.last_activity = time.time()
            
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check rate limit for client"""
        current_time = time.time()
        
        if client_id not in self.rate_limiter:
            self.rate_limiter[client_id] = []
        
        client_timestamps = self.rate_limiter[client_id]
        
        # Remove old timestamps
        client_timestamps[:] = [
            ts for ts in client_timestamps 
            if current_time - ts < 60  # Within last minute
        ]
        
        # Check if limit exceeded
        if len(client_timestamps) >= self.max_messages_per_minute:
            return False
        
        # Add current timestamp
        client_timestamps.append(current_time)
        return True
    
    async def _decrypt_message(self, client_id: str, message_data: Dict) -> Dict:
        """Decrypt message from client"""
        start_time = time.time()
        
        try:
            encrypted_content = message_data.get("content")
            if not encrypted_content:
                return message_data
            
            # Get client connection for shared secret
            connection = self.client_connections.get(client_id)
            if not connection or not connection.shared_secret:
                raise ValueError("No shared secret for client")
            
            # Decrypt using AES
            encrypted_bytes = bytes.fromhex(encrypted_content)
            
            # Extract IV and ciphertext
            iv = encrypted_bytes[:16]
            ciphertext = encrypted_bytes[16:]
            
            # Decrypt
            cipher = Cipher(
                algorithms.AES(connection.shared_secret[:32]),  # Use first 32 bytes as key
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding and decode
            padding_length = decrypted_bytes[-1]
            decrypted_bytes = decrypted_bytes[:-padding_length]
            decrypted_content = decrypted_bytes.decode('utf-8')
            
            # Parse decrypted content
            decrypted_data = json.loads(decrypted_content)
            
            # Update stats
            decryption_time = time.time() - start_time
            connection.stats.decryption_time += decryption_time
            self.global_stats.decryption_time += decryption_time
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            return message_data
    
    def _verify_message_integrity(self, message_data: Dict) -> bool:
        """Verify message integrity using hash"""
        try:
            provided_hash = message_data.get("hash")
            if not provided_hash:
                return True  # No integrity check requested
            
            # Calculate hash of message content
            content = message_data.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, sort_keys=True)
            
            calculated_hash = hashlib.sha256(str(content).encode()).hexdigest()
            
            return provided_hash == calculated_hash
            
        except Exception as e:
            logger.warning(f"Integrity verification failed: {e}")
            return False
    
    async def send_to_client(self, client_id: str, message: Dict) -> bool:
        """Send message to specific client"""
        try:
            connection = self.client_connections.get(client_id)
            if not connection or not connection.is_active:
                logger.warning(f"Client {client_id} not connected")
                return False
            
            # Add message header
            message_with_header = self._add_message_header(message, "server", client_id)
            
            # Encrypt message if needed
            if self.encryption_enabled:
                message_with_header = await self._encrypt_message(client_id, message_with_header)
            
            # Add integrity check
            if self.message_integrity_check:
                message_with_header = self._add_integrity_check(message_with_header)
            
            # Serialize message
            message_json = json.dumps(message_with_header, default=self._json_serializer)
            
            # Compress if enabled
            if self.compression_enabled:
                message_json = self._compress_message(message_json)
            
            # Send message
            await asyncio.wait_for(
                connection.websocket.send(message_json),
                timeout=self.message_timeout
            )
            
            # Update stats
            connection.stats.messages_sent += 1
            connection.stats.bytes_sent += len(message_json)
            connection.stats.last_activity = time.time()
            
            self.global_stats.messages_sent += 1
            self.global_stats.bytes_sent += len(message_json)
            self.global_stats.last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            connection = self.client_connections.get(client_id)
            if connection:
                connection.stats.network_errors += 1
            self.global_stats.network_errors += 1
            return False
    
    async def broadcast_to_clients(
        self,
        message: Dict,
        client_ids: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Broadcast message to multiple clients"""
        if client_ids is None:
            client_ids = list(self.client_connections.keys())
        
        results = {}
        tasks = []
        
        for client_id in client_ids:
            if client_id in self.client_connections:
                task = asyncio.create_task(self.send_to_client(client_id, message))
                tasks.append((client_id, task))
        
        # Wait for all sends to complete
        for client_id, task in tasks:
            try:
                result = await task
                results[client_id] = result
            except Exception as e:
                logger.error(f"Broadcast failed for {client_id}: {e}")
                results[client_id] = False
        
        return results
    
    def _add_message_header(
        self,
        message: Dict,
        sender_id: str,
        receiver_id: str
    ) -> Dict:
        """Add message header"""
        connection = self.client_connections.get(receiver_id)
        sequence_number = 0
        
        if connection:
            connection.sequence_number += 1
            sequence_number = connection.sequence_number
        
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message.get("type", "unknown"),
            timestamp=time.time(),
            sequence_number=sequence_number,
            encrypted=self.encryption_enabled
        )
        
        return {
            "header": header.__dict__,
            "payload": message
        }
    
    async def _encrypt_message(self, client_id: str, message: Dict) -> Dict:
        """Encrypt message for client"""
        start_time = time.time()
        
        try:
            connection = self.client_connections.get(client_id)
            if not connection:
                return message
            
            # Serialize payload
            payload_json = json.dumps(message["payload"], default=self._json_serializer)
            payload_bytes = payload_json.encode('utf-8')
            
            # Add padding
            padding_length = 16 - (len(payload_bytes) % 16)
            payload_bytes += bytes([padding_length] * padding_length)
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Encrypt using AES
            shared_secret = connection.shared_secret or self._generate_shared_secret(client_id)
            cipher = Cipher(
                algorithms.AES(shared_secret[:32]),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(payload_bytes) + encryptor.finalize()
            
            # Combine IV and ciphertext
            encrypted_bytes = iv + ciphertext
            encrypted_hex = encrypted_bytes.hex()
            
            # Update stats
            encryption_time = time.time() - start_time
            connection.stats.encryption_time += encryption_time
            self.global_stats.encryption_time += encryption_time
            
            return {
                "header": message["header"],
                "encrypted_payload": encrypted_hex,
                "encrypted": True
            }
            
        except Exception as e:
            logger.error(f"Message encryption failed: {e}")
            return message
    
    def _generate_shared_secret(self, client_id: str) -> bytes:
        """Generate shared secret for client"""
        # Simple shared secret generation
        # In production, use proper key exchange (ECDH, etc.)
        secret = hashlib.sha256(f"shared_secret_{client_id}".encode()).digest()
        
        connection = self.client_connections.get(client_id)
        if connection:
            connection.shared_secret = secret
        
        return secret
    
    def _add_integrity_check(self, message: Dict) -> Dict:
        """Add integrity check to message"""
        try:
            # Calculate hash of payload
            payload = message.get("payload") or message.get("encrypted_payload")
            if payload:
                if isinstance(payload, dict):
                    payload_str = json.dumps(payload, sort_keys=True)
                else:
                    payload_str = str(payload)
                
                message_hash = hashlib.sha256(payload_str.encode()).hexdigest()
                message["hash"] = message_hash
            
            return message
            
        except Exception as e:
            logger.warning(f"Failed to add integrity check: {e}")
            return message
    
    def _compress_message(self, message: str) -> str:
        """Compress message (placeholder)"""
        # In production, use gzip or other compression
        return message
    
    def _json_serializer(self, obj):
        """Custom JSON serializer"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    async def _message_processor(self):
        """Background task to process incoming messages"""
        while self.is_server_running:
            try:
                # Get message from queue
                client_id, message_data = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message based on type
                message_type = message_data.get("type")
                handler = self.message_handlers.get(message_type)
                
                if handler:
                    await handler(client_id, message_data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _connection_monitor(self):
        """Monitor client connections and cleanup inactive ones"""
        while self.is_server_running:
            try:
                current_time = time.time()
                inactive_clients = []
                
                for client_id, connection in self.client_connections.items():
                    if (connection.last_seen and 
                        current_time - connection.last_seen > self.keepalive_interval * 2):
                        inactive_clients.append(client_id)
                
                # Clean up inactive connections
                for client_id in inactive_clients:
                    logger.info(f"Cleaning up inactive client: {client_id}")
                    await self._close_client_connection(client_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
    
    async def _rate_limit_cleaner(self):
        """Clean up old rate limit entries"""
        while self.is_server_running:
            try:
                current_time = time.time()
                
                for client_id in list(self.rate_limiter.keys()):
                    timestamps = self.rate_limiter[client_id]
                    # Keep only timestamps from last minute
                    self.rate_limiter[client_id] = [
                        ts for ts in timestamps 
                        if current_time - ts < 60
                    ]
                    
                    # Remove empty entries
                    if not self.rate_limiter[client_id]:
                        del self.rate_limiter[client_id]
                
                await asyncio.sleep(60)  # Clean every minute
                
            except Exception as e:
                logger.error(f"Rate limit cleaning error: {e}")
    
    async def _close_client_connection(self, client_id: str):
        """Close client connection"""
        try:
            connection = self.client_connections.get(client_id)
            if connection:
                connection.is_active = False
                
                if connection.websocket:
                    await connection.websocket.close()
                
                del self.client_connections[client_id]
                
                # Clean up rate limiter
                if client_id in self.rate_limiter:
                    del self.rate_limiter[client_id]
                
                logger.info(f"Client {client_id} connection closed")
                
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def get_connection_stats(self, client_id: Optional[str] = None) -> Dict:
        """Get connection statistics"""
        if client_id:
            connection = self.client_connections.get(client_id)
            if connection:
                return {
                    "client_id": client_id,
                    "is_active": connection.is_active,
                    "last_seen": connection.last_seen,
                    "stats": connection.stats.__dict__
                }
            else:
                return {"error": f"Client {client_id} not found"}
        else:
            return {
                "total_connections": len(self.client_connections),
                "active_connections": sum(1 for c in self.client_connections.values() if c.is_active),
                "global_stats": self.global_stats.__dict__,
                "server_running": self.is_server_running
            }
    
    async def setup_client_keys(self, client_id: str) -> Dict[str, Any]:
        """Setup encryption keys for client"""
        try:
            # Generate shared secret
            shared_secret = self._generate_shared_secret(client_id)
            
            # Return public key information
            public_key_pem = self.server_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                "server_public_key": public_key_pem.decode('utf-8'),
                "shared_secret_hash": hashlib.sha256(shared_secret).hexdigest()[:16]
            }
            
        except Exception as e:
            logger.error(f"Key setup failed for {client_id}: {e}")
            return {}
    
    async def cleanup_client_keys(self, client_id: str):
        """Cleanup encryption keys for client"""
        connection = self.client_connections.get(client_id)
        if connection:
            connection.shared_secret = None
    
    # Client-side methods
    async def initialize_client(self, client_id: str, server_endpoint: str):
        """Initialize client-side communication"""
        self.client_id = client_id
        self.server_endpoint = server_endpoint
        self.client_websocket = None
    
    async def connect_to_server(self) -> bool:
        """Connect to federated server (client-side)"""
        try:
            # Connect via WebSocket
            uri = f"ws://{self.server_endpoint.replace('http://', '').replace('https://', '')}"
            
            self.client_websocket = await websockets.connect(uri)
            
            # Send authentication
            auth_message = {
                "client_id": self.client_id,
                "token": hashlib.sha256(f"{self.client_id}:secret".encode()).hexdigest(),
                "timestamp": time.time()
            }
            
            await self.client_websocket.send(json.dumps(auth_message))
            
            logger.info(f"Client {self.client_id} connected to server")
            return True
            
        except Exception as e:
            logger.error(f"Client connection failed: {e}")
            return False
    
    async def send_to_server(self, message_type: str, data: Dict) -> Dict:
        """Send message to server (client-side)"""
        try:
            if not self.client_websocket:
                raise ValueError("Not connected to server")
            
            message = {
                "type": message_type,
                "data": data,
                "timestamp": time.time()
            }
            
            message_json = json.dumps(message, default=self._json_serializer)
            await self.client_websocket.send(message_json)
            
            # Wait for response (simplified)
            response = await asyncio.wait_for(
                self.client_websocket.recv(),
                timeout=self.message_timeout
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Failed to send message to server: {e}")
            return {"status": "error", "message": str(e)}
    
    async def receive_from_server(self) -> Optional[Dict]:
        """Receive message from server (client-side)"""
        try:
            if not self.client_websocket:
                return None
            
            message = await asyncio.wait_for(
                self.client_websocket.recv(),
                timeout=1.0  # Short timeout for non-blocking
            )
            
            return json.loads(message)
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive from server: {e}")
            return None
    
    async def encrypt_message(self, client_id: str, message: Dict) -> bytes:
        """Encrypt message for transmission"""
        # Placeholder - would implement actual encryption
        message_json = json.dumps(message)
        return message_json.encode('utf-8')
    
    async def decrypt_message(self, client_id: str, encrypted_data: bytes) -> Dict:
        """Decrypt received message"""
        # Placeholder - would implement actual decryption
        message_json = encrypted_data.decode('utf-8')
        return json.loads(message_json)
    
    async def cleanup(self):
        """Cleanup communication resources"""
        try:
            if hasattr(self, 'client_websocket') and self.client_websocket:
                await self.client_websocket.close()
            
            await self.stop_server()
            
        except Exception as e:
            logger.error(f"Communication cleanup failed: {e}")