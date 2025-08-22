"""
IPFS (InterPlanetary File System) integration for distributed document storage.

Provides decentralized document storage, content addressing, and retrieval
for the blockchain audit trail system.
"""

import json
import logging
import asyncio
import hashlib
import mimetypes
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import base64

# IPFS client imports
try:
    import ipfshttpclient
    from ipfshttpclient.client.base import ResponseBase
except ImportError:
    # Mock for development
    class ipfshttpclient:
        @staticmethod
        def connect(*args, **kwargs):
            return MockIPFSClient()
    
    class ResponseBase:
        pass
    
    class MockIPFSClient:
        def add(self, *args, **kwargs):
            return {'Hash': 'QmMockHash123'}
        
        def cat(self, *args, **kwargs):
            return b'mock content'
        
        def pin(self, *args, **kwargs):
            return {'Pins': ['QmMockHash123']}

logger = logging.getLogger(__name__)

@dataclass
class IPFSDocument:
    """IPFS document metadata."""
    ipfs_hash: str
    filename: str
    size: int
    content_type: str
    upload_timestamp: int
    uploader_id: str
    encryption_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    pin_status: bool = False

@dataclass
class IPFSCluster:
    """IPFS cluster node configuration."""
    node_id: str
    endpoint: str
    api_port: int
    gateway_port: int
    is_bootstrap: bool
    region: str
    status: str

class IPFSStorage:
    """
    IPFS distributed storage system for blockchain audit trail.
    
    Handles document upload, retrieval, pinning, and cluster management
    with encryption and redundancy features.
    """
    
    def __init__(self,
                 ipfs_api_endpoint: str = '/ip4/127.0.0.1/tcp/5001',
                 ipfs_gateway: str = 'http://127.0.0.1:8080',
                 cluster_endpoints: Optional[List[str]] = None,
                 encryption_enabled: bool = True,
                 auto_pin: bool = True,
                 replication_factor: int = 3):
        """
        Initialize IPFS storage system.
        
        Args:
            ipfs_api_endpoint: IPFS API endpoint
            ipfs_gateway: IPFS gateway URL
            cluster_endpoints: List of IPFS cluster endpoints
            encryption_enabled: Enable client-side encryption
            auto_pin: Automatically pin uploaded content
            replication_factor: Number of replicas across cluster
        """
        self.ipfs_api_endpoint = ipfs_api_endpoint
        self.ipfs_gateway = ipfs_gateway
        self.cluster_endpoints = cluster_endpoints or []
        self.encryption_enabled = encryption_enabled
        self.auto_pin = auto_pin
        self.replication_factor = replication_factor
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize IPFS client
        self.client = None
        self.cluster_nodes: List[IPFSCluster] = []
        self.session = None
        
    async def initialize(self) -> bool:
        """
        Initialize IPFS client and cluster connections.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize IPFS client
            self.client = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: ipfshttpclient.connect(self.ipfs_api_endpoint)
            )
            
            # Initialize HTTP session for gateway operations
            self.session = aiohttp.ClientSession()
            
            # Test connection
            node_info = await self._get_node_info()
            logger.info(f"Connected to IPFS node: {node_info.get('ID', 'Unknown')}")
            
            # Initialize cluster connections
            await self._initialize_cluster_nodes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IPFS storage: {e}")
            return False
    
    async def _get_node_info(self) -> Dict[str, Any]:
        """Get IPFS node information."""
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.client.id
            )
            return info
        except Exception as e:
            logger.error(f"Failed to get node info: {e}")
            return {}
    
    async def _initialize_cluster_nodes(self):
        """Initialize connections to IPFS cluster nodes."""
        for endpoint in self.cluster_endpoints:
            try:
                # Parse endpoint
                parts = endpoint.split(':')
                host = parts[0]
                api_port = int(parts[1]) if len(parts) > 1 else 9094
                gateway_port = api_port + 1
                
                # Get cluster node info
                async with self.session.get(f'http://{host}:{api_port}/id') as response:
                    if response.status == 200:
                        node_data = await response.json()
                        
                        cluster_node = IPFSCluster(
                            node_id=node_data.get('id', 'unknown'),
                            endpoint=endpoint,
                            api_port=api_port,
                            gateway_port=gateway_port,
                            is_bootstrap=node_data.get('addresses', []) != [],
                            region=node_data.get('region', 'unknown'),
                            status='online'
                        )
                        
                        self.cluster_nodes.append(cluster_node)
                        logger.info(f"Connected to cluster node: {cluster_node.node_id}")
                        
            except Exception as e:
                logger.warning(f"Failed to connect to cluster node {endpoint}: {e}")
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key for document."""
        return base64.b64encode(hashlib.sha256(
            str(datetime.now().timestamp()).encode()
        ).digest()).decode()
    
    def _encrypt_content(self, content: bytes, key: str) -> bytes:
        """
        Encrypt content using AES encryption.
        
        Args:
            content: Content to encrypt
            key: Encryption key
            
        Returns:
            bytes: Encrypted content
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import os
            
            # Generate salt and derive key
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            
            # Encrypt content
            fernet = Fernet(derived_key)
            encrypted_content = fernet.encrypt(content)
            
            # Prepend salt to encrypted content
            return salt + encrypted_content
            
        except ImportError:
            logger.warning("Cryptography library not available, using base64 encoding")
            return base64.b64encode(content)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return content
    
    def _decrypt_content(self, encrypted_content: bytes, key: str) -> bytes:
        """
        Decrypt content using AES decryption.
        
        Args:
            encrypted_content: Encrypted content
            key: Encryption key
            
        Returns:
            bytes: Decrypted content
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            # Extract salt and encrypted data
            salt = encrypted_content[:16]
            encrypted_data = encrypted_content[16:]
            
            # Derive key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            
            # Decrypt content
            fernet = Fernet(derived_key)
            return fernet.decrypt(encrypted_data)
            
        except ImportError:
            logger.warning("Cryptography library not available, using base64 decoding")
            return base64.b64decode(encrypted_content)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_content
    
    async def upload_document(self,
                            content: Union[bytes, str, BinaryIO],
                            filename: str,
                            uploader_id: str,
                            metadata: Optional[Dict[str, Any]] = None) -> IPFSDocument:
        """
        Upload document to IPFS with optional encryption.
        
        Args:
            content: Document content (bytes, string, or file-like object)
            filename: Original filename
            uploader_id: User ID of uploader
            metadata: Additional metadata
            
        Returns:
            IPFSDocument: Document information with IPFS hash
        """
        try:
            # Prepare content
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif hasattr(content, 'read'):
                content_bytes = content.read()
                if isinstance(content_bytes, str):
                    content_bytes = content_bytes.encode('utf-8')
            else:
                content_bytes = content
            
            # Generate encryption key if enabled
            encryption_key = None
            if self.encryption_enabled:
                encryption_key = self._generate_encryption_key()
                content_bytes = self._encrypt_content(content_bytes, encryption_key)
            
            # Upload to IPFS
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.add(content_bytes, pin=self.auto_pin)
            )
            
            ipfs_hash = result['Hash']
            
            # Determine content type
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            
            # Create document metadata
            ipfs_document = IPFSDocument(
                ipfs_hash=ipfs_hash,
                filename=filename,
                size=len(content_bytes),
                content_type=content_type,
                upload_timestamp=int(datetime.now().timestamp()),
                uploader_id=uploader_id,
                encryption_key=encryption_key,
                metadata=metadata or {},
                pin_status=self.auto_pin
            )
            
            # Replicate to cluster if available
            if self.cluster_nodes:
                await self._replicate_to_cluster(ipfs_hash)
            
            logger.info(f"Document uploaded to IPFS: {ipfs_hash}")
            return ipfs_document
            
        except Exception as e:
            logger.error(f"Failed to upload document to IPFS: {e}")
            raise
    
    async def retrieve_document(self,
                              ipfs_hash: str,
                              encryption_key: Optional[str] = None) -> bytes:
        """
        Retrieve document from IPFS.
        
        Args:
            ipfs_hash: IPFS hash of the document
            encryption_key: Decryption key if document is encrypted
            
        Returns:
            bytes: Document content
        """
        try:
            # Retrieve from IPFS
            content = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.cat(ipfs_hash)
            )
            
            # Decrypt if key provided
            if encryption_key:
                content = self._decrypt_content(content, encryption_key)
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve document from IPFS: {e}")
            
            # Try cluster nodes as fallback
            if self.cluster_nodes:
                return await self._retrieve_from_cluster(ipfs_hash, encryption_key)
            
            raise
    
    async def _retrieve_from_cluster(self,
                                   ipfs_hash: str,
                                   encryption_key: Optional[str] = None) -> bytes:
        """Retrieve document from cluster nodes as fallback."""
        for node in self.cluster_nodes:
            try:
                gateway_url = f'http://{node.endpoint.split(":")[0]}:{node.gateway_port}'
                
                async with self.session.get(f'{gateway_url}/ipfs/{ipfs_hash}') as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        if encryption_key:
                            content = self._decrypt_content(content, encryption_key)
                        
                        return content
                        
            except Exception as e:
                logger.warning(f"Failed to retrieve from cluster node {node.node_id}: {e}")
                continue
        
        raise Exception("Failed to retrieve document from all cluster nodes")
    
    async def pin_document(self, ipfs_hash: str) -> bool:
        """
        Pin document to prevent garbage collection.
        
        Args:
            ipfs_hash: IPFS hash to pin
            
        Returns:
            bool: True if pinned successfully
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.pin.add(ipfs_hash)
            )
            
            logger.info(f"Document pinned: {ipfs_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pin document: {e}")
            return False
    
    async def unpin_document(self, ipfs_hash: str) -> bool:
        """
        Unpin document to allow garbage collection.
        
        Args:
            ipfs_hash: IPFS hash to unpin
            
        Returns:
            bool: True if unpinned successfully
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.pin.rm(ipfs_hash)
            )
            
            logger.info(f"Document unpinned: {ipfs_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unpin document: {e}")
            return False
    
    async def _replicate_to_cluster(self, ipfs_hash: str):
        """Replicate content to cluster nodes for redundancy."""
        replication_count = 0
        target_replications = min(self.replication_factor, len(self.cluster_nodes))
        
        for node in self.cluster_nodes:
            if replication_count >= target_replications:
                break
            
            try:
                # Pin content on cluster node
                api_url = f'http://{node.endpoint.split(":")[0]}:{node.api_port}'
                
                async with self.session.post(
                    f'{api_url}/pins/{ipfs_hash}',
                    json={'replication_factor_min': 1, 'replication_factor_max': 3}
                ) as response:
                    if response.status in [200, 201]:
                        replication_count += 1
                        logger.info(f"Content replicated to cluster node: {node.node_id}")
                        
            except Exception as e:
                logger.warning(f"Failed to replicate to cluster node {node.node_id}: {e}")
    
    async def get_document_stats(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get statistics for a document.
        
        Args:
            ipfs_hash: IPFS hash of the document
            
        Returns:
            Dict: Document statistics
        """
        try:
            # Get object stats
            stats = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.object.stat(ipfs_hash)
            )
            
            # Check pin status
            pins = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.pin.ls()
            )
            
            is_pinned = ipfs_hash in [pin['Hash'] for pin in pins.get('Keys', {})]
            
            return {
                'hash': ipfs_hash,
                'size': stats.get('CumulativeSize', 0),
                'num_links': stats.get('NumLinks', 0),
                'block_size': stats.get('BlockSize', 0),
                'data_size': stats.get('DataSize', 0),
                'is_pinned': is_pinned,
                'cluster_replications': await self._count_cluster_replications(ipfs_hash)
            }
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {}
    
    async def _count_cluster_replications(self, ipfs_hash: str) -> int:
        """Count number of cluster replications for a hash."""
        replication_count = 0
        
        for node in self.cluster_nodes:
            try:
                api_url = f'http://{node.endpoint.split(":")[0]}:{node.api_port}'
                
                async with self.session.get(f'{api_url}/pins/{ipfs_hash}') as response:
                    if response.status == 200:
                        replication_count += 1
                        
            except Exception:
                continue
        
        return replication_count
    
    async def list_pinned_documents(self) -> List[Dict[str, Any]]:
        """
        List all pinned documents.
        
        Returns:
            List[Dict]: Pinned documents information
        """
        try:
            pins = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.pin.ls()
            )
            
            pinned_docs = []
            for pin_hash, pin_info in pins.get('Keys', {}).items():
                stats = await self.get_document_stats(pin_hash)
                pinned_docs.append({
                    'hash': pin_hash,
                    'type': pin_info.get('Type', 'unknown'),
                    'stats': stats
                })
            
            return pinned_docs
            
        except Exception as e:
            logger.error(f"Failed to list pinned documents: {e}")
            return []
    
    async def garbage_collect(self) -> Dict[str, Any]:
        """
        Run garbage collection to free up space.
        
        Returns:
            Dict: Garbage collection results
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.repo.gc()
            )
            
            logger.info("IPFS garbage collection completed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run garbage collection: {e}")
            return {}
    
    async def get_cluster_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all cluster nodes.
        
        Returns:
            List[Dict]: Cluster node status information
        """
        cluster_status = []
        
        for node in self.cluster_nodes:
            try:
                api_url = f'http://{node.endpoint.split(":")[0]}:{node.api_port}'
                
                async with self.session.get(f'{api_url}/id') as response:
                    if response.status == 200:
                        node_info = await response.json()
                        node_dict = asdict(node)
                        node_dict.update({
                            'status': 'online',
                            'version': node_info.get('version', 'unknown'),
                            'peer_count': len(node_info.get('addresses', []))
                        })
                    else:
                        node_dict = asdict(node)
                        node_dict['status'] = 'offline'
                
                cluster_status.append(node_dict)
                
            except Exception as e:
                node_dict = asdict(node)
                node_dict.update({
                    'status': 'error',
                    'error': str(e)
                })
                cluster_status.append(node_dict)
        
        return cluster_status
    
    async def backup_to_cluster(self, ipfs_hashes: List[str]) -> Dict[str, bool]:
        """
        Backup multiple documents to cluster nodes.
        
        Args:
            ipfs_hashes: List of IPFS hashes to backup
            
        Returns:
            Dict: Backup status for each hash
        """
        backup_results = {}
        
        for ipfs_hash in ipfs_hashes:
            try:
                await self._replicate_to_cluster(ipfs_hash)
                backup_results[ipfs_hash] = True
                
            except Exception as e:
                logger.error(f"Failed to backup {ipfs_hash}: {e}")
                backup_results[ipfs_hash] = False
        
        return backup_results
    
    async def cleanup(self):
        """Cleanup IPFS connections and resources."""
        try:
            if self.session:
                await self.session.close()
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            logger.info("IPFS storage connections closed")
            
        except Exception as e:
            logger.error(f"Error during IPFS cleanup: {e}")