"""
Blockchain service layer for audit trail system.

Provides high-level blockchain operations including audit record management,
transaction handling, monitoring, and integration with multiple blockchain networks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from ..blockchain.ethereum import EthereumAuditTrail, AuditRecord, BlockchainTransaction
from ..blockchain.hyperledger import HyperledgerAuditTrail, PrivateAuditRecord
from ..blockchain.ipfs import IPFSStorage, IPFSDocument as IPFSDoc
from ..blockchain.monitoring import BlockchainMonitor
from ..models.blockchain import (
    BlockchainNetwork, SmartContract, BlockchainAuditRecord, 
    BlockchainTransaction as DBTransaction, IPFSDocument, 
    DocumentCertificate, BlockchainEvent, NetworkMonitoring
)
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class AuditSubmissionResult:
    """Result of audit record submission."""
    success: bool
    transaction_hash: str
    block_number: Optional[int]
    gas_used: Optional[int]
    error_message: Optional[str]
    ipfs_hash: Optional[str]

@dataclass
class BlockchainStatus:
    """Blockchain network status."""
    network_name: str
    is_connected: bool
    latest_block: int
    gas_price: float
    peer_count: int
    sync_status: bool

class BlockchainService:
    """
    High-level blockchain service for audit trail operations.
    
    Manages multiple blockchain networks, smart contracts, and IPFS storage
    with database persistence and monitoring capabilities.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.ethereum_clients: Dict[str, EthereumAuditTrail] = {}
        self.hyperledger_clients: Dict[str, HyperledgerAuditTrail] = {}
        self.ipfs_storage: Optional[IPFSStorage] = None
        self.monitor: Optional[BlockchainMonitor] = None
        
    async def initialize(self) -> bool:
        """Initialize blockchain connections and services."""
        try:
            # Initialize IPFS storage
            await self._initialize_ipfs()
            
            # Initialize blockchain networks
            await self._initialize_networks()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            logger.info("Blockchain service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain service: {e}")
            return False
    
    async def _initialize_ipfs(self):
        """Initialize IPFS storage system."""
        ipfs_config = getattr(settings, 'IPFS_CONFIG', {})
        self.ipfs_storage = IPFSStorage(
            ipfs_api_endpoint=ipfs_config.get('api_endpoint', '/ip4/127.0.0.1/tcp/5001'),
            ipfs_gateway=ipfs_config.get('gateway', 'http://127.0.0.1:8080'),
            cluster_endpoints=ipfs_config.get('cluster_endpoints', []),
            encryption_enabled=ipfs_config.get('encryption_enabled', True)
        )
        
        await self.ipfs_storage.initialize()
    
    async def _initialize_networks(self):
        """Initialize connections to configured blockchain networks."""
        # Get active networks from database
        networks = self.db.query(BlockchainNetwork).filter(
            BlockchainNetwork.is_active == True
        ).all()
        
        for network in networks:
            try:
                if network.network_type == 'ethereum':
                    await self._initialize_ethereum_network(network)
                elif network.network_type == 'hyperledger':
                    await self._initialize_hyperledger_network(network)
                    
            except Exception as e:
                logger.error(f"Failed to initialize network {network.name}: {e}")
    
    async def _initialize_ethereum_network(self, network: BlockchainNetwork):
        """Initialize Ethereum-compatible network."""
        # Get network configuration
        config = getattr(settings, 'BLOCKCHAIN_NETWORKS', {}).get(network.name, {})
        
        if not config:
            logger.warning(f"No configuration found for network: {network.name}")
            return
        
        ethereum_client = EthereumAuditTrail(
            web3_provider=config['rpc_url'],
            private_key=config['private_key'],
            contract_address=config.get('contract_address'),
            gas_limit=config.get('gas_limit', 3000000),
            gas_price_gwei=config.get('gas_price_gwei', 20)
        )
        
        if await ethereum_client.connect():
            self.ethereum_clients[network.name] = ethereum_client
            logger.info(f"Connected to Ethereum network: {network.name}")
    
    async def _initialize_hyperledger_network(self, network: BlockchainNetwork):
        """Initialize Hyperledger Fabric network."""
        config = getattr(settings, 'HYPERLEDGER_NETWORKS', {}).get(network.name, {})
        
        if not config:
            logger.warning(f"No Hyperledger configuration found for: {network.name}")
            return
        
        hyperledger_client = HyperledgerAuditTrail(
            network_config_path=config['network_config_path'],
            user_cert_path=config['user_cert_path'],
            user_key_path=config['user_key_path'],
            channel_name=config.get('channel_name', 'audit-trail-channel'),
            chaincode_name=config.get('chaincode_name', 'audit-trail-chaincode'),
            organization=config.get('organization', 'Org1')
        )
        
        if await hyperledger_client.initialize():
            self.hyperledger_clients[network.name] = hyperledger_client
            logger.info(f"Connected to Hyperledger network: {network.name}")
    
    async def _initialize_monitoring(self):
        """Initialize blockchain monitoring system."""
        networks_config = {}
        
        # Build monitoring configuration from database
        networks = self.db.query(BlockchainNetwork).filter(
            BlockchainNetwork.is_active == True
        ).all()
        
        for network in networks:
            networks_config[network.name] = {
                'rpc_url': network.rpc_url,
                'websocket_url': network.websocket_url
            }
        
        if networks_config:
            self.monitor = BlockchainMonitor(
                networks=networks_config,
                monitoring_interval=30,
                enable_websocket=True
            )
            
            # Set up event handlers
            self.monitor.add_event_handler('new_block', self._handle_new_block)
            self.monitor.add_event_handler('alert_created', self._handle_monitoring_alert)
            
            await self.monitor.start_monitoring()
    
    async def store_audit_record(self,
                                document_content: str,
                                analysis_result: Dict[str, Any],
                                decision_type: str,
                                compliance_status: str,
                                user_id: str,
                                network_name: str = 'ethereum',
                                metadata: Optional[Dict[str, Any]] = None) -> AuditSubmissionResult:
        """
        Store audit record on blockchain with IPFS backup.
        
        Args:
            document_content: Original document content
            analysis_result: AI analysis results
            decision_type: Type of decision made
            compliance_status: Compliance determination
            user_id: User submitting the record
            network_name: Target blockchain network
            metadata: Additional metadata
            
        Returns:
            AuditSubmissionResult: Submission results
        """
        try:
            # Store document on IPFS
            ipfs_document = None
            if self.ipfs_storage:
                ipfs_document = await self.ipfs_storage.upload_document(
                    content=document_content,
                    filename=f"audit_document_{datetime.now().isoformat()}.txt",
                    uploader_id=user_id,
                    metadata=metadata or {}
                )
            
            # Create document hash
            import hashlib
            document_hash = hashlib.sha256(document_content.encode()).hexdigest()
            
            # Prepare audit record
            audit_record = AuditRecord(
                document_hash=document_hash,
                analysis_result=analysis_result,
                timestamp=int(datetime.now().timestamp()),
                user_id=user_id,
                decision_type=decision_type,
                compliance_status=compliance_status,
                metadata=metadata or {}
            )
            
            # Submit to blockchain
            blockchain_tx = None
            if network_name in self.ethereum_clients:
                blockchain_tx = await self.ethereum_clients[network_name].store_audit_record(audit_record)
            elif network_name in self.hyperledger_clients:
                # Convert to private audit record
                private_record = PrivateAuditRecord(
                    document_hash=document_hash,
                    analysis_result=analysis_result,
                    timestamp=audit_record.timestamp,
                    user_id=user_id,
                    organization=metadata.get('organization', 'Default'),
                    decision_type=decision_type,
                    compliance_status=compliance_status,
                    metadata=metadata or {},
                    confidentiality_level=metadata.get('confidentiality_level', 'internal'),
                    access_control=metadata.get('access_control', [user_id])
                )
                blockchain_tx = await self.hyperledger_clients[network_name].store_private_audit_record(private_record)
            else:
                raise ValueError(f"Network not found or not initialized: {network_name}")
            
            # Store in database
            await self._store_audit_record_db(
                audit_record=audit_record,
                blockchain_tx=blockchain_tx,
                ipfs_document=ipfs_document,
                network_name=network_name
            )
            
            return AuditSubmissionResult(
                success=True,
                transaction_hash=blockchain_tx.tx_hash,
                block_number=blockchain_tx.block_number,
                gas_used=blockchain_tx.gas_used,
                error_message=None,
                ipfs_hash=ipfs_document.ipfs_hash if ipfs_document else None
            )
            
        except Exception as e:
            logger.error(f"Failed to store audit record: {e}")
            return AuditSubmissionResult(
                success=False,
                transaction_hash="",
                block_number=None,
                gas_used=None,
                error_message=str(e),
                ipfs_hash=None
            )
    
    async def _store_audit_record_db(self,
                                   audit_record: AuditRecord,
                                   blockchain_tx: BlockchainTransaction,
                                   ipfs_document: Optional[IPFSDoc],
                                   network_name: str):
        """Store audit record in database."""
        # Get network and contract
        network = self.db.query(BlockchainNetwork).filter(
            BlockchainNetwork.name == network_name
        ).first()
        
        if not network:
            raise ValueError(f"Network not found in database: {network_name}")
        
        # For simplicity, get the first audit trail contract for this network
        contract = self.db.query(SmartContract).filter(
            and_(
                SmartContract.network_id == network.id,
                SmartContract.contract_type == 'audit_trail',
                SmartContract.is_active == True
            )
        ).first()
        
        # Create audit record
        db_audit_record = BlockchainAuditRecord(
            network_id=network.id,
            contract_id=contract.id if contract else None,
            document_hash=audit_record.document_hash,
            ipfs_hash=ipfs_document.ipfs_hash if ipfs_document else None,
            transaction_hash=blockchain_tx.tx_hash,
            block_number=blockchain_tx.block_number,
            gas_used=blockchain_tx.gas_used,
            analysis_result=audit_record.analysis_result,
            decision_type=audit_record.decision_type,
            compliance_status=audit_record.compliance_status,
            metadata=audit_record.metadata,
            submitter_address="0x" + "0" * 40,  # Would get from blockchain client
            user_id=audit_record.user_id,
            blockchain_timestamp=datetime.fromtimestamp(audit_record.timestamp),
            is_verified=True,
            verification_status='confirmed'
        )
        
        self.db.add(db_audit_record)
        
        # Store IPFS document if exists
        if ipfs_document:
            db_ipfs_document = IPFSDocument(
                ipfs_hash=ipfs_document.ipfs_hash,
                filename=ipfs_document.filename,
                content_type=ipfs_document.content_type,
                file_size=ipfs_document.size,
                document_hash=audit_record.document_hash,
                encryption_key=ipfs_document.encryption_key,
                is_encrypted=ipfs_document.encryption_key is not None,
                uploader_user_id=audit_record.user_id,
                uploaded_at=datetime.fromtimestamp(ipfs_document.upload_timestamp),
                pin_status='pinned'
            )
            self.db.add(db_ipfs_document)
        
        self.db.commit()
    
    async def get_audit_record(self, 
                             document_hash: str, 
                             network_name: str = 'ethereum') -> Optional[Dict[str, Any]]:
        """Retrieve audit record from blockchain and database."""
        try:
            # First check database
            db_record = self.db.query(BlockchainAuditRecord).filter(
                BlockchainAuditRecord.document_hash == document_hash
            ).first()
            
            if not db_record:
                return None
            
            # Get blockchain record for verification
            blockchain_record = None
            if network_name in self.ethereum_clients:
                blockchain_record = await self.ethereum_clients[network_name].get_audit_record(document_hash)
            elif network_name in self.hyperledger_clients:
                blockchain_record = await self.hyperledger_clients[network_name].get_private_audit_record(
                    document_hash, "requesting_org"  # Would get from context
                )
            
            return {
                'document_hash': db_record.document_hash,
                'analysis_result': db_record.analysis_result,
                'decision_type': db_record.decision_type,
                'compliance_status': db_record.compliance_status,
                'metadata': db_record.metadata,
                'submitter_address': db_record.submitter_address,
                'user_id': str(db_record.user_id) if db_record.user_id else None,
                'transaction_hash': db_record.transaction_hash,
                'block_number': db_record.block_number,
                'blockchain_timestamp': db_record.blockchain_timestamp.isoformat() if db_record.blockchain_timestamp else None,
                'is_verified': db_record.is_verified,
                'ipfs_hash': db_record.ipfs_hash,
                'blockchain_verified': blockchain_record is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit record: {e}")
            return None
    
    async def verify_document(self, 
                            document_hash: str, 
                            network_name: str = 'ethereum') -> bool:
        """Verify document exists on blockchain."""
        try:
            if network_name in self.ethereum_clients:
                return await self.ethereum_clients[network_name].verify_document(document_hash)
            elif network_name in self.hyperledger_clients:
                return await self.hyperledger_clients[network_name].verify_document_exists(document_hash)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify document: {e}")
            return False
    
    async def get_network_status(self, network_name: str) -> Optional[BlockchainStatus]:
        """Get blockchain network status."""
        try:
            if self.monitor:
                summary = self.monitor.get_monitoring_summary()
                network_data = summary.get('networks', {}).get(network_name)
                
                if network_data:
                    return BlockchainStatus(
                        network_name=network_name,
                        is_connected=True,
                        latest_block=network_data.get('block_height', 0),
                        gas_price=network_data.get('gas_price', 0.0),
                        peer_count=network_data.get('peer_count', 0),
                        sync_status=network_data.get('status') == 'healthy'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return None
    
    async def get_audit_records_by_user(self, 
                                      user_id: str, 
                                      limit: int = 50) -> List[Dict[str, Any]]:
        """Get audit records submitted by a specific user."""
        try:
            records = self.db.query(BlockchainAuditRecord).filter(
                BlockchainAuditRecord.user_id == user_id
            ).order_by(desc(BlockchainAuditRecord.created_at)).limit(limit).all()
            
            return [
                {
                    'id': str(record.id),
                    'document_hash': record.document_hash,
                    'decision_type': record.decision_type,
                    'compliance_status': record.compliance_status,
                    'transaction_hash': record.transaction_hash,
                    'block_number': record.block_number,
                    'blockchain_timestamp': record.blockchain_timestamp.isoformat() if record.blockchain_timestamp else None,
                    'is_verified': record.is_verified,
                    'ipfs_hash': record.ipfs_hash
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Failed to get user audit records: {e}")
            return []
    
    async def get_compliance_statistics(self) -> Dict[str, Any]:
        """Get compliance statistics across all networks."""
        try:
            # Query database for statistics
            from sqlalchemy import func
            
            stats = self.db.query(
                BlockchainAuditRecord.compliance_status,
                func.count(BlockchainAuditRecord.id).label('count')
            ).group_by(BlockchainAuditRecord.compliance_status).all()
            
            total_records = sum(stat.count for stat in stats)
            
            compliance_stats = {
                'total_records': total_records,
                'by_status': {stat.compliance_status: stat.count for stat in stats},
                'compliance_rate': 0.0
            }
            
            # Calculate compliance rate
            compliant_count = compliance_stats['by_status'].get('compliant', 0)
            if total_records > 0:
                compliance_stats['compliance_rate'] = (compliant_count / total_records) * 100
            
            return compliance_stats
            
        except Exception as e:
            logger.error(f"Failed to get compliance statistics: {e}")
            return {
                'total_records': 0,
                'by_status': {},
                'compliance_rate': 0.0
            }
    
    async def _handle_new_block(self, event_data: Dict[str, Any]):
        """Handle new block events from monitoring."""
        try:
            network_name = event_data.get('network')
            block_number = event_data.get('block_number')
            
            # Update network monitoring data
            network = self.db.query(BlockchainNetwork).filter(
                BlockchainNetwork.name == network_name
            ).first()
            
            if network:
                network.latest_block_number = block_number
                network.updated_at = datetime.now()
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Failed to handle new block event: {e}")
    
    async def _handle_monitoring_alert(self, alert_data: Dict[str, Any]):
        """Handle monitoring alerts."""
        try:
            # Log alert and potentially notify administrators
            logger.warning(f"Blockchain monitoring alert: {alert_data}")
            
            # Could implement alert notifications here
            
        except Exception as e:
            logger.error(f"Failed to handle monitoring alert: {e}")
    
    async def cleanup(self):
        """Cleanup blockchain connections and resources."""
        try:
            # Cleanup Ethereum clients
            for client in self.ethereum_clients.values():
                client.cleanup()
            
            # Cleanup Hyperledger clients
            for client in self.hyperledger_clients.values():
                await client.cleanup()
            
            # Cleanup IPFS storage
            if self.ipfs_storage:
                await self.ipfs_storage.cleanup()
            
            # Cleanup monitoring
            if self.monitor:
                await self.monitor.cleanup()
                
            logger.info("Blockchain service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during blockchain service cleanup: {e}")

# Factory function for creating blockchain service
def create_blockchain_service(db_session: Session) -> BlockchainService:
    """Create and initialize blockchain service."""
    return BlockchainService(db_session)