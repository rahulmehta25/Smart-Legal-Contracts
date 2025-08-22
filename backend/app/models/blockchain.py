"""
Blockchain-specific database models for audit trail system.

This module defines SQLAlchemy models for storing blockchain-related data,
including audit records, transactions, smart contracts, and network information.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, BigInteger, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class BlockchainNetwork(Base):
    """Blockchain network configuration and status."""
    __tablename__ = 'blockchain_networks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    chain_id = Column(Integer, unique=True, nullable=False)
    network_type = Column(String(50), nullable=False)  # ethereum, polygon, bsc, hyperledger
    rpc_url = Column(String(500), nullable=False)
    websocket_url = Column(String(500))
    explorer_url = Column(String(500))
    is_testnet = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Network statistics
    latest_block_number = Column(BigInteger)
    avg_block_time = Column(Integer)  # in seconds
    gas_price_gwei = Column(Integer)
    peer_count = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    smart_contracts = relationship("SmartContract", back_populates="network")
    audit_records = relationship("BlockchainAuditRecord", back_populates="network")
    transactions = relationship("BlockchainTransaction", back_populates="network")
    
    def __repr__(self):
        return f"<BlockchainNetwork(name='{self.name}', chain_id={self.chain_id})>"

class SmartContract(Base):
    """Smart contract deployments and metadata."""
    __tablename__ = 'smart_contracts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    address = Column(String(42), nullable=False, index=True)  # Ethereum address format
    name = Column(String(100), nullable=False)
    contract_type = Column(String(50), nullable=False)  # audit_trail, verification, dispute
    abi = Column(JSON, nullable=False)
    bytecode = Column(Text)
    deployment_tx_hash = Column(String(66))
    deployment_block = Column(BigInteger)
    deployer_address = Column(String(42))
    
    # Contract metadata
    version = Column(String(20), default="1.0.0")
    description = Column(Text)
    source_code = Column(Text)
    compiler_version = Column(String(50))
    optimization_enabled = Column(Boolean, default=False)
    
    # Status
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    deployed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    network = relationship("BlockchainNetwork", back_populates="smart_contracts")
    audit_records = relationship("BlockchainAuditRecord", back_populates="contract")
    
    __table_args__ = (
        Index('idx_contract_network_address', 'network_id', 'address'),
    )
    
    def __repr__(self):
        return f"<SmartContract(name='{self.name}', address='{self.address[:10]}...')>"

class BlockchainAuditRecord(Base):
    """Audit records stored on blockchain with local metadata."""
    __tablename__ = 'blockchain_audit_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    contract_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Document identification
    document_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    ipfs_hash = Column(String(100))  # IPFS content hash
    
    # Blockchain transaction details
    transaction_hash = Column(String(66), nullable=False, unique=True, index=True)
    block_number = Column(BigInteger, nullable=False, index=True)
    block_hash = Column(String(66))
    transaction_index = Column(Integer)
    gas_used = Column(BigInteger)
    gas_price = Column(BigInteger)
    
    # Audit data
    analysis_result = Column(JSONB)
    decision_type = Column(String(50))  # approved, rejected, conditional, escalated
    compliance_status = Column(String(50))  # compliant, non-compliant, pending, under-review
    metadata = Column(JSONB)
    
    # User and context
    submitter_address = Column(String(42), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), index=True)  # Internal user ID if available
    organization = Column(String(100))
    
    # Status and verification
    is_verified = Column(Boolean, default=False)
    verification_status = Column(String(50), default='pending')
    confirmation_count = Column(Integer, default=0)
    
    # Timestamps
    blockchain_timestamp = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    network = relationship("BlockchainNetwork", back_populates="audit_records")
    contract = relationship("SmartContract", back_populates="audit_records")
    certificates = relationship("DocumentCertificate", back_populates="audit_record")
    
    __table_args__ = (
        Index('idx_audit_document_hash', 'document_hash'),
        Index('idx_audit_submitter_timestamp', 'submitter_address', 'blockchain_timestamp'),
        Index('idx_audit_compliance_status', 'compliance_status', 'created_at'),
    )
    
    def __repr__(self):
        return f"<BlockchainAuditRecord(document_hash='{self.document_hash[:10]}...', status='{self.compliance_status}')>"

class BlockchainTransaction(Base):
    """Blockchain transaction tracking and monitoring."""
    __tablename__ = 'blockchain_transactions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Transaction details
    transaction_hash = Column(String(66), nullable=False, unique=True, index=True)
    block_number = Column(BigInteger, index=True)
    block_hash = Column(String(66))
    transaction_index = Column(Integer)
    
    # Transaction data
    from_address = Column(String(42), nullable=False, index=True)
    to_address = Column(String(42), index=True)
    value = Column(BigInteger, default=0)  # in wei
    gas_limit = Column(BigInteger)
    gas_used = Column(BigInteger)
    gas_price = Column(BigInteger)
    
    # Status and confirmation
    status = Column(String(20), default='pending')  # pending, confirmed, failed
    confirmation_count = Column(Integer, default=0)
    nonce = Column(BigInteger)
    
    # Input data and logs
    input_data = Column(Text)
    logs = Column(JSONB)
    
    # Classification
    transaction_type = Column(String(50))  # audit_record, certificate, verification
    related_record_id = Column(UUID(as_uuid=True))  # ID of related audit record
    
    # Error handling
    error_message = Column(Text)
    revert_reason = Column(Text)
    
    # Timestamps
    blockchain_timestamp = Column(DateTime(timezone=True))
    submitted_at = Column(DateTime(timezone=True))
    confirmed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    network = relationship("BlockchainNetwork", back_populates="transactions")
    
    __table_args__ = (
        Index('idx_tx_from_address_timestamp', 'from_address', 'blockchain_timestamp'),
        Index('idx_tx_status_block', 'status', 'block_number'),
    )
    
    def __repr__(self):
        return f"<BlockchainTransaction(hash='{self.transaction_hash[:10]}...', status='{self.status}')>"

class DocumentCertificate(Base):
    """Document verification certificates."""
    __tablename__ = 'document_certificates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_record_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Certificate details
    certificate_hash = Column(String(64), unique=True, nullable=False, index=True)
    merkle_root = Column(String(64))
    merkle_proof = Column(JSONB)  # Array of merkle proof hashes
    
    # Certificate metadata
    certificate_type = Column(String(50), nullable=False)  # compliance, authenticity, integrity
    issuer_address = Column(String(42), nullable=False)
    issuer_name = Column(String(200))
    
    # Validity
    issued_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True))
    is_revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime(timezone=True))
    revocation_reason = Column(Text)
    
    # IPFS storage
    ipfs_hash = Column(String(100))
    
    # Verification results
    last_verification_at = Column(DateTime(timezone=True))
    verification_result = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    audit_record = relationship("BlockchainAuditRecord", back_populates="certificates")
    
    __table_args__ = (
        Index('idx_cert_issuer_type', 'issuer_address', 'certificate_type'),
        Index('idx_cert_validity', 'expires_at', 'is_revoked'),
    )
    
    def __repr__(self):
        return f"<DocumentCertificate(type='{self.certificate_type}', issuer='{self.issuer_address[:10]}...')>"

class IPFSDocument(Base):
    """IPFS document storage metadata."""
    __tablename__ = 'ipfs_documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # IPFS details
    ipfs_hash = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(500))
    content_type = Column(String(100))
    file_size = Column(BigInteger)
    
    # Document metadata
    document_hash = Column(String(64), index=True)  # SHA-256 of content
    encryption_key = Column(String(100))  # Encrypted key for document
    is_encrypted = Column(Boolean, default=False)
    
    # Upload information
    uploader_address = Column(String(42), index=True)
    uploader_user_id = Column(UUID(as_uuid=True))
    pin_status = Column(String(20), default='pinned')  # pinned, unpinned, failed
    
    # Replication and redundancy
    replication_count = Column(Integer, default=1)
    cluster_nodes = Column(JSONB)  # List of cluster nodes storing the document
    
    # Access control
    is_public = Column(Boolean, default=False)
    access_list = Column(JSONB)  # List of addresses with access
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), nullable=False)
    last_accessed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_ipfs_uploader_timestamp', 'uploader_address', 'uploaded_at'),
        Index('idx_ipfs_document_hash', 'document_hash'),
    )
    
    def __repr__(self):
        return f"<IPFSDocument(hash='{self.ipfs_hash[:10]}...', filename='{self.filename}')>"

class BlockchainEvent(Base):
    """Blockchain event monitoring and logging."""
    __tablename__ = 'blockchain_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    contract_id = Column(UUID(as_uuid=True), index=True)
    
    # Event details
    event_name = Column(String(100), nullable=False, index=True)
    transaction_hash = Column(String(66), nullable=False, index=True)
    block_number = Column(BigInteger, nullable=False, index=True)
    log_index = Column(Integer)
    
    # Event data
    event_data = Column(JSONB)
    topics = Column(JSONB)  # Event topics/indexed parameters
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True))
    processing_error = Column(Text)
    
    # Timestamps
    blockchain_timestamp = Column(DateTime(timezone=True))
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_event_name_block', 'event_name', 'block_number'),
        Index('idx_event_processed', 'is_processed', 'detected_at'),
    )
    
    def __repr__(self):
        return f"<BlockchainEvent(name='{self.event_name}', block={self.block_number})>"

class NetworkMonitoring(Base):
    """Network health and performance monitoring."""
    __tablename__ = 'network_monitoring'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Metrics
    block_height = Column(BigInteger)
    block_time = Column(Integer)  # Average block time in seconds
    gas_price_gwei = Column(Integer)
    peer_count = Column(Integer)
    pending_transactions = Column(Integer)
    
    # Performance metrics
    rpc_response_time = Column(Integer)  # in milliseconds
    sync_status = Column(Boolean)
    network_hashrate = Column(BigInteger)
    difficulty = Column(String(100))
    
    # Status
    status = Column(String(20), default='healthy')  # healthy, degraded, unhealthy, offline
    
    # Timestamp
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_monitoring_network_time', 'network_id', 'measured_at'),
    )
    
    def __repr__(self):
        return f"<NetworkMonitoring(network_id='{self.network_id}', status='{self.status}')>"

class DisputeRecord(Base):
    """Blockchain-based dispute resolution records."""
    __tablename__ = 'dispute_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_record_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Dispute details
    dispute_id = Column(String(100), unique=True, nullable=False, index=True)
    document_hash = Column(String(64), nullable=False, index=True)
    dispute_type = Column(String(50), nullable=False)  # authenticity, accuracy, compliance
    
    # Parties
    complainant_address = Column(String(42), nullable=False)
    respondent_address = Column(String(42), nullable=False)
    arbitrator_address = Column(String(42))
    
    # Status and resolution
    status = Column(String(30), default='pending')  # pending, in_review, arbitration, resolved
    resolution = Column(String(30))  # upheld, overturned, partial_upheld, settlement
    resolution_details = Column(Text)
    
    # Evidence and metadata
    evidence = Column(JSONB)
    settlement_terms = Column(Text)
    
    # Financial details
    staking_amount = Column(BigInteger)  # in wei
    arbitration_fee = Column(BigInteger)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))
    deadline_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_dispute_document_hash', 'document_hash'),
        Index('idx_dispute_status_created', 'status', 'created_at'),
    )
    
    def __repr__(self):
        return f"<DisputeRecord(dispute_id='{self.dispute_id}', status='{self.status}')>"

class GasOptimization(Base):
    """Gas usage tracking and optimization metrics."""
    __tablename__ = 'gas_optimization'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    contract_id = Column(UUID(as_uuid=True), index=True)
    
    # Function details
    function_name = Column(String(100), nullable=False)
    function_selector = Column(String(10))  # 4-byte function selector
    
    # Gas metrics
    gas_limit = Column(BigInteger)
    gas_used = Column(BigInteger)
    gas_price = Column(BigInteger)
    gas_cost_wei = Column(BigInteger)
    gas_efficiency = Column(Integer)  # Percentage
    
    # Optimization suggestions
    optimization_score = Column(Integer)  # 1-100
    optimization_suggestions = Column(JSONB)
    
    # Context
    input_size = Column(Integer)  # Size of input data
    output_size = Column(Integer)  # Size of output data
    execution_time = Column(Integer)  # in milliseconds
    
    # Timestamp
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_gas_function_network', 'function_name', 'network_id'),
        Index('idx_gas_efficiency', 'gas_efficiency', 'measured_at'),
    )
    
    def __repr__(self):
        return f"<GasOptimization(function='{self.function_name}', efficiency={self.gas_efficiency}%)>"