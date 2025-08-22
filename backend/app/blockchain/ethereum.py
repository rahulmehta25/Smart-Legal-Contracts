"""
Ethereum blockchain integration for audit trail system.

Provides Ethereum smart contract integration for immutable audit trails,
document verification, and compliance tracking.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import TransactionNotFound, BlockNotFound
from eth_account import Account
from eth_utils import to_checksum_address
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class AuditRecord:
    """Audit record structure for blockchain storage."""
    document_hash: str
    analysis_result: Dict[str, Any]
    timestamp: int
    user_id: str
    decision_type: str
    compliance_status: str
    metadata: Dict[str, Any]

@dataclass
class BlockchainTransaction:
    """Blockchain transaction details."""
    tx_hash: str
    block_number: int
    gas_used: int
    status: str
    timestamp: int

class EthereumAuditTrail:
    """
    Ethereum blockchain integration for audit trail management.
    
    Handles smart contract deployment, transaction management,
    and audit record storage on Ethereum blockchain.
    """
    
    def __init__(self, 
                 web3_provider: str,
                 private_key: str,
                 contract_address: Optional[str] = None,
                 gas_limit: int = 3000000,
                 gas_price_gwei: int = 20):
        """
        Initialize Ethereum audit trail system.
        
        Args:
            web3_provider: Ethereum node URL (Infura, Alchemy, etc.)
            private_key: Private key for transaction signing
            contract_address: Deployed contract address (if exists)
            gas_limit: Maximum gas limit for transactions
            gas_price_gwei: Gas price in Gwei
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.account = Account.from_key(private_key)
        self.contract_address = contract_address
        self.gas_limit = gas_limit
        self.gas_price = Web3.to_wei(gas_price_gwei, 'gwei')
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Contract ABI for audit trail smart contract
        self.contract_abi = [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "bytes32", "name": "documentHash", "type": "bytes32"},
                    {"indexed": True, "internalType": "address", "name": "submitter", "type": "address"},
                    {"indexed": False, "internalType": "string", "name": "analysisResult", "type": "string"},
                    {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                ],
                "name": "AuditRecordStored",
                "type": "event"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "documentHash", "type": "bytes32"},
                    {"internalType": "string", "name": "analysisResult", "type": "string"},
                    {"internalType": "string", "name": "decisionType", "type": "string"},
                    {"internalType": "string", "name": "complianceStatus", "type": "string"},
                    {"internalType": "string", "name": "metadata", "type": "string"}
                ],
                "name": "storeAuditRecord",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "documentHash", "type": "bytes32"}
                ],
                "name": "getAuditRecord",
                "outputs": [
                    {"internalType": "string", "name": "analysisResult", "type": "string"},
                    {"internalType": "string", "name": "decisionType", "type": "string"},
                    {"internalType": "string", "name": "complianceStatus", "type": "string"},
                    {"internalType": "string", "name": "metadata", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "submitter", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "documentHash", "type": "bytes32"}
                ],
                "name": "verifyDocument",
                "outputs": [
                    {"internalType": "bool", "name": "exists", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getAuditRecordCount",
                "outputs": [
                    {"internalType": "uint256", "name": "count", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Initialize contract if address provided
        self.contract = None
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=to_checksum_address(self.contract_address),
                abi=self.contract_abi
            )
    
    async def connect(self) -> bool:
        """
        Test connection to Ethereum network.
        
        Returns:
            bool: True if connected successfully
        """
        try:
            latest_block = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.w3.eth.get_block, 'latest'
            )
            logger.info(f"Connected to Ethereum network. Latest block: {latest_block['number']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ethereum network: {e}")
            return False
    
    def create_document_hash(self, document_content: str) -> str:
        """
        Create SHA-256 hash of document content.
        
        Args:
            document_content: Document content to hash
            
        Returns:
            str: Hexadecimal hash string
        """
        return hashlib.sha256(document_content.encode()).hexdigest()
    
    async def store_audit_record(self, audit_record: AuditRecord) -> BlockchainTransaction:
        """
        Store audit record on Ethereum blockchain.
        
        Args:
            audit_record: Audit record to store
            
        Returns:
            BlockchainTransaction: Transaction details
        """
        if not self.contract:
            raise ValueError("Contract not initialized. Deploy contract first.")
        
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = bytes.fromhex(audit_record.document_hash)
            
            # Prepare transaction
            function = self.contract.functions.storeAuditRecord(
                doc_hash_bytes,
                json.dumps(audit_record.analysis_result),
                audit_record.decision_type,
                audit_record.compliance_status,
                json.dumps(audit_record.metadata)
            )
            
            # Build transaction
            transaction = function.build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': self.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.w3.eth.send_raw_transaction, signed_txn.rawTransaction
            )
            
            # Wait for transaction receipt
            tx_receipt = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.w3.eth.wait_for_transaction_receipt, tx_hash
            )
            
            blockchain_tx = BlockchainTransaction(
                tx_hash=tx_receipt['transactionHash'].hex(),
                block_number=tx_receipt['blockNumber'],
                gas_used=tx_receipt['gasUsed'],
                status='success' if tx_receipt['status'] == 1 else 'failed',
                timestamp=int(datetime.now().timestamp())
            )
            
            logger.info(f"Audit record stored on blockchain: {blockchain_tx.tx_hash}")
            return blockchain_tx
            
        except Exception as e:
            logger.error(f"Failed to store audit record: {e}")
            raise
    
    async def get_audit_record(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve audit record from blockchain.
        
        Args:
            document_hash: Document hash to retrieve
            
        Returns:
            Optional[Dict]: Audit record data if found
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        try:
            doc_hash_bytes = bytes.fromhex(document_hash)
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.contract.functions.getAuditRecord(doc_hash_bytes).call
            )
            
            if result[4] == 0:  # timestamp is 0, record doesn't exist
                return None
            
            return {
                'analysis_result': json.loads(result[0]),
                'decision_type': result[1],
                'compliance_status': result[2],
                'metadata': json.loads(result[3]),
                'timestamp': result[4],
                'submitter': result[5]
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit record: {e}")
            return None
    
    async def verify_document(self, document_hash: str) -> bool:
        """
        Verify if document exists on blockchain.
        
        Args:
            document_hash: Document hash to verify
            
        Returns:
            bool: True if document exists
        """
        if not self.contract:
            return False
        
        try:
            doc_hash_bytes = bytes.fromhex(document_hash)
            exists = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.contract.functions.verifyDocument(doc_hash_bytes).call
            )
            return exists
            
        except Exception as e:
            logger.error(f"Failed to verify document: {e}")
            return False
    
    async def get_audit_record_count(self) -> int:
        """
        Get total number of audit records stored.
        
        Returns:
            int: Number of audit records
        """
        if not self.contract:
            return 0
        
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.contract.functions.getAuditRecordCount().call
            )
            return count
            
        except Exception as e:
            logger.error(f"Failed to get audit record count: {e}")
            return 0
    
    async def get_transaction_history(self, 
                                    from_block: int = 0, 
                                    to_block: str = 'latest') -> List[Dict[str, Any]]:
        """
        Get transaction history for audit records.
        
        Args:
            from_block: Starting block number
            to_block: Ending block number or 'latest'
            
        Returns:
            List[Dict]: Transaction history
        """
        if not self.contract:
            return []
        
        try:
            # Get AuditRecordStored events
            event_filter = self.contract.events.AuditRecordStored.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            events = await asyncio.get_event_loop().run_in_executor(
                self.executor, event_filter.get_all_entries
            )
            
            history = []
            for event in events:
                history.append({
                    'transaction_hash': event['transactionHash'].hex(),
                    'block_number': event['blockNumber'],
                    'document_hash': event['args']['documentHash'].hex(),
                    'submitter': event['args']['submitter'],
                    'analysis_result': event['args']['analysisResult'],
                    'timestamp': event['args']['timestamp']
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get transaction history: {e}")
            return []
    
    async def estimate_gas_cost(self, audit_record: AuditRecord) -> Dict[str, Any]:
        """
        Estimate gas cost for storing audit record.
        
        Args:
            audit_record: Audit record to estimate cost for
            
        Returns:
            Dict: Gas estimation details
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        try:
            doc_hash_bytes = bytes.fromhex(audit_record.document_hash)
            
            # Estimate gas
            gas_estimate = self.contract.functions.storeAuditRecord(
                doc_hash_bytes,
                json.dumps(audit_record.analysis_result),
                audit_record.decision_type,
                audit_record.compliance_status,
                json.dumps(audit_record.metadata)
            ).estimate_gas({'from': self.account.address})
            
            # Calculate costs
            cost_wei = gas_estimate * self.gas_price
            cost_eth = self.w3.from_wei(cost_wei, 'ether')
            
            return {
                'gas_estimate': gas_estimate,
                'gas_price_gwei': self.w3.from_wei(self.gas_price, 'gwei'),
                'cost_wei': cost_wei,
                'cost_eth': float(cost_eth),
                'cost_usd': None  # Would need price oracle integration
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate gas cost: {e}")
            raise
    
    async def deploy_contract(self, bytecode: str) -> str:
        """
        Deploy audit trail smart contract.
        
        Args:
            bytecode: Compiled contract bytecode
            
        Returns:
            str: Deployed contract address
        """
        try:
            # Create contract instance
            contract = self.w3.eth.contract(abi=self.contract_abi, bytecode=bytecode)
            
            # Build deployment transaction
            transaction = contract.constructor().build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': self.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.w3.eth.send_raw_transaction, signed_txn.rawTransaction
            )
            
            # Wait for deployment
            tx_receipt = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.w3.eth.wait_for_transaction_receipt, tx_hash
            )
            
            contract_address = tx_receipt.contractAddress
            logger.info(f"Contract deployed at address: {contract_address}")
            
            # Initialize contract instance
            self.contract_address = contract_address
            self.contract = self.w3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )
            
            return contract_address
            
        except Exception as e:
            logger.error(f"Failed to deploy contract: {e}")
            raise
    
    async def batch_store_audit_records(self, audit_records: List[AuditRecord]) -> List[BlockchainTransaction]:
        """
        Store multiple audit records in batch.
        
        Args:
            audit_records: List of audit records to store
            
        Returns:
            List[BlockchainTransaction]: Transaction details for each record
        """
        transactions = []
        
        for record in audit_records:
            try:
                tx = await self.store_audit_record(record)
                transactions.append(tx)
                
                # Small delay to avoid nonce conflicts
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to store audit record in batch: {e}")
                # Continue with next record
                continue
        
        return transactions
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)