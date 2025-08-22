/**
 * Smart contract interaction utilities for audit trail system
 */

import { MetaMaskUtils, ContractUtils, TransactionUtils, ErrorUtils } from './web3Utils';

// Contract ABIs
export const AUDIT_TRAIL_ABI = [
  {
    "inputs": [
      {"type": "bytes32", "name": "documentHash"},
      {"type": "string", "name": "analysisResult"},
      {"type": "string", "name": "decisionType"},
      {"type": "string", "name": "complianceStatus"},
      {"type": "string", "name": "metadata"}
    ],
    "name": "storeAuditRecord",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{"type": "bytes32", "name": "documentHash"}],
    "name": "getAuditRecord",
    "outputs": [
      {"type": "string", "name": "analysisResult"},
      {"type": "string", "name": "decisionType"},
      {"type": "string", "name": "complianceStatus"},
      {"type": "string", "name": "metadata"},
      {"type": "uint256", "name": "timestamp"},
      {"type": "address", "name": "submitter"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"type": "bytes32", "name": "documentHash"}],
    "name": "verifyDocument",
    "outputs": [{"type": "bool", "name": "exists"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "anonymous": false,
    "inputs": [
      {"indexed": true, "type": "bytes32", "name": "documentHash"},
      {"indexed": true, "type": "address", "name": "submitter"},
      {"indexed": false, "type": "string", "name": "analysisResult"},
      {"indexed": false, "type": "uint256", "name": "timestamp"}
    ],
    "name": "AuditRecordStored",
    "type": "event"
  }
];

export const VERIFICATION_CONTRACT_ABI = [
  {
    "inputs": [
      {"type": "bytes32", "name": "documentHash"},
      {"type": "bytes32", "name": "merkleRoot"},
      {"type": "string", "name": "certificateType"},
      {"type": "uint256", "name": "expiryPeriod"},
      {"type": "string", "name": "ipfsHash"}
    ],
    "name": "issueCertificate",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"type": "bytes32", "name": "documentHash"},
      {"type": "bytes32[]", "name": "merkleProof"}
    ],
    "name": "verifyDocument",
    "outputs": [
      {
        "components": [
          {"type": "bool", "name": "isValid"},
          {"type": "bool", "name": "exists"},
          {"type": "bool", "name": "isRevoked"},
          {"type": "bool", "name": "isExpired"},
          {"type": "address", "name": "issuer"},
          {"type": "uint256", "name": "verifiedAt"},
          {"type": "string", "name": "reason"}
        ],
        "type": "tuple",
        "name": "result"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  }
];

export interface AuditRecord {
  documentHash: string;
  analysisResult: any;
  decisionType: string;
  complianceStatus: string;
  metadata: any;
  timestamp: number;
  submitter: string;
}

export interface VerificationResult {
  isValid: boolean;
  exists: boolean;
  isRevoked: boolean;
  isExpired: boolean;
  issuer: string;
  verifiedAt: number;
  reason: string;
}

export class AuditTrailContract {
  private web3: any;
  private contractAddress: string;
  private account: string | null = null;

  constructor(web3: any, contractAddress: string) {
    this.web3 = web3;
    this.contractAddress = contractAddress;
  }

  async initialize(): Promise<void> {
    this.account = await MetaMaskUtils.getCurrentAccount();
    if (!this.account) {
      throw new Error('No wallet connected');
    }
  }

  async storeAuditRecord(
    documentHash: string,
    analysisResult: any,
    decisionType: string,
    complianceStatus: string,
    metadata: any = {},
    onProgress?: (stage: string) => void
  ): Promise<string> {
    if (!this.account) {
      throw new Error('Wallet not connected');
    }

    try {
      onProgress?.('Preparing transaction...');

      // Convert document hash to bytes32 format
      const documentHashBytes = this.web3.utils.padLeft(documentHash, 64);
      
      onProgress?.('Estimating gas...');
      
      const gasEstimate = await ContractUtils.callMethod(
        this.web3,
        this.contractAddress,
        AUDIT_TRAIL_ABI,
        'storeAuditRecord',
        [
          documentHashBytes,
          JSON.stringify(analysisResult),
          decisionType,
          complianceStatus,
          JSON.stringify(metadata)
        ],
        this.account
      );

      onProgress?.('Sending transaction...');

      const tx = await ContractUtils.sendTransaction(
        this.web3,
        this.contractAddress,
        AUDIT_TRAIL_ABI,
        'storeAuditRecord',
        [
          documentHashBytes,
          JSON.stringify(analysisResult),
          decisionType,
          complianceStatus,
          JSON.stringify(metadata)
        ],
        this.account
      );

      onProgress?.('Waiting for confirmation...');

      await TransactionUtils.waitForConfirmation(this.web3, tx.transactionHash, 1);

      onProgress?.('Transaction confirmed');

      return tx.transactionHash;
    } catch (error) {
      const parsedError = ErrorUtils.parseError(error);
      throw new Error(`Failed to store audit record: ${parsedError.message}`);
    }
  }

  async getAuditRecord(documentHash: string): Promise<AuditRecord | null> {
    try {
      const documentHashBytes = this.web3.utils.padLeft(documentHash, 64);
      
      const result = await ContractUtils.callMethod(
        this.web3,
        this.contractAddress,
        AUDIT_TRAIL_ABI,
        'getAuditRecord',
        [documentHashBytes]
      );

      if (!result || result[4] === '0') {
        return null; // Record doesn't exist
      }

      return {
        documentHash,
        analysisResult: JSON.parse(result[0]),
        decisionType: result[1],
        complianceStatus: result[2],
        metadata: JSON.parse(result[3] || '{}'),
        timestamp: parseInt(result[4]),
        submitter: result[5]
      };
    } catch (error) {
      console.error('Error getting audit record:', error);
      return null;
    }
  }

  async verifyDocument(documentHash: string): Promise<boolean> {
    try {
      const documentHashBytes = this.web3.utils.padLeft(documentHash, 64);
      
      return await ContractUtils.callMethod(
        this.web3,
        this.contractAddress,
        AUDIT_TRAIL_ABI,
        'verifyDocument',
        [documentHashBytes]
      );
    } catch (error) {
      console.error('Error verifying document:', error);
      return false;
    }
  }

  async getAuditEvents(
    fromBlock: number | string = 'earliest',
    toBlock: number | string = 'latest',
    documentHash?: string
  ): Promise<any[]> {
    try {
      const filter: any = {};
      
      if (documentHash) {
        filter.documentHash = this.web3.utils.padLeft(documentHash, 64);
      }

      return await ContractUtils.getPastEvents(
        this.web3,
        this.contractAddress,
        AUDIT_TRAIL_ABI,
        'AuditRecordStored',
        fromBlock,
        toBlock,
        filter
      );
    } catch (error) {
      console.error('Error getting audit events:', error);
      return [];
    }
  }

  subscribeToAuditEvents(
    callback: (event: any) => void,
    documentHash?: string
  ): any {
    const filter: any = {};
    
    if (documentHash) {
      filter.documentHash = this.web3.utils.padLeft(documentHash, 64);
    }

    return ContractUtils.subscribeToEvents(
      this.web3,
      this.contractAddress,
      AUDIT_TRAIL_ABI,
      'AuditRecordStored',
      callback,
      filter
    );
  }

  async batchVerifyDocuments(documentHashes: string[]): Promise<boolean[]> {
    const results = await Promise.all(
      documentHashes.map(hash => this.verifyDocument(hash))
    );
    return results;
  }

  async getContractInfo(): Promise<{
    totalRecords: number;
    isOperational: boolean;
    version: string;
  }> {
    try {
      // These would be actual contract methods
      return {
        totalRecords: 0, // Would call getAuditRecordCount()
        isOperational: true, // Would check contract state
        version: '1.0.0' // Would call getContractInfo()
      };
    } catch (error) {
      return {
        totalRecords: 0,
        isOperational: false,
        version: 'unknown'
      };
    }
  }
}

export class DocumentVerificationContract {
  private web3: any;
  private contractAddress: string;
  private account: string | null = null;

  constructor(web3: any, contractAddress: string) {
    this.web3 = web3;
    this.contractAddress = contractAddress;
  }

  async initialize(): Promise<void> {
    this.account = await MetaMaskUtils.getCurrentAccount();
    if (!this.account) {
      throw new Error('No wallet connected');
    }
  }

  async issueCertificate(
    documentHash: string,
    merkleRoot: string,
    certificateType: string,
    expiryPeriod: number,
    ipfsHash: string,
    onProgress?: (stage: string) => void
  ): Promise<string> {
    if (!this.account) {
      throw new Error('Wallet not connected');
    }

    try {
      onProgress?.('Preparing certificate issuance...');

      const documentHashBytes = this.web3.utils.padLeft(documentHash, 64);
      const merkleRootBytes = this.web3.utils.padLeft(merkleRoot, 64);

      onProgress?.('Sending transaction...');

      const tx = await ContractUtils.sendTransaction(
        this.web3,
        this.contractAddress,
        VERIFICATION_CONTRACT_ABI,
        'issueCertificate',
        [documentHashBytes, merkleRootBytes, certificateType, expiryPeriod, ipfsHash],
        this.account
      );

      onProgress?.('Waiting for confirmation...');

      await TransactionUtils.waitForConfirmation(this.web3, tx.transactionHash, 1);

      onProgress?.('Certificate issued successfully');

      return tx.transactionHash;
    } catch (error) {
      const parsedError = ErrorUtils.parseError(error);
      throw new Error(`Failed to issue certificate: ${parsedError.message}`);
    }
  }

  async verifyDocument(
    documentHash: string,
    merkleProof: string[]
  ): Promise<VerificationResult> {
    try {
      const documentHashBytes = this.web3.utils.padLeft(documentHash, 64);
      const merkleProofBytes = merkleProof.map(proof => 
        this.web3.utils.padLeft(proof, 64)
      );

      const result = await ContractUtils.callMethod(
        this.web3,
        this.contractAddress,
        VERIFICATION_CONTRACT_ABI,
        'verifyDocument',
        [documentHashBytes, merkleProofBytes]
      );

      return {
        isValid: result.isValid,
        exists: result.exists,
        isRevoked: result.isRevoked,
        isExpired: result.isExpired,
        issuer: result.issuer,
        verifiedAt: parseInt(result.verifiedAt),
        reason: result.reason
      };
    } catch (error) {
      console.error('Error verifying document:', error);
      return {
        isValid: false,
        exists: false,
        isRevoked: false,
        isExpired: false,
        issuer: '',
        verifiedAt: 0,
        reason: 'Verification failed'
      };
    }
  }

  async batchVerifyDocuments(
    documentHashes: string[],
    merkleProofs: string[][]
  ): Promise<VerificationResult[]> {
    if (documentHashes.length !== merkleProofs.length) {
      throw new Error('Document hashes and merkle proofs arrays must have the same length');
    }

    const results = await Promise.all(
      documentHashes.map((hash, index) => 
        this.verifyDocument(hash, merkleProofs[index])
      )
    );

    return results;
  }
}

export class ContractFactory {
  private web3: any;

  constructor(web3: any) {
    this.web3 = web3;
  }

  createAuditTrailContract(contractAddress: string): AuditTrailContract {
    return new AuditTrailContract(this.web3, contractAddress);
  }

  createVerificationContract(contractAddress: string): DocumentVerificationContract {
    return new DocumentVerificationContract(this.web3, contractAddress);
  }

  async deployAuditTrailContract(
    adminAddress: string,
    auditorAddresses: string[],
    complianceOfficerAddresses: string[],
    onProgress?: (stage: string) => void
  ): Promise<string> {
    // This would deploy the contract with constructor parameters
    // Implementation would depend on contract deployment service
    throw new Error('Contract deployment not implemented');
  }

  async deployVerificationContract(
    ownerAddress: string,
    onProgress?: (stage: string) => void
  ): Promise<string> {
    // This would deploy the verification contract
    // Implementation would depend on contract deployment service
    throw new Error('Contract deployment not implemented');
  }
}

// Utility functions for contract interactions
export class ContractHelpers {
  static createDocumentHash(content: string): string {
    // This would typically use keccak256 or another hash function
    return `0x${require('crypto').createHash('sha256').update(content).digest('hex')}`;
  }

  static createMerkleTree(documents: string[]): {
    root: string;
    proofs: { [key: string]: string[] };
  } {
    // Simplified merkle tree implementation
    // In production, use a proper merkle tree library
    const leaves = documents.map(doc => this.createDocumentHash(doc));
    
    return {
      root: this.createDocumentHash(leaves.join('')),
      proofs: {} // Would contain actual merkle proofs
    };
  }

  static validateContractAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
  }

  static estimateGasCost(
    gasEstimate: number,
    gasPriceGwei: number
  ): { eth: string; usd: string } {
    const gasCostWei = gasEstimate * gasPriceGwei * 1e9;
    const gasCostEth = gasCostWei / 1e18;
    
    // Would need to fetch current ETH price for USD conversion
    return {
      eth: gasCostEth.toFixed(6),
      usd: '0.00' // Would calculate based on current ETH price
    };
  }

  static formatContractError(error: any): string {
    if (error.message.includes('revert')) {
      // Extract revert reason
      const reason = error.message.match(/revert (.+)/)?.[1];
      return reason || 'Transaction reverted';
    }
    
    if (error.message.includes('out of gas')) {
      return 'Transaction ran out of gas';
    }
    
    if (error.message.includes('insufficient funds')) {
      return 'Insufficient funds for transaction';
    }
    
    return 'Contract interaction failed';
  }
}

export default {
  AuditTrailContract,
  DocumentVerificationContract,
  ContractFactory,
  ContractHelpers,
  AUDIT_TRAIL_ABI,
  VERIFICATION_CONTRACT_ABI
};