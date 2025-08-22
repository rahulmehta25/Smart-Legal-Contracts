// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/**
 * @title DocumentVerificationContract
 * @dev Smart contract for document verification and integrity checking
 * @author Blockchain Audit Trail System
 */
contract DocumentVerificationContract is Ownable, ReentrancyGuard {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;
    
    // Structures
    struct DocumentCertificate {
        bytes32 documentHash;
        bytes32 merkleRoot;
        address issuer;
        uint256 issuedAt;
        uint256 expiresAt;
        string certificateType;
        bool isRevoked;
        string ipfsHash;
    }
    
    struct VerificationResult {
        bool isValid;
        bool exists;
        bool isRevoked;
        bool isExpired;
        address issuer;
        uint256 verifiedAt;
        string reason;
    }
    
    struct TrustedIssuer {
        address issuerAddress;
        string organizationName;
        bool isActive;
        uint256 addedAt;
        string publicKey;
    }
    
    // State variables
    mapping(bytes32 => DocumentCertificate) public certificates;
    mapping(address => TrustedIssuer) public trustedIssuers;
    mapping(bytes32 => bytes32[]) public documentVersions;
    mapping(bytes32 => mapping(address => bool)) public issuerApprovals;
    
    uint256 public certificateCount;
    uint256 public defaultExpiryPeriod = 365 days;
    
    // Events
    event CertificateIssued(
        bytes32 indexed documentHash,
        address indexed issuer,
        uint256 issuedAt,
        uint256 expiresAt,
        string certificateType
    );
    
    event CertificateRevoked(
        bytes32 indexed documentHash,
        address indexed revoker,
        uint256 revokedAt,
        string reason
    );
    
    event DocumentVerified(
        bytes32 indexed documentHash,
        address indexed verifier,
        bool isValid,
        uint256 verifiedAt
    );
    
    event TrustedIssuerAdded(
        address indexed issuer,
        string organizationName,
        uint256 addedAt
    );
    
    event TrustedIssuerRemoved(
        address indexed issuer,
        uint256 removedAt
    );
    
    // Modifiers
    modifier onlyTrustedIssuer() {
        require(trustedIssuers[msg.sender].isActive, "Not a trusted issuer");
        _;
    }
    
    modifier certificateExists(bytes32 _documentHash) {
        require(certificates[_documentHash].issuer != address(0), "Certificate does not exist");
        _;
    }
    
    modifier certificateNotRevoked(bytes32 _documentHash) {
        require(!certificates[_documentHash].isRevoked, "Certificate is revoked");
        _;
    }
    
    /**
     * @dev Constructor
     * @param _owner Contract owner address
     */
    constructor(address _owner) Ownable(_owner) {
        // Contract initialization
    }
    
    /**
     * @dev Issue a new document certificate
     * @param _documentHash Hash of the document
     * @param _merkleRoot Merkle root for document integrity
     * @param _certificateType Type of certificate
     * @param _expiryPeriod Expiry period in seconds (0 for default)
     * @param _ipfsHash IPFS hash for additional document storage
     */
    function issueCertificate(
        bytes32 _documentHash,
        bytes32 _merkleRoot,
        string memory _certificateType,
        uint256 _expiryPeriod,
        string memory _ipfsHash
    )
        external
        onlyTrustedIssuer
        nonReentrant
    {
        require(certificates[_documentHash].issuer == address(0), "Certificate already exists");
        require(bytes(_certificateType).length > 0, "Certificate type cannot be empty");
        
        uint256 expiryTime = _expiryPeriod > 0 ? 
            block.timestamp + _expiryPeriod : 
            block.timestamp + defaultExpiryPeriod;
        
        DocumentCertificate storage cert = certificates[_documentHash];
        cert.documentHash = _documentHash;
        cert.merkleRoot = _merkleRoot;
        cert.issuer = msg.sender;
        cert.issuedAt = block.timestamp;
        cert.expiresAt = expiryTime;
        cert.certificateType = _certificateType;
        cert.isRevoked = false;
        cert.ipfsHash = _ipfsHash;
        
        certificateCount++;
        
        emit CertificateIssued(
            _documentHash,
            msg.sender,
            block.timestamp,
            expiryTime,
            _certificateType
        );
    }
    
    /**
     * @dev Verify document certificate
     * @param _documentHash Hash of the document to verify
     * @param _merkleProof Merkle proof for document integrity
     * @return result Verification result structure
     */
    function verifyDocument(
        bytes32 _documentHash,
        bytes32[] memory _merkleProof
    )
        external
        view
        returns (VerificationResult memory result)
    {
        DocumentCertificate storage cert = certificates[_documentHash];
        
        // Check if certificate exists
        if (cert.issuer == address(0)) {
            return VerificationResult({
                isValid: false,
                exists: false,
                isRevoked: false,
                isExpired: false,
                issuer: address(0),
                verifiedAt: block.timestamp,
                reason: "Certificate does not exist"
            });
        }
        
        // Check if certificate is revoked
        if (cert.isRevoked) {
            return VerificationResult({
                isValid: false,
                exists: true,
                isRevoked: true,
                isExpired: false,
                issuer: cert.issuer,
                verifiedAt: block.timestamp,
                reason: "Certificate is revoked"
            });
        }
        
        // Check if certificate is expired
        if (block.timestamp > cert.expiresAt) {
            return VerificationResult({
                isValid: false,
                exists: true,
                isRevoked: false,
                isExpired: true,
                issuer: cert.issuer,
                verifiedAt: block.timestamp,
                reason: "Certificate is expired"
            });
        }
        
        // Verify merkle proof
        bool merkleValid = _verifyMerkleProof(_documentHash, cert.merkleRoot, _merkleProof);
        
        return VerificationResult({
            isValid: merkleValid,
            exists: true,
            isRevoked: false,
            isExpired: false,
            issuer: cert.issuer,
            verifiedAt: block.timestamp,
            reason: merkleValid ? "Valid certificate" : "Invalid merkle proof"
        });
    }
    
    /**
     * @dev Verify merkle proof for document integrity
     * @param _documentHash Document hash
     * @param _merkleRoot Merkle root
     * @param _proof Merkle proof
     * @return isValid True if proof is valid
     */
    function _verifyMerkleProof(
        bytes32 _documentHash,
        bytes32 _merkleRoot,
        bytes32[] memory _proof
    )
        internal
        pure
        returns (bool isValid)
    {
        bytes32 computedHash = _documentHash;
        
        for (uint256 i = 0; i < _proof.length; i++) {
            bytes32 proofElement = _proof[i];
            
            if (computedHash <= proofElement) {
                computedHash = keccak256(abi.encodePacked(computedHash, proofElement));
            } else {
                computedHash = keccak256(abi.encodePacked(proofElement, computedHash));
            }
        }
        
        return computedHash == _merkleRoot;
    }
    
    /**
     * @dev Revoke a certificate
     * @param _documentHash Hash of the document
     * @param _reason Reason for revocation
     */
    function revokeCertificate(
        bytes32 _documentHash,
        string memory _reason
    )
        external
        certificateExists(_documentHash)
        certificateNotRevoked(_documentHash)
        nonReentrant
    {
        DocumentCertificate storage cert = certificates[_documentHash];
        
        // Only issuer or owner can revoke
        require(
            msg.sender == cert.issuer || msg.sender == owner(),
            "Not authorized to revoke"
        );
        
        cert.isRevoked = true;
        
        emit CertificateRevoked(_documentHash, msg.sender, block.timestamp, _reason);
    }
    
    /**
     * @dev Add trusted issuer
     * @param _issuer Issuer address
     * @param _organizationName Organization name
     * @param _publicKey Public key for verification
     */
    function addTrustedIssuer(
        address _issuer,
        string memory _organizationName,
        string memory _publicKey
    )
        external
        onlyOwner
    {
        require(_issuer != address(0), "Invalid issuer address");
        require(bytes(_organizationName).length > 0, "Organization name cannot be empty");
        require(!trustedIssuers[_issuer].isActive, "Issuer already exists");
        
        TrustedIssuer storage issuer = trustedIssuers[_issuer];
        issuer.issuerAddress = _issuer;
        issuer.organizationName = _organizationName;
        issuer.isActive = true;
        issuer.addedAt = block.timestamp;
        issuer.publicKey = _publicKey;
        
        emit TrustedIssuerAdded(_issuer, _organizationName, block.timestamp);
    }
    
    /**
     * @dev Remove trusted issuer
     * @param _issuer Issuer address to remove
     */
    function removeTrustedIssuer(address _issuer) external onlyOwner {
        require(trustedIssuers[_issuer].isActive, "Issuer does not exist");
        
        trustedIssuers[_issuer].isActive = false;
        
        emit TrustedIssuerRemoved(_issuer, block.timestamp);
    }
    
    /**
     * @dev Create new document version
     * @param _originalHash Original document hash
     * @param _newHash New version hash
     * @param _merkleRoot Merkle root for new version
     * @param _certificateType Certificate type
     * @param _ipfsHash IPFS hash for new version
     */
    function createDocumentVersion(
        bytes32 _originalHash,
        bytes32 _newHash,
        bytes32 _merkleRoot,
        string memory _certificateType,
        string memory _ipfsHash
    )
        external
        onlyTrustedIssuer
        certificateExists(_originalHash)
        nonReentrant
    {
        require(certificates[_newHash].issuer == address(0), "New version already exists");
        
        // Issue certificate for new version
        DocumentCertificate storage newCert = certificates[_newHash];
        newCert.documentHash = _newHash;
        newCert.merkleRoot = _merkleRoot;
        newCert.issuer = msg.sender;
        newCert.issuedAt = block.timestamp;
        newCert.expiresAt = block.timestamp + defaultExpiryPeriod;
        newCert.certificateType = _certificateType;
        newCert.isRevoked = false;
        newCert.ipfsHash = _ipfsHash;
        
        // Link versions
        documentVersions[_originalHash].push(_newHash);
        
        certificateCount++;
        
        emit CertificateIssued(
            _newHash,
            msg.sender,
            block.timestamp,
            newCert.expiresAt,
            _certificateType
        );
    }
    
    /**
     * @dev Get document versions
     * @param _documentHash Original document hash
     * @return versions Array of version hashes
     */
    function getDocumentVersions(bytes32 _documentHash)
        external
        view
        returns (bytes32[] memory versions)
    {
        return documentVersions[_documentHash];
    }
    
    /**
     * @dev Batch verify multiple documents
     * @param _documentHashes Array of document hashes
     * @param _merkleProofs Array of merkle proofs
     * @return results Array of verification results
     */
    function batchVerifyDocuments(
        bytes32[] memory _documentHashes,
        bytes32[][] memory _merkleProofs
    )
        external
        view
        returns (VerificationResult[] memory results)
    {
        require(_documentHashes.length == _merkleProofs.length, "Array length mismatch");
        
        results = new VerificationResult[](_documentHashes.length);
        
        for (uint256 i = 0; i < _documentHashes.length; i++) {
            results[i] = this.verifyDocument(_documentHashes[i], _merkleProofs[i]);
        }
        
        return results;
    }
    
    /**
     * @dev Check if issuer is trusted
     * @param _issuer Issuer address
     * @return isTrusted True if issuer is trusted
     */
    function isTrustedIssuer(address _issuer) external view returns (bool isTrusted) {
        return trustedIssuers[_issuer].isActive;
    }
    
    /**
     * @dev Get certificate details
     * @param _documentHash Document hash
     * @return certificate Full certificate details
     */
    function getCertificate(bytes32 _documentHash)
        external
        view
        certificateExists(_documentHash)
        returns (DocumentCertificate memory certificate)
    {
        return certificates[_documentHash];
    }
    
    /**
     * @dev Update default expiry period
     * @param _newExpiryPeriod New expiry period in seconds
     */
    function updateDefaultExpiryPeriod(uint256 _newExpiryPeriod) external onlyOwner {
        require(_newExpiryPeriod > 0, "Expiry period must be greater than 0");
        defaultExpiryPeriod = _newExpiryPeriod;
    }
    
    /**
     * @dev Get contract statistics
     * @return totalCertificates Total number of certificates
     * @return activeCertificates Number of active certificates
     * @return revokedCertificates Number of revoked certificates
     */
    function getContractStats()
        external
        view
        returns (
            uint256 totalCertificates,
            uint256 activeCertificates,
            uint256 revokedCertificates
        )
    {
        // Note: This is a simplified version. In production, maintain counters.
        return (certificateCount, 0, 0);
    }
    
    /**
     * @dev Verify document signature
     * @param _documentHash Document hash
     * @param _signature Digital signature
     * @param _signerAddress Expected signer address
     * @return isValid True if signature is valid
     */
    function verifySignature(
        bytes32 _documentHash,
        bytes memory _signature,
        address _signerAddress
    )
        external
        pure
        returns (bool isValid)
    {
        bytes32 messageHash = _documentHash.toEthSignedMessageHash();
        address recoveredSigner = messageHash.recover(_signature);
        return recoveredSigner == _signerAddress;
    }
}