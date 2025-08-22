// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title AuditTrailContract
 * @dev Smart contract for immutable audit trail management with multi-signature support
 * @author Blockchain Audit Trail System
 */
contract AuditTrailContract is AccessControl, ReentrancyGuard, Pausable {
    using Counters for Counters.Counter;
    
    // Role definitions
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");
    bytes32 public constant COMPLIANCE_OFFICER_ROLE = keccak256("COMPLIANCE_OFFICER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // Counter for audit records
    Counters.Counter private _auditRecordIds;
    
    // Structures
    struct AuditRecord {
        bytes32 documentHash;
        string analysisResult;
        string decisionType;
        string complianceStatus;
        string metadata;
        uint256 timestamp;
        address submitter;
        uint256 blockNumber;
        bool isActive;
    }
    
    struct MultiSigApproval {
        bytes32 documentHash;
        address[] approvers;
        uint256 requiredSignatures;
        uint256 currentSignatures;
        mapping(address => bool) hasApproved;
        bool isExecuted;
        uint256 deadline;
    }
    
    struct ComplianceRule {
        string ruleName;
        string ruleDescription;
        bool isActive;
        uint256 createdAt;
        address createdBy;
    }
    
    // State variables
    mapping(bytes32 => AuditRecord) public auditRecords;
    mapping(bytes32 => bool) public documentExists;
    mapping(uint256 => MultiSigApproval) public multiSigApprovals;
    mapping(bytes32 => ComplianceRule) public complianceRules;
    mapping(address => uint256) public userAuditCounts;
    
    uint256 public totalAuditRecords;
    uint256 public requiredApprovals = 2;
    uint256 public approvalDeadline = 7 days;
    
    // Events
    event AuditRecordStored(
        bytes32 indexed documentHash,
        address indexed submitter,
        string analysisResult,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    event MultiSigApprovalCreated(
        uint256 indexed approvalId,
        bytes32 indexed documentHash,
        address indexed creator,
        uint256 requiredSignatures,
        uint256 deadline
    );
    
    event MultiSigApprovalSigned(
        uint256 indexed approvalId,
        bytes32 indexed documentHash,
        address indexed signer,
        uint256 currentSignatures
    );
    
    event MultiSigApprovalExecuted(
        uint256 indexed approvalId,
        bytes32 indexed documentHash,
        uint256 executedAt
    );
    
    event ComplianceRuleAdded(
        bytes32 indexed ruleId,
        string ruleName,
        address indexed creator
    );
    
    event AuditRecordUpdated(
        bytes32 indexed documentHash,
        address indexed updater,
        string newComplianceStatus
    );
    
    event EmergencyPaused(address indexed pauser, uint256 timestamp);
    event EmergencyUnpaused(address indexed unpauser, uint256 timestamp);
    
    // Modifiers
    modifier onlyAuditor() {
        require(hasRole(AUDITOR_ROLE, msg.sender), "Caller is not an auditor");
        _;
    }
    
    modifier onlyComplianceOfficer() {
        require(hasRole(COMPLIANCE_OFFICER_ROLE, msg.sender), "Caller is not a compliance officer");
        _;
    }
    
    modifier onlyAdmin() {
        require(hasRole(ADMIN_ROLE, msg.sender), "Caller is not an admin");
        _;
    }
    
    modifier documentNotExists(bytes32 _documentHash) {
        require(!documentExists[_documentHash], "Document already exists");
        _;
    }
    
    modifier documentExistsCheck(bytes32 _documentHash) {
        require(documentExists[_documentHash], "Document does not exist");
        _;
    }
    
    /**
     * @dev Constructor sets up roles and initial configuration
     * @param _admin Admin address
     * @param _auditors Array of auditor addresses
     * @param _complianceOfficers Array of compliance officer addresses
     */
    constructor(
        address _admin,
        address[] memory _auditors,
        address[] memory _complianceOfficers
    ) {
        // Setup roles
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(ADMIN_ROLE, _admin);
        
        // Grant auditor roles
        for (uint256 i = 0; i < _auditors.length; i++) {
            _grantRole(AUDITOR_ROLE, _auditors[i]);
        }
        
        // Grant compliance officer roles
        for (uint256 i = 0; i < _complianceOfficers.length; i++) {
            _grantRole(COMPLIANCE_OFFICER_ROLE, _complianceOfficers[i]);
        }
    }
    
    /**
     * @dev Store audit record on blockchain
     * @param _documentHash Hash of the document
     * @param _analysisResult JSON string of analysis results
     * @param _decisionType Type of decision made
     * @param _complianceStatus Compliance status
     * @param _metadata Additional metadata
     */
    function storeAuditRecord(
        bytes32 _documentHash,
        string memory _analysisResult,
        string memory _decisionType,
        string memory _complianceStatus,
        string memory _metadata
    ) 
        external 
        onlyAuditor 
        whenNotPaused 
        documentNotExists(_documentHash) 
        nonReentrant 
    {
        AuditRecord storage record = auditRecords[_documentHash];
        record.documentHash = _documentHash;
        record.analysisResult = _analysisResult;
        record.decisionType = _decisionType;
        record.complianceStatus = _complianceStatus;
        record.metadata = _metadata;
        record.timestamp = block.timestamp;
        record.submitter = msg.sender;
        record.blockNumber = block.number;
        record.isActive = true;
        
        documentExists[_documentHash] = true;
        totalAuditRecords++;
        userAuditCounts[msg.sender]++;
        
        emit AuditRecordStored(
            _documentHash,
            msg.sender,
            _analysisResult,
            block.timestamp,
            block.number
        );
    }
    
    /**
     * @dev Get audit record by document hash
     * @param _documentHash Hash of the document
     * @return All audit record details
     */
    function getAuditRecord(bytes32 _documentHash)
        external
        view
        documentExistsCheck(_documentHash)
        returns (
            string memory analysisResult,
            string memory decisionType,
            string memory complianceStatus,
            string memory metadata,
            uint256 timestamp,
            address submitter,
            uint256 blockNumber,
            bool isActive
        )
    {
        AuditRecord storage record = auditRecords[_documentHash];
        return (
            record.analysisResult,
            record.decisionType,
            record.complianceStatus,
            record.metadata,
            record.timestamp,
            record.submitter,
            record.blockNumber,
            record.isActive
        );
    }
    
    /**
     * @dev Verify if document exists in audit trail
     * @param _documentHash Hash of the document
     * @return exists True if document exists
     */
    function verifyDocument(bytes32 _documentHash)
        external
        view
        returns (bool exists)
    {
        return documentExists[_documentHash];
    }
    
    /**
     * @dev Get total number of audit records
     * @return count Total audit record count
     */
    function getAuditRecordCount() external view returns (uint256 count) {
        return totalAuditRecords;
    }
    
    /**
     * @dev Create multi-signature approval for sensitive operations
     * @param _documentHash Document hash requiring approval
     * @param _requiredSignatures Number of required signatures
     */
    function createMultiSigApproval(
        bytes32 _documentHash,
        uint256 _requiredSignatures
    )
        external
        onlyComplianceOfficer
        whenNotPaused
        returns (uint256 approvalId)
    {
        require(_requiredSignatures > 0, "Required signatures must be greater than 0");
        require(_requiredSignatures <= getRoleMemberCount(COMPLIANCE_OFFICER_ROLE), 
               "Required signatures exceeds available officers");
        
        _auditRecordIds.increment();
        approvalId = _auditRecordIds.current();
        
        MultiSigApproval storage approval = multiSigApprovals[approvalId];
        approval.documentHash = _documentHash;
        approval.requiredSignatures = _requiredSignatures;
        approval.currentSignatures = 0;
        approval.isExecuted = false;
        approval.deadline = block.timestamp + approvalDeadline;
        
        emit MultiSigApprovalCreated(
            approvalId,
            _documentHash,
            msg.sender,
            _requiredSignatures,
            approval.deadline
        );
        
        return approvalId;
    }
    
    /**
     * @dev Sign multi-signature approval
     * @param _approvalId ID of the approval to sign
     */
    function signMultiSigApproval(uint256 _approvalId)
        external
        onlyComplianceOfficer
        whenNotPaused
        nonReentrant
    {
        MultiSigApproval storage approval = multiSigApprovals[_approvalId];
        
        require(!approval.isExecuted, "Approval already executed");
        require(block.timestamp <= approval.deadline, "Approval deadline passed");
        require(!approval.hasApproved[msg.sender], "Already approved by sender");
        
        approval.hasApproved[msg.sender] = true;
        approval.currentSignatures++;
        approval.approvers.push(msg.sender);
        
        emit MultiSigApprovalSigned(
            _approvalId,
            approval.documentHash,
            msg.sender,
            approval.currentSignatures
        );
        
        // Auto-execute if required signatures reached
        if (approval.currentSignatures >= approval.requiredSignatures) {
            _executeMultiSigApproval(_approvalId);
        }
    }
    
    /**
     * @dev Execute multi-signature approval
     * @param _approvalId ID of the approval to execute
     */
    function _executeMultiSigApproval(uint256 _approvalId) internal {
        MultiSigApproval storage approval = multiSigApprovals[_approvalId];
        
        require(approval.currentSignatures >= approval.requiredSignatures, 
               "Insufficient signatures");
        require(!approval.isExecuted, "Already executed");
        
        approval.isExecuted = true;
        
        emit MultiSigApprovalExecuted(
            _approvalId,
            approval.documentHash,
            block.timestamp
        );
    }
    
    /**
     * @dev Add compliance rule
     * @param _ruleId Unique identifier for the rule
     * @param _ruleName Name of the compliance rule
     * @param _ruleDescription Description of the rule
     */
    function addComplianceRule(
        bytes32 _ruleId,
        string memory _ruleName,
        string memory _ruleDescription
    )
        external
        onlyComplianceOfficer
        whenNotPaused
    {
        require(bytes(_ruleName).length > 0, "Rule name cannot be empty");
        require(!complianceRules[_ruleId].isActive, "Rule already exists");
        
        ComplianceRule storage rule = complianceRules[_ruleId];
        rule.ruleName = _ruleName;
        rule.ruleDescription = _ruleDescription;
        rule.isActive = true;
        rule.createdAt = block.timestamp;
        rule.createdBy = msg.sender;
        
        emit ComplianceRuleAdded(_ruleId, _ruleName, msg.sender);
    }
    
    /**
     * @dev Update audit record compliance status (requires multi-sig)
     * @param _documentHash Document hash to update
     * @param _newComplianceStatus New compliance status
     * @param _approvalId Multi-sig approval ID
     */
    function updateAuditRecordCompliance(
        bytes32 _documentHash,
        string memory _newComplianceStatus,
        uint256 _approvalId
    )
        external
        onlyComplianceOfficer
        whenNotPaused
        documentExistsCheck(_documentHash)
        nonReentrant
    {
        MultiSigApproval storage approval = multiSigApprovals[_approvalId];
        
        require(approval.isExecuted, "Multi-sig approval not executed");
        require(approval.documentHash == _documentHash, "Document hash mismatch");
        
        auditRecords[_documentHash].complianceStatus = _newComplianceStatus;
        
        emit AuditRecordUpdated(_documentHash, msg.sender, _newComplianceStatus);
    }
    
    /**
     * @dev Deactivate audit record (emergency function)
     * @param _documentHash Document hash to deactivate
     */
    function deactivateAuditRecord(bytes32 _documentHash)
        external
        onlyAdmin
        whenNotPaused
        documentExistsCheck(_documentHash)
    {
        auditRecords[_documentHash].isActive = false;
    }
    
    /**
     * @dev Get user audit statistics
     * @param _user User address
     * @return auditCount Number of audits performed by user
     */
    function getUserAuditStats(address _user)
        external
        view
        returns (uint256 auditCount)
    {
        return userAuditCounts[_user];
    }
    
    /**
     * @dev Batch verify multiple documents
     * @param _documentHashes Array of document hashes
     * @return results Array of verification results
     */
    function batchVerifyDocuments(bytes32[] memory _documentHashes)
        external
        view
        returns (bool[] memory results)
    {
        results = new bool[](_documentHashes.length);
        
        for (uint256 i = 0; i < _documentHashes.length; i++) {
            results[i] = documentExists[_documentHashes[i]];
        }
        
        return results;
    }
    
    /**
     * @dev Emergency pause function
     */
    function emergencyPause() external onlyAdmin {
        _pause();
        emit EmergencyPaused(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Emergency unpause function
     */
    function emergencyUnpause() external onlyAdmin {
        _unpause();
        emit EmergencyUnpaused(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Update required approvals count
     * @param _newRequiredApprovals New number of required approvals
     */
    function updateRequiredApprovals(uint256 _newRequiredApprovals)
        external
        onlyAdmin
    {
        require(_newRequiredApprovals > 0, "Required approvals must be greater than 0");
        requiredApprovals = _newRequiredApprovals;
    }
    
    /**
     * @dev Update approval deadline
     * @param _newDeadline New deadline in seconds
     */
    function updateApprovalDeadline(uint256 _newDeadline)
        external
        onlyAdmin
    {
        require(_newDeadline > 0, "Deadline must be greater than 0");
        approvalDeadline = _newDeadline;
    }
    
    /**
     * @dev Get contract version and info
     * @return version Contract version
     * @return totalRecords Total audit records
     * @return activeRecords Active audit records
     */
    function getContractInfo()
        external
        view
        returns (
            string memory version,
            uint256 totalRecords,
            uint256 activeRecords
        )
    {
        // Count active records (this could be gas-intensive for large datasets)
        uint256 active = 0;
        // Note: In production, maintain a separate counter for active records
        
        return ("1.0.0", totalAuditRecords, active);
    }
    
    /**
     * @dev Check if address has specific role
     * @param role Role to check
     * @param account Account to check
     * @return hasRoleCheck True if account has role
     */
    function checkRole(bytes32 role, address account)
        external
        view
        returns (bool hasRoleCheck)
    {
        return hasRole(role, account);
    }
}