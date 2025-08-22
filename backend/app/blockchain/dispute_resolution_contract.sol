// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title DisputeResolutionContract
 * @dev Smart contract for automated dispute resolution and arbitration
 * @author Blockchain Audit Trail System
 */
contract DisputeResolutionContract is AccessControl, ReentrancyGuard, Pausable {
    using Counters for Counters.Counter;
    
    // Role definitions
    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");
    bytes32 public constant MEDIATOR_ROLE = keccak256("MEDIATOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // Enums
    enum DisputeStatus {
        PENDING,
        IN_REVIEW,
        EVIDENCE_SUBMISSION,
        ARBITRATION,
        MEDIATION,
        RESOLVED,
        CANCELLED,
        APPEALED
    }
    
    enum DisputeType {
        DOCUMENT_AUTHENTICITY,
        ANALYSIS_ACCURACY,
        COMPLIANCE_INTERPRETATION,
        PROCEDURAL_ERROR,
        DATA_INTEGRITY,
        ACCESS_VIOLATION
    }
    
    enum Resolution {
        NONE,
        UPHELD,
        OVERTURNED,
        PARTIAL_UPHELD,
        REMANDED,
        SETTLEMENT
    }
    
    // Structures
    struct Dispute {
        uint256 disputeId;
        bytes32 documentHash;
        address complainant;
        address respondent;
        DisputeType disputeType;
        DisputeStatus status;
        string description;
        string evidence;
        uint256 createdAt;
        uint256 deadlineAt;
        address assignedArbitrator;
        address assignedMediator;
        Resolution resolution;
        string resolutionDetails;
        uint256 resolvedAt;
        uint256 stakingAmount;
        bool isAppealed;
    }
    
    struct Evidence {
        uint256 disputeId;
        address submitter;
        string evidenceType;
        string evidenceHash;
        string description;
        uint256 submittedAt;
        bool isVerified;
    }
    
    struct ArbitratorProfile {
        address arbitratorAddress;
        string name;
        string specialization;
        uint256 casesHandled;
        uint256 successRate;
        bool isActive;
        uint256 stakingRequired;
        string certificationHash;
    }
    
    struct Settlement {
        uint256 disputeId;
        address proposer;
        string terms;
        uint256 proposedAt;
        uint256 expiresAt;
        bool isAccepted;
        address[] signatories;
        mapping(address => bool) hasAccepted;
    }
    
    // State variables
    Counters.Counter private _disputeIds;
    Counters.Counter private _evidenceIds;
    
    mapping(uint256 => Dispute) public disputes;
    mapping(uint256 => Evidence[]) public disputeEvidence;
    mapping(address => ArbitratorProfile) public arbitrators;
    mapping(uint256 => Settlement) public settlements;
    mapping(address => uint256[]) public userDisputes;
    mapping(bytes32 => uint256[]) public documentDisputes;
    
    IERC20 public stakingToken;
    uint256 public baseStakingAmount = 1000 * 10**18; // 1000 tokens
    uint256 public arbitrationFee = 100 * 10**18; // 100 tokens
    uint256 public mediationFee = 50 * 10**18; // 50 tokens
    uint256 public evidenceSubmissionPeriod = 7 days;
    uint256 public arbitrationPeriod = 14 days;
    uint256 public appealPeriod = 3 days;
    
    // Events
    event DisputeCreated(
        uint256 indexed disputeId,
        bytes32 indexed documentHash,
        address indexed complainant,
        address respondent,
        DisputeType disputeType
    );
    
    event EvidenceSubmitted(
        uint256 indexed disputeId,
        uint256 indexed evidenceId,
        address indexed submitter,
        string evidenceType
    );
    
    event ArbitratorAssigned(
        uint256 indexed disputeId,
        address indexed arbitrator
    );
    
    event DisputeResolved(
        uint256 indexed disputeId,
        Resolution resolution,
        address indexed resolver,
        uint256 resolvedAt
    );
    
    event SettlementProposed(
        uint256 indexed disputeId,
        address indexed proposer,
        uint256 expiresAt
    );
    
    event SettlementAccepted(
        uint256 indexed disputeId,
        address indexed acceptor
    );
    
    event DisputeAppealed(
        uint256 indexed disputeId,
        address indexed appellant,
        uint256 appealedAt
    );
    
    event ArbitratorRegistered(
        address indexed arbitrator,
        string name,
        string specialization
    );
    
    // Modifiers
    modifier onlyArbitrator() {
        require(hasRole(ARBITRATOR_ROLE, msg.sender), "Caller is not an arbitrator");
        _;
    }
    
    modifier onlyMediator() {
        require(hasRole(MEDIATOR_ROLE, msg.sender), "Caller is not a mediator");
        _;
    }
    
    modifier onlyAdmin() {
        require(hasRole(ADMIN_ROLE, msg.sender), "Caller is not an admin");
        _;
    }
    
    modifier disputeExists(uint256 _disputeId) {
        require(_disputeId <= _disputeIds.current() && _disputeId > 0, "Dispute does not exist");
        _;
    }
    
    modifier onlyDisputeParty(uint256 _disputeId) {
        Dispute storage dispute = disputes[_disputeId];
        require(
            msg.sender == dispute.complainant || msg.sender == dispute.respondent,
            "Not a dispute party"
        );
        _;
    }
    
    /**
     * @dev Constructor
     * @param _stakingToken ERC20 token for staking
     * @param _admin Admin address
     */
    constructor(
        address _stakingToken,
        address _admin
    ) {
        stakingToken = IERC20(_stakingToken);
        
        // Setup roles
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(ADMIN_ROLE, _admin);
    }
    
    /**
     * @dev Create a new dispute
     * @param _documentHash Hash of the disputed document
     * @param _respondent Address of the respondent
     * @param _disputeType Type of dispute
     * @param _description Description of the dispute
     */
    function createDispute(
        bytes32 _documentHash,
        address _respondent,
        DisputeType _disputeType,
        string memory _description
    )
        external
        whenNotPaused
        nonReentrant
        returns (uint256 disputeId)
    {
        require(_respondent != address(0), "Invalid respondent address");
        require(_respondent != msg.sender, "Cannot dispute with yourself");
        require(bytes(_description).length > 0, "Description cannot be empty");
        
        // Transfer staking amount
        require(
            stakingToken.transferFrom(msg.sender, address(this), baseStakingAmount),
            "Staking transfer failed"
        );
        
        _disputeIds.increment();
        disputeId = _disputeIds.current();
        
        Dispute storage dispute = disputes[disputeId];
        dispute.disputeId = disputeId;
        dispute.documentHash = _documentHash;
        dispute.complainant = msg.sender;
        dispute.respondent = _respondent;
        dispute.disputeType = _disputeType;
        dispute.status = DisputeStatus.PENDING;
        dispute.description = _description;
        dispute.createdAt = block.timestamp;
        dispute.deadlineAt = block.timestamp + evidenceSubmissionPeriod;
        dispute.stakingAmount = baseStakingAmount;
        dispute.resolution = Resolution.NONE;
        
        // Track disputes
        userDisputes[msg.sender].push(disputeId);
        userDisputes[_respondent].push(disputeId);
        documentDisputes[_documentHash].push(disputeId);
        
        emit DisputeCreated(
            disputeId,
            _documentHash,
            msg.sender,
            _respondent,
            _disputeType
        );
        
        return disputeId;
    }
    
    /**
     * @dev Submit evidence for a dispute
     * @param _disputeId ID of the dispute
     * @param _evidenceType Type of evidence
     * @param _evidenceHash Hash of the evidence document
     * @param _description Description of the evidence
     */
    function submitEvidence(
        uint256 _disputeId,
        string memory _evidenceType,
        string memory _evidenceHash,
        string memory _description
    )
        external
        disputeExists(_disputeId)
        onlyDisputeParty(_disputeId)
        whenNotPaused
        nonReentrant
    {
        Dispute storage dispute = disputes[_disputeId];
        require(
            dispute.status == DisputeStatus.PENDING || 
            dispute.status == DisputeStatus.EVIDENCE_SUBMISSION,
            "Cannot submit evidence at this stage"
        );
        require(block.timestamp <= dispute.deadlineAt, "Evidence submission period expired");
        
        _evidenceIds.increment();
        uint256 evidenceId = _evidenceIds.current();
        
        Evidence memory evidence = Evidence({
            disputeId: _disputeId,
            submitter: msg.sender,
            evidenceType: _evidenceType,
            evidenceHash: _evidenceHash,
            description: _description,
            submittedAt: block.timestamp,
            isVerified: false
        });
        
        disputeEvidence[_disputeId].push(evidence);
        
        // Update dispute status
        if (dispute.status == DisputeStatus.PENDING) {
            dispute.status = DisputeStatus.EVIDENCE_SUBMISSION;
        }
        
        emit EvidenceSubmitted(_disputeId, evidenceId, msg.sender, _evidenceType);
    }
    
    /**
     * @dev Assign arbitrator to a dispute
     * @param _disputeId ID of the dispute
     * @param _arbitrator Address of the arbitrator
     */
    function assignArbitrator(
        uint256 _disputeId,
        address _arbitrator
    )
        external
        onlyAdmin
        disputeExists(_disputeId)
        whenNotPaused
    {
        require(arbitrators[_arbitrator].isActive, "Arbitrator is not active");
        
        Dispute storage dispute = disputes[_disputeId];
        require(
            dispute.status == DisputeStatus.EVIDENCE_SUBMISSION ||
            dispute.status == DisputeStatus.PENDING,
            "Cannot assign arbitrator at this stage"
        );
        
        dispute.assignedArbitrator = _arbitrator;
        dispute.status = DisputeStatus.ARBITRATION;
        dispute.deadlineAt = block.timestamp + arbitrationPeriod;
        
        emit ArbitratorAssigned(_disputeId, _arbitrator);
    }
    
    /**
     * @dev Resolve a dispute (arbitrator only)
     * @param _disputeId ID of the dispute
     * @param _resolution Resolution decision
     * @param _resolutionDetails Details of the resolution
     */
    function resolveDispute(
        uint256 _disputeId,
        Resolution _resolution,
        string memory _resolutionDetails
    )
        external
        disputeExists(_disputeId)
        whenNotPaused
        nonReentrant
    {
        Dispute storage dispute = disputes[_disputeId];
        require(
            msg.sender == dispute.assignedArbitrator || hasRole(ADMIN_ROLE, msg.sender),
            "Not authorized to resolve dispute"
        );
        require(dispute.status == DisputeStatus.ARBITRATION, "Dispute not in arbitration");
        require(_resolution != Resolution.NONE, "Invalid resolution");
        
        dispute.resolution = _resolution;
        dispute.resolutionDetails = _resolutionDetails;
        dispute.status = DisputeStatus.RESOLVED;
        dispute.resolvedAt = block.timestamp;
        
        // Handle staking rewards/penalties
        _distributeStaking(_disputeId, _resolution);
        
        // Update arbitrator stats
        arbitrators[dispute.assignedArbitrator].casesHandled++;
        
        emit DisputeResolved(_disputeId, _resolution, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Propose settlement for a dispute
     * @param _disputeId ID of the dispute
     * @param _terms Settlement terms
     * @param _expiryPeriod Expiry period for the settlement
     */
    function proposeSettlement(
        uint256 _disputeId,
        string memory _terms,
        uint256 _expiryPeriod
    )
        external
        disputeExists(_disputeId)
        onlyDisputeParty(_disputeId)
        whenNotPaused
    {
        Dispute storage dispute = disputes[_disputeId];
        require(
            dispute.status == DisputeStatus.PENDING ||
            dispute.status == DisputeStatus.EVIDENCE_SUBMISSION ||
            dispute.status == DisputeStatus.MEDIATION,
            "Cannot propose settlement at this stage"
        );
        
        Settlement storage settlement = settlements[_disputeId];
        settlement.disputeId = _disputeId;
        settlement.proposer = msg.sender;
        settlement.terms = _terms;
        settlement.proposedAt = block.timestamp;
        settlement.expiresAt = block.timestamp + _expiryPeriod;
        settlement.isAccepted = false;
        
        emit SettlementProposed(_disputeId, msg.sender, settlement.expiresAt);
    }
    
    /**
     * @dev Accept a settlement proposal
     * @param _disputeId ID of the dispute
     */
    function acceptSettlement(uint256 _disputeId)
        external
        disputeExists(_disputeId)
        onlyDisputeParty(_disputeId)
        whenNotPaused
        nonReentrant
    {
        Settlement storage settlement = settlements[_disputeId];
        require(settlement.proposer != address(0), "No settlement proposed");
        require(settlement.proposer != msg.sender, "Cannot accept own proposal");
        require(block.timestamp <= settlement.expiresAt, "Settlement expired");
        require(!settlement.hasAccepted[msg.sender], "Already accepted");
        
        settlement.hasAccepted[msg.sender] = true;
        settlement.signatories.push(msg.sender);
        
        // If both parties accepted, finalize settlement
        Dispute storage dispute = disputes[_disputeId];
        if (settlement.hasAccepted[dispute.complainant] && 
            settlement.hasAccepted[dispute.respondent]) {
            
            settlement.isAccepted = true;
            dispute.status = DisputeStatus.RESOLVED;
            dispute.resolution = Resolution.SETTLEMENT;
            dispute.resolutionDetails = settlement.terms;
            dispute.resolvedAt = block.timestamp;
            
            // Return staking amounts
            stakingToken.transfer(dispute.complainant, dispute.stakingAmount / 2);
            stakingToken.transfer(dispute.respondent, dispute.stakingAmount / 2);
        }
        
        emit SettlementAccepted(_disputeId, msg.sender);
    }
    
    /**
     * @dev Appeal a dispute resolution
     * @param _disputeId ID of the dispute
     * @param _appealReason Reason for appeal
     */
    function appealDispute(
        uint256 _disputeId,
        string memory _appealReason
    )
        external
        disputeExists(_disputeId)
        onlyDisputeParty(_disputeId)
        whenNotPaused
        nonReentrant
    {
        Dispute storage dispute = disputes[_disputeId];
        require(dispute.status == DisputeStatus.RESOLVED, "Dispute not resolved");
        require(!dispute.isAppealed, "Already appealed");
        require(
            block.timestamp <= dispute.resolvedAt + appealPeriod,
            "Appeal period expired"
        );
        
        // Additional staking for appeal
        require(
            stakingToken.transferFrom(msg.sender, address(this), baseStakingAmount),
            "Appeal staking transfer failed"
        );
        
        dispute.isAppealed = true;
        dispute.status = DisputeStatus.APPEALED;
        dispute.stakingAmount += baseStakingAmount;
        
        emit DisputeAppealed(_disputeId, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Register as an arbitrator
     * @param _name Arbitrator name
     * @param _specialization Area of specialization
     * @param _certificationHash Hash of certification documents
     */
    function registerArbitrator(
        string memory _name,
        string memory _specialization,
        string memory _certificationHash
    )
        external
        whenNotPaused
        nonReentrant
    {
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(!arbitrators[msg.sender].isActive, "Already registered");
        
        // Stake tokens to become arbitrator
        require(
            stakingToken.transferFrom(msg.sender, address(this), baseStakingAmount * 5),
            "Arbitrator staking failed"
        );
        
        ArbitratorProfile storage profile = arbitrators[msg.sender];
        profile.arbitratorAddress = msg.sender;
        profile.name = _name;
        profile.specialization = _specialization;
        profile.casesHandled = 0;
        profile.successRate = 0;
        profile.isActive = true;
        profile.stakingRequired = baseStakingAmount * 5;
        profile.certificationHash = _certificationHash;
        
        // Grant arbitrator role
        _grantRole(ARBITRATOR_ROLE, msg.sender);
        
        emit ArbitratorRegistered(msg.sender, _name, _specialization);
    }
    
    /**
     * @dev Distribute staking based on resolution
     * @param _disputeId ID of the dispute
     * @param _resolution Resolution decision
     */
    function _distributeStaking(uint256 _disputeId, Resolution _resolution) internal {
        Dispute storage dispute = disputes[_disputeId];
        uint256 amount = dispute.stakingAmount;
        
        if (_resolution == Resolution.UPHELD) {
            // Complainant wins, gets staking back + penalty from respondent
            stakingToken.transfer(dispute.complainant, amount);
        } else if (_resolution == Resolution.OVERTURNED) {
            // Respondent wins, gets staking back + penalty from complainant
            stakingToken.transfer(dispute.respondent, amount);
        } else if (_resolution == Resolution.PARTIAL_UPHELD) {
            // Split staking
            stakingToken.transfer(dispute.complainant, amount * 60 / 100);
            stakingToken.transfer(dispute.respondent, amount * 40 / 100);
        } else {
            // Other resolutions - return to both parties
            stakingToken.transfer(dispute.complainant, amount / 2);
            stakingToken.transfer(dispute.respondent, amount / 2);
        }
    }
    
    /**
     * @dev Get dispute details
     * @param _disputeId ID of the dispute
     * @return dispute Dispute information
     */
    function getDispute(uint256 _disputeId)
        external
        view
        disputeExists(_disputeId)
        returns (Dispute memory dispute)
    {
        return disputes[_disputeId];
    }
    
    /**
     * @dev Get evidence for a dispute
     * @param _disputeId ID of the dispute
     * @return evidence Array of evidence
     */
    function getDisputeEvidence(uint256 _disputeId)
        external
        view
        disputeExists(_disputeId)
        returns (Evidence[] memory evidence)
    {
        return disputeEvidence[_disputeId];
    }
    
    /**
     * @dev Get user's disputes
     * @param _user User address
     * @return disputeIds Array of dispute IDs
     */
    function getUserDisputes(address _user)
        external
        view
        returns (uint256[] memory disputeIds)
    {
        return userDisputes[_user];
    }
    
    /**
     * @dev Get disputes for a document
     * @param _documentHash Document hash
     * @return disputeIds Array of dispute IDs
     */
    function getDocumentDisputes(bytes32 _documentHash)
        external
        view
        returns (uint256[] memory disputeIds)
    {
        return documentDisputes[_documentHash];
    }
    
    /**
     * @dev Update contract parameters
     * @param _baseStakingAmount New base staking amount
     * @param _arbitrationFee New arbitration fee
     * @param _evidenceSubmissionPeriod New evidence submission period
     */
    function updateParameters(
        uint256 _baseStakingAmount,
        uint256 _arbitrationFee,
        uint256 _evidenceSubmissionPeriod
    )
        external
        onlyAdmin
    {
        baseStakingAmount = _baseStakingAmount;
        arbitrationFee = _arbitrationFee;
        evidenceSubmissionPeriod = _evidenceSubmissionPeriod;
    }
    
    /**
     * @dev Emergency pause
     */
    function emergencyPause() external onlyAdmin {
        _pause();
    }
    
    /**
     * @dev Emergency unpause
     */
    function emergencyUnpause() external onlyAdmin {
        _unpause();
    }
}