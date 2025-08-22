// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "./ArbitrationRegistry.sol";

/**
 * @title DisputeResolution
 * @dev Automated dispute handling system with multi-stage resolution
 */
contract DisputeResolution is AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");
    bytes32 public constant MEDIATOR_ROLE = keccak256("MEDIATOR_ROLE");

    enum DisputeStatus {
        Created,
        UnderReview,
        InMediation,
        InArbitration,
        Resolved,
        Appealed,
        Closed
    }

    enum DisputeType {
        Contract,
        Payment,
        Intellectual,
        Employment,
        Commercial,
        Consumer
    }

    struct Dispute {
        bytes32 id;
        address claimant;
        address respondent;
        DisputeType disputeType;
        DisputeStatus status;
        string description;
        string[] evidence;
        uint256 disputeAmount;
        uint256 createdAt;
        uint256 deadline;
        address assignedArbitrator;
        address assignedMediator;
        uint256 resolutionTime;
        bytes32 arbitrationClauseId;
        mapping(address => bool) hasVoted;
        mapping(address => string) statements;
    }

    struct Resolution {
        bytes32 disputeId;
        address resolver;
        string decision;
        uint256 awardAmount;
        address awardRecipient;
        uint256 resolvedAt;
        bool isAppealable;
        uint256 appealDeadline;
        string reasoning;
    }

    struct Appeal {
        bytes32 disputeId;
        address appellant;
        string reason;
        uint256 appealFee;
        uint256 createdAt;
        address[] appealPanelArbitrators;
        bool resolved;
    }

    ArbitrationRegistry public immutable arbitrationRegistry;
    
    mapping(bytes32 => Dispute) public disputes;
    mapping(bytes32 => Resolution) public resolutions;
    mapping(bytes32 => Appeal) public appeals;
    mapping(bytes32 => address[]) public disputeArbitrators;
    
    bytes32[] public allDisputeIds;

    uint256 public constant MEDIATION_DURATION = 7 days;
    uint256 public constant ARBITRATION_DURATION = 14 days;
    uint256 public constant APPEAL_WINDOW = 3 days;
    uint256 public constant MIN_DISPUTE_AMOUNT = 0.01 ether;
    
    uint256 public mediationFee = 0.05 ether;
    uint256 public arbitrationFeePercentage = 3; // 3% of dispute amount
    uint256 public appealFeeMultiplier = 2; // 2x original arbitration fee

    event DisputeCreated(bytes32 indexed disputeId, address indexed claimant, address indexed respondent);
    event DisputeStatusChanged(bytes32 indexed disputeId, DisputeStatus newStatus);
    event ArbitratorAssigned(bytes32 indexed disputeId, address indexed arbitrator);
    event MediatorAssigned(bytes32 indexed disputeId, address indexed mediator);
    event DisputeResolved(bytes32 indexed disputeId, address indexed resolver, uint256 awardAmount);
    event AppealFiled(bytes32 indexed disputeId, address indexed appellant);
    event EvidenceSubmitted(bytes32 indexed disputeId, address indexed submitter);

    modifier onlyDisputeParty(bytes32 disputeId) {
        require(
            disputes[disputeId].claimant == msg.sender || 
            disputes[disputeId].respondent == msg.sender,
            "Not a party to dispute"
        );
        _;
    }

    modifier onlyAssignedResolver(bytes32 disputeId) {
        require(
            disputes[disputeId].assignedArbitrator == msg.sender || 
            disputes[disputeId].assignedMediator == msg.sender,
            "Not assigned resolver"
        );
        _;
    }

    modifier validDispute(bytes32 disputeId) {
        require(disputes[disputeId].claimant != address(0), "Dispute does not exist");
        _;
    }

    constructor(address _arbitrationRegistry) {
        arbitrationRegistry = ArbitrationRegistry(_arbitrationRegistry);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Create a new dispute
     */
    function createDispute(
        address respondent,
        DisputeType disputeType,
        string memory description,
        bytes32 arbitrationClauseId,
        uint256 disputeAmount
    ) external payable nonReentrant whenNotPaused returns (bytes32) {
        require(respondent != address(0) && respondent != msg.sender, "Invalid respondent");
        require(bytes(description).length > 0, "Empty description");
        require(disputeAmount >= MIN_DISPUTE_AMOUNT, "Dispute amount too low");
        
        uint256 requiredFee = (disputeAmount * arbitrationFeePercentage) / 100;
        require(msg.value >= requiredFee, "Insufficient arbitration fee");

        bytes32 disputeId = keccak256(abi.encodePacked(
            msg.sender,
            respondent,
            description,
            block.timestamp
        ));

        require(disputes[disputeId].claimant == address(0), "Dispute already exists");

        Dispute storage newDispute = disputes[disputeId];
        newDispute.id = disputeId;
        newDispute.claimant = msg.sender;
        newDispute.respondent = respondent;
        newDispute.disputeType = disputeType;
        newDispute.status = DisputeStatus.Created;
        newDispute.description = description;
        newDispute.disputeAmount = disputeAmount;
        newDispute.createdAt = block.timestamp;
        newDispute.arbitrationClauseId = arbitrationClauseId;
        newDispute.deadline = block.timestamp + ARBITRATION_DURATION;

        allDisputeIds.push(disputeId);

        // Automatically assign mediator for initial mediation attempt
        _assignMediator(disputeId);

        emit DisputeCreated(disputeId, msg.sender, respondent);
        return disputeId;
    }

    /**
     * @dev Submit evidence to a dispute
     */
    function submitEvidence(bytes32 disputeId, string memory evidence) 
        external 
        validDispute(disputeId) 
        onlyDisputeParty(disputeId) 
        whenNotPaused 
    {
        require(
            disputes[disputeId].status == DisputeStatus.UnderReview || 
            disputes[disputeId].status == DisputeStatus.InMediation ||
            disputes[disputeId].status == DisputeStatus.InArbitration,
            "Cannot submit evidence in current status"
        );
        require(bytes(evidence).length > 0, "Empty evidence");
        require(block.timestamp < disputes[disputeId].deadline, "Evidence submission deadline passed");

        disputes[disputeId].evidence.push(evidence);
        emit EvidenceSubmitted(disputeId, msg.sender);
    }

    /**
     * @dev Submit statement for dispute
     */
    function submitStatement(bytes32 disputeId, string memory statement) 
        external 
        validDispute(disputeId) 
        onlyDisputeParty(disputeId) 
        whenNotPaused 
    {
        require(bytes(statement).length > 0, "Empty statement");
        require(!disputes[disputeId].hasVoted[msg.sender], "Statement already submitted");

        disputes[disputeId].statements[msg.sender] = statement;
        disputes[disputeId].hasVoted[msg.sender] = true;
    }

    /**
     * @dev Escalate dispute to arbitration
     */
    function escalateToArbitration(bytes32 disputeId) 
        external 
        validDispute(disputeId) 
        onlyDisputeParty(disputeId) 
        whenNotPaused 
    {
        require(
            disputes[disputeId].status == DisputeStatus.InMediation,
            "Can only escalate from mediation"
        );

        disputes[disputeId].status = DisputeStatus.InArbitration;
        disputes[disputeId].deadline = block.timestamp + ARBITRATION_DURATION;

        _assignArbitrator(disputeId);
        emit DisputeStatusChanged(disputeId, DisputeStatus.InArbitration);
    }

    /**
     * @dev Resolve dispute (arbitrator/mediator only)
     */
    function resolveDispute(
        bytes32 disputeId,
        string memory decision,
        uint256 awardAmount,
        address awardRecipient,
        string memory reasoning
    ) external validDispute(disputeId) onlyAssignedResolver(disputeId) whenNotPaused {
        require(
            disputes[disputeId].status == DisputeStatus.InMediation || 
            disputes[disputeId].status == DisputeStatus.InArbitration,
            "Dispute not in resolvable status"
        );
        require(bytes(decision).length > 0, "Empty decision");
        require(awardAmount <= disputes[disputeId].disputeAmount, "Award exceeds dispute amount");

        disputes[disputeId].status = DisputeStatus.Resolved;
        disputes[disputeId].resolutionTime = block.timestamp;

        Resolution storage resolution = resolutions[disputeId];
        resolution.disputeId = disputeId;
        resolution.resolver = msg.sender;
        resolution.decision = decision;
        resolution.awardAmount = awardAmount;
        resolution.awardRecipient = awardRecipient;
        resolution.resolvedAt = block.timestamp;
        resolution.isAppealable = disputes[disputeId].status == DisputeStatus.InArbitration;
        resolution.appealDeadline = block.timestamp + APPEAL_WINDOW;
        resolution.reasoning = reasoning;

        // Transfer award if applicable
        if (awardAmount > 0 && awardRecipient != address(0)) {
            payable(awardRecipient).transfer(awardAmount);
        }

        // Update arbitrator statistics
        if (disputes[disputeId].assignedArbitrator != address(0)) {
            _updateArbitratorStats(disputes[disputeId].assignedArbitrator, true);
        }

        emit DisputeResolved(disputeId, msg.sender, awardAmount);
    }

    /**
     * @dev File an appeal
     */
    function fileAppeal(bytes32 disputeId, string memory reason) 
        external 
        payable 
        validDispute(disputeId) 
        onlyDisputeParty(disputeId) 
        nonReentrant 
        whenNotPaused 
    {
        require(disputes[disputeId].status == DisputeStatus.Resolved, "Dispute not resolved");
        require(resolutions[disputeId].isAppealable, "Dispute not appealable");
        require(block.timestamp <= resolutions[disputeId].appealDeadline, "Appeal window closed");
        require(appeals[disputeId].appellant == address(0), "Appeal already filed");

        uint256 appealFee = ((disputes[disputeId].disputeAmount * arbitrationFeePercentage) / 100) * appealFeeMultiplier;
        require(msg.value >= appealFee, "Insufficient appeal fee");

        disputes[disputeId].status = DisputeStatus.Appealed;
        
        Appeal storage appeal = appeals[disputeId];
        appeal.disputeId = disputeId;
        appeal.appellant = msg.sender;
        appeal.reason = reason;
        appeal.appealFee = msg.value;
        appeal.createdAt = block.timestamp;

        // Assign appeal panel (3 arbitrators)
        _assignAppealPanel(disputeId);

        emit AppealFiled(disputeId, msg.sender);
        emit DisputeStatusChanged(disputeId, DisputeStatus.Appealed);
    }

    /**
     * @dev Auto-resolve disputes that exceed deadline
     */
    function autoResolveExpiredDispute(bytes32 disputeId) 
        external 
        validDispute(disputeId) 
        whenNotPaused 
    {
        require(block.timestamp > disputes[disputeId].deadline, "Dispute not expired");
        require(
            disputes[disputeId].status == DisputeStatus.InMediation || 
            disputes[disputeId].status == DisputeStatus.InArbitration,
            "Dispute not in active resolution"
        );

        // Auto-resolve in favor of respondent if no resolution
        disputes[disputeId].status = DisputeStatus.Resolved;
        disputes[disputeId].resolutionTime = block.timestamp;

        Resolution storage resolution = resolutions[disputeId];
        resolution.disputeId = disputeId;
        resolution.resolver = address(this); // System resolution
        resolution.decision = "Auto-resolved due to timeout";
        resolution.awardAmount = 0;
        resolution.awardRecipient = disputes[disputeId].respondent;
        resolution.resolvedAt = block.timestamp;
        resolution.isAppealable = false;

        emit DisputeResolved(disputeId, address(this), 0);
    }

    /**
     * @dev Internal function to assign mediator
     */
    function _assignMediator(bytes32 disputeId) internal {
        // Simple assignment logic - in production, implement more sophisticated matching
        if (allArbitrators.length > 0) {
            address mediator = allArbitrators[uint256(disputeId) % allArbitrators.length];
            disputes[disputeId].assignedMediator = mediator;
            disputes[disputeId].status = DisputeStatus.InMediation;
            disputes[disputeId].deadline = block.timestamp + MEDIATION_DURATION;
            emit MediatorAssigned(disputeId, mediator);
        }
    }

    /**
     * @dev Internal function to assign arbitrator
     */
    function _assignArbitrator(bytes32 disputeId) internal {
        bytes32 clauseId = disputes[disputeId].arbitrationClauseId;
        address[] memory qualifiedArbitrators = arbitrationRegistry.getClauseArbitrators(clauseId);
        
        if (qualifiedArbitrators.length > 0) {
            address arbitrator = qualifiedArbitrators[uint256(disputeId) % qualifiedArbitrators.length];
            disputes[disputeId].assignedArbitrator = arbitrator;
            emit ArbitratorAssigned(disputeId, arbitrator);
        }
    }

    /**
     * @dev Internal function to assign appeal panel
     */
    function _assignAppealPanel(bytes32 disputeId) internal {
        bytes32 clauseId = disputes[disputeId].arbitrationClauseId;
        address[] memory qualifiedArbitrators = arbitrationRegistry.getClauseArbitrators(clauseId);
        
        // Select 3 different arbitrators for appeal panel
        if (qualifiedArbitrators.length >= 3) {
            Appeal storage appeal = appeals[disputeId];
            
            // Simple selection - in production, implement more sophisticated selection
            for (uint i = 0; i < 3 && i < qualifiedArbitrators.length; i++) {
                uint256 index = (uint256(disputeId) + i) % qualifiedArbitrators.length;
                appeal.appealPanelArbitrators.push(qualifiedArbitrators[index]);
            }
        }
    }

    /**
     * @dev Internal function to update arbitrator statistics
     */
    function _updateArbitratorStats(address arbitrator, bool successful) internal {
        // This would interact with ArbitrationRegistry to update stats
        // Implementation depends on registry interface
    }

    /**
     * @dev Get dispute details
     */
    function getDisputeDetails(bytes32 disputeId) external view returns (
        address claimant,
        address respondent,
        DisputeType disputeType,
        DisputeStatus status,
        string memory description,
        uint256 disputeAmount,
        uint256 createdAt,
        address assignedArbitrator
    ) {
        Dispute storage dispute = disputes[disputeId];
        return (
            dispute.claimant,
            dispute.respondent,
            dispute.disputeType,
            dispute.status,
            dispute.description,
            dispute.disputeAmount,
            dispute.createdAt,
            dispute.assignedArbitrator
        );
    }

    /**
     * @dev Get dispute evidence
     */
    function getDisputeEvidence(bytes32 disputeId) external view returns (string[] memory) {
        return disputes[disputeId].evidence;
    }

    /**
     * @dev Get resolution details
     */
    function getResolution(bytes32 disputeId) external view returns (
        address resolver,
        string memory decision,
        uint256 awardAmount,
        address awardRecipient,
        uint256 resolvedAt,
        bool isAppealable
    ) {
        Resolution storage resolution = resolutions[disputeId];
        return (
            resolution.resolver,
            resolution.decision,
            resolution.awardAmount,
            resolution.awardRecipient,
            resolution.resolvedAt,
            resolution.isAppealable
        );
    }

    /**
     * @dev Get total disputes
     */
    function getTotalDisputes() external view returns (uint256) {
        return allDisputeIds.length;
    }

    /**
     * @dev Emergency functions
     */
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }

    function setMediationFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        mediationFee = newFee;
    }

    function setArbitrationFeePercentage(uint256 newPercentage) external onlyRole(ADMIN_ROLE) {
        require(newPercentage <= 10, "Fee too high"); // Max 10%
        arbitrationFeePercentage = newPercentage;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}