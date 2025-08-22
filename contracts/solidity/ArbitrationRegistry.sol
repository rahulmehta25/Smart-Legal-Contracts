// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/draft-EIP712.sol";

/**
 * @title ArbitrationRegistry
 * @dev Registry for managing arbitration clauses and arbitrators with reputation system
 */
contract ArbitrationRegistry is AccessControl, ReentrancyGuard, Pausable, EIP712 {
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");

    struct ArbitrationClause {
        bytes32 id;
        address creator;
        string clauseText;
        string[] applicableJurisdictions;
        uint256 fee;
        bool active;
        uint256 createdAt;
        uint256 updatedAt;
        bytes32[] requirements;
        mapping(address => bool) approvedArbitrators;
    }

    struct Arbitrator {
        address arbitratorAddress;
        string name;
        string[] specializations;
        string[] jurisdictions;
        uint256 totalCases;
        uint256 successfulCases;
        uint256 averageResolutionTime;
        uint256 stakeAmount;
        bool certified;
        uint256 reputation;
        mapping(bytes32 => bool) qualifiedClauses;
    }

    struct Reputation {
        uint256 score;
        uint256 totalVotes;
        uint256 positiveVotes;
        mapping(address => bool) hasVoted;
    }

    mapping(bytes32 => ArbitrationClause) public arbitrationClauses;
    mapping(address => Arbitrator) public arbitrators;
    mapping(address => Reputation) public reputations;
    mapping(bytes32 => address[]) public clauseArbitrators;
    
    bytes32[] public allClauseIds;
    address[] public allArbitrators;

    uint256 public constant MIN_STAKE = 10 ether;
    uint256 public constant REPUTATION_THRESHOLD = 80;
    uint256 public registrationFee = 0.1 ether;

    event ClauseRegistered(bytes32 indexed clauseId, address indexed creator);
    event ArbitratorRegistered(address indexed arbitrator, uint256 stake);
    event ArbitratorApproved(bytes32 indexed clauseId, address indexed arbitrator);
    event ReputationUpdated(address indexed arbitrator, uint256 newScore);
    event ClauseUpdated(bytes32 indexed clauseId);
    event ArbitratorStakeUpdated(address indexed arbitrator, uint256 newStake);

    modifier onlyRegisteredArbitrator() {
        require(arbitrators[msg.sender].arbitratorAddress != address(0), "Not registered arbitrator");
        _;
    }

    modifier onlyActiveClause(bytes32 clauseId) {
        require(arbitrationClauses[clauseId].active, "Clause not active");
        _;
    }

    constructor() EIP712("ArbitrationRegistry", "1") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Register a new arbitration clause
     */
    function registerClause(
        string memory clauseText,
        string[] memory applicableJurisdictions,
        uint256 fee,
        bytes32[] memory requirements
    ) external payable nonReentrant whenNotPaused returns (bytes32) {
        require(msg.value >= registrationFee, "Insufficient registration fee");
        require(bytes(clauseText).length > 0, "Empty clause text");
        
        bytes32 clauseId = keccak256(abi.encodePacked(
            clauseText,
            msg.sender,
            block.timestamp
        ));
        
        require(arbitrationClauses[clauseId].creator == address(0), "Clause already exists");

        ArbitrationClause storage newClause = arbitrationClauses[clauseId];
        newClause.id = clauseId;
        newClause.creator = msg.sender;
        newClause.clauseText = clauseText;
        newClause.applicableJurisdictions = applicableJurisdictions;
        newClause.fee = fee;
        newClause.active = true;
        newClause.createdAt = block.timestamp;
        newClause.updatedAt = block.timestamp;
        newClause.requirements = requirements;

        allClauseIds.push(clauseId);

        emit ClauseRegistered(clauseId, msg.sender);
        return clauseId;
    }

    /**
     * @dev Register as an arbitrator with stake
     */
    function registerArbitrator(
        string memory name,
        string[] memory specializations,
        string[] memory jurisdictions
    ) external payable nonReentrant whenNotPaused {
        require(msg.value >= MIN_STAKE, "Insufficient stake amount");
        require(arbitrators[msg.sender].arbitratorAddress == address(0), "Already registered");
        require(bytes(name).length > 0, "Empty name");

        Arbitrator storage newArbitrator = arbitrators[msg.sender];
        newArbitrator.arbitratorAddress = msg.sender;
        newArbitrator.name = name;
        newArbitrator.specializations = specializations;
        newArbitrator.jurisdictions = jurisdictions;
        newArbitrator.stakeAmount = msg.value;
        newArbitrator.certified = false;
        newArbitrator.reputation = 50; // Start with neutral reputation

        allArbitrators.push(msg.sender);

        emit ArbitratorRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Approve arbitrator for specific clause
     */
    function approveArbitratorForClause(bytes32 clauseId, address arbitrator) 
        external 
        onlyActiveClause(clauseId) 
        whenNotPaused 
    {
        require(
            arbitrationClauses[clauseId].creator == msg.sender || 
            hasRole(ADMIN_ROLE, msg.sender), 
            "Not authorized"
        );
        require(arbitrators[arbitrator].arbitratorAddress != address(0), "Arbitrator not registered");
        require(!arbitrationClauses[clauseId].approvedArbitrators[arbitrator], "Already approved");

        arbitrationClauses[clauseId].approvedArbitrators[arbitrator] = true;
        clauseArbitrators[clauseId].push(arbitrator);
        arbitrators[arbitrator].qualifiedClauses[clauseId] = true;

        emit ArbitratorApproved(clauseId, arbitrator);
    }

    /**
     * @dev Update arbitrator reputation
     */
    function updateReputation(address arbitrator, bool positive) 
        external 
        onlyRegisteredArbitrator 
        whenNotPaused 
    {
        require(arbitrator != msg.sender, "Cannot vote for yourself");
        require(!reputations[arbitrator].hasVoted[msg.sender], "Already voted");

        Reputation storage rep = reputations[arbitrator];
        rep.hasVoted[msg.sender] = true;
        rep.totalVotes++;
        
        if (positive) {
            rep.positiveVotes++;
        }

        // Calculate new reputation score (0-100)
        rep.score = (rep.positiveVotes * 100) / rep.totalVotes;
        arbitrators[arbitrator].reputation = rep.score;

        // Auto-certify if reputation is high enough
        if (rep.score >= REPUTATION_THRESHOLD && !arbitrators[arbitrator].certified) {
            arbitrators[arbitrator].certified = true;
        }

        emit ReputationUpdated(arbitrator, rep.score);
    }

    /**
     * @dev Add more stake to arbitrator account
     */
    function addStake() external payable onlyRegisteredArbitrator whenNotPaused {
        require(msg.value > 0, "Must stake positive amount");
        arbitrators[msg.sender].stakeAmount += msg.value;
        emit ArbitratorStakeUpdated(msg.sender, arbitrators[msg.sender].stakeAmount);
    }

    /**
     * @dev Withdraw stake (partial)
     */
    function withdrawStake(uint256 amount) external onlyRegisteredArbitrator nonReentrant whenNotPaused {
        require(amount > 0, "Must withdraw positive amount");
        require(
            arbitrators[msg.sender].stakeAmount - amount >= MIN_STAKE,
            "Cannot withdraw below minimum stake"
        );

        arbitrators[msg.sender].stakeAmount -= amount;
        payable(msg.sender).transfer(amount);
        
        emit ArbitratorStakeUpdated(msg.sender, arbitrators[msg.sender].stakeAmount);
    }

    /**
     * @dev Update clause (only creator or admin)
     */
    function updateClause(
        bytes32 clauseId,
        string memory newClauseText,
        uint256 newFee
    ) external onlyActiveClause(clauseId) whenNotPaused {
        require(
            arbitrationClauses[clauseId].creator == msg.sender || 
            hasRole(ADMIN_ROLE, msg.sender),
            "Not authorized"
        );
        require(bytes(newClauseText).length > 0, "Empty clause text");

        arbitrationClauses[clauseId].clauseText = newClauseText;
        arbitrationClauses[clauseId].fee = newFee;
        arbitrationClauses[clauseId].updatedAt = block.timestamp;

        emit ClauseUpdated(clauseId);
    }

    /**
     * @dev Deactivate clause
     */
    function deactivateClause(bytes32 clauseId) external onlyActiveClause(clauseId) whenNotPaused {
        require(
            arbitrationClauses[clauseId].creator == msg.sender || 
            hasRole(ADMIN_ROLE, msg.sender),
            "Not authorized"
        );

        arbitrationClauses[clauseId].active = false;
        arbitrationClauses[clauseId].updatedAt = block.timestamp;

        emit ClauseUpdated(clauseId);
    }

    /**
     * @dev Get all arbitrators for a clause
     */
    function getClauseArbitrators(bytes32 clauseId) external view returns (address[] memory) {
        return clauseArbitrators[clauseId];
    }

    /**
     * @dev Get arbitrator details
     */
    function getArbitratorDetails(address arbitrator) external view returns (
        string memory name,
        string[] memory specializations,
        string[] memory jurisdictions,
        uint256 totalCases,
        uint256 successfulCases,
        uint256 reputation,
        uint256 stakeAmount,
        bool certified
    ) {
        Arbitrator storage arb = arbitrators[arbitrator];
        return (
            arb.name,
            arb.specializations,
            arb.jurisdictions,
            arb.totalCases,
            arb.successfulCases,
            arb.reputation,
            arb.stakeAmount,
            arb.certified
        );
    }

    /**
     * @dev Check if arbitrator is approved for clause
     */
    function isArbitratorApproved(bytes32 clauseId, address arbitrator) external view returns (bool) {
        return arbitrationClauses[clauseId].approvedArbitrators[arbitrator];
    }

    /**
     * @dev Get total number of clauses
     */
    function getTotalClauses() external view returns (uint256) {
        return allClauseIds.length;
    }

    /**
     * @dev Get total number of arbitrators
     */
    function getTotalArbitrators() external view returns (uint256) {
        return allArbitrators.length;
    }

    /**
     * @dev Emergency functions (admin only)
     */
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }

    function setRegistrationFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        registrationFee = newFee;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}