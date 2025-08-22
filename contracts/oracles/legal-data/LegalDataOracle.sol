// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title LegalDataOracle
 * @dev Advanced oracle system for legal data aggregation and verification
 */
contract LegalDataOracle is AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;

    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant CONSUMER_ROLE = keccak256("CONSUMER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    enum DataStatus {
        Pending,
        Verified,
        Disputed,
        Rejected,
        Expired
    }

    enum DataType {
        CourtDecision,
        RegulatoryUpdate,
        LegalPrecedent,
        ComplianceRule,
        JurisdictionChange,
        LawAmendment,
        CaseOutcome,
        LegalOpinion
    }

    struct DataSubmission {
        bytes32 id;
        address oracle;
        DataType dataType;
        string jurisdiction;
        string caseNumber;
        string dataHash;
        string metadataURI;
        uint256 timestamp;
        uint256 expiryTime;
        DataStatus status;
        uint256 validatorCount;
        uint256 disputeCount;
        mapping(address => bool) validators;
        mapping(address => bool) disputes;
        bytes32[] relatedCases;
        string[] tags;
    }

    struct OracleNode {
        address nodeAddress;
        string name;
        string[] specializations;
        string[] jurisdictions;
        uint256 totalSubmissions;
        uint256 verifiedSubmissions;
        uint256 rejectedSubmissions;
        uint256 reputation;
        uint256 stake;
        bool active;
        uint256 lastActiveTime;
        mapping(string => bool) authorizedJurisdictions;
        mapping(DataType => bool) authorizedDataTypes;
    }

    struct DataRequest {
        bytes32 id;
        address requester;
        DataType dataType;
        string jurisdiction;
        string query;
        uint256 bounty;
        uint256 deadline;
        bool fulfilled;
        bytes32 responseId;
        uint256 createdAt;
    }

    struct ValidationResult {
        address validator;
        bool approved;
        string comments;
        uint256 timestamp;
        uint256 confidence; // 0-100
    }

    struct AggregatedData {
        bytes32 id;
        DataType dataType;
        string jurisdiction;
        string summary;
        bytes32[] sourceSubmissions;
        uint256 consensusScore;
        uint256 lastUpdated;
        mapping(address => ValidationResult) validations;
        address[] validators;
    }

    mapping(bytes32 => DataSubmission) public dataSubmissions;
    mapping(address => OracleNode) public oracleNodes;
    mapping(bytes32 => DataRequest) public dataRequests;
    mapping(bytes32 => AggregatedData) public aggregatedData;
    mapping(string => mapping(DataType => bytes32[])) public jurisdictionData;
    mapping(address => bytes32[]) public userRequests;
    mapping(address => bytes32[]) public oracleSubmissions;
    
    bytes32[] public allSubmissionIds;
    bytes32[] public allRequestIds;
    bytes32[] public allAggregatedIds;
    address[] public allOracleNodes;

    uint256 public constant MIN_STAKE = 5 ether;
    uint256 public constant MIN_VALIDATORS = 3;
    uint256 public constant DISPUTE_THRESHOLD = 2;
    uint256 public constant DATA_EXPIRY_PERIOD = 365 days;
    uint256 public constant REPUTATION_THRESHOLD = 70;

    uint256 public submissionFee = 0.01 ether;
    uint256 public validationReward = 0.005 ether;
    uint256 public requestFee = 0.05 ether;

    event DataSubmitted(bytes32 indexed submissionId, address indexed oracle, DataType dataType, string jurisdiction);
    event DataValidated(bytes32 indexed submissionId, address indexed validator, bool approved);
    event DataDisputed(bytes32 indexed submissionId, address indexed disputer, string reason);
    event DataAggregated(bytes32 indexed aggregatedId, DataType dataType, string jurisdiction);
    event DataRequested(bytes32 indexed requestId, address indexed requester, DataType dataType);
    event RequestFulfilled(bytes32 indexed requestId, bytes32 indexed responseId);
    event OracleRegistered(address indexed oracle, string name);
    event ReputationUpdated(address indexed oracle, uint256 newReputation);

    modifier onlyRegisteredOracle() {
        require(oracleNodes[msg.sender].active, "Not registered oracle");
        _;
    }

    modifier onlyAuthorizedValidator() {
        require(hasRole(VALIDATOR_ROLE, msg.sender), "Not authorized validator");
        _;
    }

    modifier validSubmission(bytes32 submissionId) {
        require(dataSubmissions[submissionId].oracle != address(0), "Submission does not exist");
        _;
    }

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Register as an oracle node
     */
    function registerOracle(
        string memory name,
        string[] memory specializations,
        string[] memory jurisdictions,
        DataType[] memory authorizedTypes
    ) external payable nonReentrant whenNotPaused {
        require(msg.value >= MIN_STAKE, "Insufficient stake");
        require(!oracleNodes[msg.sender].active, "Oracle already registered");
        require(bytes(name).length > 0, "Empty name");

        OracleNode storage newOracle = oracleNodes[msg.sender];
        newOracle.nodeAddress = msg.sender;
        newOracle.name = name;
        newOracle.specializations = specializations;
        newOracle.jurisdictions = jurisdictions;
        newOracle.reputation = 50; // Start with neutral reputation
        newOracle.stake = msg.value;
        newOracle.active = true;
        newOracle.lastActiveTime = block.timestamp;

        // Set authorized jurisdictions
        for (uint256 i = 0; i < jurisdictions.length; i++) {
            newOracle.authorizedJurisdictions[jurisdictions[i]] = true;
        }

        // Set authorized data types
        for (uint256 i = 0; i < authorizedTypes.length; i++) {
            newOracle.authorizedDataTypes[authorizedTypes[i]] = true;
        }

        allOracleNodes.push(msg.sender);
        _grantRole(ORACLE_ROLE, msg.sender);

        emit OracleRegistered(msg.sender, name);
    }

    /**
     * @dev Submit legal data
     */
    function submitData(
        DataType dataType,
        string memory jurisdiction,
        string memory caseNumber,
        string memory dataHash,
        string memory metadataURI,
        uint256 expiryTime,
        bytes32[] memory relatedCases,
        string[] memory tags
    ) external payable onlyRegisteredOracle whenNotPaused returns (bytes32) {
        require(msg.value >= submissionFee, "Insufficient submission fee");
        require(oracleNodes[msg.sender].authorizedJurisdictions[jurisdiction], "Not authorized for jurisdiction");
        require(oracleNodes[msg.sender].authorizedDataTypes[dataType], "Not authorized for data type");
        require(bytes(dataHash).length > 0, "Empty data hash");
        require(expiryTime > block.timestamp, "Invalid expiry time");

        bytes32 submissionId = keccak256(abi.encodePacked(
            msg.sender,
            dataType,
            jurisdiction,
            caseNumber,
            dataHash,
            block.timestamp
        ));

        require(dataSubmissions[submissionId].oracle == address(0), "Submission already exists");

        DataSubmission storage submission = dataSubmissions[submissionId];
        submission.id = submissionId;
        submission.oracle = msg.sender;
        submission.dataType = dataType;
        submission.jurisdiction = jurisdiction;
        submission.caseNumber = caseNumber;
        submission.dataHash = dataHash;
        submission.metadataURI = metadataURI;
        submission.timestamp = block.timestamp;
        submission.expiryTime = expiryTime;
        submission.status = DataStatus.Pending;
        submission.relatedCases = relatedCases;
        submission.tags = tags;

        allSubmissionIds.push(submissionId);
        jurisdictionData[jurisdiction][dataType].push(submissionId);
        oracleSubmissions[msg.sender].push(submissionId);

        // Update oracle stats
        oracleNodes[msg.sender].totalSubmissions++;
        oracleNodes[msg.sender].lastActiveTime = block.timestamp;

        emit DataSubmitted(submissionId, msg.sender, dataType, jurisdiction);
        return submissionId;
    }

    /**
     * @dev Validate submitted data
     */
    function validateData(
        bytes32 submissionId,
        bool approved,
        string memory comments,
        uint256 confidence
    ) external validSubmission(submissionId) onlyAuthorizedValidator whenNotPaused {
        DataSubmission storage submission = dataSubmissions[submissionId];
        require(submission.oracle != msg.sender, "Cannot validate own submission");
        require(!submission.validators[msg.sender], "Already validated");
        require(submission.status == DataStatus.Pending, "Submission not pending");
        require(confidence <= 100, "Invalid confidence level");

        submission.validators[msg.sender] = true;
        submission.validatorCount++;

        // Store validation result in aggregated data if needed
        _storeValidationResult(submissionId, msg.sender, approved, comments, confidence);

        emit DataValidated(submissionId, msg.sender, approved);

        // Check if enough validators have validated
        if (submission.validatorCount >= MIN_VALIDATORS) {
            _finalizeValidation(submissionId);
        }

        // Reward validator
        payable(msg.sender).transfer(validationReward);
    }

    /**
     * @dev Dispute submitted data
     */
    function disputeData(
        bytes32 submissionId,
        string memory reason
    ) external validSubmission(submissionId) whenNotPaused {
        require(hasRole(VALIDATOR_ROLE, msg.sender) || hasRole(ORACLE_ROLE, msg.sender), "Not authorized");
        require(!dataSubmissions[submissionId].disputes[msg.sender], "Already disputed");
        require(bytes(reason).length > 0, "Empty dispute reason");

        DataSubmission storage submission = dataSubmissions[submissionId];
        submission.disputes[msg.sender] = true;
        submission.disputeCount++;

        emit DataDisputed(submissionId, msg.sender, reason);

        // Mark as disputed if threshold reached
        if (submission.disputeCount >= DISPUTE_THRESHOLD) {
            submission.status = DataStatus.Disputed;
            _updateOracleReputation(submission.oracle, false);
        }
    }

    /**
     * @dev Request specific legal data
     */
    function requestData(
        DataType dataType,
        string memory jurisdiction,
        string memory query,
        uint256 deadline
    ) external payable nonReentrant whenNotPaused returns (bytes32) {
        require(msg.value >= requestFee, "Insufficient request fee");
        require(deadline > block.timestamp, "Invalid deadline");
        require(bytes(query).length > 0, "Empty query");

        bytes32 requestId = keccak256(abi.encodePacked(
            msg.sender,
            dataType,
            jurisdiction,
            query,
            block.timestamp
        ));

        DataRequest storage request = dataRequests[requestId];
        request.id = requestId;
        request.requester = msg.sender;
        request.dataType = dataType;
        request.jurisdiction = jurisdiction;
        request.query = query;
        request.bounty = msg.value;
        request.deadline = deadline;
        request.createdAt = block.timestamp;

        allRequestIds.push(requestId);
        userRequests[msg.sender].push(requestId);

        emit DataRequested(requestId, msg.sender, dataType);
        return requestId;
    }

    /**
     * @dev Fulfill a data request
     */
    function fulfillRequest(
        bytes32 requestId,
        bytes32 submissionId
    ) external onlyRegisteredOracle whenNotPaused {
        DataRequest storage request = dataRequests[requestId];
        require(!request.fulfilled, "Request already fulfilled");
        require(block.timestamp <= request.deadline, "Request expired");
        require(dataSubmissions[submissionId].oracle == msg.sender, "Not your submission");
        require(dataSubmissions[submissionId].status == DataStatus.Verified, "Submission not verified");

        request.fulfilled = true;
        request.responseId = submissionId;

        // Transfer bounty to oracle
        payable(msg.sender).transfer(request.bounty);

        emit RequestFulfilled(requestId, submissionId);
    }

    /**
     * @dev Aggregate data from multiple sources
     */
    function aggregateData(
        DataType dataType,
        string memory jurisdiction,
        string memory summary,
        bytes32[] memory sourceSubmissions
    ) external onlyRole(ADMIN_ROLE) whenNotPaused returns (bytes32) {
        require(sourceSubmissions.length > 0, "No source submissions");

        bytes32 aggregatedId = keccak256(abi.encodePacked(
            dataType,
            jurisdiction,
            summary,
            block.timestamp
        ));

        AggregatedData storage aggregated = aggregatedData[aggregatedId];
        aggregated.id = aggregatedId;
        aggregated.dataType = dataType;
        aggregated.jurisdiction = jurisdiction;
        aggregated.summary = summary;
        aggregated.sourceSubmissions = sourceSubmissions;
        aggregated.lastUpdated = block.timestamp;

        // Calculate consensus score
        aggregated.consensusScore = _calculateConsensusScore(sourceSubmissions);

        allAggregatedIds.push(aggregatedId);

        emit DataAggregated(aggregatedId, dataType, jurisdiction);
        return aggregatedId;
    }

    /**
     * @dev Internal function to store validation result
     */
    function _storeValidationResult(
        bytes32 submissionId,
        address validator,
        bool approved,
        string memory comments,
        uint256 confidence
    ) internal {
        // This could be expanded to store validation results in aggregated data
        // For now, just emit the validation event
    }

    /**
     * @dev Internal function to finalize validation
     */
    function _finalizeValidation(bytes32 submissionId) internal {
        DataSubmission storage submission = dataSubmissions[submissionId];
        
        // Simple majority voting - can be made more sophisticated
        uint256 approvals = 0;
        // Count approvals from validators (simplified for this example)
        
        if (approvals > submission.validatorCount / 2) {
            submission.status = DataStatus.Verified;
            _updateOracleReputation(submission.oracle, true);
            oracleNodes[submission.oracle].verifiedSubmissions++;
        } else {
            submission.status = DataStatus.Rejected;
            _updateOracleReputation(submission.oracle, false);
            oracleNodes[submission.oracle].rejectedSubmissions++;
        }
    }

    /**
     * @dev Internal function to update oracle reputation
     */
    function _updateOracleReputation(address oracle, bool positive) internal {
        OracleNode storage node = oracleNodes[oracle];
        
        if (positive) {
            node.reputation = node.reputation + 5 > 100 ? 100 : node.reputation + 5;
        } else {
            node.reputation = node.reputation < 5 ? 0 : node.reputation - 5;
        }

        emit ReputationUpdated(oracle, node.reputation);
    }

    /**
     * @dev Internal function to calculate consensus score
     */
    function _calculateConsensusScore(bytes32[] memory sourceSubmissions) internal view returns (uint256) {
        if (sourceSubmissions.length == 0) return 0;
        
        uint256 totalScore = 0;
        uint256 validSubmissions = 0;
        
        for (uint256 i = 0; i < sourceSubmissions.length; i++) {
            DataSubmission storage submission = dataSubmissions[sourceSubmissions[i]];
            if (submission.status == DataStatus.Verified) {
                totalScore += oracleNodes[submission.oracle].reputation;
                validSubmissions++;
            }
        }
        
        return validSubmissions > 0 ? totalScore / validSubmissions : 0;
    }

    /**
     * @dev Get submission details
     */
    function getSubmissionDetails(bytes32 submissionId) external view returns (
        address oracle,
        DataType dataType,
        string memory jurisdiction,
        string memory caseNumber,
        string memory dataHash,
        DataStatus status,
        uint256 timestamp,
        uint256 validatorCount
    ) {
        DataSubmission storage submission = dataSubmissions[submissionId];
        return (
            submission.oracle,
            submission.dataType,
            submission.jurisdiction,
            submission.caseNumber,
            submission.dataHash,
            submission.status,
            submission.timestamp,
            submission.validatorCount
        );
    }

    /**
     * @dev Get oracle node details
     */
    function getOracleDetails(address oracle) external view returns (
        string memory name,
        string[] memory specializations,
        string[] memory jurisdictions,
        uint256 totalSubmissions,
        uint256 verifiedSubmissions,
        uint256 reputation,
        uint256 stake,
        bool active
    ) {
        OracleNode storage node = oracleNodes[oracle];
        return (
            node.name,
            node.specializations,
            node.jurisdictions,
            node.totalSubmissions,
            node.verifiedSubmissions,
            node.reputation,
            node.stake,
            node.active
        );
    }

    /**
     * @dev Get data by jurisdiction and type
     */
    function getJurisdictionData(
        string memory jurisdiction,
        DataType dataType
    ) external view returns (bytes32[] memory) {
        return jurisdictionData[jurisdiction][dataType];
    }

    /**
     * @dev Get user requests
     */
    function getUserRequests(address user) external view returns (bytes32[] memory) {
        return userRequests[user];
    }

    /**
     * @dev Get oracle submissions
     */
    function getOracleSubmissions(address oracle) external view returns (bytes32[] memory) {
        return oracleSubmissions[oracle];
    }

    /**
     * @dev Get total submissions
     */
    function getTotalSubmissions() external view returns (uint256) {
        return allSubmissionIds.length;
    }

    /**
     * @dev Get total requests
     */
    function getTotalRequests() external view returns (uint256) {
        return allRequestIds.length;
    }

    /**
     * @dev Get total oracles
     */
    function getTotalOracles() external view returns (uint256) {
        return allOracleNodes.length;
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

    function setSubmissionFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        submissionFee = newFee;
    }

    function setValidationReward(uint256 newReward) external onlyRole(ADMIN_ROLE) {
        validationReward = newReward;
    }

    function setRequestFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        requestFee = newFee;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}