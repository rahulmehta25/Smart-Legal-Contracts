// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title ComplianceOracle
 * @dev Oracle system for legal compliance verification and regulatory data feeds
 */
contract ComplianceOracle is AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant CONSUMER_ROLE = keccak256("CONSUMER_ROLE");

    enum DataStatus {
        Pending,
        Verified,
        Disputed,
        Invalid,
        Expired
    }

    enum ComplianceLevel {
        NonCompliant,
        PartiallyCompliant,
        FullyCompliant,
        Unknown
    }

    struct OracleData {
        bytes32 id;
        string jurisdiction;
        string regulation;
        string dataType;
        bytes data;
        uint256 timestamp;
        uint256 expiryTime;
        DataStatus status;
        address oracle;
        uint256 validatorCount;
        mapping(address => bool) validators;
        mapping(address => bool) disputes;
        uint256 disputeCount;
    }

    struct ComplianceCheck {
        bytes32 id;
        address subject;
        string jurisdiction;
        string[] regulations;
        ComplianceLevel level;
        string[] violations;
        uint256 checkedAt;
        uint256 validUntil;
        address auditor;
        bytes32[] supportingDataIds;
    }

    struct OracleNode {
        address nodeAddress;
        string name;
        string[] specializations;
        uint256 totalSubmissions;
        uint256 verifiedSubmissions;
        uint256 stake;
        uint256 reputation;
        bool active;
        mapping(string => bool) authorizedJurisdictions;
    }

    struct Subscription {
        bytes32 id;
        address subscriber;
        string[] dataTypes;
        string[] jurisdictions;
        uint256 paidAmount;
        uint256 expiryTime;
        bool active;
        uint256 requestCount;
        uint256 requestLimit;
    }

    mapping(bytes32 => OracleData) public oracleData;
    mapping(bytes32 => ComplianceCheck) public complianceChecks;
    mapping(address => OracleNode) public oracleNodes;
    mapping(bytes32 => Subscription) public subscriptions;
    mapping(string => mapping(string => bytes32[])) public regulationData; // jurisdiction => regulation => dataIds
    mapping(address => bytes32[]) public userComplianceHistory;
    mapping(string => uint256) public subscriptionPrices; // dataType => price
    
    bytes32[] public allDataIds;
    address[] public allOracleNodes;
    
    uint256 public constant MIN_VALIDATORS = 3;
    uint256 public constant MIN_STAKE = 5 ether;
    uint256 public constant DISPUTE_THRESHOLD = 2;
    uint256 public constant DATA_VALIDITY_PERIOD = 30 days;
    
    uint256 public oracleReward = 0.01 ether;
    uint256 public validatorReward = 0.005 ether;

    event DataSubmitted(bytes32 indexed dataId, address indexed oracle, string jurisdiction, string regulation);
    event DataVerified(bytes32 indexed dataId, address indexed validator);
    event DataDisputed(bytes32 indexed dataId, address indexed disputer);
    event ComplianceChecked(bytes32 indexed checkId, address indexed subject, ComplianceLevel level);
    event OracleRegistered(address indexed oracle, uint256 stake);
    event SubscriptionCreated(bytes32 indexed subscriptionId, address indexed subscriber);
    event ReputationUpdated(address indexed oracle, uint256 newReputation);

    modifier onlyActiveOracle() {
        require(oracleNodes[msg.sender].active, "Oracle not active");
        _;
    }

    modifier validDataId(bytes32 dataId) {
        require(oracleData[dataId].oracle != address(0), "Data does not exist");
        _;
    }

    modifier onlySubscriber(bytes32 subscriptionId) {
        require(subscriptions[subscriptionId].subscriber == msg.sender, "Not subscriber");
        require(subscriptions[subscriptionId].active, "Subscription not active");
        require(block.timestamp <= subscriptions[subscriptionId].expiryTime, "Subscription expired");
        _;
    }

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        
        // Set default subscription prices
        subscriptionPrices["regulatory_updates"] = 0.1 ether;
        subscriptionPrices["compliance_status"] = 0.05 ether;
        subscriptionPrices["court_decisions"] = 0.15 ether;
        subscriptionPrices["sanctions_list"] = 0.08 ether;
    }

    /**
     * @dev Register as an oracle node
     */
    function registerOracle(
        string memory name,
        string[] memory specializations,
        string[] memory jurisdictions
    ) external payable nonReentrant whenNotPaused {
        require(msg.value >= MIN_STAKE, "Insufficient stake");
        require(!oracleNodes[msg.sender].active, "Oracle already registered");
        require(bytes(name).length > 0, "Empty name");

        OracleNode storage newOracle = oracleNodes[msg.sender];
        newOracle.nodeAddress = msg.sender;
        newOracle.name = name;
        newOracle.specializations = specializations;
        newOracle.stake = msg.value;
        newOracle.reputation = 50; // Start with neutral reputation
        newOracle.active = true;

        // Set authorized jurisdictions
        for (uint256 i = 0; i < jurisdictions.length; i++) {
            newOracle.authorizedJurisdictions[jurisdictions[i]] = true;
        }

        allOracleNodes.push(msg.sender);
        _grantRole(ORACLE_ROLE, msg.sender);

        emit OracleRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Submit regulatory data
     */
    function submitData(
        string memory jurisdiction,
        string memory regulation,
        string memory dataType,
        bytes memory data,
        uint256 validityPeriod
    ) external onlyActiveOracle whenNotPaused returns (bytes32) {
        require(oracleNodes[msg.sender].authorizedJurisdictions[jurisdiction], "Not authorized for jurisdiction");
        require(bytes(regulation).length > 0, "Empty regulation");
        require(data.length > 0, "Empty data");
        require(validityPeriod <= DATA_VALIDITY_PERIOD, "Validity period too long");

        bytes32 dataId = keccak256(abi.encodePacked(
            jurisdiction,
            regulation,
            dataType,
            data,
            msg.sender,
            block.timestamp
        ));

        require(oracleData[dataId].oracle == address(0), "Data already exists");

        OracleData storage newData = oracleData[dataId];
        newData.id = dataId;
        newData.jurisdiction = jurisdiction;
        newData.regulation = regulation;
        newData.dataType = dataType;
        newData.data = data;
        newData.timestamp = block.timestamp;
        newData.expiryTime = block.timestamp + validityPeriod;
        newData.status = DataStatus.Pending;
        newData.oracle = msg.sender;

        allDataIds.push(dataId);
        regulationData[jurisdiction][regulation].push(dataId);

        // Update oracle stats
        oracleNodes[msg.sender].totalSubmissions++;

        emit DataSubmitted(dataId, msg.sender, jurisdiction, regulation);
        return dataId;
    }

    /**
     * @dev Validate submitted data
     */
    function validateData(bytes32 dataId) external validDataId(dataId) whenNotPaused {
        require(hasRole(VALIDATOR_ROLE, msg.sender) || hasRole(ORACLE_ROLE, msg.sender), "Not authorized validator");
        require(oracleData[dataId].oracle != msg.sender, "Cannot validate own data");
        require(!oracleData[dataId].validators[msg.sender], "Already validated");
        require(oracleData[dataId].status == DataStatus.Pending, "Data not pending validation");

        oracleData[dataId].validators[msg.sender] = true;
        oracleData[dataId].validatorCount++;

        emit DataVerified(dataId, msg.sender);

        // Auto-verify if enough validators
        if (oracleData[dataId].validatorCount >= MIN_VALIDATORS) {
            oracleData[dataId].status = DataStatus.Verified;
            
            // Reward oracle and validators
            _rewardOracle(oracleData[dataId].oracle);
            _updateOracleReputation(oracleData[dataId].oracle, true);
        }
    }

    /**
     * @dev Dispute submitted data
     */
    function disputeData(bytes32 dataId, string memory reason) 
        external 
        validDataId(dataId) 
        whenNotPaused 
    {
        require(hasRole(VALIDATOR_ROLE, msg.sender) || hasRole(ORACLE_ROLE, msg.sender), "Not authorized");
        require(!oracleData[dataId].disputes[msg.sender], "Already disputed");
        require(bytes(reason).length > 0, "Empty dispute reason");

        oracleData[dataId].disputes[msg.sender] = true;
        oracleData[dataId].disputeCount++;

        emit DataDisputed(dataId, msg.sender);

        // Mark as disputed if threshold reached
        if (oracleData[dataId].disputeCount >= DISPUTE_THRESHOLD) {
            oracleData[dataId].status = DataStatus.Disputed;
            _updateOracleReputation(oracleData[dataId].oracle, false);
        }
    }

    /**
     * @dev Perform compliance check
     */
    function checkCompliance(
        address subject,
        string memory jurisdiction,
        string[] memory regulations
    ) external payable whenNotPaused returns (bytes32) {
        require(subject != address(0), "Invalid subject");
        require(regulations.length > 0, "No regulations specified");
        
        // Check if caller has valid subscription or pays per use
        bool hasAccess = _checkAccess(msg.sender, "compliance_status");
        if (!hasAccess) {
            require(msg.value >= subscriptionPrices["compliance_status"], "Insufficient payment");
        }

        bytes32 checkId = keccak256(abi.encodePacked(
            subject,
            jurisdiction,
            regulations,
            msg.sender,
            block.timestamp
        ));

        ComplianceCheck storage newCheck = complianceChecks[checkId];
        newCheck.id = checkId;
        newCheck.subject = subject;
        newCheck.jurisdiction = jurisdiction;
        newCheck.regulations = regulations;
        newCheck.checkedAt = block.timestamp;
        newCheck.validUntil = block.timestamp + 7 days; // 7-day validity
        newCheck.auditor = msg.sender;

        // Perform compliance analysis
        (ComplianceLevel level, string[] memory violations) = _analyzeCompliance(subject, jurisdiction, regulations);
        newCheck.level = level;
        newCheck.violations = violations;

        userComplianceHistory[subject].push(checkId);

        emit ComplianceChecked(checkId, subject, level);
        return checkId;
    }

    /**
     * @dev Create subscription
     */
    function createSubscription(
        string[] memory dataTypes,
        string[] memory jurisdictions,
        uint256 duration
    ) external payable nonReentrant whenNotPaused returns (bytes32) {
        require(dataTypes.length > 0, "No data types specified");
        require(duration >= 30 days && duration <= 365 days, "Invalid duration");

        uint256 totalPrice = 0;
        for (uint256 i = 0; i < dataTypes.length; i++) {
            totalPrice += subscriptionPrices[dataTypes[i]];
        }
        totalPrice = (totalPrice * duration) / 30 days; // Monthly pricing

        require(msg.value >= totalPrice, "Insufficient payment");

        bytes32 subscriptionId = keccak256(abi.encodePacked(
            msg.sender,
            dataTypes,
            jurisdictions,
            block.timestamp
        ));

        Subscription storage newSubscription = subscriptions[subscriptionId];
        newSubscription.id = subscriptionId;
        newSubscription.subscriber = msg.sender;
        newSubscription.dataTypes = dataTypes;
        newSubscription.jurisdictions = jurisdictions;
        newSubscription.paidAmount = msg.value;
        newSubscription.expiryTime = block.timestamp + duration;
        newSubscription.active = true;
        newSubscription.requestLimit = (duration / 1 days) * 10; // 10 requests per day

        _grantRole(CONSUMER_ROLE, msg.sender);

        emit SubscriptionCreated(subscriptionId, msg.sender);
        return subscriptionId;
    }

    /**
     * @dev Get regulatory data
     */
    function getRegulatoryData(
        bytes32 subscriptionId,
        string memory jurisdiction,
        string memory regulation
    ) external onlySubscriber(subscriptionId) whenNotPaused returns (bytes32[] memory) {
        require(subscriptions[subscriptionId].requestCount < subscriptions[subscriptionId].requestLimit, "Request limit exceeded");
        
        subscriptions[subscriptionId].requestCount++;
        return regulationData[jurisdiction][regulation];
    }

    /**
     * @dev Get oracle data details
     */
    function getOracleData(bytes32 dataId) external view returns (
        string memory jurisdiction,
        string memory regulation,
        string memory dataType,
        bytes memory data,
        uint256 timestamp,
        DataStatus status,
        address oracle
    ) {
        OracleData storage oData = oracleData[dataId];
        return (
            oData.jurisdiction,
            oData.regulation,
            oData.dataType,
            oData.data,
            oData.timestamp,
            oData.status,
            oData.oracle
        );
    }

    /**
     * @dev Get compliance check details
     */
    function getComplianceCheck(bytes32 checkId) external view returns (
        address subject,
        string memory jurisdiction,
        string[] memory regulations,
        ComplianceLevel level,
        string[] memory violations,
        uint256 checkedAt,
        uint256 validUntil
    ) {
        ComplianceCheck storage check = complianceChecks[checkId];
        return (
            check.subject,
            check.jurisdiction,
            check.regulations,
            check.level,
            check.violations,
            check.checkedAt,
            check.validUntil
        );
    }

    /**
     * @dev Internal function to analyze compliance
     */
    function _analyzeCompliance(
        address subject,
        string memory jurisdiction,
        string[] memory regulations
    ) internal view returns (ComplianceLevel, string[] memory) {
        // Simplified compliance analysis - in production, implement comprehensive logic
        string[] memory violations = new string[](0);
        
        // Check against known data
        uint256 compliantCount = 0;
        for (uint256 i = 0; i < regulations.length; i++) {
            bytes32[] memory dataIds = regulationData[jurisdiction][regulations[i]];
            if (dataIds.length > 0) {
                // Check latest data
                bytes32 latestDataId = dataIds[dataIds.length - 1];
                if (oracleData[latestDataId].status == DataStatus.Verified) {
                    compliantCount++;
                }
            }
        }

        if (compliantCount == regulations.length) {
            return (ComplianceLevel.FullyCompliant, violations);
        } else if (compliantCount > 0) {
            return (ComplianceLevel.PartiallyCompliant, violations);
        } else {
            return (ComplianceLevel.NonCompliant, violations);
        }
    }

    /**
     * @dev Internal function to check access
     */
    function _checkAccess(address user, string memory dataType) internal view returns (bool) {
        // Check if user has active subscription for this data type
        // Simplified implementation
        return hasRole(CONSUMER_ROLE, user);
    }

    /**
     * @dev Internal function to reward oracle
     */
    function _rewardOracle(address oracle) internal {
        if (address(this).balance >= oracleReward) {
            payable(oracle).transfer(oracleReward);
            oracleNodes[oracle].verifiedSubmissions++;
        }
    }

    /**
     * @dev Internal function to update oracle reputation
     */
    function _updateOracleReputation(address oracle, bool positive) internal {
        if (positive) {
            if (oracleNodes[oracle].reputation < 100) {
                oracleNodes[oracle].reputation = oracleNodes[oracle].reputation + 1;
            }
        } else {
            if (oracleNodes[oracle].reputation > 0) {
                oracleNodes[oracle].reputation = oracleNodes[oracle].reputation - 2;
            }
        }

        emit ReputationUpdated(oracle, oracleNodes[oracle].reputation);
    }

    /**
     * @dev Get oracle node details
     */
    function getOracleNodeDetails(address oracle) external view returns (
        string memory name,
        string[] memory specializations,
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
            node.totalSubmissions,
            node.verifiedSubmissions,
            node.reputation,
            node.stake,
            node.active
        );
    }

    /**
     * @dev Get user compliance history
     */
    function getUserComplianceHistory(address user) external view returns (bytes32[] memory) {
        return userComplianceHistory[user];
    }

    /**
     * @dev Get total data entries
     */
    function getTotalDataEntries() external view returns (uint256) {
        return allDataIds.length;
    }

    /**
     * @dev Get total oracle nodes
     */
    function getTotalOracleNodes() external view returns (uint256) {
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

    function setSubscriptionPrice(string memory dataType, uint256 price) external onlyRole(ADMIN_ROLE) {
        subscriptionPrices[dataType] = price;
    }

    function setOracleReward(uint256 newReward) external onlyRole(ADMIN_ROLE) {
        oracleReward = newReward;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}