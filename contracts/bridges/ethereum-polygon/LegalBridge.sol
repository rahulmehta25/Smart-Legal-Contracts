// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title LegalBridge
 * @dev Cross-chain bridge for legal contracts and data between Ethereum and Polygon
 */
contract LegalBridge is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant RELAYER_ROLE = keccak256("RELAYER_ROLE");
    bytes32 public constant BRIDGE_OPERATOR_ROLE = keccak256("BRIDGE_OPERATOR_ROLE");

    enum MessageStatus {
        Pending,
        Validated,
        Executed,
        Failed,
        Reverted
    }

    enum MessageType {
        ContractTransfer,
        DisputeResolution,
        ComplianceUpdate,
        EvidenceSubmission,
        TokenTransfer,
        GovernanceProposal,
        OracleUpdate,
        EmergencyAction
    }

    struct CrossChainMessage {
        bytes32 messageId;
        uint256 sourceChainId;
        uint256 destinationChainId;
        address sender;
        address receiver;
        MessageType messageType;
        bytes payload;
        uint256 timestamp;
        uint256 blockNumber;
        MessageStatus status;
        bytes32 payloadHash;
        uint256 validatorCount;
        mapping(address => bool) validatedBy;
        uint256 executionGas;
        uint256 fee;
    }

    struct Validator {
        address validatorAddress;
        uint256 stake;
        uint256 reputation;
        bool active;
        uint256 validatedMessages;
        uint256 lastActivity;
        string endpoint;
        bytes32[] supportedChains;
    }

    struct BridgeConfig {
        uint256 minValidators;
        uint256 validationThreshold; // Percentage (e.g., 67 = 67%)
        uint256 messageTimeout;
        uint256 minStake;
        uint256 baseFee;
        uint256 gasMultiplier;
        bool emergencyMode;
        bytes32 merkleRoot;
    }

    struct RelayerInfo {
        address relayer;
        uint256 totalRelayed;
        uint256 successfulRelays;
        uint256 stake;
        bool active;
        uint256 reputation;
        mapping(uint256 => bool) authorizedChains;
    }

    struct ChainInfo {
        uint256 chainId;
        string name;
        address bridgeContract;
        bool active;
        uint256 lastBlockSynced;
        uint256 totalMessages;
        bytes32 stateRoot;
    }

    mapping(bytes32 => CrossChainMessage) public messages;
    mapping(address => Validator) public validators;
    mapping(address => RelayerInfo) public relayers;
    mapping(uint256 => ChainInfo) public supportedChains;
    mapping(bytes32 => bool) public processedMessages;
    mapping(address => uint256) public nonces;
    mapping(bytes32 => uint256) public messageToChain;
    
    bytes32[] public allMessageIds;
    address[] public allValidators;
    address[] public allRelayers;
    uint256[] public chainIds;
    
    BridgeConfig public config;
    uint256 public totalStaked;
    uint256 public totalMessages;
    address public treasury;

    event MessageSent(
        bytes32 indexed messageId,
        uint256 indexed sourceChain,
        uint256 indexed destinationChain,
        address sender,
        MessageType messageType
    );
    
    event MessageValidated(
        bytes32 indexed messageId,
        address indexed validator,
        bool approved
    );
    
    event MessageExecuted(
        bytes32 indexed messageId,
        bool success,
        bytes returnData
    );
    
    event ValidatorRegistered(address indexed validator, uint256 stake);
    event ValidatorSlashed(address indexed validator, uint256 amount, string reason);
    event RelayerRegistered(address indexed relayer, uint256 stake);
    event ChainAdded(uint256 indexed chainId, string name);
    event EmergencyModeToggled(bool enabled);

    modifier onlyActiveValidator() {
        require(validators[msg.sender].active, "Not active validator");
        require(validators[msg.sender].stake >= config.minStake, "Insufficient stake");
        _;
    }

    modifier onlyActiveRelayer() {
        require(relayers[msg.sender].active, "Not active relayer");
        _;
    }

    modifier validMessage(bytes32 messageId) {
        require(messages[messageId].sender != address(0), "Message does not exist");
        _;
    }

    modifier notInEmergencyMode() {
        require(!config.emergencyMode, "Bridge in emergency mode");
        _;
    }

    constructor(
        address _treasury,
        uint256 _minValidators,
        uint256 _validationThreshold
    ) {
        require(_treasury != address(0), "Invalid treasury");
        require(_minValidators >= 3, "Need at least 3 validators");
        require(_validationThreshold >= 51 && _validationThreshold <= 100, "Invalid threshold");

        treasury = _treasury;
        
        config = BridgeConfig({
            minValidators: _minValidators,
            validationThreshold: _validationThreshold,
            messageTimeout: 24 hours,
            minStake: 100 ether,
            baseFee: 0.01 ether,
            gasMultiplier: 2,
            emergencyMode: false,
            merkleRoot: bytes32(0)
        });

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Register as a validator
     */
    function registerValidator(
        string memory endpoint,
        bytes32[] memory supportedChainIds
    ) external payable nonReentrant whenNotPaused {
        require(msg.value >= config.minStake, "Insufficient stake");
        require(!validators[msg.sender].active, "Already registered");
        require(bytes(endpoint).length > 0, "Empty endpoint");

        validators[msg.sender] = Validator({
            validatorAddress: msg.sender,
            stake: msg.value,
            reputation: 100, // Start with perfect reputation
            active: true,
            validatedMessages: 0,
            lastActivity: block.timestamp,
            endpoint: endpoint,
            supportedChains: supportedChainIds
        });

        allValidators.push(msg.sender);
        totalStaked += msg.value;
        _grantRole(VALIDATOR_ROLE, msg.sender);

        emit ValidatorRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Register as a relayer
     */
    function registerRelayer(uint256[] memory authorizedChainIds) 
        external 
        payable 
        nonReentrant 
        whenNotPaused 
    {
        require(msg.value >= config.minStake / 2, "Insufficient stake"); // Lower stake for relayers
        require(!relayers[msg.sender].active, "Already registered");

        RelayerInfo storage relayer = relayers[msg.sender];
        relayer.relayer = msg.sender;
        relayer.stake = msg.value;
        relayer.active = true;
        relayer.reputation = 100;

        for (uint256 i = 0; i < authorizedChainIds.length; i++) {
            relayer.authorizedChains[authorizedChainIds[i]] = true;
        }

        allRelayers.push(msg.sender);
        _grantRole(RELAYER_ROLE, msg.sender);

        emit RelayerRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Send a cross-chain message
     */
    function sendMessage(
        uint256 destinationChainId,
        address receiver,
        MessageType messageType,
        bytes memory payload,
        uint256 executionGas
    ) external payable nonReentrant notInEmergencyMode whenNotPaused returns (bytes32) {
        require(supportedChains[destinationChainId].active, "Destination chain not supported");
        require(receiver != address(0), "Invalid receiver");
        require(payload.length > 0, "Empty payload");

        // Calculate fee
        uint256 fee = calculateFee(payload.length, executionGas, destinationChainId);
        require(msg.value >= fee, "Insufficient fee");

        bytes32 messageId = keccak256(abi.encodePacked(
            block.chainid,
            destinationChainId,
            msg.sender,
            receiver,
            nonces[msg.sender]++,
            block.timestamp
        ));

        require(messages[messageId].sender == address(0), "Message ID collision");

        CrossChainMessage storage message = messages[messageId];
        message.messageId = messageId;
        message.sourceChainId = block.chainid;
        message.destinationChainId = destinationChainId;
        message.sender = msg.sender;
        message.receiver = receiver;
        message.messageType = messageType;
        message.payload = payload;
        message.timestamp = block.timestamp;
        message.blockNumber = block.number;
        message.status = MessageStatus.Pending;
        message.payloadHash = keccak256(payload);
        message.executionGas = executionGas;
        message.fee = fee;

        allMessageIds.push(messageId);
        messageToChain[messageId] = destinationChainId;
        totalMessages++;

        // Transfer fee to treasury
        payable(treasury).transfer(fee);

        // Refund excess
        if (msg.value > fee) {
            payable(msg.sender).transfer(msg.value - fee);
        }

        emit MessageSent(messageId, block.chainid, destinationChainId, msg.sender, messageType);
        return messageId;
    }

    /**
     * @dev Validate a cross-chain message
     */
    function validateMessage(
        bytes32 messageId,
        bool approved,
        bytes memory signature
    ) external validMessage(messageId) onlyActiveValidator whenNotPaused {
        CrossChainMessage storage message = messages[messageId];
        require(message.status == MessageStatus.Pending, "Message not pending");
        require(!message.validatedBy[msg.sender], "Already validated");
        require(block.timestamp <= message.timestamp + config.messageTimeout, "Message expired");

        // Verify signature
        bytes32 messageHash = keccak256(abi.encodePacked(
            messageId,
            approved,
            msg.sender
        ));
        address signer = messageHash.toEthSignedMessageHash().recover(signature);
        require(signer == msg.sender, "Invalid signature");

        message.validatedBy[msg.sender] = true;
        message.validatorCount++;

        validators[msg.sender].validatedMessages++;
        validators[msg.sender].lastActivity = block.timestamp;

        emit MessageValidated(messageId, msg.sender, approved);

        // Check if validation threshold reached
        uint256 requiredValidations = (allValidators.length * config.validationThreshold) / 100;
        if (message.validatorCount >= requiredValidations) {
            message.status = MessageStatus.Validated;
        }
    }

    /**
     * @dev Execute a validated message
     */
    function executeMessage(
        bytes32 messageId,
        bytes memory proof
    ) external validMessage(messageId) onlyActiveRelayer nonReentrant whenNotPaused {
        CrossChainMessage storage message = messages[messageId];
        require(message.status == MessageStatus.Validated, "Message not validated");
        require(message.destinationChainId == block.chainid, "Wrong destination chain");
        require(!processedMessages[messageId], "Message already processed");

        // Verify merkle proof if required
        if (config.merkleRoot != bytes32(0)) {
            require(_verifyMerkleProof(messageId, proof), "Invalid merkle proof");
        }

        processedMessages[messageId] = true;

        // Execute based on message type
        bool success = _executeMessagePayload(message);

        message.status = success ? MessageStatus.Executed : MessageStatus.Failed;

        // Update relayer stats
        relayers[msg.sender].totalRelayed++;
        if (success) {
            relayers[msg.sender].successfulRelays++;
        }

        emit MessageExecuted(messageId, success, "");
    }

    /**
     * @dev Add support for a new chain
     */
    function addSupportedChain(
        uint256 chainId,
        string memory name,
        address bridgeContract
    ) external onlyRole(ADMIN_ROLE) {
        require(chainId != 0, "Invalid chain ID");
        require(bridgeContract != address(0), "Invalid bridge contract");
        require(!supportedChains[chainId].active, "Chain already supported");

        supportedChains[chainId] = ChainInfo({
            chainId: chainId,
            name: name,
            bridgeContract: bridgeContract,
            active: true,
            lastBlockSynced: 0,
            totalMessages: 0,
            stateRoot: bytes32(0)
        });

        chainIds.push(chainId);

        emit ChainAdded(chainId, name);
    }

    /**
     * @dev Calculate bridge fee
     */
    function calculateFee(
        uint256 payloadSize,
        uint256 executionGas,
        uint256 destinationChainId
    ) public view returns (uint256) {
        uint256 baseFee = config.baseFee;
        uint256 sizeFee = (payloadSize * 1e15); // 0.001 ETH per byte
        uint256 gasFee = (executionGas * config.gasMultiplier * 1e9); // Gas price estimation
        
        // Chain-specific multiplier (can be different for each chain)
        uint256 chainMultiplier = 100; // 1.0x (can be adjusted per chain)
        
        return (baseFee + sizeFee + gasFee) * chainMultiplier / 100;
    }

    /**
     * @dev Slash a misbehaving validator
     */
    function slashValidator(
        address validatorAddress,
        uint256 slashAmount,
        string memory reason
    ) external onlyRole(ADMIN_ROLE) {
        Validator storage validator = validators[validatorAddress];
        require(validator.active, "Validator not active");
        require(slashAmount <= validator.stake, "Slash amount exceeds stake");

        validator.stake -= slashAmount;
        validator.reputation = validator.reputation > 10 ? validator.reputation - 10 : 0;
        totalStaked -= slashAmount;

        if (validator.stake < config.minStake) {
            validator.active = false;
            _revokeRole(VALIDATOR_ROLE, validatorAddress);
        }

        // Transfer slashed amount to treasury
        payable(treasury).transfer(slashAmount);

        emit ValidatorSlashed(validatorAddress, slashAmount, reason);
    }

    /**
     * @dev Toggle emergency mode
     */
    function toggleEmergencyMode() external onlyRole(ADMIN_ROLE) {
        config.emergencyMode = !config.emergencyMode;
        emit EmergencyModeToggled(config.emergencyMode);
    }

    /**
     * @dev Update bridge configuration
     */
    function updateConfig(
        uint256 _minValidators,
        uint256 _validationThreshold,
        uint256 _messageTimeout,
        uint256 _minStake,
        uint256 _baseFee
    ) external onlyRole(ADMIN_ROLE) {
        require(_minValidators >= 3, "Need at least 3 validators");
        require(_validationThreshold >= 51 && _validationThreshold <= 100, "Invalid threshold");

        config.minValidators = _minValidators;
        config.validationThreshold = _validationThreshold;
        config.messageTimeout = _messageTimeout;
        config.minStake = _minStake;
        config.baseFee = _baseFee;
    }

    /**
     * @dev Internal function to execute message payload
     */
    function _executeMessagePayload(CrossChainMessage storage message) internal returns (bool) {
        try this.processMessagePayload(
            message.messageType,
            message.sender,
            message.receiver,
            message.payload
        ) {
            return true;
        } catch {
            return false;
        }
    }

    /**
     * @dev Process message payload based on type
     */
    function processMessagePayload(
        MessageType messageType,
        address sender,
        address receiver,
        bytes memory payload
    ) external {
        require(msg.sender == address(this), "Internal call only");

        if (messageType == MessageType.ContractTransfer) {
            _processContractTransfer(sender, receiver, payload);
        } else if (messageType == MessageType.DisputeResolution) {
            _processDisputeResolution(sender, receiver, payload);
        } else if (messageType == MessageType.ComplianceUpdate) {
            _processComplianceUpdate(sender, receiver, payload);
        } else if (messageType == MessageType.TokenTransfer) {
            _processTokenTransfer(sender, receiver, payload);
        }
        // Add more message type handlers as needed
    }

    /**
     * @dev Process contract transfer
     */
    function _processContractTransfer(
        address sender,
        address receiver,
        bytes memory payload
    ) internal {
        // Decode payload and execute contract transfer logic
        (bytes32 contractId, uint256 value, bytes memory data) = abi.decode(payload, (bytes32, uint256, bytes));
        
        // Execute the contract transfer (implementation depends on specific requirements)
        // This could involve calling other contracts, updating state, etc.
    }

    /**
     * @dev Process dispute resolution
     */
    function _processDisputeResolution(
        address sender,
        address receiver,
        bytes memory payload
    ) internal {
        // Decode and process dispute resolution data
        (bytes32 disputeId, bool resolved, address winner, uint256 amount) = abi.decode(
            payload, 
            (bytes32, bool, address, uint256)
        );
        
        // Update local dispute state or execute resolution actions
    }

    /**
     * @dev Process compliance update
     */
    function _processComplianceUpdate(
        address sender,
        address receiver,
        bytes memory payload
    ) internal {
        // Decode and process compliance data
        (string memory jurisdiction, bytes32 ruleHash, bool active) = abi.decode(
            payload, 
            (string, bytes32, bool)
        );
        
        // Update compliance rules or trigger compliance checks
    }

    /**
     * @dev Process token transfer
     */
    function _processTokenTransfer(
        address sender,
        address receiver,
        bytes memory payload
    ) internal {
        // Decode token transfer data
        (address token, uint256 amount) = abi.decode(payload, (address, uint256));
        
        // Execute token transfer (could involve minting, burning, or transferring)
        if (token != address(0) && amount > 0) {
            IERC20(token).safeTransfer(receiver, amount);
        }
    }

    /**
     * @dev Verify merkle proof
     */
    function _verifyMerkleProof(bytes32 messageId, bytes memory proof) internal view returns (bool) {
        bytes32[] memory merkleProof = abi.decode(proof, (bytes32[]));
        return MerkleProof.verify(merkleProof, config.merkleRoot, messageId);
    }

    /**
     * @dev Get message details
     */
    function getMessageDetails(bytes32 messageId) external view returns (
        uint256 sourceChainId,
        uint256 destinationChainId,
        address sender,
        address receiver,
        MessageType messageType,
        MessageStatus status,
        uint256 timestamp,
        uint256 validatorCount
    ) {
        CrossChainMessage storage message = messages[messageId];
        return (
            message.sourceChainId,
            message.destinationChainId,
            message.sender,
            message.receiver,
            message.messageType,
            message.status,
            message.timestamp,
            message.validatorCount
        );
    }

    /**
     * @dev Get validator info
     */
    function getValidatorInfo(address validatorAddress) external view returns (
        uint256 stake,
        uint256 reputation,
        bool active,
        uint256 validatedMessages,
        string memory endpoint
    ) {
        Validator storage validator = validators[validatorAddress];
        return (
            validator.stake,
            validator.reputation,
            validator.active,
            validator.validatedMessages,
            validator.endpoint
        );
    }

    /**
     * @dev Get all message IDs
     */
    function getAllMessageIds() external view returns (bytes32[] memory) {
        return allMessageIds;
    }

    /**
     * @dev Get all validators
     */
    function getAllValidators() external view returns (address[] memory) {
        return allValidators;
    }

    /**
     * @dev Get supported chain IDs
     */
    function getSupportedChains() external view returns (uint256[] memory) {
        return chainIds;
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

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(treasury).transfer(address(this).balance);
    }

    receive() external payable {}
}