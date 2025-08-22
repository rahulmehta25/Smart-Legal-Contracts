// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "./ComplianceOracle.sol";

/**
 * @title ContractExecution
 * @dev Self-executing smart contracts with automated condition checking and compliance verification
 */
contract ContractExecution is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");

    enum ContractStatus {
        Draft,
        Active,
        Paused,
        Executed,
        Breached,
        Terminated,
        Disputed
    }

    enum ConditionType {
        TimeBase,
        EventBased,
        OracleData,
        PaymentReceived,
        ComplianceCheck,
        MultiSignature,
        ExternalAPI
    }

    enum ActionType {
        Transfer,
        MintToken,
        BurnToken,
        StateChange,
        Notification,
        ComplianceReport,
        DisputeEscalation
    }

    struct Condition {
        bytes32 id;
        ConditionType conditionType;
        bytes parameters;
        bool isMet;
        uint256 deadline;
        address verifier;
        mapping(address => bool) confirmations;
        uint256 confirmationCount;
        uint256 requiredConfirmations;
    }

    struct Action {
        bytes32 id;
        ActionType actionType;
        address target;
        bytes data;
        uint256 value;
        bool executed;
        uint256 executedAt;
        bytes32[] requiredConditions;
    }

    struct SmartContract {
        bytes32 id;
        string name;
        string description;
        address[] parties;
        ContractStatus status;
        uint256 createdAt;
        uint256 activatedAt;
        uint256 expiryTime;
        bytes32[] conditionIds;
        bytes32[] actionIds;
        mapping(address => bool) signatures;
        uint256 signatureCount;
        uint256 requiredSignatures;
        string jurisdiction;
        bytes32 complianceCheckId;
        mapping(bytes32 => uint256) conditionDeadlines;
        uint256 executionFee;
        bool autoExecute;
    }

    struct ExecutionContext {
        bytes32 contractId;
        bytes32 triggeredCondition;
        address executor;
        uint256 executionTime;
        string reason;
        bytes32[] executedActions;
    }

    ComplianceOracle public immutable complianceOracle;
    
    mapping(bytes32 => SmartContract) public smartContracts;
    mapping(bytes32 => Condition) public conditions;
    mapping(bytes32 => Action) public actions;
    mapping(bytes32 => ExecutionContext) public executionHistory;
    mapping(address => bytes32[]) public userContracts;
    
    bytes32[] public allContractIds;
    bytes32[] public pendingExecutions;
    
    uint256 public executionFee = 0.001 ether;
    uint256 public constant MAX_CONDITIONS = 50;
    uint256 public constant MAX_ACTIONS = 50;
    uint256 public constant MAX_PARTIES = 20;

    event ContractCreated(bytes32 indexed contractId, address indexed creator, string name);
    event ContractSigned(bytes32 indexed contractId, address indexed signer);
    event ContractActivated(bytes32 indexed contractId);
    event ConditionMet(bytes32 indexed contractId, bytes32 indexed conditionId);
    event ActionExecuted(bytes32 indexed contractId, bytes32 indexed actionId);
    event ContractExecuted(bytes32 indexed contractId, address indexed executor);
    event ContractBreached(bytes32 indexed contractId, string reason);
    event ComplianceVerified(bytes32 indexed contractId, bytes32 complianceCheckId);

    modifier onlyContractParty(bytes32 contractId) {
        bool isParty = false;
        for (uint256 i = 0; i < smartContracts[contractId].parties.length; i++) {
            if (smartContracts[contractId].parties[i] == msg.sender) {
                isParty = true;
                break;
            }
        }
        require(isParty, "Not a contract party");
        _;
    }

    modifier validContract(bytes32 contractId) {
        require(smartContracts[contractId].parties.length > 0, "Contract does not exist");
        _;
    }

    modifier activeContract(bytes32 contractId) {
        require(smartContracts[contractId].status == ContractStatus.Active, "Contract not active");
        _;
    }

    constructor(address _complianceOracle) {
        require(_complianceOracle != address(0), "Invalid compliance oracle");
        complianceOracle = ComplianceOracle(_complianceOracle);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(EXECUTOR_ROLE, msg.sender);
    }

    /**
     * @dev Create a new smart contract
     */
    function createSmartContract(
        string memory name,
        string memory description,
        address[] memory parties,
        uint256 expiryTime,
        string memory jurisdiction,
        bool autoExecute
    ) external payable nonReentrant whenNotPaused returns (bytes32) {
        require(bytes(name).length > 0, "Empty contract name");
        require(parties.length >= 2 && parties.length <= MAX_PARTIES, "Invalid party count");
        require(expiryTime > block.timestamp, "Invalid expiry time");
        require(msg.value >= executionFee, "Insufficient execution fee");

        bytes32 contractId = keccak256(abi.encodePacked(
            name,
            parties,
            msg.sender,
            block.timestamp
        ));

        require(smartContracts[contractId].parties.length == 0, "Contract already exists");

        SmartContract storage newContract = smartContracts[contractId];
        newContract.id = contractId;
        newContract.name = name;
        newContract.description = description;
        newContract.parties = parties;
        newContract.status = ContractStatus.Draft;
        newContract.createdAt = block.timestamp;
        newContract.expiryTime = expiryTime;
        newContract.requiredSignatures = parties.length;
        newContract.jurisdiction = jurisdiction;
        newContract.executionFee = msg.value;
        newContract.autoExecute = autoExecute;

        allContractIds.push(contractId);
        
        // Add to user contracts for all parties
        for (uint256 i = 0; i < parties.length; i++) {
            userContracts[parties[i]].push(contractId);
        }

        emit ContractCreated(contractId, msg.sender, name);
        return contractId;
    }

    /**
     * @dev Add condition to contract
     */
    function addCondition(
        bytes32 contractId,
        ConditionType conditionType,
        bytes memory parameters,
        uint256 deadline,
        uint256 requiredConfirmations
    ) external validContract(contractId) onlyContractParty(contractId) whenNotPaused returns (bytes32) {
        require(smartContracts[contractId].status == ContractStatus.Draft, "Contract not in draft");
        require(smartContracts[contractId].conditionIds.length < MAX_CONDITIONS, "Too many conditions");
        require(requiredConfirmations > 0, "Invalid confirmation count");

        bytes32 conditionId = keccak256(abi.encodePacked(
            contractId,
            conditionType,
            parameters,
            block.timestamp
        ));

        require(conditions[conditionId].id == bytes32(0), "Condition already exists");

        Condition storage newCondition = conditions[conditionId];
        newCondition.id = conditionId;
        newCondition.conditionType = conditionType;
        newCondition.parameters = parameters;
        newCondition.deadline = deadline;
        newCondition.requiredConfirmations = requiredConfirmations;

        smartContracts[contractId].conditionIds.push(conditionId);
        smartContracts[contractId].conditionDeadlines[conditionId] = deadline;

        return conditionId;
    }

    /**
     * @dev Add action to contract
     */
    function addAction(
        bytes32 contractId,
        ActionType actionType,
        address target,
        bytes memory data,
        uint256 value,
        bytes32[] memory requiredConditions
    ) external validContract(contractId) onlyContractParty(contractId) whenNotPaused returns (bytes32) {
        require(smartContracts[contractId].status == ContractStatus.Draft, "Contract not in draft");
        require(smartContracts[contractId].actionIds.length < MAX_ACTIONS, "Too many actions");

        bytes32 actionId = keccak256(abi.encodePacked(
            contractId,
            actionType,
            target,
            data,
            block.timestamp
        ));

        require(actions[actionId].id == bytes32(0), "Action already exists");

        Action storage newAction = actions[actionId];
        newAction.id = actionId;
        newAction.actionType = actionType;
        newAction.target = target;
        newAction.data = data;
        newAction.value = value;
        newAction.requiredConditions = requiredConditions;

        smartContracts[contractId].actionIds.push(actionId);

        return actionId;
    }

    /**
     * @dev Sign contract
     */
    function signContract(bytes32 contractId) 
        external 
        validContract(contractId) 
        onlyContractParty(contractId) 
        whenNotPaused 
    {
        require(smartContracts[contractId].status == ContractStatus.Draft, "Contract not in draft");
        require(!smartContracts[contractId].signatures[msg.sender], "Already signed");

        smartContracts[contractId].signatures[msg.sender] = true;
        smartContracts[contractId].signatureCount++;

        emit ContractSigned(contractId, msg.sender);

        // Auto-activate if all parties signed
        if (smartContracts[contractId].signatureCount >= smartContracts[contractId].requiredSignatures) {
            _activateContract(contractId);
        }
    }

    /**
     * @dev Confirm condition
     */
    function confirmCondition(bytes32 conditionId) 
        external 
        whenNotPaused 
    {
        require(conditions[conditionId].id != bytes32(0), "Condition does not exist");
        require(!conditions[conditionId].confirmations[msg.sender], "Already confirmed");
        require(block.timestamp <= conditions[conditionId].deadline, "Condition deadline passed");

        conditions[conditionId].confirmations[msg.sender] = true;
        conditions[conditionId].confirmationCount++;

        // Check if condition is met
        if (conditions[conditionId].confirmationCount >= conditions[conditionId].requiredConfirmations) {
            conditions[conditionId].isMet = true;
            
            // Find associated contract and trigger execution check
            bytes32 contractId = _findContractByCondition(conditionId);
            emit ConditionMet(contractId, conditionId);
            
            if (smartContracts[contractId].autoExecute) {
                _checkAndExecuteActions(contractId);
            }
        }
    }

    /**
     * @dev Execute contract manually
     */
    function executeContract(bytes32 contractId) 
        external 
        validContract(contractId) 
        activeContract(contractId) 
        whenNotPaused 
    {
        require(
            hasRole(EXECUTOR_ROLE, msg.sender) || 
            _isContractParty(contractId, msg.sender),
            "Not authorized to execute"
        );

        _checkAndExecuteActions(contractId);
    }

    /**
     * @dev Verify compliance before activation
     */
    function verifyCompliance(bytes32 contractId) 
        external 
        payable 
        validContract(contractId) 
        whenNotPaused 
    {
        require(smartContracts[contractId].status == ContractStatus.Draft, "Contract not in draft");
        
        // Get required compliance data from oracle
        string[] memory regulations = new string[](1);
        regulations[0] = "SMART_CONTRACT_COMPLIANCE";
        
        bytes32 complianceCheckId = complianceOracle.checkCompliance{value: msg.value}(
            address(this),
            smartContracts[contractId].jurisdiction,
            regulations
        );
        
        smartContracts[contractId].complianceCheckId = complianceCheckId;
        emit ComplianceVerified(contractId, complianceCheckId);
    }

    /**
     * @dev Report contract breach
     */
    function reportBreach(bytes32 contractId, string memory reason) 
        external 
        validContract(contractId) 
        onlyContractParty(contractId) 
        whenNotPaused 
    {
        require(bytes(reason).length > 0, "Empty breach reason");
        require(
            smartContracts[contractId].status == ContractStatus.Active ||
            smartContracts[contractId].status == ContractStatus.Paused,
            "Invalid contract status"
        );

        smartContracts[contractId].status = ContractStatus.Breached;
        emit ContractBreached(contractId, reason);
    }

    /**
     * @dev Internal function to activate contract
     */
    function _activateContract(bytes32 contractId) internal {
        smartContracts[contractId].status = ContractStatus.Active;
        smartContracts[contractId].activatedAt = block.timestamp;
        
        // Add to pending executions if auto-execute enabled
        if (smartContracts[contractId].autoExecute) {
            pendingExecutions.push(contractId);
        }
        
        emit ContractActivated(contractId);
    }

    /**
     * @dev Internal function to check and execute actions
     */
    function _checkAndExecuteActions(bytes32 contractId) internal {
        bytes32[] memory actionIds = smartContracts[contractId].actionIds;
        
        for (uint256 i = 0; i < actionIds.length; i++) {
            bytes32 actionId = actionIds[i];
            
            if (!actions[actionId].executed && _areConditionsMet(actionId)) {
                _executeAction(contractId, actionId);
            }
        }

        // Check if all actions are executed
        bool allExecuted = true;
        for (uint256 i = 0; i < actionIds.length; i++) {
            if (!actions[actionIds[i]].executed) {
                allExecuted = false;
                break;
            }
        }

        if (allExecuted) {
            smartContracts[contractId].status = ContractStatus.Executed;
            emit ContractExecuted(contractId, msg.sender);
        }
    }

    /**
     * @dev Internal function to execute action
     */
    function _executeAction(bytes32 contractId, bytes32 actionId) internal {
        Action storage action = actions[actionId];
        
        if (action.actionType == ActionType.Transfer) {
            _executeTransfer(action);
        } else if (action.actionType == ActionType.StateChange) {
            _executeStateChange(action);
        } else if (action.actionType == ActionType.Notification) {
            _executeNotification(action);
        }
        // Add more action types as needed

        action.executed = true;
        action.executedAt = block.timestamp;

        emit ActionExecuted(contractId, actionId);
    }

    /**
     * @dev Internal function to execute transfer
     */
    function _executeTransfer(Action storage action) internal {
        if (action.value > 0 && address(this).balance >= action.value) {
            payable(action.target).transfer(action.value);
        }
    }

    /**
     * @dev Internal function to execute state change
     */
    function _executeStateChange(Action storage action) internal {
        // Implement state change logic based on action.data
        // This could involve calling external contracts, updating internal state, etc.
    }

    /**
     * @dev Internal function to execute notification
     */
    function _executeNotification(Action storage action) internal {
        // Emit event or trigger external notification system
        // Implementation depends on notification infrastructure
    }

    /**
     * @dev Internal function to check if all required conditions are met
     */
    function _areConditionsMet(bytes32 actionId) internal view returns (bool) {
        bytes32[] memory requiredConditions = actions[actionId].requiredConditions;
        
        for (uint256 i = 0; i < requiredConditions.length; i++) {
            if (!conditions[requiredConditions[i]].isMet) {
                return false;
            }
        }
        
        return true;
    }

    /**
     * @dev Internal function to find contract by condition
     */
    function _findContractByCondition(bytes32 conditionId) internal view returns (bytes32) {
        for (uint256 i = 0; i < allContractIds.length; i++) {
            bytes32 contractId = allContractIds[i];
            bytes32[] memory conditionIds = smartContracts[contractId].conditionIds;
            
            for (uint256 j = 0; j < conditionIds.length; j++) {
                if (conditionIds[j] == conditionId) {
                    return contractId;
                }
            }
        }
        return bytes32(0);
    }

    /**
     * @dev Internal function to check if address is contract party
     */
    function _isContractParty(bytes32 contractId, address addr) internal view returns (bool) {
        for (uint256 i = 0; i < smartContracts[contractId].parties.length; i++) {
            if (smartContracts[contractId].parties[i] == addr) {
                return true;
            }
        }
        return false;
    }

    /**
     * @dev Get contract details
     */
    function getContractDetails(bytes32 contractId) external view returns (
        string memory name,
        string memory description,
        address[] memory parties,
        ContractStatus status,
        uint256 createdAt,
        uint256 expiryTime,
        uint256 signatureCount,
        uint256 requiredSignatures
    ) {
        SmartContract storage contractData = smartContracts[contractId];
        return (
            contractData.name,
            contractData.description,
            contractData.parties,
            contractData.status,
            contractData.createdAt,
            contractData.expiryTime,
            contractData.signatureCount,
            contractData.requiredSignatures
        );
    }

    /**
     * @dev Get condition details
     */
    function getConditionDetails(bytes32 conditionId) external view returns (
        ConditionType conditionType,
        bytes memory parameters,
        bool isMet,
        uint256 deadline,
        uint256 confirmationCount,
        uint256 requiredConfirmations
    ) {
        Condition storage condition = conditions[conditionId];
        return (
            condition.conditionType,
            condition.parameters,
            condition.isMet,
            condition.deadline,
            condition.confirmationCount,
            condition.requiredConfirmations
        );
    }

    /**
     * @dev Get action details
     */
    function getActionDetails(bytes32 actionId) external view returns (
        ActionType actionType,
        address target,
        bytes memory data,
        uint256 value,
        bool executed,
        uint256 executedAt
    ) {
        Action storage action = actions[actionId];
        return (
            action.actionType,
            action.target,
            action.data,
            action.value,
            action.executed,
            action.executedAt
        );
    }

    /**
     * @dev Get user contracts
     */
    function getUserContracts(address user) external view returns (bytes32[] memory) {
        return userContracts[user];
    }

    /**
     * @dev Get total contracts
     */
    function getTotalContracts() external view returns (uint256) {
        return allContractIds.length;
    }

    /**
     * @dev Get pending executions
     */
    function getPendingExecutions() external view returns (bytes32[] memory) {
        return pendingExecutions;
    }

    /**
     * @dev Batch execute pending contracts (for automation)
     */
    function batchExecutePending(uint256 maxExecutions) external onlyRole(EXECUTOR_ROLE) whenNotPaused {
        uint256 executed = 0;
        uint256 i = 0;
        
        while (i < pendingExecutions.length && executed < maxExecutions) {
            bytes32 contractId = pendingExecutions[i];
            
            if (smartContracts[contractId].status == ContractStatus.Active) {
                _checkAndExecuteActions(contractId);
                executed++;
            }
            
            i++;
        }
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

    function setExecutionFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        executionFee = newFee;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}