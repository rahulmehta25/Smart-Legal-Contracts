// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/Address.sol";

/**
 * @title LegalMultiSigWallet
 * @dev Advanced multi-signature wallet with role-based access, time-locked transactions, and legal compliance features
 */
contract LegalMultiSigWallet is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;
    using Address for address;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant SIGNER_ROLE = keccak256("SIGNER_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");

    enum TransactionStatus {
        Pending,
        Approved,
        Executed,
        Cancelled,
        Expired,
        Rejected
    }

    enum TransactionType {
        EtherTransfer,
        TokenTransfer,
        ContractCall,
        AddSigner,
        RemoveSigner,
        ChangeRequirement,
        EmergencyAction,
        ComplianceAction,
        GovernanceAction
    }

    enum Priority {
        Low,
        Medium,
        High,
        Critical,
        Emergency
    }

    struct Transaction {
        uint256 id;
        address destination;
        uint256 value;
        bytes data;
        TransactionType txType;
        Priority priority;
        uint256 createdAt;
        uint256 deadline;
        uint256 timelock;
        address proposer;
        TransactionStatus status;
        uint256 confirmations;
        uint256 rejections;
        mapping(address => bool) isConfirmed;
        mapping(address => bool) isRejected;
        mapping(address => uint256) confirmationTime;
        string description;
        bytes32 complianceHash;
        bool complianceApproved;
        uint256 executedAt;
        bytes executionResult;
    }

    struct Signer {
        address signerAddress;
        string name;
        string role;
        uint256 addedAt;
        uint256 weight; // Voting weight (1-10)
        bool active;
        uint256 totalSigned;
        uint256 lastActivity;
        mapping(TransactionType => bool) permissions;
    }

    struct ComplianceRule {
        uint256 id;
        string description;
        uint256 minApprovals;
        uint256 timeDelay;
        bool requiresCompliance;
        mapping(TransactionType => bool) applicableTypes;
        bool active;
    }

    struct EmergencyConfig {
        bool emergencyMode;
        uint256 emergencyThreshold;
        uint256 emergencyDelay;
        address emergencyContact;
        uint256 lastEmergencyAction;
        mapping(address => bool) emergencySigners;
    }

    mapping(uint256 => Transaction) public transactions;
    mapping(address => Signer) public signers;
    mapping(uint256 => ComplianceRule) public complianceRules;
    mapping(bytes32 => bool) public usedNonces;
    mapping(address => uint256) public dailyLimits;
    mapping(address => mapping(uint256 => uint256)) public dailySpent; // address => day => amount
    
    address[] public signerList;
    uint256[] public transactionList;
    uint256 public transactionCount;
    uint256 public requiredSignatures;
    uint256 public totalWeight;
    uint256 public requiredWeight;
    
    EmergencyConfig public emergencyConfig;
    
    uint256 public constant MAX_SIGNERS = 50;
    uint256 public constant MIN_CONFIRMATION_TIME = 1 hours;
    uint256 public constant MAX_TRANSACTION_DELAY = 30 days;
    uint256 public constant EMERGENCY_DELAY = 24 hours;

    event TransactionSubmitted(
        uint256 indexed transactionId,
        address indexed proposer,
        address indexed destination,
        uint256 value,
        TransactionType txType
    );
    
    event TransactionConfirmed(
        uint256 indexed transactionId,
        address indexed signer,
        uint256 confirmations
    );
    
    event TransactionRejected(
        uint256 indexed transactionId,
        address indexed signer,
        uint256 rejections
    );
    
    event TransactionExecuted(
        uint256 indexed transactionId,
        address indexed executor,
        bool success
    );
    
    event SignerAdded(address indexed signer, string name, uint256 weight);
    event SignerRemoved(address indexed signer);
    event RequirementChanged(uint256 required, uint256 requiredWeight);
    event EmergencyModeToggled(bool enabled);
    event ComplianceRuleAdded(uint256 indexed ruleId, string description);
    event DailyLimitChanged(address indexed token, uint256 newLimit);

    modifier onlySigners() {
        require(signers[msg.sender].active, "Not an active signer");
        _;
    }

    modifier transactionExists(uint256 transactionId) {
        require(transactionId < transactionCount, "Transaction does not exist");
        _;
    }

    modifier notExecuted(uint256 transactionId) {
        require(transactions[transactionId].status != TransactionStatus.Executed, "Transaction already executed");
        _;
    }

    modifier notExpired(uint256 transactionId) {
        require(
            transactions[transactionId].deadline == 0 || 
            block.timestamp <= transactions[transactionId].deadline,
            "Transaction expired"
        );
        _;
    }

    modifier canExecute(uint256 transactionId) {
        Transaction storage txn = transactions[transactionId];
        require(txn.status == TransactionStatus.Approved, "Transaction not approved");
        require(
            txn.timelock == 0 || block.timestamp >= txn.timelock,
            "Transaction still timelocked"
        );
        _;
    }

    constructor(
        address[] memory _signers,
        string[] memory _signerNames,
        uint256[] memory _signerWeights,
        uint256 _requiredSignatures,
        uint256 _requiredWeight
    ) {
        require(_signers.length == _signerNames.length, "Mismatched arrays");
        require(_signers.length == _signerWeights.length, "Mismatched arrays");
        require(_signers.length <= MAX_SIGNERS, "Too many signers");
        require(_requiredSignatures > 0 && _requiredSignatures <= _signers.length, "Invalid required signatures");

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);

        uint256 _totalWeight = 0;
        for (uint256 i = 0; i < _signers.length; i++) {
            require(_signers[i] != address(0), "Invalid signer address");
            require(_signerWeights[i] > 0 && _signerWeights[i] <= 10, "Invalid signer weight");
            
            signers[_signers[i]].signerAddress = _signers[i];
            signers[_signers[i]].name = _signerNames[i];
            signers[_signers[i]].addedAt = block.timestamp;
            signers[_signers[i]].weight = _signerWeights[i];
            signers[_signers[i]].active = true;
            
            // Grant default permissions
            signers[_signers[i]].permissions[TransactionType.EtherTransfer] = true;
            signers[_signers[i]].permissions[TransactionType.TokenTransfer] = true;
            signers[_signers[i]].permissions[TransactionType.ContractCall] = true;
            
            signerList.push(_signers[i]);
            _grantRole(SIGNER_ROLE, _signers[i]);
            _totalWeight += _signerWeights[i];
        }

        requiredSignatures = _requiredSignatures;
        requiredWeight = _requiredWeight;
        totalWeight = _totalWeight;
        
        // Initialize emergency config
        emergencyConfig.emergencyThreshold = _requiredSignatures;
        emergencyConfig.emergencyDelay = EMERGENCY_DELAY;
    }

    /**
     * @dev Submit a new transaction
     */
    function submitTransaction(
        address destination,
        uint256 value,
        bytes memory data,
        TransactionType txType,
        Priority priority,
        uint256 deadline,
        string memory description
    ) external onlySigners whenNotPaused returns (uint256) {
        require(destination != address(0), "Invalid destination");
        require(signers[msg.sender].permissions[txType], "No permission for transaction type");
        require(deadline == 0 || deadline > block.timestamp, "Invalid deadline");

        uint256 transactionId = transactionCount++;
        
        Transaction storage txn = transactions[transactionId];
        txn.id = transactionId;
        txn.destination = destination;
        txn.value = value;
        txn.data = data;
        txn.txType = txType;
        txn.priority = priority;
        txn.createdAt = block.timestamp;
        txn.deadline = deadline;
        txn.proposer = msg.sender;
        txn.status = TransactionStatus.Pending;
        txn.description = description;
        
        // Apply timelock based on transaction type and priority
        txn.timelock = _calculateTimelock(txType, priority, value);
        
        transactionList.push(transactionId);

        emit TransactionSubmitted(transactionId, msg.sender, destination, value, txType);
        
        // Auto-confirm by proposer
        _confirmTransaction(transactionId);
        
        return transactionId;
    }

    /**
     * @dev Confirm a transaction
     */
    function confirmTransaction(uint256 transactionId)
        external
        onlySigners
        transactionExists(transactionId)
        notExecuted(transactionId)
        notExpired(transactionId)
        whenNotPaused
    {
        _confirmTransaction(transactionId);
    }

    /**
     * @dev Reject a transaction
     */
    function rejectTransaction(uint256 transactionId)
        external
        onlySigners
        transactionExists(transactionId)
        notExecuted(transactionId)
        notExpired(transactionId)
        whenNotPaused
    {
        Transaction storage txn = transactions[transactionId];
        require(!txn.isRejected[msg.sender], "Already rejected");
        require(!txn.isConfirmed[msg.sender], "Already confirmed");

        txn.isRejected[msg.sender] = true;
        txn.rejections += signers[msg.sender].weight;

        emit TransactionRejected(transactionId, msg.sender, txn.rejections);

        // Check if transaction should be rejected
        if (txn.rejections > totalWeight - requiredWeight) {
            txn.status = TransactionStatus.Rejected;
        }
    }

    /**
     * @dev Execute an approved transaction
     */
    function executeTransaction(uint256 transactionId)
        external
        transactionExists(transactionId)
        canExecute(transactionId)
        nonReentrant
        whenNotPaused
        returns (bool success)
    {
        Transaction storage txn = transactions[transactionId];
        
        // Check compliance if required
        if (txn.complianceHash != bytes32(0)) {
            require(txn.complianceApproved, "Compliance approval required");
        }
        
        // Check daily limits for transfers
        if (txn.txType == TransactionType.EtherTransfer || txn.txType == TransactionType.TokenTransfer) {
            _checkDailyLimit(txn.destination, txn.value);
        }

        txn.status = TransactionStatus.Executed;
        txn.executedAt = block.timestamp;

        // Execute the transaction
        if (txn.txType == TransactionType.EtherTransfer) {
            success = _executeEtherTransfer(txn);
        } else if (txn.txType == TransactionType.TokenTransfer) {
            success = _executeTokenTransfer(txn);
        } else if (txn.txType == TransactionType.ContractCall) {
            success = _executeContractCall(txn);
        } else if (txn.txType == TransactionType.AddSigner) {
            success = _executeAddSigner(txn);
        } else if (txn.txType == TransactionType.RemoveSigner) {
            success = _executeRemoveSigner(txn);
        } else if (txn.txType == TransactionType.ChangeRequirement) {
            success = _executeChangeRequirement(txn);
        }

        emit TransactionExecuted(transactionId, msg.sender, success);
        return success;
    }

    /**
     * @dev Add a new signer
     */
    function addSigner(
        address signerAddress,
        string memory name,
        uint256 weight
    ) external onlyRole(ADMIN_ROLE) {
        require(signerAddress != address(0), "Invalid signer address");
        require(!signers[signerAddress].active, "Signer already exists");
        require(signerList.length < MAX_SIGNERS, "Too many signers");
        require(weight > 0 && weight <= 10, "Invalid weight");

        signers[signerAddress] = Signer({
            signerAddress: signerAddress,
            name: name,
            role: "Signer",
            addedAt: block.timestamp,
            weight: weight,
            active: true,
            totalSigned: 0,
            lastActivity: block.timestamp
        });

        // Set default permissions
        signers[signerAddress].permissions[TransactionType.EtherTransfer] = true;
        signers[signerAddress].permissions[TransactionType.TokenTransfer] = true;
        signers[signerAddress].permissions[TransactionType.ContractCall] = true;

        signerList.push(signerAddress);
        totalWeight += weight;
        _grantRole(SIGNER_ROLE, signerAddress);

        emit SignerAdded(signerAddress, name, weight);
    }

    /**
     * @dev Remove a signer
     */
    function removeSigner(address signerAddress) external onlyRole(ADMIN_ROLE) {
        require(signers[signerAddress].active, "Signer not active");
        require(signerList.length > requiredSignatures, "Cannot remove - would break requirement");

        signers[signerAddress].active = false;
        totalWeight -= signers[signerAddress].weight;
        _revokeRole(SIGNER_ROLE, signerAddress);

        // Remove from signer list
        for (uint256 i = 0; i < signerList.length; i++) {
            if (signerList[i] == signerAddress) {
                signerList[i] = signerList[signerList.length - 1];
                signerList.pop();
                break;
            }
        }

        emit SignerRemoved(signerAddress);
    }

    /**
     * @dev Change signature requirements
     */
    function changeRequirement(uint256 _required, uint256 _requiredWeight) external onlyRole(ADMIN_ROLE) {
        require(_required <= signerList.length, "Required signatures too high");
        require(_requiredWeight <= totalWeight, "Required weight too high");
        require(_required > 0 && _requiredWeight > 0, "Requirements must be positive");

        requiredSignatures = _required;
        requiredWeight = _requiredWeight;

        emit RequirementChanged(_required, _requiredWeight);
    }

    /**
     * @dev Set daily spending limit
     */
    function setDailyLimit(address token, uint256 limit) external onlyRole(ADMIN_ROLE) {
        dailyLimits[token] = limit;
        emit DailyLimitChanged(token, limit);
    }

    /**
     * @dev Toggle emergency mode
     */
    function toggleEmergencyMode() external onlyRole(EMERGENCY_ROLE) {
        emergencyConfig.emergencyMode = !emergencyConfig.emergencyMode;
        emergencyConfig.lastEmergencyAction = block.timestamp;
        emit EmergencyModeToggled(emergencyConfig.emergencyMode);
    }

    /**
     * @dev Add compliance rule
     */
    function addComplianceRule(
        string memory description,
        uint256 minApprovals,
        uint256 timeDelay,
        TransactionType[] memory applicableTypes
    ) external onlyRole(COMPLIANCE_ROLE) returns (uint256) {
        uint256 ruleId = block.timestamp; // Simple ID generation
        
        ComplianceRule storage rule = complianceRules[ruleId];
        rule.id = ruleId;
        rule.description = description;
        rule.minApprovals = minApprovals;
        rule.timeDelay = timeDelay;
        rule.requiresCompliance = true;
        rule.active = true;

        for (uint256 i = 0; i < applicableTypes.length; i++) {
            rule.applicableTypes[applicableTypes[i]] = true;
        }

        emit ComplianceRuleAdded(ruleId, description);
        return ruleId;
    }

    /**
     * @dev Internal function to confirm transaction
     */
    function _confirmTransaction(uint256 transactionId) internal {
        Transaction storage txn = transactions[transactionId];
        require(!txn.isConfirmed[msg.sender], "Already confirmed");
        require(!txn.isRejected[msg.sender], "Already rejected");

        txn.isConfirmed[msg.sender] = true;
        txn.confirmationTime[msg.sender] = block.timestamp;
        txn.confirmations += signers[msg.sender].weight;

        signers[msg.sender].totalSigned++;
        signers[msg.sender].lastActivity = block.timestamp;

        emit TransactionConfirmed(transactionId, msg.sender, txn.confirmations);

        // Check if transaction can be approved
        bool hasEnoughSignatures = _countConfirmations(transactionId) >= requiredSignatures;
        bool hasEnoughWeight = txn.confirmations >= requiredWeight;

        if (hasEnoughSignatures && hasEnoughWeight) {
            txn.status = TransactionStatus.Approved;
        }
    }

    /**
     * @dev Calculate timelock for transaction
     */
    function _calculateTimelock(
        TransactionType txType,
        Priority priority,
        uint256 value
    ) internal view returns (uint256) {
        if (priority == Priority.Emergency) return 0;
        
        uint256 baseDelay = MIN_CONFIRMATION_TIME;
        
        if (txType == TransactionType.AddSigner || txType == TransactionType.RemoveSigner) {
            baseDelay = 24 hours;
        } else if (txType == TransactionType.ChangeRequirement) {
            baseDelay = 48 hours;
        } else if (value > 100 ether) {
            baseDelay = 12 hours;
        }
        
        // Adjust based on priority
        if (priority == Priority.Low) {
            baseDelay *= 2;
        } else if (priority == Priority.High) {
            baseDelay /= 2;
        }
        
        return block.timestamp + baseDelay;
    }

    /**
     * @dev Check daily spending limit
     */
    function _checkDailyLimit(address token, uint256 amount) internal {
        uint256 today = block.timestamp / 1 days;
        uint256 limit = dailyLimits[token];
        
        if (limit > 0) {
            require(dailySpent[token][today] + amount <= limit, "Daily limit exceeded");
            dailySpent[token][today] += amount;
        }
    }

    /**
     * @dev Count actual confirmations (not weighted)
     */
    function _countConfirmations(uint256 transactionId) internal view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < signerList.length; i++) {
            if (transactions[transactionId].isConfirmed[signerList[i]]) {
                count++;
            }
        }
        return count;
    }

    /**
     * @dev Execute ether transfer
     */
    function _executeEtherTransfer(Transaction storage txn) internal returns (bool) {
        require(address(this).balance >= txn.value, "Insufficient balance");
        (bool success,) = txn.destination.call{value: txn.value}("");
        return success;
    }

    /**
     * @dev Execute token transfer
     */
    function _executeTokenTransfer(Transaction storage txn) internal returns (bool) {
        (address token, uint256 amount) = abi.decode(txn.data, (address, uint256));
        try IERC20(token).transfer(txn.destination, amount) {
            return true;
        } catch {
            return false;
        }
    }

    /**
     * @dev Execute contract call
     */
    function _executeContractCall(Transaction storage txn) internal returns (bool) {
        (bool success, bytes memory result) = txn.destination.call{value: txn.value}(txn.data);
        txn.executionResult = result;
        return success;
    }

    /**
     * @dev Execute add signer
     */
    function _executeAddSigner(Transaction storage txn) internal returns (bool) {
        (address newSigner, string memory name, uint256 weight) = abi.decode(
            txn.data, 
            (address, string, uint256)
        );
        
        if (!signers[newSigner].active && signerList.length < MAX_SIGNERS) {
            // Add signer logic here
            return true;
        }
        return false;
    }

    /**
     * @dev Execute remove signer
     */
    function _executeRemoveSigner(Transaction storage txn) internal returns (bool) {
        address signerToRemove = abi.decode(txn.data, (address));
        
        if (signers[signerToRemove].active && signerList.length > requiredSignatures) {
            // Remove signer logic here
            return true;
        }
        return false;
    }

    /**
     * @dev Execute change requirement
     */
    function _executeChangeRequirement(Transaction storage txn) internal returns (bool) {
        (uint256 newRequired, uint256 newRequiredWeight) = abi.decode(txn.data, (uint256, uint256));
        
        if (newRequired <= signerList.length && newRequiredWeight <= totalWeight) {
            requiredSignatures = newRequired;
            requiredWeight = newRequiredWeight;
            return true;
        }
        return false;
    }

    /**
     * @dev Get transaction details
     */
    function getTransactionDetails(uint256 transactionId) external view returns (
        address destination,
        uint256 value,
        TransactionType txType,
        TransactionStatus status,
        uint256 confirmations,
        uint256 rejections,
        address proposer,
        uint256 createdAt
    ) {
        Transaction storage txn = transactions[transactionId];
        return (
            txn.destination,
            txn.value,
            txn.txType,
            txn.status,
            txn.confirmations,
            txn.rejections,
            txn.proposer,
            txn.createdAt
        );
    }

    /**
     * @dev Get signer info
     */
    function getSignerInfo(address signerAddress) external view returns (
        string memory name,
        uint256 weight,
        bool active,
        uint256 totalSigned,
        uint256 lastActivity
    ) {
        Signer storage signer = signers[signerAddress];
        return (
            signer.name,
            signer.weight,
            signer.active,
            signer.totalSigned,
            signer.lastActivity
        );
    }

    /**
     * @dev Get all signers
     */
    function getSigners() external view returns (address[] memory) {
        return signerList;
    }

    /**
     * @dev Get transaction count
     */
    function getTransactionCount() external view returns (uint256) {
        return transactionCount;
    }

    /**
     * @dev Emergency functions
     */
    function pause() external onlyRole(EMERGENCY_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(EMERGENCY_ROLE) {
        _unpause();
    }

    /**
     * @dev Receive function to accept Ether
     */
    receive() external payable {}

    /**
     * @dev Fallback function
     */
    fallback() external payable {}
}