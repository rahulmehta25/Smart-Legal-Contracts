// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/governance/TimelockController.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title LegalTimelockController
 * @dev Enhanced timelock controller with legal-specific features and emergency capabilities
 */
contract LegalTimelockController is TimelockController, ReentrancyGuard, Pausable {
    bytes32 public constant LEGAL_ADMIN_ROLE = keccak256("LEGAL_ADMIN_ROLE");
    bytes32 public constant COMPLIANCE_OFFICER_ROLE = keccak256("COMPLIANCE_OFFICER_ROLE");
    bytes32 public constant EMERGENCY_GUARDIAN_ROLE = keccak256("EMERGENCY_GUARDIAN_ROLE");
    bytes32 public constant SECURITY_COUNCIL_ROLE = keccak256("SECURITY_COUNCIL_ROLE");

    enum OperationType {
        Standard,
        Emergency,
        Compliance,
        Security,
        Governance
    }

    enum OperationStatus {
        Pending,
        Ready,
        Executed,
        Cancelled,
        Blocked,
        Emergency
    }

    struct EnhancedOperation {
        bytes32 id;
        OperationType operationType;
        OperationStatus status;
        uint256 createdAt;
        uint256 scheduledFor;
        uint256 emergencyDelay;
        address proposer;
        string description;
        bool requiresCompliance;
        bool complianceApproved;
        address complianceOfficer;
        mapping(address => bool) guardianApprovals;
        uint256 guardianApprovalsCount;
        mapping(address => bool) securityCouncilApprovals;
        uint256 securityCouncilApprovalsCount;
        bytes32[] dependencies;
        uint256 executionWindow;
        bool canBeEmergencyExecuted;
    }

    struct ComplianceCheck {
        bytes32 operationId;
        address officer;
        bool approved;
        string comments;
        uint256 checkedAt;
        string[] requirements;
        bool[] requirementsMet;
    }

    struct EmergencyConfig {
        bool emergencyMode;
        uint256 emergencyDelay;
        uint256 minGuardianApprovals;
        uint256 emergencyDuration;
        uint256 emergencyActivatedAt;
        mapping(address => bool) emergencyGuardians;
        uint256 totalGuardians;
    }

    mapping(bytes32 => EnhancedOperation) public enhancedOperations;
    mapping(bytes32 => ComplianceCheck) public complianceChecks;
    mapping(OperationType => uint256) public typeDelays;
    mapping(OperationType => uint256) public typeRequiredApprovals;
    
    EmergencyConfig public emergencyConfig;
    
    uint256 public constant EMERGENCY_DELAY = 6 hours;
    uint256 public constant COMPLIANCE_WINDOW = 72 hours;
    uint256 public constant SECURITY_DELAY = 48 hours;
    uint256 public constant MAX_EMERGENCY_DURATION = 7 days;

    event OperationScheduledEnhanced(
        bytes32 indexed id,
        OperationType operationType,
        address indexed proposer,
        uint256 delay
    );
    
    event ComplianceCheckCompleted(
        bytes32 indexed operationId,
        address indexed officer,
        bool approved
    );
    
    event GuardianApprovalGiven(
        bytes32 indexed operationId,
        address indexed guardian
    );
    
    event EmergencyModeActivated(uint256 activatedAt, uint256 duration);
    event EmergencyModeDeactivated(uint256 deactivatedAt);
    
    event OperationBlocked(
        bytes32 indexed operationId,
        address indexed blocker,
        string reason
    );

    modifier onlyInEmergency() {
        require(emergencyConfig.emergencyMode, "Not in emergency mode");
        require(
            block.timestamp <= emergencyConfig.emergencyActivatedAt + emergencyConfig.emergencyDuration,
            "Emergency period expired"
        );
        _;
    }

    modifier onlyGuardian() {
        require(emergencyConfig.emergencyGuardians[msg.sender], "Not an emergency guardian");
        _;
    }

    modifier validOperation(bytes32 id) {
        require(enhancedOperations[id].proposer != address(0), "Operation does not exist");
        _;
    }

    constructor(
        uint256 minDelay,
        address[] memory proposers,
        address[] memory executors,
        address[] memory emergencyGuardians,
        address admin
    ) TimelockController(minDelay, proposers, executors, admin) {
        _grantRole(LEGAL_ADMIN_ROLE, admin);
        
        // Initialize operation type delays
        typeDelays[OperationType.Standard] = minDelay;
        typeDelays[OperationType.Emergency] = EMERGENCY_DELAY;
        typeDelays[OperationType.Compliance] = COMPLIANCE_WINDOW;
        typeDelays[OperationType.Security] = SECURITY_DELAY;
        typeDelays[OperationType.Governance] = minDelay * 2;

        // Initialize required approvals
        typeRequiredApprovals[OperationType.Emergency] = 3;
        typeRequiredApprovals[OperationType.Security] = 2;
        typeRequiredApprovals[OperationType.Compliance] = 1;

        // Set up emergency guardians
        emergencyConfig.emergencyDelay = EMERGENCY_DELAY;
        emergencyConfig.minGuardianApprovals = (emergencyGuardians.length * 60) / 100; // 60% of guardians
        emergencyConfig.emergencyDuration = MAX_EMERGENCY_DURATION;

        for (uint256 i = 0; i < emergencyGuardians.length; i++) {
            emergencyConfig.emergencyGuardians[emergencyGuardians[i]] = true;
            _grantRole(EMERGENCY_GUARDIAN_ROLE, emergencyGuardians[i]);
        }
        emergencyConfig.totalGuardians = emergencyGuardians.length;
    }

    /**
     * @dev Schedule an enhanced operation with additional metadata
     */
    function scheduleEnhanced(
        address target,
        uint256 value,
        bytes calldata data,
        bytes32 predecessor,
        bytes32 salt,
        OperationType operationType,
        string memory description,
        bool requiresCompliance,
        bytes32[] memory dependencies,
        uint256 executionWindow
    ) external returns (bytes32) {
        require(hasRole(PROPOSER_ROLE, msg.sender), "Not authorized proposer");
        
        bytes32 id = hashOperation(target, value, data, predecessor, salt);
        require(enhancedOperations[id].proposer == address(0), "Operation already exists");

        uint256 delay = typeDelays[operationType];
        
        // Schedule the base operation
        schedule(target, value, data, predecessor, salt, delay);

        // Create enhanced operation record
        EnhancedOperation storage op = enhancedOperations[id];
        op.id = id;
        op.operationType = operationType;
        op.status = OperationStatus.Pending;
        op.createdAt = block.timestamp;
        op.scheduledFor = block.timestamp + delay;
        op.proposer = msg.sender;
        op.description = description;
        op.requiresCompliance = requiresCompliance;
        op.dependencies = dependencies;
        op.executionWindow = executionWindow;
        op.canBeEmergencyExecuted = (operationType == OperationType.Emergency);

        if (operationType == OperationType.Emergency) {
            op.emergencyDelay = EMERGENCY_DELAY;
        }

        emit OperationScheduledEnhanced(id, operationType, msg.sender, delay);
        return id;
    }

    /**
     * @dev Perform compliance check
     */
    function performComplianceCheck(
        bytes32 operationId,
        bool approved,
        string memory comments,
        string[] memory requirements,
        bool[] memory requirementsMet
    ) external onlyRole(COMPLIANCE_OFFICER_ROLE) validOperation(operationId) {
        EnhancedOperation storage op = enhancedOperations[operationId];
        require(op.requiresCompliance, "Operation does not require compliance");
        require(!op.complianceApproved, "Compliance already checked");

        op.complianceApproved = approved;
        op.complianceOfficer = msg.sender;

        complianceChecks[operationId] = ComplianceCheck({
            operationId: operationId,
            officer: msg.sender,
            approved: approved,
            comments: comments,
            checkedAt: block.timestamp,
            requirements: requirements,
            requirementsMet: requirementsMet
        });

        if (approved) {
            op.status = OperationStatus.Ready;
        } else {
            op.status = OperationStatus.Blocked;
        }

        emit ComplianceCheckCompleted(operationId, msg.sender, approved);
    }

    /**
     * @dev Provide guardian approval for emergency operations
     */
    function provideGuardianApproval(bytes32 operationId) 
        external 
        onlyGuardian 
        validOperation(operationId) 
    {
        EnhancedOperation storage op = enhancedOperations[operationId];
        require(op.operationType == OperationType.Emergency, "Not an emergency operation");
        require(!op.guardianApprovals[msg.sender], "Already approved");

        op.guardianApprovals[msg.sender] = true;
        op.guardianApprovalsCount++;

        emit GuardianApprovalGiven(operationId, msg.sender);

        // Check if enough approvals received
        if (op.guardianApprovalsCount >= typeRequiredApprovals[OperationType.Emergency]) {
            op.status = OperationStatus.Ready;
        }
    }

    /**
     * @dev Provide security council approval
     */
    function provideSecurityApproval(bytes32 operationId) 
        external 
        onlyRole(SECURITY_COUNCIL_ROLE) 
        validOperation(operationId) 
    {
        EnhancedOperation storage op = enhancedOperations[operationId];
        require(op.operationType == OperationType.Security, "Not a security operation");
        require(!op.securityCouncilApprovals[msg.sender], "Already approved");

        op.securityCouncilApprovals[msg.sender] = true;
        op.securityCouncilApprovalsCount++;

        // Check if enough approvals received
        if (op.securityCouncilApprovalsCount >= typeRequiredApprovals[OperationType.Security]) {
            op.status = OperationStatus.Ready;
        }
    }

    /**
     * @dev Execute an enhanced operation with additional checks
     */
    function executeEnhanced(
        address target,
        uint256 value,
        bytes calldata payload,
        bytes32 predecessor,
        bytes32 salt
    ) external payable nonReentrant {
        bytes32 id = hashOperation(target, value, payload, predecessor, salt);
        EnhancedOperation storage op = enhancedOperations[id];
        
        require(op.proposer != address(0), "Operation does not exist");
        require(op.status == OperationStatus.Ready, "Operation not ready");
        
        // Check compliance if required
        if (op.requiresCompliance) {
            require(op.complianceApproved, "Compliance approval required");
        }

        // Check execution window
        if (op.executionWindow > 0) {
            require(
                block.timestamp <= op.scheduledFor + op.executionWindow,
                "Execution window expired"
            );
        }

        // Check dependencies
        for (uint256 i = 0; i < op.dependencies.length; i++) {
            require(
                enhancedOperations[op.dependencies[i]].status == OperationStatus.Executed,
                "Dependency not executed"
            );
        }

        // Execute the base operation
        execute(target, value, payload, predecessor, salt);
        
        op.status = OperationStatus.Executed;
    }

    /**
     * @dev Emergency execution with guardian approval
     */
    function emergencyExecute(
        address target,
        uint256 value,
        bytes calldata payload,
        bytes32 predecessor,
        bytes32 salt
    ) external payable onlyInEmergency nonReentrant {
        bytes32 id = hashOperation(target, value, payload, predecessor, salt);
        EnhancedOperation storage op = enhancedOperations[id];
        
        require(op.canBeEmergencyExecuted, "Cannot be emergency executed");
        require(
            op.guardianApprovalsCount >= emergencyConfig.minGuardianApprovals,
            "Insufficient guardian approvals"
        );

        // Cancel the scheduled operation and execute immediately
        cancel(id);
        
        // Direct execution
        (bool success,) = target.call{value: value}(payload);
        require(success, "Emergency execution failed");
        
        op.status = OperationStatus.Emergency;
    }

    /**
     * @dev Activate emergency mode
     */
    function activateEmergencyMode(uint256 duration) 
        external 
        onlyRole(EMERGENCY_GUARDIAN_ROLE) 
    {
        require(!emergencyConfig.emergencyMode, "Already in emergency mode");
        require(duration <= MAX_EMERGENCY_DURATION, "Duration too long");

        emergencyConfig.emergencyMode = true;
        emergencyConfig.emergencyActivatedAt = block.timestamp;
        emergencyConfig.emergencyDuration = duration;

        emit EmergencyModeActivated(block.timestamp, duration);
    }

    /**
     * @dev Deactivate emergency mode
     */
    function deactivateEmergencyMode() external onlyRole(LEGAL_ADMIN_ROLE) {
        require(emergencyConfig.emergencyMode, "Not in emergency mode");

        emergencyConfig.emergencyMode = false;
        emergencyConfig.emergencyActivatedAt = 0;

        emit EmergencyModeDeactivated(block.timestamp);
    }

    /**
     * @dev Block an operation
     */
    function blockOperation(bytes32 operationId, string memory reason) 
        external 
        onlyRole(SECURITY_COUNCIL_ROLE) 
        validOperation(operationId) 
    {
        EnhancedOperation storage op = enhancedOperations[operationId];
        require(op.status != OperationStatus.Executed, "Operation already executed");

        op.status = OperationStatus.Blocked;
        
        // Cancel the underlying timelock operation
        cancel(operationId);

        emit OperationBlocked(operationId, msg.sender, reason);
    }

    /**
     * @dev Update operation type delays
     */
    function updateTypeDelay(OperationType operationType, uint256 newDelay) 
        external 
        onlyRole(LEGAL_ADMIN_ROLE) 
    {
        require(newDelay >= getMinDelay(), "Delay too short");
        typeDelays[operationType] = newDelay;
    }

    /**
     * @dev Update required approvals for operation types
     */
    function updateRequiredApprovals(OperationType operationType, uint256 required) 
        external 
        onlyRole(LEGAL_ADMIN_ROLE) 
    {
        typeRequiredApprovals[operationType] = required;
    }

    /**
     * @dev Add emergency guardian
     */
    function addEmergencyGuardian(address guardian) external onlyRole(LEGAL_ADMIN_ROLE) {
        require(!emergencyConfig.emergencyGuardians[guardian], "Already a guardian");
        
        emergencyConfig.emergencyGuardians[guardian] = true;
        emergencyConfig.totalGuardians++;
        _grantRole(EMERGENCY_GUARDIAN_ROLE, guardian);
    }

    /**
     * @dev Remove emergency guardian
     */
    function removeEmergencyGuardian(address guardian) external onlyRole(LEGAL_ADMIN_ROLE) {
        require(emergencyConfig.emergencyGuardians[guardian], "Not a guardian");
        
        emergencyConfig.emergencyGuardians[guardian] = false;
        emergencyConfig.totalGuardians--;
        _revokeRole(EMERGENCY_GUARDIAN_ROLE, guardian);
    }

    /**
     * @dev Get enhanced operation details
     */
    function getEnhancedOperation(bytes32 operationId) external view returns (
        OperationType operationType,
        OperationStatus status,
        uint256 createdAt,
        uint256 scheduledFor,
        address proposer,
        string memory description,
        bool requiresCompliance,
        bool complianceApproved
    ) {
        EnhancedOperation storage op = enhancedOperations[operationId];
        return (
            op.operationType,
            op.status,
            op.createdAt,
            op.scheduledFor,
            op.proposer,
            op.description,
            op.requiresCompliance,
            op.complianceApproved
        );
    }

    /**
     * @dev Get compliance check details
     */
    function getComplianceCheck(bytes32 operationId) external view returns (
        address officer,
        bool approved,
        string memory comments,
        uint256 checkedAt,
        string[] memory requirements,
        bool[] memory requirementsMet
    ) {
        ComplianceCheck storage check = complianceChecks[operationId];
        return (
            check.officer,
            check.approved,
            check.comments,
            check.checkedAt,
            check.requirements,
            check.requirementsMet
        );
    }

    /**
     * @dev Check if address is emergency guardian
     */
    function isEmergencyGuardian(address addr) external view returns (bool) {
        return emergencyConfig.emergencyGuardians[addr];
    }

    /**
     * @dev Get emergency configuration
     */
    function getEmergencyConfig() external view returns (
        bool emergencyMode,
        uint256 emergencyDelay,
        uint256 minGuardianApprovals,
        uint256 emergencyDuration,
        uint256 emergencyActivatedAt,
        uint256 totalGuardians
    ) {
        return (
            emergencyConfig.emergencyMode,
            emergencyConfig.emergencyDelay,
            emergencyConfig.minGuardianApprovals,
            emergencyConfig.emergencyDuration,
            emergencyConfig.emergencyActivatedAt,
            emergencyConfig.totalGuardians
        );
    }

    /**
     * @dev Pause contract (emergency only)
     */
    function pause() external onlyRole(EMERGENCY_GUARDIAN_ROLE) {
        _pause();
    }

    /**
     * @dev Unpause contract
     */
    function unpause() external onlyRole(LEGAL_ADMIN_ROLE) {
        _unpause();
    }
}