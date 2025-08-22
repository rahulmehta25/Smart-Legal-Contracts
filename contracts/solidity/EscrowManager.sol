// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title EscrowManager
 * @dev Multi-party escrow system with milestone-based releases and dispute resolution
 */
contract EscrowManager is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");
    bytes32 public constant AGENT_ROLE = keccak256("AGENT_ROLE");

    enum EscrowStatus {
        Created,
        Funded,
        Active,
        Disputed,
        Completed,
        Cancelled,
        Released
    }

    enum MilestoneStatus {
        Pending,
        Submitted,
        Approved,
        Disputed,
        Released
    }

    struct Escrow {
        bytes32 id;
        address payer;
        address payee;
        address arbitrator;
        address token; // address(0) for ETH
        uint256 totalAmount;
        uint256 releasedAmount;
        EscrowStatus status;
        uint256 createdAt;
        uint256 deadline;
        string terms;
        bytes32[] milestoneIds;
        uint256 disputeId;
        bool autoRelease;
        uint256 autoReleaseTime;
        mapping(address => bool) hasApproved;
    }

    struct Milestone {
        bytes32 id;
        bytes32 escrowId;
        string description;
        uint256 amount;
        uint256 deadline;
        MilestoneStatus status;
        address submitter;
        string deliverable;
        uint256 submittedAt;
        uint256 approvedAt;
        mapping(address => bool) approvals;
        uint256 approvalCount;
        uint256 requiredApprovals;
    }

    struct MultiPartyEscrow {
        bytes32 id;
        address[] parties;
        mapping(address => uint256) contributions;
        mapping(address => uint256) allocations;
        uint256 totalContributed;
        uint256 totalAllocated;
        EscrowStatus status;
        uint256 votingThreshold;
        mapping(address => mapping(bytes32 => bool)) votes;
        mapping(bytes32 => uint256) voteCount;
    }

    mapping(bytes32 => Escrow) public escrows;
    mapping(bytes32 => Milestone) public milestones;
    mapping(bytes32 => MultiPartyEscrow) public multiPartyEscrows;
    mapping(address => bytes32[]) public userEscrows;
    
    bytes32[] public allEscrowIds;
    
    uint256 public constant MAX_PARTIES = 10;
    uint256 public constant MIN_AUTO_RELEASE_TIME = 1 days;
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public constant FEE_DENOMINATOR = 10000;
    
    address public feeCollector;

    event EscrowCreated(bytes32 indexed escrowId, address indexed payer, address indexed payee, uint256 amount);
    event EscrowFunded(bytes32 indexed escrowId, uint256 amount);
    event MilestoneCreated(bytes32 indexed milestoneId, bytes32 indexed escrowId, uint256 amount);
    event MilestoneSubmitted(bytes32 indexed milestoneId, address indexed submitter);
    event MilestoneApproved(bytes32 indexed milestoneId, address indexed approver);
    event MilestoneReleased(bytes32 indexed milestoneId, uint256 amount);
    event EscrowDisputed(bytes32 indexed escrowId, address indexed disputeInitiator);
    event EscrowResolved(bytes32 indexed escrowId, address indexed resolver);
    event AutoReleaseTriggered(bytes32 indexed escrowId);
    event MultiPartyEscrowCreated(bytes32 indexed escrowId, address[] parties);
    event ContributionMade(bytes32 indexed escrowId, address indexed contributor, uint256 amount);

    modifier onlyEscrowParty(bytes32 escrowId) {
        require(
            escrows[escrowId].payer == msg.sender || 
            escrows[escrowId].payee == msg.sender ||
            escrows[escrowId].arbitrator == msg.sender,
            "Not an escrow party"
        );
        _;
    }

    modifier validEscrow(bytes32 escrowId) {
        require(escrows[escrowId].payer != address(0), "Escrow does not exist");
        _;
    }

    modifier validMilestone(bytes32 milestoneId) {
        require(milestones[milestoneId].escrowId != bytes32(0), "Milestone does not exist");
        _;
    }

    constructor(address _feeCollector) {
        require(_feeCollector != address(0), "Invalid fee collector");
        feeCollector = _feeCollector;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Create a new escrow
     */
    function createEscrow(
        address payee,
        address arbitrator,
        address token,
        uint256 totalAmount,
        uint256 deadline,
        string memory terms,
        bool autoRelease,
        uint256 autoReleaseTime
    ) external nonReentrant whenNotPaused returns (bytes32) {
        require(payee != address(0) && payee != msg.sender, "Invalid payee");
        require(totalAmount > 0, "Invalid amount");
        require(deadline > block.timestamp, "Invalid deadline");
        require(bytes(terms).length > 0, "Empty terms");
        
        if (autoRelease) {
            require(autoReleaseTime >= MIN_AUTO_RELEASE_TIME, "Auto release time too short");
        }

        bytes32 escrowId = keccak256(abi.encodePacked(
            msg.sender,
            payee,
            totalAmount,
            block.timestamp
        ));

        require(escrows[escrowId].payer == address(0), "Escrow already exists");

        Escrow storage newEscrow = escrows[escrowId];
        newEscrow.id = escrowId;
        newEscrow.payer = msg.sender;
        newEscrow.payee = payee;
        newEscrow.arbitrator = arbitrator;
        newEscrow.token = token;
        newEscrow.totalAmount = totalAmount;
        newEscrow.status = EscrowStatus.Created;
        newEscrow.createdAt = block.timestamp;
        newEscrow.deadline = deadline;
        newEscrow.terms = terms;
        newEscrow.autoRelease = autoRelease;
        newEscrow.autoReleaseTime = autoReleaseTime;

        allEscrowIds.push(escrowId);
        userEscrows[msg.sender].push(escrowId);
        userEscrows[payee].push(escrowId);

        emit EscrowCreated(escrowId, msg.sender, payee, totalAmount);
        return escrowId;
    }

    /**
     * @dev Fund an escrow
     */
    function fundEscrow(bytes32 escrowId) 
        external 
        payable 
        validEscrow(escrowId) 
        nonReentrant 
        whenNotPaused 
    {
        require(escrows[escrowId].payer == msg.sender, "Only payer can fund");
        require(escrows[escrowId].status == EscrowStatus.Created, "Escrow already funded");

        if (escrows[escrowId].token == address(0)) {
            require(msg.value == escrows[escrowId].totalAmount, "Incorrect ETH amount");
        } else {
            require(msg.value == 0, "ETH not needed for token escrow");
            IERC20(escrows[escrowId].token).safeTransferFrom(
                msg.sender,
                address(this),
                escrows[escrowId].totalAmount
            );
        }

        escrows[escrowId].status = EscrowStatus.Funded;
        emit EscrowFunded(escrowId, escrows[escrowId].totalAmount);
    }

    /**
     * @dev Create milestone for escrow
     */
    function createMilestone(
        bytes32 escrowId,
        string memory description,
        uint256 amount,
        uint256 deadline,
        uint256 requiredApprovals
    ) external validEscrow(escrowId) whenNotPaused returns (bytes32) {
        require(
            escrows[escrowId].payer == msg.sender || 
            escrows[escrowId].payee == msg.sender,
            "Not authorized"
        );
        require(bytes(description).length > 0, "Empty description");
        require(amount > 0, "Invalid amount");
        require(deadline > block.timestamp, "Invalid deadline");
        require(requiredApprovals >= 1, "Invalid approval count");

        bytes32 milestoneId = keccak256(abi.encodePacked(
            escrowId,
            description,
            amount,
            block.timestamp
        ));

        require(milestones[milestoneId].escrowId == bytes32(0), "Milestone already exists");

        Milestone storage newMilestone = milestones[milestoneId];
        newMilestone.id = milestoneId;
        newMilestone.escrowId = escrowId;
        newMilestone.description = description;
        newMilestone.amount = amount;
        newMilestone.deadline = deadline;
        newMilestone.status = MilestoneStatus.Pending;
        newMilestone.requiredApprovals = requiredApprovals;

        escrows[escrowId].milestoneIds.push(milestoneId);

        emit MilestoneCreated(milestoneId, escrowId, amount);
        return milestoneId;
    }

    /**
     * @dev Submit deliverable for milestone
     */
    function submitMilestone(bytes32 milestoneId, string memory deliverable) 
        external 
        validMilestone(milestoneId) 
        whenNotPaused 
    {
        bytes32 escrowId = milestones[milestoneId].escrowId;
        require(escrows[escrowId].payee == msg.sender, "Only payee can submit");
        require(milestones[milestoneId].status == MilestoneStatus.Pending, "Milestone not pending");
        require(bytes(deliverable).length > 0, "Empty deliverable");
        require(block.timestamp <= milestones[milestoneId].deadline, "Milestone deadline passed");

        milestones[milestoneId].status = MilestoneStatus.Submitted;
        milestones[milestoneId].submitter = msg.sender;
        milestones[milestoneId].deliverable = deliverable;
        milestones[milestoneId].submittedAt = block.timestamp;

        emit MilestoneSubmitted(milestoneId, msg.sender);
    }

    /**
     * @dev Approve milestone
     */
    function approveMilestone(bytes32 milestoneId) 
        external 
        validMilestone(milestoneId) 
        whenNotPaused 
    {
        bytes32 escrowId = milestones[milestoneId].escrowId;
        require(
            escrows[escrowId].payer == msg.sender || 
            escrows[escrowId].arbitrator == msg.sender,
            "Not authorized to approve"
        );
        require(milestones[milestoneId].status == MilestoneStatus.Submitted, "Milestone not submitted");
        require(!milestones[milestoneId].approvals[msg.sender], "Already approved");

        milestones[milestoneId].approvals[msg.sender] = true;
        milestones[milestoneId].approvalCount++;

        emit MilestoneApproved(milestoneId, msg.sender);

        // Auto-release if enough approvals
        if (milestones[milestoneId].approvalCount >= milestones[milestoneId].requiredApprovals) {
            _releaseMilestone(milestoneId);
        }
    }

    /**
     * @dev Release milestone payment
     */
    function _releaseMilestone(bytes32 milestoneId) internal {
        milestones[milestoneId].status = MilestoneStatus.Released;
        milestones[milestoneId].approvedAt = block.timestamp;

        bytes32 escrowId = milestones[milestoneId].escrowId;
        uint256 amount = milestones[milestoneId].amount;
        
        // Calculate platform fee
        uint256 platformFee = (amount * platformFeePercentage) / FEE_DENOMINATOR;
        uint256 payoutAmount = amount - platformFee;

        escrows[escrowId].releasedAmount += amount;

        // Transfer payment
        if (escrows[escrowId].token == address(0)) {
            payable(escrows[escrowId].payee).transfer(payoutAmount);
            if (platformFee > 0) {
                payable(feeCollector).transfer(platformFee);
            }
        } else {
            IERC20(escrows[escrowId].token).safeTransfer(escrows[escrowId].payee, payoutAmount);
            if (platformFee > 0) {
                IERC20(escrows[escrowId].token).safeTransfer(feeCollector, platformFee);
            }
        }

        emit MilestoneReleased(milestoneId, payoutAmount);

        // Check if escrow is complete
        if (escrows[escrowId].releasedAmount >= escrows[escrowId].totalAmount) {
            escrows[escrowId].status = EscrowStatus.Completed;
        }
    }

    /**
     * @dev Dispute a milestone or escrow
     */
    function initiateDispute(bytes32 escrowId, string memory reason) 
        external 
        validEscrow(escrowId) 
        onlyEscrowParty(escrowId) 
        whenNotPaused 
    {
        require(escrows[escrowId].status == EscrowStatus.Funded || escrows[escrowId].status == EscrowStatus.Active, "Cannot dispute");
        require(bytes(reason).length > 0, "Empty dispute reason");

        escrows[escrowId].status = EscrowStatus.Disputed;
        emit EscrowDisputed(escrowId, msg.sender);
    }

    /**
     * @dev Resolve dispute (arbitrator only)
     */
    function resolveDispute(
        bytes32 escrowId,
        address recipient,
        uint256 amount,
        string memory resolution
    ) external validEscrow(escrowId) whenNotPaused {
        require(escrows[escrowId].arbitrator == msg.sender, "Only arbitrator can resolve");
        require(escrows[escrowId].status == EscrowStatus.Disputed, "Not disputed");
        require(recipient != address(0), "Invalid recipient");
        require(amount <= escrows[escrowId].totalAmount - escrows[escrowId].releasedAmount, "Amount exceeds available");

        // Calculate platform fee
        uint256 platformFee = (amount * platformFeePercentage) / FEE_DENOMINATOR;
        uint256 payoutAmount = amount - platformFee;

        escrows[escrowId].releasedAmount += amount;
        escrows[escrowId].status = EscrowStatus.Released;

        // Transfer resolution amount
        if (escrows[escrowId].token == address(0)) {
            payable(recipient).transfer(payoutAmount);
            if (platformFee > 0) {
                payable(feeCollector).transfer(platformFee);
            }
        } else {
            IERC20(escrows[escrowId].token).safeTransfer(recipient, payoutAmount);
            if (platformFee > 0) {
                IERC20(escrows[escrowId].token).safeTransfer(feeCollector, platformFee);
            }
        }

        emit EscrowResolved(escrowId, msg.sender);
    }

    /**
     * @dev Auto-release escrow after time delay
     */
    function triggerAutoRelease(bytes32 escrowId) external validEscrow(escrowId) whenNotPaused {
        require(escrows[escrowId].autoRelease, "Auto release not enabled");
        require(escrows[escrowId].status == EscrowStatus.Funded, "Invalid status for auto release");
        require(
            block.timestamp >= escrows[escrowId].createdAt + escrows[escrowId].autoReleaseTime,
            "Auto release time not reached"
        );

        uint256 remainingAmount = escrows[escrowId].totalAmount - escrows[escrowId].releasedAmount;
        
        // Calculate platform fee
        uint256 platformFee = (remainingAmount * platformFeePercentage) / FEE_DENOMINATOR;
        uint256 payoutAmount = remainingAmount - platformFee;

        escrows[escrowId].releasedAmount = escrows[escrowId].totalAmount;
        escrows[escrowId].status = EscrowStatus.Released;

        // Transfer remaining amount to payee
        if (escrows[escrowId].token == address(0)) {
            payable(escrows[escrowId].payee).transfer(payoutAmount);
            if (platformFee > 0) {
                payable(feeCollector).transfer(platformFee);
            }
        } else {
            IERC20(escrows[escrowId].token).safeTransfer(escrows[escrowId].payee, payoutAmount);
            if (platformFee > 0) {
                IERC20(escrows[escrowId].token).safeTransfer(feeCollector, platformFee);
            }
        }

        emit AutoReleaseTriggered(escrowId);
    }

    /**
     * @dev Create multi-party escrow
     */
    function createMultiPartyEscrow(
        address[] memory parties,
        uint256[] memory allocations,
        uint256 votingThreshold
    ) external nonReentrant whenNotPaused returns (bytes32) {
        require(parties.length >= 2 && parties.length <= MAX_PARTIES, "Invalid party count");
        require(parties.length == allocations.length, "Array length mismatch");
        require(votingThreshold > 0 && votingThreshold <= parties.length, "Invalid voting threshold");

        bytes32 escrowId = keccak256(abi.encodePacked(
            parties,
            allocations,
            block.timestamp
        ));

        MultiPartyEscrow storage newEscrow = multiPartyEscrows[escrowId];
        newEscrow.id = escrowId;
        newEscrow.parties = parties;
        newEscrow.status = EscrowStatus.Created;
        newEscrow.votingThreshold = votingThreshold;

        uint256 totalAllocation = 0;
        for (uint256 i = 0; i < parties.length; i++) {
            require(parties[i] != address(0), "Invalid party address");
            newEscrow.allocations[parties[i]] = allocations[i];
            totalAllocation += allocations[i];
        }
        newEscrow.totalAllocated = totalAllocation;

        emit MultiPartyEscrowCreated(escrowId, parties);
        return escrowId;
    }

    /**
     * @dev Contribute to multi-party escrow
     */
    function contributeToMultiParty(bytes32 escrowId) 
        external 
        payable 
        nonReentrant 
        whenNotPaused 
    {
        MultiPartyEscrow storage escrow = multiPartyEscrows[escrowId];
        require(escrow.id != bytes32(0), "Escrow does not exist");
        require(msg.value > 0, "Must contribute positive amount");

        bool isParty = false;
        for (uint256 i = 0; i < escrow.parties.length; i++) {
            if (escrow.parties[i] == msg.sender) {
                isParty = true;
                break;
            }
        }
        require(isParty, "Not a party to this escrow");

        escrow.contributions[msg.sender] += msg.value;
        escrow.totalContributed += msg.value;

        emit ContributionMade(escrowId, msg.sender, msg.value);
    }

    /**
     * @dev Get escrow details
     */
    function getEscrowDetails(bytes32 escrowId) external view returns (
        address payer,
        address payee,
        address arbitrator,
        uint256 totalAmount,
        uint256 releasedAmount,
        EscrowStatus status,
        uint256 deadline
    ) {
        Escrow storage escrow = escrows[escrowId];
        return (
            escrow.payer,
            escrow.payee,
            escrow.arbitrator,
            escrow.totalAmount,
            escrow.releasedAmount,
            escrow.status,
            escrow.deadline
        );
    }

    /**
     * @dev Get milestone details
     */
    function getMilestoneDetails(bytes32 milestoneId) external view returns (
        bytes32 escrowId,
        string memory description,
        uint256 amount,
        MilestoneStatus status,
        uint256 deadline,
        uint256 approvalCount,
        uint256 requiredApprovals
    ) {
        Milestone storage milestone = milestones[milestoneId];
        return (
            milestone.escrowId,
            milestone.description,
            milestone.amount,
            milestone.status,
            milestone.deadline,
            milestone.approvalCount,
            milestone.requiredApprovals
        );
    }

    /**
     * @dev Get user escrows
     */
    function getUserEscrows(address user) external view returns (bytes32[] memory) {
        return userEscrows[user];
    }

    /**
     * @dev Get total escrows
     */
    function getTotalEscrows() external view returns (uint256) {
        return allEscrowIds.length;
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

    function setPlatformFee(uint256 newFeePercentage) external onlyRole(ADMIN_ROLE) {
        require(newFeePercentage <= 1000, "Fee too high"); // Max 10%
        platformFeePercentage = newFeePercentage;
    }

    function setFeeCollector(address newFeeCollector) external onlyRole(ADMIN_ROLE) {
        require(newFeeCollector != address(0), "Invalid fee collector");
        feeCollector = newFeeCollector;
    }

    function emergencyWithdraw(address token) external onlyRole(ADMIN_ROLE) {
        if (token == address(0)) {
            payable(msg.sender).transfer(address(this).balance);
        } else {
            IERC20(token).safeTransfer(msg.sender, IERC20(token).balanceOf(address(this)));
        }
    }

    receive() external payable {}
}