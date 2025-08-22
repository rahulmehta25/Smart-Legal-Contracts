// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC1155/IERC1155.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title RoyaltyDistribution
 * @dev Automated royalty distribution system for intellectual property and creative works
 */
contract RoyaltyDistribution is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant CREATOR_ROLE = keccak256("CREATOR_ROLE");
    bytes32 public constant DISTRIBUTOR_ROLE = keccak256("DISTRIBUTOR_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");

    enum RoyaltyType {
        FixedAmount,
        Percentage,
        Tiered,
        Progressive,
        TimeDecaying,
        UsageBased
    }

    enum PaymentStatus {
        Pending,
        Distributed,
        Failed,
        Disputed,
        Withheld
    }

    struct RoyaltyScheme {
        bytes32 id;
        string name;
        address creator;
        RoyaltyType royaltyType;
        uint256[] rates; // Basis points (10000 = 100%)
        uint256[] thresholds;
        uint256 totalSupply;
        uint256 totalDistributed;
        bool active;
        uint256 createdAt;
        mapping(address => uint256) stakeholderShares;
        address[] stakeholders;
        string ipfsHash; // For storing detailed terms
    }

    struct RoyaltyPayment {
        bytes32 id;
        bytes32 schemeId;
        address payer;
        uint256 grossAmount;
        uint256 netAmount;
        uint256 platformFee;
        PaymentStatus status;
        uint256 timestamp;
        string source; // Source of revenue (sale, streaming, etc.)
        mapping(address => uint256) distributions;
        mapping(address => bool) claimed;
        address[] recipients;
        uint256 totalClaimed;
    }

    struct Stakeholder {
        address wallet;
        uint256 share; // Basis points
        string role; // Creator, Producer, Publisher, etc.
        bool verified;
        uint256 totalEarned;
        uint256 totalClaimed;
        string metadata; // IPFS hash for additional data
    }

    struct CreativeWork {
        bytes32 id;
        string title;
        string creator;
        address creatorWallet;
        bytes32 royaltySchemeId;
        uint256 createdAt;
        string ipfsHash;
        mapping(address => bool) collaborators;
        address[] collaboratorList;
        bool registered;
    }

    struct RevenueStream {
        bytes32 id;
        bytes32 workId;
        string source;
        uint256 totalRevenue;
        uint256 distributedRevenue;
        uint256 pendingRevenue;
        mapping(uint256 => uint256) monthlyRevenue; // timestamp => amount
        bool active;
    }

    mapping(bytes32 => RoyaltyScheme) public royaltySchemes;
    mapping(bytes32 => RoyaltyPayment) public royaltyPayments;
    mapping(address => Stakeholder) public stakeholders;
    mapping(bytes32 => CreativeWork) public creativeWorks;
    mapping(bytes32 => RevenueStream) public revenueStreams;
    mapping(address => bytes32[]) public userSchemes;
    mapping(address => bytes32[]) public userPayments;
    mapping(bytes32 => bytes32[]) public schemePayments;
    
    bytes32[] public allSchemeIds;
    bytes32[] public allPaymentIds;
    bytes32[] public allWorkIds;
    
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public constant FEE_DENOMINATOR = 10000;
    uint256 public constant MAX_STAKEHOLDERS = 100;
    
    address public treasury;
    uint256 public totalFeesCollected;

    event RoyaltySchemeCreated(bytes32 indexed schemeId, address indexed creator, string name);
    event PaymentReceived(bytes32 indexed paymentId, bytes32 indexed schemeId, uint256 amount);
    event RoyaltyDistributed(bytes32 indexed paymentId, address indexed recipient, uint256 amount);
    event StakeholderAdded(bytes32 indexed schemeId, address indexed stakeholder, uint256 share);
    event CreativeWorkRegistered(bytes32 indexed workId, address indexed creator, string title);
    event RevenueStreamCreated(bytes32 indexed streamId, bytes32 indexed workId, string source);
    event RoyaltyClaimed(bytes32 indexed paymentId, address indexed claimant, uint256 amount);

    modifier onlySchemeCreator(bytes32 schemeId) {
        require(royaltySchemes[schemeId].creator == msg.sender, "Not scheme creator");
        _;
    }

    modifier validScheme(bytes32 schemeId) {
        require(royaltySchemes[schemeId].creator != address(0), "Scheme does not exist");
        _;
    }

    modifier validPayment(bytes32 paymentId) {
        require(royaltyPayments[paymentId].payer != address(0), "Payment does not exist");
        _;
    }

    constructor(address _treasury) {
        require(_treasury != address(0), "Invalid treasury address");
        treasury = _treasury;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @dev Create a new royalty scheme
     */
    function createRoyaltyScheme(
        string memory name,
        RoyaltyType royaltyType,
        uint256[] memory rates,
        uint256[] memory thresholds,
        address[] memory stakeholderAddresses,
        uint256[] memory shares,
        string memory ipfsHash
    ) external nonReentrant whenNotPaused returns (bytes32) {
        require(bytes(name).length > 0, "Empty scheme name");
        require(stakeholderAddresses.length == shares.length, "Array length mismatch");
        require(stakeholderAddresses.length <= MAX_STAKEHOLDERS, "Too many stakeholders");
        
        // Validate shares sum to 100%
        uint256 totalShares = 0;
        for (uint256 i = 0; i < shares.length; i++) {
            totalShares += shares[i];
        }
        require(totalShares == FEE_DENOMINATOR, "Shares must sum to 100%");

        bytes32 schemeId = keccak256(abi.encodePacked(
            name,
            msg.sender,
            block.timestamp
        ));

        require(royaltySchemes[schemeId].creator == address(0), "Scheme already exists");

        RoyaltyScheme storage newScheme = royaltySchemes[schemeId];
        newScheme.id = schemeId;
        newScheme.name = name;
        newScheme.creator = msg.sender;
        newScheme.royaltyType = royaltyType;
        newScheme.rates = rates;
        newScheme.thresholds = thresholds;
        newScheme.active = true;
        newScheme.createdAt = block.timestamp;
        newScheme.ipfsHash = ipfsHash;

        // Add stakeholders
        for (uint256 i = 0; i < stakeholderAddresses.length; i++) {
            address stakeholder = stakeholderAddresses[i];
            uint256 share = shares[i];
            
            newScheme.stakeholderShares[stakeholder] = share;
            newScheme.stakeholders.push(stakeholder);
            
            // Initialize stakeholder if new
            if (!stakeholders[stakeholder].verified) {
                stakeholders[stakeholder].wallet = stakeholder;
                stakeholders[stakeholder].verified = true;
            }
        }

        allSchemeIds.push(schemeId);
        userSchemes[msg.sender].push(schemeId);
        _grantRole(CREATOR_ROLE, msg.sender);

        emit RoyaltySchemeCreated(schemeId, msg.sender, name);
        return schemeId;
    }

    /**
     * @dev Register a creative work
     */
    function registerCreativeWork(
        string memory title,
        string memory creator,
        bytes32 royaltySchemeId,
        address[] memory collaborators,
        string memory ipfsHash
    ) external validScheme(royaltySchemeId) whenNotPaused returns (bytes32) {
        require(bytes(title).length > 0, "Empty title");
        require(bytes(creator).length > 0, "Empty creator");

        bytes32 workId = keccak256(abi.encodePacked(
            title,
            creator,
            msg.sender,
            block.timestamp
        ));

        require(!creativeWorks[workId].registered, "Work already registered");

        CreativeWork storage newWork = creativeWorks[workId];
        newWork.id = workId;
        newWork.title = title;
        newWork.creator = creator;
        newWork.creatorWallet = msg.sender;
        newWork.royaltySchemeId = royaltySchemeId;
        newWork.createdAt = block.timestamp;
        newWork.ipfsHash = ipfsHash;
        newWork.registered = true;

        // Add collaborators
        for (uint256 i = 0; i < collaborators.length; i++) {
            newWork.collaborators[collaborators[i]] = true;
            newWork.collaboratorList.push(collaborators[i]);
        }

        allWorkIds.push(workId);

        emit CreativeWorkRegistered(workId, msg.sender, title);
        return workId;
    }

    /**
     * @dev Submit royalty payment
     */
    function submitRoyaltyPayment(
        bytes32 schemeId,
        string memory source
    ) external payable validScheme(schemeId) nonReentrant whenNotPaused returns (bytes32) {
        require(msg.value > 0, "Must send payment");
        require(bytes(source).length > 0, "Empty source");
        require(royaltySchemes[schemeId].active, "Scheme not active");

        uint256 platformFee = (msg.value * platformFeePercentage) / FEE_DENOMINATOR;
        uint256 netAmount = msg.value - platformFee;

        bytes32 paymentId = keccak256(abi.encodePacked(
            schemeId,
            msg.sender,
            msg.value,
            block.timestamp
        ));

        require(royaltyPayments[paymentId].payer == address(0), "Payment already exists");

        RoyaltyPayment storage newPayment = royaltyPayments[paymentId];
        newPayment.id = paymentId;
        newPayment.schemeId = schemeId;
        newPayment.payer = msg.sender;
        newPayment.grossAmount = msg.value;
        newPayment.netAmount = netAmount;
        newPayment.platformFee = platformFee;
        newPayment.status = PaymentStatus.Pending;
        newPayment.timestamp = block.timestamp;
        newPayment.source = source;

        // Calculate distributions
        _calculateDistributions(paymentId, schemeId, netAmount);

        allPaymentIds.push(paymentId);
        userPayments[msg.sender].push(paymentId);
        schemePayments[schemeId].push(paymentId);
        
        // Transfer platform fee to treasury
        if (platformFee > 0) {
            payable(treasury).transfer(platformFee);
            totalFeesCollected += platformFee;
        }

        emit PaymentReceived(paymentId, schemeId, msg.value);
        
        // Auto-distribute if enabled
        _autoDistribute(paymentId);
        
        return paymentId;
    }

    /**
     * @dev Internal function to calculate distributions
     */
    function _calculateDistributions(bytes32 paymentId, bytes32 schemeId, uint256 netAmount) internal {
        RoyaltyScheme storage scheme = royaltySchemes[schemeId];
        RoyaltyPayment storage payment = royaltyPayments[paymentId];

        for (uint256 i = 0; i < scheme.stakeholders.length; i++) {
            address stakeholder = scheme.stakeholders[i];
            uint256 share = scheme.stakeholderShares[stakeholder];
            uint256 distribution = (netAmount * share) / FEE_DENOMINATOR;
            
            payment.distributions[stakeholder] = distribution;
            payment.recipients.push(stakeholder);
        }
    }

    /**
     * @dev Internal function to auto-distribute payments
     */
    function _autoDistribute(bytes32 paymentId) internal {
        RoyaltyPayment storage payment = royaltyPayments[paymentId];
        
        for (uint256 i = 0; i < payment.recipients.length; i++) {
            address recipient = payment.recipients[i];
            uint256 amount = payment.distributions[recipient];
            
            if (amount > 0) {
                payable(recipient).transfer(amount);
                payment.claimed[recipient] = true;
                payment.totalClaimed += amount;
                
                // Update stakeholder stats
                stakeholders[recipient].totalEarned += amount;
                stakeholders[recipient].totalClaimed += amount;
                
                emit RoyaltyDistributed(paymentId, recipient, amount);
            }
        }
        
        payment.status = PaymentStatus.Distributed;
        royaltySchemes[payment.schemeId].totalDistributed += payment.netAmount;
    }

    /**
     * @dev Claim royalty payment manually
     */
    function claimRoyalty(bytes32 paymentId) 
        external 
        validPayment(paymentId) 
        nonReentrant 
        whenNotPaused 
    {
        RoyaltyPayment storage payment = royaltyPayments[paymentId];
        require(!payment.claimed[msg.sender], "Already claimed");
        require(payment.distributions[msg.sender] > 0, "No distribution for this address");

        uint256 amount = payment.distributions[msg.sender];
        payment.claimed[msg.sender] = true;
        payment.totalClaimed += amount;

        // Update stakeholder stats
        stakeholders[msg.sender].totalClaimed += amount;

        payable(msg.sender).transfer(amount);

        emit RoyaltyClaimed(paymentId, msg.sender, amount);
    }

    /**
     * @dev Batch claim multiple payments
     */
    function batchClaimRoyalties(bytes32[] memory paymentIds) 
        external 
        nonReentrant 
        whenNotPaused 
    {
        uint256 totalAmount = 0;
        
        for (uint256 i = 0; i < paymentIds.length; i++) {
            bytes32 paymentId = paymentIds[i];
            RoyaltyPayment storage payment = royaltyPayments[paymentId];
            
            if (!payment.claimed[msg.sender] && payment.distributions[msg.sender] > 0) {
                uint256 amount = payment.distributions[msg.sender];
                payment.claimed[msg.sender] = true;
                payment.totalClaimed += amount;
                totalAmount += amount;
                
                emit RoyaltyClaimed(paymentId, msg.sender, amount);
            }
        }
        
        if (totalAmount > 0) {
            stakeholders[msg.sender].totalClaimed += totalAmount;
            payable(msg.sender).transfer(totalAmount);
        }
    }

    /**
     * @dev Create revenue stream for creative work
     */
    function createRevenueStream(
        bytes32 workId,
        string memory source
    ) external whenNotPaused returns (bytes32) {
        require(creativeWorks[workId].registered, "Work not registered");
        require(
            creativeWorks[workId].creatorWallet == msg.sender ||
            creativeWorks[workId].collaborators[msg.sender],
            "Not authorized"
        );

        bytes32 streamId = keccak256(abi.encodePacked(
            workId,
            source,
            block.timestamp
        ));

        RevenueStream storage newStream = revenueStreams[streamId];
        newStream.id = streamId;
        newStream.workId = workId;
        newStream.source = source;
        newStream.active = true;

        emit RevenueStreamCreated(streamId, workId, source);
        return streamId;
    }

    /**
     * @dev Add stakeholder to existing scheme
     */
    function addStakeholder(
        bytes32 schemeId,
        address stakeholder,
        uint256 share,
        string memory role
    ) external validScheme(schemeId) onlySchemeCreator(schemeId) whenNotPaused {
        require(stakeholder != address(0), "Invalid stakeholder");
        require(share > 0, "Invalid share");
        require(royaltySchemes[schemeId].stakeholderShares[stakeholder] == 0, "Stakeholder already exists");

        // Check if adding this share would exceed 100%
        uint256 currentTotal = 0;
        for (uint256 i = 0; i < royaltySchemes[schemeId].stakeholders.length; i++) {
            currentTotal += royaltySchemes[schemeId].stakeholderShares[royaltySchemes[schemeId].stakeholders[i]];
        }
        require(currentTotal + share <= FEE_DENOMINATOR, "Share would exceed 100%");

        royaltySchemes[schemeId].stakeholderShares[stakeholder] = share;
        royaltySchemes[schemeId].stakeholders.push(stakeholder);

        // Initialize or update stakeholder
        stakeholders[stakeholder].wallet = stakeholder;
        stakeholders[stakeholder].role = role;
        stakeholders[stakeholder].verified = true;

        emit StakeholderAdded(schemeId, stakeholder, share);
    }

    /**
     * @dev Update stakeholder share
     */
    function updateStakeholderShare(
        bytes32 schemeId,
        address stakeholder,
        uint256 newShare
    ) external validScheme(schemeId) onlySchemeCreator(schemeId) whenNotPaused {
        require(royaltySchemes[schemeId].stakeholderShares[stakeholder] > 0, "Stakeholder not found");
        
        uint256 oldShare = royaltySchemes[schemeId].stakeholderShares[stakeholder];
        royaltySchemes[schemeId].stakeholderShares[stakeholder] = newShare;
        
        // Validate total shares don't exceed 100%
        uint256 currentTotal = 0;
        for (uint256 i = 0; i < royaltySchemes[schemeId].stakeholders.length; i++) {
            currentTotal += royaltySchemes[schemeId].stakeholderShares[royaltySchemes[schemeId].stakeholders[i]];
        }
        require(currentTotal <= FEE_DENOMINATOR, "Total shares exceed 100%");
    }

    /**
     * @dev Get scheme details
     */
    function getSchemeDetails(bytes32 schemeId) external view returns (
        string memory name,
        address creator,
        RoyaltyType royaltyType,
        uint256[] memory rates,
        bool active,
        uint256 totalDistributed,
        address[] memory stakeholders
    ) {
        RoyaltyScheme storage scheme = royaltySchemes[schemeId];
        return (
            scheme.name,
            scheme.creator,
            scheme.royaltyType,
            scheme.rates,
            scheme.active,
            scheme.totalDistributed,
            scheme.stakeholders
        );
    }

    /**
     * @dev Get payment details
     */
    function getPaymentDetails(bytes32 paymentId) external view returns (
        bytes32 schemeId,
        address payer,
        uint256 grossAmount,
        uint256 netAmount,
        PaymentStatus status,
        uint256 timestamp,
        string memory source,
        uint256 totalClaimed
    ) {
        RoyaltyPayment storage payment = royaltyPayments[paymentId];
        return (
            payment.schemeId,
            payment.payer,
            payment.grossAmount,
            payment.netAmount,
            payment.status,
            payment.timestamp,
            payment.source,
            payment.totalClaimed
        );
    }

    /**
     * @dev Get stakeholder details
     */
    function getStakeholderDetails(address stakeholderAddr) external view returns (
        address wallet,
        string memory role,
        bool verified,
        uint256 totalEarned,
        uint256 totalClaimed
    ) {
        Stakeholder storage stakeholder = stakeholders[stakeholderAddr];
        return (
            stakeholder.wallet,
            stakeholder.role,
            stakeholder.verified,
            stakeholder.totalEarned,
            stakeholder.totalClaimed
        );
    }

    /**
     * @dev Get user schemes
     */
    function getUserSchemes(address user) external view returns (bytes32[] memory) {
        return userSchemes[user];
    }

    /**
     * @dev Get user payments
     */
    function getUserPayments(address user) external view returns (bytes32[] memory) {
        return userPayments[user];
    }

    /**
     * @dev Get scheme payments
     */
    function getSchemePayments(bytes32 schemeId) external view returns (bytes32[] memory) {
        return schemePayments[schemeId];
    }

    /**
     * @dev Get total schemes
     */
    function getTotalSchemes() external view returns (uint256) {
        return allSchemeIds.length;
    }

    /**
     * @dev Get total payments
     */
    function getTotalPayments() external view returns (uint256) {
        return allPaymentIds.length;
    }

    /**
     * @dev Check pending distribution for address
     */
    function getPendingDistribution(address user) external view returns (uint256) {
        uint256 pending = 0;
        
        for (uint256 i = 0; i < allPaymentIds.length; i++) {
            bytes32 paymentId = allPaymentIds[i];
            RoyaltyPayment storage payment = royaltyPayments[paymentId];
            
            if (!payment.claimed[user] && payment.distributions[user] > 0) {
                pending += payment.distributions[user];
            }
        }
        
        return pending;
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

    function setTreasury(address newTreasury) external onlyRole(ADMIN_ROLE) {
        require(newTreasury != address(0), "Invalid treasury");
        treasury = newTreasury;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}