// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title DisputeInsurancePool
 * @dev Decentralized insurance pool for legal dispute coverage with risk assessment and automated payouts
 */
contract DisputeInsurancePool is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ASSESSOR_ROLE = keccak256("ASSESSOR_ROLE");
    bytes32 public constant UNDERWRITER_ROLE = keccak256("UNDERWRITER_ROLE");
    bytes32 public constant CLAIMS_ADJUSTER_ROLE = keccak256("CLAIMS_ADJUSTER_ROLE");

    enum PolicyStatus {
        Active,
        Lapsed,
        Claimed,
        Cancelled,
        Suspended
    }

    enum ClaimStatus {
        Submitted,
        UnderReview,
        Investigating,
        Approved,
        Denied,
        Paid,
        Disputed
    }

    enum RiskLevel {
        VeryLow,
        Low,
        Medium,
        High,
        VeryHigh
    }

    enum DisputeType {
        ContractBreach,
        IntellectualProperty,
        Employment,
        Commercial,
        RealEstate,
        Partnership,
        Negligence,
        Professional,
        ProductLiability,
        Regulatory
    }

    struct InsurancePolicy {
        uint256 policyId;
        address policyholder;
        DisputeType[] coveredTypes;
        uint256 coverageAmount;
        uint256 premium;
        uint256 deductible;
        uint256 startDate;
        uint256 endDate;
        PolicyStatus status;
        RiskLevel riskLevel;
        uint256 premiumsPaid;
        uint256 claimsCount;
        uint256 totalClaimsAmount;
        string jurisdiction;
        bytes32 termsHash;
        mapping(address => bool) authorizedClaimants;
    }

    struct Claim {
        uint256 claimId;
        uint256 policyId;
        address claimant;
        DisputeType disputeType;
        uint256 claimAmount;
        uint256 incidentDate;
        uint256 claimDate;
        ClaimStatus status;
        string description;
        string[] evidenceHashes;
        uint256 assessmentScore;
        uint256 approvedAmount;
        uint256 paidAmount;
        address assignedAdjuster;
        string denialReason;
        mapping(address => bool) assessorVotes;
        mapping(address => string) assessorComments;
        uint256 assessorCount;
        uint256 approvalVotes;
    }

    struct LiquidityProvider {
        address provider;
        uint256 stakedAmount;
        uint256 sharePercentage;
        uint256 totalRewards;
        uint256 lastRewardClaim;
        uint256 lockupEnd;
        bool active;
    }

    struct RiskAssessment {
        uint256 policyId;
        address assessor;
        RiskLevel riskLevel;
        uint256 score; // 0-100
        string factors;
        uint256 assessmentDate;
        bool finalized;
    }

    struct PoolMetrics {
        uint256 totalPoolValue;
        uint256 totalStaked;
        uint256 totalClaims;
        uint256 totalPremiums;
        uint256 utilizationRatio; // Claims/Premiums * 100
        uint256 solvencyRatio; // Pool Value/Outstanding Claims * 100
        uint256 averageClaimAmount;
        uint256 claimFrequency;
    }

    mapping(uint256 => InsurancePolicy) public policies;
    mapping(uint256 => Claim) public claims;
    mapping(address => LiquidityProvider) public liquidityProviders;
    mapping(uint256 => RiskAssessment) public riskAssessments;
    mapping(address => uint256[]) public userPolicies;
    mapping(address => uint256[]) public userClaims;
    mapping(DisputeType => uint256) public basePremiumRates;
    mapping(RiskLevel => uint256) public riskMultipliers;
    
    uint256 public totalPolicies;
    uint256 public totalClaims;
    uint256 public poolBalance;
    uint256 public totalStaked;
    uint256 public platformFee = 300; // 3%
    uint256 public constant FEE_DENOMINATOR = 10000;
    uint256 public constant MIN_STAKE = 1000e18;
    uint256 public constant CLAIM_VOTING_PERIOD = 7 days;
    uint256 public constant MIN_ASSESSORS = 3;
    
    address public treasury;
    IERC20 public stakingToken;
    AggregatorV3Interface public priceFeed;

    event PolicyCreated(uint256 indexed policyId, address indexed policyholder, uint256 coverageAmount);
    event PremiumPaid(uint256 indexed policyId, uint256 amount);
    event ClaimSubmitted(uint256 indexed claimId, uint256 indexed policyId, address indexed claimant);
    event ClaimAssessed(uint256 indexed claimId, address indexed assessor, bool approved);
    event ClaimPaid(uint256 indexed claimId, uint256 amount);
    event ClaimDenied(uint256 indexed claimId, string reason);
    event LiquidityAdded(address indexed provider, uint256 amount);
    event LiquidityRemoved(address indexed provider, uint256 amount);
    event RewardsClaimed(address indexed provider, uint256 amount);
    event RiskAssessed(uint256 indexed policyId, address indexed assessor, RiskLevel riskLevel);

    modifier onlyPolicyholder(uint256 policyId) {
        require(policies[policyId].policyholder == msg.sender, "Not policyholder");
        _;
    }

    modifier validPolicy(uint256 policyId) {
        require(policyId < totalPolicies, "Invalid policy ID");
        _;
    }

    modifier validClaim(uint256 claimId) {
        require(claimId < totalClaims, "Invalid claim ID");
        _;
    }

    modifier onlyActivePolicy(uint256 policyId) {
        require(policies[policyId].status == PolicyStatus.Active, "Policy not active");
        require(block.timestamp <= policies[policyId].endDate, "Policy expired");
        _;
    }

    constructor(
        address _stakingToken,
        address _treasury,
        address _priceFeed
    ) {
        require(_stakingToken != address(0), "Invalid staking token");
        require(_treasury != address(0), "Invalid treasury");
        
        stakingToken = IERC20(_stakingToken);
        treasury = _treasury;
        priceFeed = AggregatorV3Interface(_priceFeed);
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        
        // Initialize base premium rates (annual premium per $1000 coverage)
        basePremiumRates[DisputeType.ContractBreach] = 50e18; // $50
        basePremiumRates[DisputeType.IntellectualProperty] = 75e18; // $75
        basePremiumRates[DisputeType.Employment] = 40e18; // $40
        basePremiumRates[DisputeType.Commercial] = 60e18; // $60
        basePremiumRates[DisputeType.RealEstate] = 30e18; // $30
        basePremiumRates[DisputeType.Partnership] = 45e18; // $45
        basePremiumRates[DisputeType.Negligence] = 80e18; // $80
        basePremiumRates[DisputeType.Professional] = 70e18; // $70
        basePremiumRates[DisputeType.ProductLiability] = 90e18; // $90
        basePremiumRates[DisputeType.Regulatory] = 85e18; // $85
        
        // Initialize risk multipliers
        riskMultipliers[RiskLevel.VeryLow] = 50; // 0.5x
        riskMultipliers[RiskLevel.Low] = 75; // 0.75x
        riskMultipliers[RiskLevel.Medium] = 100; // 1.0x
        riskMultipliers[RiskLevel.High] = 150; // 1.5x
        riskMultipliers[RiskLevel.VeryHigh] = 250; // 2.5x
    }

    /**
     * @dev Create a new insurance policy
     */
    function createPolicy(
        DisputeType[] memory coveredTypes,
        uint256 coverageAmount,
        uint256 duration,
        uint256 deductible,
        string memory jurisdiction,
        bytes32 termsHash
    ) external nonReentrant whenNotPaused returns (uint256) {
        require(coveredTypes.length > 0, "No coverage types specified");
        require(coverageAmount > 0, "Invalid coverage amount");
        require(duration >= 30 days, "Minimum 30-day duration");
        require(deductible < coverageAmount, "Deductible too high");

        uint256 policyId = totalPolicies++;
        
        InsurancePolicy storage policy = policies[policyId];
        policy.policyId = policyId;
        policy.policyholder = msg.sender;
        policy.coveredTypes = coveredTypes;
        policy.coverageAmount = coverageAmount;
        policy.deductible = deductible;
        policy.startDate = block.timestamp;
        policy.endDate = block.timestamp + duration;
        policy.status = PolicyStatus.Active;
        policy.riskLevel = RiskLevel.Medium; // Default, will be assessed
        policy.jurisdiction = jurisdiction;
        policy.termsHash = termsHash;
        
        // Calculate premium based on coverage types and amount
        uint256 premium = calculatePremium(coveredTypes, coverageAmount, duration, RiskLevel.Medium);
        policy.premium = premium;

        userPolicies[msg.sender].push(policyId);

        emit PolicyCreated(policyId, msg.sender, coverageAmount);
        return policyId;
    }

    /**
     * @dev Pay premium for a policy
     */
    function payPremium(uint256 policyId) 
        external 
        validPolicy(policyId) 
        onlyPolicyholder(policyId) 
        nonReentrant 
        whenNotPaused 
    {
        InsurancePolicy storage policy = policies[policyId];
        require(policy.status == PolicyStatus.Active || policy.status == PolicyStatus.Lapsed, "Invalid policy status");
        
        uint256 premiumAmount = policy.premium;
        stakingToken.safeTransferFrom(msg.sender, address(this), premiumAmount);
        
        policy.premiumsPaid += premiumAmount;
        policy.status = PolicyStatus.Active;
        
        // Distribute premium to pool and treasury
        uint256 fee = (premiumAmount * platformFee) / FEE_DENOMINATOR;
        uint256 poolAmount = premiumAmount - fee;
        
        poolBalance += poolAmount;
        stakingToken.safeTransfer(treasury, fee);

        emit PremiumPaid(policyId, premiumAmount);
    }

    /**
     * @dev Submit a claim
     */
    function submitClaim(
        uint256 policyId,
        DisputeType disputeType,
        uint256 claimAmount,
        uint256 incidentDate,
        string memory description,
        string[] memory evidenceHashes
    ) external validPolicy(policyId) onlyActivePolicy(policyId) nonReentrant whenNotPaused returns (uint256) {
        InsurancePolicy storage policy = policies[policyId];
        require(
            msg.sender == policy.policyholder || policy.authorizedClaimants[msg.sender],
            "Not authorized to submit claim"
        );
        require(_isCoveredType(policy.coveredTypes, disputeType), "Dispute type not covered");
        require(claimAmount > policy.deductible, "Claim below deductible");
        require(claimAmount <= policy.coverageAmount, "Claim exceeds coverage");
        require(incidentDate >= policy.startDate && incidentDate <= block.timestamp, "Invalid incident date");

        uint256 claimId = totalClaims++;
        
        Claim storage claim = claims[claimId];
        claim.claimId = claimId;
        claim.policyId = policyId;
        claim.claimant = msg.sender;
        claim.disputeType = disputeType;
        claim.claimAmount = claimAmount;
        claim.incidentDate = incidentDate;
        claim.claimDate = block.timestamp;
        claim.status = ClaimStatus.Submitted;
        claim.description = description;
        claim.evidenceHashes = evidenceHashes;

        policy.claimsCount++;
        userClaims[msg.sender].push(claimId);

        emit ClaimSubmitted(claimId, policyId, msg.sender);
        return claimId;
    }

    /**
     * @dev Assess a claim (assessors only)
     */
    function assessClaim(
        uint256 claimId,
        bool approved,
        uint256 assessmentScore,
        string memory comments
    ) external validClaim(claimId) onlyRole(ASSESSOR_ROLE) whenNotPaused {
        Claim storage claim = claims[claimId];
        require(claim.status == ClaimStatus.UnderReview, "Claim not under review");
        require(!claim.assessorVotes[msg.sender], "Already assessed");
        require(assessmentScore <= 100, "Invalid assessment score");

        claim.assessorVotes[msg.sender] = approved;
        claim.assessorComments[msg.sender] = comments;
        claim.assessorCount++;
        claim.assessmentScore = ((claim.assessmentScore * (claim.assessorCount - 1)) + assessmentScore) / claim.assessorCount;
        
        if (approved) {
            claim.approvalVotes++;
        }

        emit ClaimAssessed(claimId, msg.sender, approved);

        // Auto-finalize if enough assessors have voted
        if (claim.assessorCount >= MIN_ASSESSORS) {
            _finalizeClaim(claimId);
        }
    }

    /**
     * @dev Process approved claim payment
     */
    function processClaim(uint256 claimId) 
        external 
        validClaim(claimId) 
        onlyRole(CLAIMS_ADJUSTER_ROLE) 
        nonReentrant 
        whenNotPaused 
    {
        Claim storage claim = claims[claimId];
        require(claim.status == ClaimStatus.Approved, "Claim not approved");
        require(claim.approvedAmount > 0, "No approved amount");
        require(poolBalance >= claim.approvedAmount, "Insufficient pool balance");

        claim.status = ClaimStatus.Paid;
        claim.paidAmount = claim.approvedAmount;
        
        poolBalance -= claim.approvedAmount;
        policies[claim.policyId].totalClaimsAmount += claim.approvedAmount;

        stakingToken.safeTransfer(claim.claimant, claim.approvedAmount);

        emit ClaimPaid(claimId, claim.approvedAmount);
    }

    /**
     * @dev Add liquidity to the insurance pool
     */
    function addLiquidity(uint256 amount, uint256 lockupPeriod) 
        external 
        nonReentrant 
        whenNotPaused 
    {
        require(amount >= MIN_STAKE, "Amount below minimum stake");
        require(lockupPeriod >= 30 days, "Minimum 30-day lockup");

        stakingToken.safeTransferFrom(msg.sender, address(this), amount);

        LiquidityProvider storage provider = liquidityProviders[msg.sender];
        if (provider.provider == address(0)) {
            provider.provider = msg.sender;
            provider.active = true;
        }

        provider.stakedAmount += amount;
        provider.lockupEnd = block.timestamp + lockupPeriod;
        totalStaked += amount;
        poolBalance += amount;

        // Calculate share percentage
        provider.sharePercentage = (provider.stakedAmount * 10000) / totalStaked;

        emit LiquidityAdded(msg.sender, amount);
    }

    /**
     * @dev Remove liquidity from the pool
     */
    function removeLiquidity(uint256 amount) 
        external 
        nonReentrant 
        whenNotPaused 
    {
        LiquidityProvider storage provider = liquidityProviders[msg.sender];
        require(provider.active, "Not a liquidity provider");
        require(provider.stakedAmount >= amount, "Insufficient staked amount");
        require(block.timestamp >= provider.lockupEnd, "Lockup period not ended");
        require(poolBalance >= amount, "Insufficient pool balance");

        provider.stakedAmount -= amount;
        totalStaked -= amount;
        poolBalance -= amount;

        if (provider.stakedAmount == 0) {
            provider.active = false;
        } else {
            provider.sharePercentage = (provider.stakedAmount * 10000) / totalStaked;
        }

        stakingToken.safeTransfer(msg.sender, amount);

        emit LiquidityRemoved(msg.sender, amount);
    }

    /**
     * @dev Claim rewards for liquidity providers
     */
    function claimRewards() external nonReentrant whenNotPaused {
        LiquidityProvider storage provider = liquidityProviders[msg.sender];
        require(provider.active, "Not an active liquidity provider");

        uint256 rewards = calculatePendingRewards(msg.sender);
        require(rewards > 0, "No rewards to claim");

        provider.totalRewards += rewards;
        provider.lastRewardClaim = block.timestamp;

        stakingToken.safeTransfer(msg.sender, rewards);

        emit RewardsClaimed(msg.sender, rewards);
    }

    /**
     * @dev Perform risk assessment for a policy
     */
    function performRiskAssessment(
        uint256 policyId,
        RiskLevel riskLevel,
        uint256 score,
        string memory factors
    ) external validPolicy(policyId) onlyRole(UNDERWRITER_ROLE) whenNotPaused {
        require(score <= 100, "Invalid score");

        riskAssessments[policyId] = RiskAssessment({
            policyId: policyId,
            assessor: msg.sender,
            riskLevel: riskLevel,
            score: score,
            factors: factors,
            assessmentDate: block.timestamp,
            finalized: true
        });

        // Update policy risk level and recalculate premium
        InsurancePolicy storage policy = policies[policyId];
        policy.riskLevel = riskLevel;
        
        // Recalculate premium based on new risk assessment
        uint256 duration = policy.endDate - policy.startDate;
        policy.premium = calculatePremium(policy.coveredTypes, policy.coverageAmount, duration, riskLevel);

        emit RiskAssessed(policyId, msg.sender, riskLevel);
    }

    /**
     * @dev Calculate premium for a policy
     */
    function calculatePremium(
        DisputeType[] memory coveredTypes,
        uint256 coverageAmount,
        uint256 duration,
        RiskLevel riskLevel
    ) public view returns (uint256) {
        uint256 basePremium = 0;
        
        // Calculate base premium for all covered types
        for (uint256 i = 0; i < coveredTypes.length; i++) {
            basePremium += basePremiumRates[coveredTypes[i]];
        }
        
        // Apply coverage amount (per $1000)
        basePremium = (basePremium * coverageAmount) / 1000e18;
        
        // Apply duration (pro-rated from annual)
        basePremium = (basePremium * duration) / 365 days;
        
        // Apply risk multiplier
        basePremium = (basePremium * riskMultipliers[riskLevel]) / 100;
        
        return basePremium;
    }

    /**
     * @dev Calculate pending rewards for a liquidity provider
     */
    function calculatePendingRewards(address provider) public view returns (uint256) {
        LiquidityProvider storage lp = liquidityProviders[provider];
        if (!lp.active || lp.stakedAmount == 0) return 0;

        // Simplified reward calculation - 5% APY on staked amount
        uint256 timeElapsed = block.timestamp - lp.lastRewardClaim;
        uint256 annualReward = (lp.stakedAmount * 500) / 10000; // 5%
        uint256 reward = (annualReward * timeElapsed) / 365 days;
        
        return reward;
    }

    /**
     * @dev Get pool metrics
     */
    function getPoolMetrics() external view returns (PoolMetrics memory) {
        uint256 totalClaimsValue = 0;
        uint256 totalPremiumsValue = 0;
        
        // Calculate metrics (simplified)
        for (uint256 i = 0; i < totalPolicies; i++) {
            totalPremiumsValue += policies[i].premiumsPaid;
            totalClaimsValue += policies[i].totalClaimsAmount;
        }
        
        return PoolMetrics({
            totalPoolValue: poolBalance,
            totalStaked: totalStaked,
            totalClaims: totalClaimsValue,
            totalPremiums: totalPremiumsValue,
            utilizationRatio: totalPremiumsValue > 0 ? (totalClaimsValue * 100) / totalPremiumsValue : 0,
            solvencyRatio: totalClaimsValue > 0 ? (poolBalance * 100) / totalClaimsValue : 0,
            averageClaimAmount: totalClaims > 0 ? totalClaimsValue / totalClaims : 0,
            claimFrequency: totalPolicies > 0 ? (totalClaims * 100) / totalPolicies : 0
        });
    }

    /**
     * @dev Internal function to finalize claim assessment
     */
    function _finalizeClaim(uint256 claimId) internal {
        Claim storage claim = claims[claimId];
        
        // Simple majority voting
        if (claim.approvalVotes > claim.assessorCount / 2) {
            claim.status = ClaimStatus.Approved;
            
            // Calculate approved amount (could be less than claimed)
            uint256 approvedAmount = claim.claimAmount;
            if (claim.assessmentScore < 80) {
                // Reduce payout based on assessment score
                approvedAmount = (claim.claimAmount * claim.assessmentScore) / 100;
            }
            
            // Apply deductible
            InsurancePolicy storage policy = policies[claim.policyId];
            if (approvedAmount > policy.deductible) {
                claim.approvedAmount = approvedAmount - policy.deductible;
            } else {
                claim.approvedAmount = 0;
            }
        } else {
            claim.status = ClaimStatus.Denied;
            claim.denialReason = "Insufficient assessor approval";
            emit ClaimDenied(claimId, claim.denialReason);
        }
    }

    /**
     * @dev Internal function to check if dispute type is covered
     */
    function _isCoveredType(DisputeType[] memory coveredTypes, DisputeType disputeType) internal pure returns (bool) {
        for (uint256 i = 0; i < coveredTypes.length; i++) {
            if (coveredTypes[i] == disputeType) {
                return true;
            }
        }
        return false;
    }

    /**
     * @dev Get policy details
     */
    function getPolicyDetails(uint256 policyId) external view returns (
        address policyholder,
        DisputeType[] memory coveredTypes,
        uint256 coverageAmount,
        uint256 premium,
        PolicyStatus status,
        RiskLevel riskLevel,
        uint256 claimsCount
    ) {
        InsurancePolicy storage policy = policies[policyId];
        return (
            policy.policyholder,
            policy.coveredTypes,
            policy.coverageAmount,
            policy.premium,
            policy.status,
            policy.riskLevel,
            policy.claimsCount
        );
    }

    /**
     * @dev Get claim details
     */
    function getClaimDetails(uint256 claimId) external view returns (
        uint256 policyId,
        address claimant,
        DisputeType disputeType,
        uint256 claimAmount,
        ClaimStatus status,
        uint256 approvedAmount,
        uint256 assessmentScore
    ) {
        Claim storage claim = claims[claimId];
        return (
            claim.policyId,
            claim.claimant,
            claim.disputeType,
            claim.claimAmount,
            claim.status,
            claim.approvedAmount,
            claim.assessmentScore
        );
    }

    /**
     * @dev Get user policies
     */
    function getUserPolicies(address user) external view returns (uint256[] memory) {
        return userPolicies[user];
    }

    /**
     * @dev Get user claims
     */
    function getUserClaims(address user) external view returns (uint256[] memory) {
        return userClaims[user];
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

    function setPlatformFee(uint256 newFee) external onlyRole(ADMIN_ROLE) {
        require(newFee <= 1000, "Fee too high"); // Max 10%
        platformFee = newFee;
    }

    function updateBasePremiumRate(DisputeType disputeType, uint256 newRate) external onlyRole(ADMIN_ROLE) {
        basePremiumRates[disputeType] = newRate;
    }

    function emergencyWithdraw(uint256 amount) external onlyRole(ADMIN_ROLE) {
        require(amount <= poolBalance, "Insufficient balance");
        poolBalance -= amount;
        stakingToken.safeTransfer(treasury, amount);
    }
}