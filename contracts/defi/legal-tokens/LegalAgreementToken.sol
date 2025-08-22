// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Burnable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title LegalAgreementToken
 * @dev NFT representing tokenized legal agreements with fractional ownership and revenue sharing
 */
contract LegalAgreementToken is 
    ERC721, 
    ERC721Enumerable, 
    ERC721URIStorage, 
    ERC721Burnable, 
    AccessControl, 
    Pausable, 
    ReentrancyGuard 
{
    using SafeERC20 for IERC20;
    using Counters for Counters.Counter;
    using ECDSA for bytes32;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");

    enum AgreementStatus {
        Draft,
        Active,
        Completed,
        Breached,
        Disputed,
        Terminated
    }

    enum AgreementType {
        ServiceContract,
        LicenseAgreement,
        PartnershipAgreement,
        EmploymentContract,
        RealEstateContract,
        IntellectualProperty,
        SupplyAgreement,
        NDAAgreement,
        ConsultingAgreement,
        MergersAcquisitions
    }

    struct Agreement {
        uint256 tokenId;
        address creator;
        AgreementType agreementType;
        AgreementStatus status;
        string title;
        string description;
        bytes32 termsHash;
        string jurisdiction;
        uint256 value; // Total value of the agreement
        uint256 createdAt;
        uint256 effectiveDate;
        uint256 expiryDate;
        address[] parties;
        mapping(address => bool) signatures;
        uint256 signatureCount;
        uint256 requiredSignatures;
        bool fractionalized;
        uint256 totalShares;
        mapping(address => uint256) shareHolders;
        uint256 totalRevenue;
        uint256 distributedRevenue;
        string metadataURI;
    }

    struct RevenueDistribution {
        uint256 agreementId;
        uint256 amount;
        uint256 timestamp;
        string source;
        mapping(address => uint256) distributions;
        mapping(address => bool) claimed;
        uint256 totalClaimed;
    }

    struct FractionalShare {
        uint256 agreementId;
        address holder;
        uint256 shares;
        uint256 purchasePrice;
        uint256 purchaseTime;
        bool locked;
        uint256 lockExpiry;
    }

    Counters.Counter private _tokenIdCounter;
    Counters.Counter private _distributionIdCounter;

    mapping(uint256 => Agreement) public agreements;
    mapping(uint256 => RevenueDistribution) public distributions;
    mapping(uint256 => mapping(address => FractionalShare)) public fractionalShares;
    mapping(address => uint256[]) public userAgreements;
    mapping(bytes32 => uint256) public termsToToken;
    mapping(uint256 => uint256[]) public agreementDistributions;
    
    // Market-related storage
    mapping(uint256 => uint256) public sharePrice; // Price per share for fractional ownership
    mapping(uint256 => bool) public sharesForSale;
    mapping(uint256 => mapping(address => uint256)) public shareOffers; // tokenId => offeror => amount
    
    uint256 public platformFee = 250; // 2.5% in basis points
    uint256 public constant FEE_DENOMINATOR = 10000;
    address public treasury;

    event AgreementCreated(uint256 indexed tokenId, address indexed creator, AgreementType agreementType);
    event AgreementSigned(uint256 indexed tokenId, address indexed signer);
    event AgreementStatusChanged(uint256 indexed tokenId, AgreementStatus newStatus);
    event SharesCreated(uint256 indexed tokenId, uint256 totalShares, uint256 pricePerShare);
    event SharesPurchased(uint256 indexed tokenId, address indexed buyer, uint256 shares, uint256 totalPrice);
    event RevenueDistributed(uint256 indexed tokenId, uint256 amount, uint256 distributionId);
    event RevenueClaimedByHolder(uint256 indexed tokenId, address indexed holder, uint256 amount);
    event SharesTransferred(uint256 indexed tokenId, address indexed from, address indexed to, uint256 shares);
    event ShareOfferMade(uint256 indexed tokenId, address indexed offeror, uint256 amount);
    event ShareOfferAccepted(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 amount);

    modifier onlyAgreementParty(uint256 tokenId) {
        require(_isAgreementParty(tokenId, msg.sender), "Not an agreement party");
        _;
    }

    modifier onlyAgreementCreator(uint256 tokenId) {
        require(agreements[tokenId].creator == msg.sender, "Not agreement creator");
        _;
    }

    modifier validAgreement(uint256 tokenId) {
        require(_exists(tokenId), "Agreement does not exist");
        _;
    }

    constructor(address _treasury) ERC721("Legal Agreement Token", "LAT") {
        require(_treasury != address(0), "Invalid treasury address");
        treasury = _treasury;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    /**
     * @dev Create a new legal agreement NFT
     */
    function createAgreement(
        AgreementType agreementType,
        string memory title,
        string memory description,
        bytes32 termsHash,
        string memory jurisdiction,
        uint256 value,
        uint256 effectiveDate,
        uint256 expiryDate,
        address[] memory parties,
        string memory metadataURI
    ) external nonReentrant whenNotPaused returns (uint256) {
        require(parties.length >= 2, "Need at least 2 parties");
        require(effectiveDate >= block.timestamp, "Invalid effective date");
        require(expiryDate > effectiveDate, "Invalid expiry date");
        require(bytes(title).length > 0, "Empty title");
        require(termsToToken[termsHash] == 0, "Terms already tokenized");

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        Agreement storage agreement = agreements[tokenId];
        agreement.tokenId = tokenId;
        agreement.creator = msg.sender;
        agreement.agreementType = agreementType;
        agreement.status = AgreementStatus.Draft;
        agreement.title = title;
        agreement.description = description;
        agreement.termsHash = termsHash;
        agreement.jurisdiction = jurisdiction;
        agreement.value = value;
        agreement.createdAt = block.timestamp;
        agreement.effectiveDate = effectiveDate;
        agreement.expiryDate = expiryDate;
        agreement.parties = parties;
        agreement.requiredSignatures = parties.length;
        agreement.metadataURI = metadataURI;

        _safeMint(msg.sender, tokenId);
        _setTokenURI(tokenId, metadataURI);
        
        termsToToken[termsHash] = tokenId;
        userAgreements[msg.sender].push(tokenId);

        // Add to user agreements for all parties
        for (uint256 i = 0; i < parties.length; i++) {
            userAgreements[parties[i]].push(tokenId);
        }

        emit AgreementCreated(tokenId, msg.sender, agreementType);
        return tokenId;
    }

    /**
     * @dev Sign an agreement
     */
    function signAgreement(uint256 tokenId) 
        external 
        validAgreement(tokenId) 
        onlyAgreementParty(tokenId) 
        whenNotPaused 
    {
        Agreement storage agreement = agreements[tokenId];
        require(agreement.status == AgreementStatus.Draft, "Agreement not in draft status");
        require(!agreement.signatures[msg.sender], "Already signed");

        agreement.signatures[msg.sender] = true;
        agreement.signatureCount++;

        emit AgreementSigned(tokenId, msg.sender);

        // Auto-activate if all parties signed
        if (agreement.signatureCount >= agreement.requiredSignatures) {
            agreement.status = AgreementStatus.Active;
            emit AgreementStatusChanged(tokenId, AgreementStatus.Active);
        }
    }

    /**
     * @dev Create fractional shares for an agreement
     */
    function createFractionalShares(
        uint256 tokenId,
        uint256 totalShares,
        uint256 pricePerShare
    ) external validAgreement(tokenId) onlyAgreementCreator(tokenId) whenNotPaused {
        Agreement storage agreement = agreements[tokenId];
        require(!agreement.fractionalized, "Already fractionalized");
        require(totalShares > 0, "Invalid total shares");
        require(pricePerShare > 0, "Invalid price per share");

        agreement.fractionalized = true;
        agreement.totalShares = totalShares;
        
        // Creator gets initial ownership
        agreement.shareHolders[msg.sender] = totalShares;
        
        fractionalShares[tokenId][msg.sender] = FractionalShare({
            agreementId: tokenId,
            holder: msg.sender,
            shares: totalShares,
            purchasePrice: 0,
            purchaseTime: block.timestamp,
            locked: false,
            lockExpiry: 0
        });

        sharePrice[tokenId] = pricePerShare;
        sharesForSale[tokenId] = true;

        emit SharesCreated(tokenId, totalShares, pricePerShare);
    }

    /**
     * @dev Purchase fractional shares
     */
    function purchaseShares(
        uint256 tokenId,
        uint256 shares
    ) external payable validAgreement(tokenId) nonReentrant whenNotPaused {
        require(agreements[tokenId].fractionalized, "Agreement not fractionalized");
        require(sharesForSale[tokenId], "Shares not for sale");
        require(shares > 0, "Invalid share amount");
        
        uint256 totalPrice = shares * sharePrice[tokenId];
        require(msg.value >= totalPrice, "Insufficient payment");

        Agreement storage agreement = agreements[tokenId];
        require(agreement.shareHolders[agreement.creator] >= shares, "Insufficient shares available");

        // Transfer shares
        agreement.shareHolders[agreement.creator] -= shares;
        agreement.shareHolders[msg.sender] += shares;

        // Update fractional share records
        if (fractionalShares[tokenId][msg.sender].holder == address(0)) {
            fractionalShares[tokenId][msg.sender] = FractionalShare({
                agreementId: tokenId,
                holder: msg.sender,
                shares: shares,
                purchasePrice: totalPrice,
                purchaseTime: block.timestamp,
                locked: false,
                lockExpiry: 0
            });
        } else {
            fractionalShares[tokenId][msg.sender].shares += shares;
            fractionalShares[tokenId][msg.sender].purchasePrice += totalPrice;
        }

        // Transfer payment to creator (minus platform fee)
        uint256 platformFeeAmount = (totalPrice * platformFee) / FEE_DENOMINATOR;
        uint256 creatorAmount = totalPrice - platformFeeAmount;

        payable(agreement.creator).transfer(creatorAmount);
        if (platformFeeAmount > 0) {
            payable(treasury).transfer(platformFeeAmount);
        }

        // Refund excess payment
        if (msg.value > totalPrice) {
            payable(msg.sender).transfer(msg.value - totalPrice);
        }

        emit SharesPurchased(tokenId, msg.sender, shares, totalPrice);
    }

    /**
     * @dev Distribute revenue to shareholders
     */
    function distributeRevenue(
        uint256 tokenId,
        string memory source
    ) external payable validAgreement(tokenId) nonReentrant whenNotPaused {
        require(msg.value > 0, "No revenue to distribute");
        Agreement storage agreement = agreements[tokenId];
        require(agreement.fractionalized, "Agreement not fractionalized");

        uint256 distributionId = _distributionIdCounter.current();
        _distributionIdCounter.increment();

        RevenueDistribution storage distribution = distributions[distributionId];
        distribution.agreementId = tokenId;
        distribution.amount = msg.value;
        distribution.timestamp = block.timestamp;
        distribution.source = source;

        // Calculate distributions for each shareholder
        for (uint256 i = 0; i < agreement.parties.length; i++) {
            address party = agreement.parties[i];
            uint256 shares = agreement.shareHolders[party];
            if (shares > 0) {
                uint256 distributionAmount = (msg.value * shares) / agreement.totalShares;
                distribution.distributions[party] = distributionAmount;
            }
        }

        agreement.totalRevenue += msg.value;
        agreementDistributions[tokenId].push(distributionId);

        emit RevenueDistributed(tokenId, msg.value, distributionId);
    }

    /**
     * @dev Claim revenue distribution
     */
    function claimRevenue(uint256 distributionId) external nonReentrant whenNotPaused {
        RevenueDistribution storage distribution = distributions[distributionId];
        require(!distribution.claimed[msg.sender], "Already claimed");
        require(distribution.distributions[msg.sender] > 0, "No distribution for this address");

        uint256 amount = distribution.distributions[msg.sender];
        distribution.claimed[msg.sender] = true;
        distribution.totalClaimed += amount;

        Agreement storage agreement = agreements[distribution.agreementId];
        agreement.distributedRevenue += amount;

        payable(msg.sender).transfer(amount);

        emit RevenueClaimedByHolder(distribution.agreementId, msg.sender, amount);
    }

    /**
     * @dev Transfer fractional shares between holders
     */
    function transferShares(
        uint256 tokenId,
        address to,
        uint256 shares
    ) external validAgreement(tokenId) whenNotPaused {
        require(to != address(0), "Invalid recipient");
        require(shares > 0, "Invalid share amount");
        
        Agreement storage agreement = agreements[tokenId];
        require(agreement.fractionalized, "Agreement not fractionalized");
        require(agreement.shareHolders[msg.sender] >= shares, "Insufficient shares");

        FractionalShare storage senderShare = fractionalShares[tokenId][msg.sender];
        require(!senderShare.locked || block.timestamp >= senderShare.lockExpiry, "Shares are locked");

        // Transfer shares
        agreement.shareHolders[msg.sender] -= shares;
        agreement.shareHolders[to] += shares;

        // Update fractional share records
        senderShare.shares -= shares;

        if (fractionalShares[tokenId][to].holder == address(0)) {
            fractionalShares[tokenId][to] = FractionalShare({
                agreementId: tokenId,
                holder: to,
                shares: shares,
                purchasePrice: 0,
                purchaseTime: block.timestamp,
                locked: false,
                lockExpiry: 0
            });
        } else {
            fractionalShares[tokenId][to].shares += shares;
        }

        emit SharesTransferred(tokenId, msg.sender, to, shares);
    }

    /**
     * @dev Make an offer to buy shares
     */
    function makeShareOffer(uint256 tokenId) external payable validAgreement(tokenId) whenNotPaused {
        require(msg.value > 0, "Invalid offer amount");
        require(agreements[tokenId].fractionalized, "Agreement not fractionalized");

        shareOffers[tokenId][msg.sender] = msg.value;

        emit ShareOfferMade(tokenId, msg.sender, msg.value);
    }

    /**
     * @dev Accept a share offer
     */
    function acceptShareOffer(
        uint256 tokenId,
        address offeror,
        uint256 shares
    ) external validAgreement(tokenId) nonReentrant whenNotPaused {
        require(shares > 0, "Invalid share amount");
        require(shareOffers[tokenId][offeror] > 0, "No offer from this address");
        
        Agreement storage agreement = agreements[tokenId];
        require(agreement.shareHolders[msg.sender] >= shares, "Insufficient shares");

        uint256 offerAmount = shareOffers[tokenId][offeror];
        uint256 pricePerShare = offerAmount / shares;
        require(pricePerShare > 0, "Insufficient offer amount");

        // Transfer shares
        agreement.shareHolders[msg.sender] -= shares;
        agreement.shareHolders[offeror] += shares;

        // Update fractional share records
        fractionalShares[tokenId][msg.sender].shares -= shares;
        
        if (fractionalShares[tokenId][offeror].holder == address(0)) {
            fractionalShares[tokenId][offeror] = FractionalShare({
                agreementId: tokenId,
                holder: offeror,
                shares: shares,
                purchasePrice: offerAmount,
                purchaseTime: block.timestamp,
                locked: false,
                lockExpiry: 0
            });
        } else {
            fractionalShares[tokenId][offeror].shares += shares;
            fractionalShares[tokenId][offeror].purchasePrice += offerAmount;
        }

        // Transfer payment
        uint256 platformFeeAmount = (offerAmount * platformFee) / FEE_DENOMINATOR;
        uint256 sellerAmount = offerAmount - platformFeeAmount;

        payable(msg.sender).transfer(sellerAmount);
        if (platformFeeAmount > 0) {
            payable(treasury).transfer(platformFeeAmount);
        }

        // Clear the offer
        shareOffers[tokenId][offeror] = 0;

        emit ShareOfferAccepted(tokenId, msg.sender, offeror, offerAmount);
    }

    /**
     * @dev Update agreement status
     */
    function updateAgreementStatus(
        uint256 tokenId,
        AgreementStatus newStatus
    ) external validAgreement(tokenId) onlyAgreementParty(tokenId) whenNotPaused {
        Agreement storage agreement = agreements[tokenId];
        require(agreement.status != newStatus, "Status already set");

        agreement.status = newStatus;
        emit AgreementStatusChanged(tokenId, newStatus);
    }

    /**
     * @dev Get agreement details
     */
    function getAgreementDetails(uint256 tokenId) external view returns (
        address creator,
        AgreementType agreementType,
        AgreementStatus status,
        string memory title,
        uint256 value,
        uint256 effectiveDate,
        uint256 expiryDate,
        address[] memory parties,
        bool fractionalized,
        uint256 totalShares
    ) {
        Agreement storage agreement = agreements[tokenId];
        return (
            agreement.creator,
            agreement.agreementType,
            agreement.status,
            agreement.title,
            agreement.value,
            agreement.effectiveDate,
            agreement.expiryDate,
            agreement.parties,
            agreement.fractionalized,
            agreement.totalShares
        );
    }

    /**
     * @dev Get user's agreements
     */
    function getUserAgreements(address user) external view returns (uint256[] memory) {
        return userAgreements[user];
    }

    /**
     * @dev Get shareholder information
     */
    function getShareholderInfo(uint256 tokenId, address holder) external view returns (
        uint256 shares,
        uint256 purchasePrice,
        uint256 purchaseTime,
        bool locked,
        uint256 lockExpiry
    ) {
        FractionalShare storage share = fractionalShares[tokenId][holder];
        return (
            share.shares,
            share.purchasePrice,
            share.purchaseTime,
            share.locked,
            share.lockExpiry
        );
    }

    /**
     * @dev Get pending revenue for holder
     */
    function getPendingRevenue(uint256 tokenId, address holder) external view returns (uint256) {
        uint256 totalPending = 0;
        uint256[] memory distributionIds = agreementDistributions[tokenId];
        
        for (uint256 i = 0; i < distributionIds.length; i++) {
            RevenueDistribution storage distribution = distributions[distributionIds[i]];
            if (!distribution.claimed[holder] && distribution.distributions[holder] > 0) {
                totalPending += distribution.distributions[holder];
            }
        }
        
        return totalPending;
    }

    /**
     * @dev Internal function to check if address is agreement party
     */
    function _isAgreementParty(uint256 tokenId, address addr) internal view returns (bool) {
        Agreement storage agreement = agreements[tokenId];
        
        // Creator is always a party
        if (agreement.creator == addr) return true;
        
        // Check if address is in parties array
        for (uint256 i = 0; i < agreement.parties.length; i++) {
            if (agreement.parties[i] == addr) return true;
        }
        
        return false;
    }

    /**
     * @dev Override required functions
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override(ERC721, ERC721Enumerable) {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
    }

    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId) public view override(ERC721, ERC721Enumerable, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
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

    function setTreasury(address newTreasury) external onlyRole(ADMIN_ROLE) {
        require(newTreasury != address(0), "Invalid treasury");
        treasury = newTreasury;
    }

    function emergencyWithdraw() external onlyRole(ADMIN_ROLE) {
        payable(treasury).transfer(address(this).balance);
    }

    receive() external payable {}
}