use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::collections::{LookupMap, UnorderedMap, Vector};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::{
    env, near_bindgen, AccountId, Balance, Promise, PromiseResult,
    PanicOnDefault, BorshStorageKey, CryptoHash, Gas,
};
use std::collections::HashMap;

/// Gas allocation for cross-contract calls
const ARBITRATION_GAS: Gas = Gas(30_000_000_000_000);
const CALLBACK_GAS: Gas = Gas(20_000_000_000_000);

#[derive(BorshSerialize, BorshStorageKey)]
enum StorageKey {
    Arbitrators,
    Disputes,
    Clauses,
    UserDisputes,
    ArbitratorDisputes,
    Resolutions,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone)]
#[serde(crate = "near_sdk::serde")]
pub enum DisputeStatus {
    Created,
    UnderReview,
    InArbitration,
    Resolved,
    Appealed,
    Closed,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, Clone)]
#[serde(crate = "near_sdk::serde")]
pub enum DisputeType {
    ContractBreach,
    PaymentDispute,
    IntellectualProperty,
    ServiceDispute,
    CommercialTrade,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct Arbitrator {
    pub account_id: AccountId,
    pub name: String,
    pub specializations: Vec<String>,
    pub jurisdictions: Vec<String>,
    pub total_cases: u64,
    pub successful_cases: u64,
    pub reputation_score: u32,
    pub stake_amount: Balance,
    pub is_active: bool,
    pub certification_hash: String,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct Dispute {
    pub id: CryptoHash,
    pub claimant: AccountId,
    pub respondent: AccountId,
    pub dispute_type: DisputeType,
    pub status: DisputeStatus,
    pub description: String,
    pub evidence: Vec<String>,
    pub dispute_amount: Balance,
    pub created_at: u64,
    pub deadline: u64,
    pub assigned_arbitrator: Option<AccountId>,
    pub resolution: Option<String>,
    pub arbitration_fee: Balance,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct ArbitrationClause {
    pub id: CryptoHash,
    pub creator: AccountId,
    pub clause_text: String,
    pub applicable_jurisdictions: Vec<String>,
    pub arbitration_fee: Balance,
    pub is_active: bool,
    pub created_at: u64,
    pub approved_arbitrators: Vec<AccountId>,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct Resolution {
    pub dispute_id: CryptoHash,
    pub arbitrator: AccountId,
    pub decision: String,
    pub award_amount: Balance,
    pub award_recipient: AccountId,
    pub resolved_at: u64,
    pub reasoning: String,
    pub is_final: bool,
}

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct LegalArbitration {
    pub owner: AccountId,
    pub arbitrators: LookupMap<AccountId, Arbitrator>,
    pub disputes: UnorderedMap<CryptoHash, Dispute>,
    pub arbitration_clauses: UnorderedMap<CryptoHash, ArbitrationClause>,
    pub user_disputes: LookupMap<AccountId, Vector<CryptoHash>>,
    pub arbitrator_disputes: LookupMap<AccountId, Vector<CryptoHash>>,
    pub resolutions: LookupMap<CryptoHash, Resolution>,
    pub platform_fee_percentage: u32, // Basis points (100 = 1%)
    pub min_stake_amount: Balance,
    pub min_arbitration_fee: Balance,
}

#[near_bindgen]
impl LegalArbitration {
    #[init]
    pub fn new(owner: AccountId) -> Self {
        Self {
            owner,
            arbitrators: LookupMap::new(StorageKey::Arbitrators),
            disputes: UnorderedMap::new(StorageKey::Disputes),
            arbitration_clauses: UnorderedMap::new(StorageKey::Clauses),
            user_disputes: LookupMap::new(StorageKey::UserDisputes),
            arbitrator_disputes: LookupMap::new(StorageKey::ArbitratorDisputes),
            resolutions: LookupMap::new(StorageKey::Resolutions),
            platform_fee_percentage: 250, // 2.5%
            min_stake_amount: near_sdk::utils::parse_near!("10"),
            min_arbitration_fee: near_sdk::utils::parse_near!("0.1"),
        }
    }

    /// Register as an arbitrator with stake
    #[payable]
    pub fn register_arbitrator(
        &mut self,
        name: String,
        specializations: Vec<String>,
        jurisdictions: Vec<String>,
        certification_hash: String,
    ) {
        let account_id = env::predecessor_account_id();
        let deposit = env::attached_deposit();
        
        assert!(deposit >= self.min_stake_amount, "Insufficient stake amount");
        assert!(!self.arbitrators.contains_key(&account_id), "Arbitrator already registered");
        assert!(!name.is_empty(), "Name cannot be empty");

        let arbitrator = Arbitrator {
            account_id: account_id.clone(),
            name,
            specializations,
            jurisdictions,
            total_cases: 0,
            successful_cases: 0,
            reputation_score: 50, // Start with neutral reputation
            stake_amount: deposit,
            is_active: true,
            certification_hash,
        };

        self.arbitrators.insert(&account_id, &arbitrator);
        
        // Initialize empty dispute vector for arbitrator
        self.arbitrator_disputes.insert(&account_id, &Vector::new(format!("arb_disputes_{}", account_id).as_bytes()));

        env::log_str(&format!("Arbitrator {} registered with stake {}", account_id, deposit));
    }

    /// Create an arbitration clause
    #[payable]
    pub fn create_arbitration_clause(
        &mut self,
        clause_text: String,
        applicable_jurisdictions: Vec<String>,
        arbitration_fee: Balance,
    ) -> CryptoHash {
        let creator = env::predecessor_account_id();
        let deposit = env::attached_deposit();
        
        assert!(deposit >= self.min_arbitration_fee, "Insufficient registration fee");
        assert!(!clause_text.is_empty(), "Clause text cannot be empty");

        let clause_id = env::keccak256(
            format!("{}{}{}", clause_text, creator, env::block_timestamp()).as_bytes()
        );

        let clause = ArbitrationClause {
            id: clause_id,
            creator,
            clause_text,
            applicable_jurisdictions,
            arbitration_fee,
            is_active: true,
            created_at: env::block_timestamp(),
            approved_arbitrators: Vec::new(),
        };

        self.arbitration_clauses.insert(&clause_id, &clause);

        env::log_str(&format!("Arbitration clause created with ID: {:?}", clause_id));
        clause_id
    }

    /// Create a new dispute
    #[payable]
    pub fn create_dispute(
        &mut self,
        respondent: AccountId,
        dispute_type: DisputeType,
        description: String,
        dispute_amount: Balance,
    ) -> CryptoHash {
        let claimant = env::predecessor_account_id();
        let arbitration_fee = env::attached_deposit();
        
        assert!(arbitration_fee >= self.min_arbitration_fee, "Insufficient arbitration fee");
        assert!(respondent != claimant, "Cannot dispute with yourself");
        assert!(!description.is_empty(), "Description cannot be empty");

        let dispute_id = env::keccak256(
            format!("{}{}{}{}", claimant, respondent, description, env::block_timestamp()).as_bytes()
        );

        let dispute = Dispute {
            id: dispute_id,
            claimant: claimant.clone(),
            respondent: respondent.clone(),
            dispute_type,
            status: DisputeStatus::Created,
            description,
            evidence: Vec::new(),
            dispute_amount,
            created_at: env::block_timestamp(),
            deadline: env::block_timestamp() + 14 * 24 * 60 * 60 * 1_000_000_000, // 14 days
            assigned_arbitrator: None,
            resolution: None,
            arbitration_fee,
        };

        self.disputes.insert(&dispute_id, &dispute);

        // Add to user disputes
        self.add_user_dispute(&claimant, dispute_id);
        self.add_user_dispute(&respondent, dispute_id);

        env::log_str(&format!("Dispute created with ID: {:?}", dispute_id));
        dispute_id
    }

    /// Submit evidence for a dispute
    pub fn submit_evidence(&mut self, dispute_id: CryptoHash, evidence: String) {
        let account_id = env::predecessor_account_id();
        let mut dispute = self.disputes.get(&dispute_id).expect("Dispute not found");
        
        assert!(
            dispute.claimant == account_id || dispute.respondent == account_id,
            "Not a party to this dispute"
        );
        assert!(!evidence.is_empty(), "Evidence cannot be empty");
        assert!(env::block_timestamp() < dispute.deadline, "Evidence submission deadline passed");

        dispute.evidence.push(evidence);
        self.disputes.insert(&dispute_id, &dispute);

        env::log_str(&format!("Evidence submitted for dispute: {:?}", dispute_id));
    }

    /// Assign arbitrator to dispute
    pub fn assign_arbitrator(&mut self, dispute_id: CryptoHash, arbitrator: AccountId) {
        let caller = env::predecessor_account_id();
        let mut dispute = self.disputes.get(&dispute_id).expect("Dispute not found");
        
        assert!(
            dispute.claimant == caller || dispute.respondent == caller,
            "Not a party to this dispute"
        );
        assert!(self.arbitrators.contains_key(&arbitrator), "Arbitrator not registered");
        assert!(dispute.assigned_arbitrator.is_none(), "Arbitrator already assigned");

        dispute.assigned_arbitrator = Some(arbitrator.clone());
        dispute.status = DisputeStatus::InArbitration;
        self.disputes.insert(&dispute_id, &dispute);

        // Add to arbitrator's disputes
        let mut arb_disputes = self.arbitrator_disputes.get(&arbitrator)
            .unwrap_or_else(|| Vector::new(format!("arb_disputes_{}", arbitrator).as_bytes()));
        arb_disputes.push(&dispute_id);
        self.arbitrator_disputes.insert(&arbitrator, &arb_disputes);

        env::log_str(&format!("Arbitrator {} assigned to dispute: {:?}", arbitrator, dispute_id));
    }

    /// Resolve dispute (arbitrator only)
    #[payable]
    pub fn resolve_dispute(
        &mut self,
        dispute_id: CryptoHash,
        decision: String,
        award_amount: Balance,
        award_recipient: AccountId,
        reasoning: String,
    ) {
        let arbitrator = env::predecessor_account_id();
        let mut dispute = self.disputes.get(&dispute_id).expect("Dispute not found");
        
        assert_eq!(
            dispute.assigned_arbitrator.as_ref().unwrap(),
            &arbitrator,
            "Not the assigned arbitrator"
        );
        assert!(matches!(dispute.status, DisputeStatus::InArbitration), "Dispute not in arbitration");
        assert!(!decision.is_empty(), "Decision cannot be empty");
        assert!(award_amount <= dispute.dispute_amount, "Award exceeds dispute amount");

        dispute.status = DisputeStatus::Resolved;
        dispute.resolution = Some(decision.clone());
        self.disputes.insert(&dispute_id, &dispute);

        let resolution = Resolution {
            dispute_id,
            arbitrator: arbitrator.clone(),
            decision,
            award_amount,
            award_recipient: award_recipient.clone(),
            resolved_at: env::block_timestamp(),
            reasoning,
            is_final: true,
        };

        self.resolutions.insert(&dispute_id, &resolution);

        // Update arbitrator statistics
        self.update_arbitrator_stats(&arbitrator, true);

        // Transfer award if applicable
        if award_amount > 0 {
            let platform_fee = (award_amount * self.platform_fee_percentage as u128) / 10000;
            let net_award = award_amount - platform_fee;
            
            Promise::new(award_recipient).transfer(net_award);
            if platform_fee > 0 {
                Promise::new(self.owner.clone()).transfer(platform_fee);
            }
        }

        env::log_str(&format!("Dispute {:?} resolved by arbitrator {}", dispute_id, arbitrator));
    }

    /// Auto-resolve expired disputes
    pub fn auto_resolve_expired_dispute(&mut self, dispute_id: CryptoHash) {
        let mut dispute = self.disputes.get(&dispute_id).expect("Dispute not found");
        
        assert!(env::block_timestamp() > dispute.deadline, "Dispute not expired");
        assert!(matches!(dispute.status, DisputeStatus::InArbitration | DisputeStatus::UnderReview), 
                "Dispute not in resolvable state");

        dispute.status = DisputeStatus::Resolved;
        dispute.resolution = Some("Auto-resolved due to timeout".to_string());
        self.disputes.insert(&dispute_id, &dispute);

        let resolution = Resolution {
            dispute_id,
            arbitrator: env::current_account_id(), // System resolution
            decision: "Auto-resolved in favor of respondent due to timeout".to_string(),
            award_amount: 0,
            award_recipient: dispute.respondent.clone(),
            resolved_at: env::block_timestamp(),
            reasoning: "Dispute exceeded deadline without resolution".to_string(),
            is_final: true,
        };

        self.resolutions.insert(&dispute_id, &resolution);

        env::log_str(&format!("Dispute {:?} auto-resolved due to expiration", dispute_id));
    }

    /// Get arbitrator details
    pub fn get_arbitrator(&self, account_id: AccountId) -> Option<Arbitrator> {
        self.arbitrators.get(&account_id)
    }

    /// Get dispute details
    pub fn get_dispute(&self, dispute_id: CryptoHash) -> Option<Dispute> {
        self.disputes.get(&dispute_id)
    }

    /// Get arbitration clause
    pub fn get_arbitration_clause(&self, clause_id: CryptoHash) -> Option<ArbitrationClause> {
        self.arbitration_clauses.get(&clause_id)
    }

    /// Get resolution details
    pub fn get_resolution(&self, dispute_id: CryptoHash) -> Option<Resolution> {
        self.resolutions.get(&dispute_id)
    }

    /// Get user disputes
    pub fn get_user_disputes(&self, account_id: AccountId) -> Vec<CryptoHash> {
        self.user_disputes.get(&account_id)
            .map(|disputes| disputes.to_vec())
            .unwrap_or_default()
    }

    /// Get arbitrator disputes
    pub fn get_arbitrator_disputes(&self, arbitrator: AccountId) -> Vec<CryptoHash> {
        self.arbitrator_disputes.get(&arbitrator)
            .map(|disputes| disputes.to_vec())
            .unwrap_or_default()
    }

    /// Get total disputes count
    pub fn get_total_disputes(&self) -> u64 {
        self.disputes.len()
    }

    /// Update arbitrator reputation
    pub fn update_arbitrator_reputation(&mut self, arbitrator: AccountId, positive: bool) {
        let caller = env::predecessor_account_id();
        assert!(self.arbitrators.contains_key(&caller), "Caller not a registered arbitrator");
        assert!(arbitrator != caller, "Cannot vote for yourself");

        if let Some(mut arb) = self.arbitrators.get(&arbitrator) {
            if positive && arb.reputation_score < 100 {
                arb.reputation_score += 1;
            } else if !positive && arb.reputation_score > 0 {
                arb.reputation_score = arb.reputation_score.saturating_sub(2);
            }
            
            self.arbitrators.insert(&arbitrator, &arb);
            env::log_str(&format!("Arbitrator {} reputation updated to {}", arbitrator, arb.reputation_score));
        }
    }

    /// Add stake to arbitrator account
    #[payable]
    pub fn add_arbitrator_stake(&mut self) {
        let arbitrator = env::predecessor_account_id();
        let additional_stake = env::attached_deposit();
        
        assert!(additional_stake > 0, "Must stake positive amount");
        
        if let Some(mut arb) = self.arbitrators.get(&arbitrator) {
            arb.stake_amount += additional_stake;
            self.arbitrators.insert(&arbitrator, &arb);
            env::log_str(&format!("Arbitrator {} added stake: {}", arbitrator, additional_stake));
        } else {
            panic!("Arbitrator not registered");
        }
    }

    /// Withdraw arbitrator stake (partial)
    pub fn withdraw_arbitrator_stake(&mut self, amount: Balance) {
        let arbitrator = env::predecessor_account_id();
        
        if let Some(mut arb) = self.arbitrators.get(&arbitrator) {
            assert!(arb.stake_amount.saturating_sub(amount) >= self.min_stake_amount, 
                    "Cannot withdraw below minimum stake");
            
            arb.stake_amount -= amount;
            self.arbitrators.insert(&arbitrator, &arb);
            
            Promise::new(arbitrator.clone()).transfer(amount);
            env::log_str(&format!("Arbitrator {} withdrew stake: {}", arbitrator, amount));
        } else {
            panic!("Arbitrator not registered");
        }
    }

    /// Private helper functions
    fn add_user_dispute(&mut self, user: &AccountId, dispute_id: CryptoHash) {
        let mut user_disputes = self.user_disputes.get(user)
            .unwrap_or_else(|| Vector::new(format!("user_disputes_{}", user).as_bytes()));
        user_disputes.push(&dispute_id);
        self.user_disputes.insert(user, &user_disputes);
    }

    fn update_arbitrator_stats(&mut self, arbitrator: &AccountId, successful: bool) {
        if let Some(mut arb) = self.arbitrators.get(arbitrator) {
            arb.total_cases += 1;
            if successful {
                arb.successful_cases += 1;
            }
            self.arbitrators.insert(arbitrator, &arb);
        }
    }

    /// Admin functions
    pub fn set_platform_fee(&mut self, new_fee_percentage: u32) {
        assert_eq!(env::predecessor_account_id(), self.owner, "Only owner can set fees");
        assert!(new_fee_percentage <= 1000, "Fee too high"); // Max 10%
        self.platform_fee_percentage = new_fee_percentage;
    }

    pub fn set_min_stake(&mut self, new_min_stake: Balance) {
        assert_eq!(env::predecessor_account_id(), self.owner, "Only owner can set minimum stake");
        self.min_stake_amount = new_min_stake;
    }

    pub fn set_min_arbitration_fee(&mut self, new_min_fee: Balance) {
        assert_eq!(env::predecessor_account_id(), self.owner, "Only owner can set minimum fee");
        self.min_arbitration_fee = new_min_fee;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use near_sdk::test_utils::{accounts, VMContextBuilder};
    use near_sdk::testing_env;

    fn get_context(predecessor_account_id: AccountId) -> VMContextBuilder {
        let mut builder = VMContextBuilder::new();
        builder
            .current_account_id(accounts(0))
            .signer_account_id(predecessor_account_id.clone())
            .predecessor_account_id(predecessor_account_id);
        builder
    }

    #[test]
    fn test_register_arbitrator() {
        let context = get_context(accounts(1));
        testing_env!(context.build());
        
        let mut contract = LegalArbitration::new(accounts(0));
        
        contract.register_arbitrator(
            "Test Arbitrator".to_string(),
            vec!["Commercial Law".to_string()],
            vec!["US".to_string()],
            "cert_hash".to_string(),
        );
        
        let arbitrator = contract.get_arbitrator(accounts(1)).unwrap();
        assert_eq!(arbitrator.name, "Test Arbitrator");
        assert_eq!(arbitrator.reputation_score, 50);
    }

    #[test]
    fn test_create_dispute() {
        let mut context = get_context(accounts(1));
        testing_env!(context.attached_deposit(near_sdk::utils::parse_near!("1")).build());
        
        let mut contract = LegalArbitration::new(accounts(0));
        
        let dispute_id = contract.create_dispute(
            accounts(2),
            DisputeType::ContractBreach,
            "Test dispute".to_string(),
            near_sdk::utils::parse_near!("100"),
        );
        
        let dispute = contract.get_dispute(dispute_id).unwrap();
        assert_eq!(dispute.claimant, accounts(1));
        assert_eq!(dispute.respondent, accounts(2));
        assert!(matches!(dispute.status, DisputeStatus::Created));
    }
}