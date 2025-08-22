#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    dispatch::{DispatchResult, DispatchError},
    traits::{Get, Currency, ReservableCurrency, ExistenceRequirement},
    codec::{Encode, Decode},
    StorageMap, StorageDoubleMap,
};
use frame_system::ensure_signed;
use sp_std::{vec::Vec, collections::btree_map::BTreeMap};
use sp_runtime::traits::{Zero, Saturating, AccountIdConversion};
use sp_runtime::{ModuleId, Permill};

/// Legal DAO pallet for decentralized legal governance and decision making
pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    
    /// The currency used for voting deposits and treasury
    type Currency: Currency<Self::AccountId> + ReservableCurrency<Self::AccountId>;
    
    /// Module ID for the DAO treasury
    type ModuleId: Get<ModuleId>;
    
    /// Minimum proposal deposit required
    type MinimumDeposit: Get<<Self::Currency as Currency<Self::AccountId>>::Balance>;
    
    /// Voting period for proposals
    type VotingPeriod: Get<Self::BlockNumber>;
    
    /// Minimum voting threshold for proposal approval
    type ApprovalThreshold: Get<Permill>;
    
    /// Maximum number of members in the DAO
    type MaxMembers: Get<u32>;
}

type BalanceOf<T> = <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub enum ProposalType {
    LegalFramework,
    ArbitrationRule,
    ComplianceStandard,
    TreasurySpend,
    MembershipChange,
    GovernanceUpdate,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub enum ProposalStatus {
    Active,
    Approved,
    Rejected,
    Executed,
    Expired,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub enum VoteType {
    Aye,
    Nay,
    Abstain,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub enum MemberRole {
    LegalExpert,
    TechnicalAdvisor,
    CommunityRepresentative,
    Arbitrator,
    Validator,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub struct Proposal<AccountId, BlockNumber, Balance> {
    pub id: u32,
    pub proposer: AccountId,
    pub proposal_type: ProposalType,
    pub title: Vec<u8>,
    pub description: Vec<u8>,
    pub call_data: Vec<u8>,
    pub deposit: Balance,
    pub status: ProposalStatus,
    pub created_at: BlockNumber,
    pub voting_end: BlockNumber,
    pub aye_votes: u32,
    pub nay_votes: u32,
    pub abstain_votes: u32,
    pub total_stake_aye: Balance,
    pub total_stake_nay: Balance,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub struct Member<AccountId, Balance> {
    pub account_id: AccountId,
    pub role: MemberRole,
    pub stake: Balance,
    pub reputation: u32,
    pub voting_power: u32,
    pub joined_at: u32,
    pub proposals_created: u32,
    pub votes_cast: u32,
    pub is_active: bool,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub struct Vote<AccountId, Balance> {
    pub voter: AccountId,
    pub vote_type: VoteType,
    pub stake: Balance,
    pub conviction: u8, // 1-6 for conviction voting
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub struct LegalFramework {
    pub id: u32,
    pub title: Vec<u8>,
    pub content: Vec<u8>,
    pub jurisdiction: Vec<u8>,
    pub version: u32,
    pub created_at: u32,
    pub last_updated: u32,
    pub is_active: bool,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, Debug)]
pub struct ArbitrationRule {
    pub id: u32,
    pub title: Vec<u8>,
    pub rule_text: Vec<u8>,
    pub applicable_types: Vec<ProposalType>,
    pub penalty: BalanceOf<T>,
    pub is_mandatory: bool,
    pub created_at: u32,
}

decl_storage! {
    trait Store for Module<T: Config> as LegalDAO {
        /// The next available proposal ID
        NextProposalId get(fn next_proposal_id): u32 = 1;
        
        /// All proposals in the DAO
        Proposals get(fn proposals): 
            map hasher(blake2_128_concat) u32 => Option<Proposal<T::AccountId, T::BlockNumber, BalanceOf<T>>>;
        
        /// Votes for each proposal
        ProposalVotes get(fn proposal_votes):
            double_map hasher(blake2_128_concat) u32, hasher(blake2_128_concat) T::AccountId 
            => Option<Vote<T::AccountId, BalanceOf<T>>>;
        
        /// DAO members and their details
        Members get(fn members):
            map hasher(blake2_128_concat) T::AccountId => Option<Member<T::AccountId, BalanceOf<T>>>;
        
        /// Member count by role
        MemberCountByRole get(fn member_count_by_role):
            map hasher(blake2_128_concat) MemberRole => u32;
        
        /// Total member count
        TotalMembers get(fn total_members): u32 = 0;
        
        /// Legal frameworks created by the DAO
        LegalFrameworks get(fn legal_frameworks):
            map hasher(blake2_128_concat) u32 => Option<LegalFramework>;
        
        /// Next framework ID
        NextFrameworkId get(fn next_framework_id): u32 = 1;
        
        /// Arbitration rules
        ArbitrationRules get(fn arbitration_rules):
            map hasher(blake2_128_concat) u32 => Option<ArbitrationRule<T>>;
        
        /// Next rule ID
        NextRuleId get(fn next_rule_id): u32 = 1;
        
        /// Treasury balance
        TreasuryBalance get(fn treasury_balance): BalanceOf<T> = Zero::zero();
        
        /// Proposal count by type
        ProposalCountByType get(fn proposal_count_by_type):
            map hasher(blake2_128_concat) ProposalType => u32;
        
        /// Member reputation scores
        ReputationScores get(fn reputation_scores):
            map hasher(blake2_128_concat) T::AccountId => u32;
        
        /// Quorum threshold (minimum participation)
        QuorumThreshold get(fn quorum_threshold): Permill = Permill::from_percent(20);
        
        /// Emergency pause state
        IsPaused get(fn is_paused): bool = false;
        
        /// DAO admin (can pause/unpause)
        DaoAdmin get(fn dao_admin): Option<T::AccountId>;
    }
}

decl_event!(
    pub enum Event<T> where 
        AccountId = <T as frame_system::Config>::AccountId,
        Balance = BalanceOf<T>,
        BlockNumber = <T as frame_system::Config>::BlockNumber,
    {
        /// A new proposal has been created [proposal_id, proposer]
        ProposalCreated(u32, AccountId),
        
        /// A vote has been cast [proposal_id, voter, vote_type]
        VoteCast(u32, AccountId, VoteType),
        
        /// A proposal has been executed [proposal_id]
        ProposalExecuted(u32),
        
        /// A proposal has been approved [proposal_id]
        ProposalApproved(u32),
        
        /// A proposal has been rejected [proposal_id]
        ProposalRejected(u32),
        
        /// A new member has joined [member, role]
        MemberJoined(AccountId, MemberRole),
        
        /// A member has left or been removed [member]
        MemberRemoved(AccountId),
        
        /// A legal framework has been created [framework_id, title]
        LegalFrameworkCreated(u32, Vec<u8>),
        
        /// An arbitration rule has been created [rule_id, title]
        ArbitrationRuleCreated(u32, Vec<u8>),
        
        /// Treasury funds have been transferred [recipient, amount]
        TreasuryTransfer(AccountId, Balance),
        
        /// Member reputation updated [member, new_reputation]
        ReputationUpdated(AccountId, u32),
        
        /// DAO has been paused
        DaoPaused,
        
        /// DAO has been unpaused
        DaoUnpaused,
    }
);

decl_error! {
    pub enum Error for Module<T: Config> {
        /// Proposal not found
        ProposalNotFound,
        /// Not a DAO member
        NotMember,
        /// Already voted on this proposal
        AlreadyVoted,
        /// Voting period has ended
        VotingEnded,
        /// Voting period is still active
        VotingStillActive,
        /// Insufficient deposit
        InsufficientDeposit,
        /// Proposal already executed
        ProposalAlreadyExecuted,
        /// Maximum members reached
        MaxMembersReached,
        /// Member already exists
        MemberAlreadyExists,
        /// Insufficient voting power
        InsufficientVotingPower,
        /// Invalid proposal type
        InvalidProposalType,
        /// Insufficient treasury balance
        InsufficientTreasuryBalance,
        /// DAO is paused
        DaoPaused,
        /// Not authorized (admin only)
        NotAuthorized,
        /// Invalid conviction level
        InvalidConviction,
        /// Quorum not reached
        QuorumNotReached,
    }
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        const MinimumDeposit: BalanceOf<T> = T::MinimumDeposit::get();
        const VotingPeriod: T::BlockNumber = T::VotingPeriod::get();
        const ApprovalThreshold: Permill = T::ApprovalThreshold::get();
        const MaxMembers: u32 = T::MaxMembers::get();

        /// Join the DAO as a member
        #[weight = 10_000]
        pub fn join_dao(
            origin,
            role: MemberRole,
            stake: BalanceOf<T>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            ensure!(!Self::is_paused(), Error::<T>::DaoPaused);
            ensure!(!Members::<T>::contains_key(&who), Error::<T>::MemberAlreadyExists);
            ensure!(Self::total_members() < T::MaxMembers::get(), Error::<T>::MaxMembersReached);
            ensure!(stake >= T::MinimumDeposit::get(), Error::<T>::InsufficientDeposit);
            
            // Reserve the stake
            T::Currency::reserve(&who, stake)?;
            
            let voting_power = Self::calculate_voting_power(&role, stake);
            
            let member = Member {
                account_id: who.clone(),
                role: role.clone(),
                stake,
                reputation: 50, // Starting reputation
                voting_power,
                joined_at: frame_system::Module::<T>::block_number().saturated_into::<u32>(),
                proposals_created: 0,
                votes_cast: 0,
                is_active: true,
            };
            
            Members::<T>::insert(&who, &member);
            TotalMembers::mutate(|count| *count += 1);
            MemberCountByRole::mutate(&role, |count| *count += 1);
            ReputationScores::<T>::insert(&who, 50);
            
            Self::deposit_event(RawEvent::MemberJoined(who, role));
            Ok(())
        }

        /// Create a new proposal
        #[weight = 10_000]
        pub fn create_proposal(
            origin,
            proposal_type: ProposalType,
            title: Vec<u8>,
            description: Vec<u8>,
            call_data: Vec<u8>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            ensure!(!Self::is_paused(), Error::<T>::DaoPaused);
            ensure!(Members::<T>::contains_key(&who), Error::<T>::NotMember);
            
            let deposit = T::MinimumDeposit::get();
            T::Currency::reserve(&who, deposit)?;
            
            let proposal_id = Self::next_proposal_id();
            let current_block = frame_system::Module::<T>::block_number();
            let voting_end = current_block + T::VotingPeriod::get();
            
            let proposal = Proposal {
                id: proposal_id,
                proposer: who.clone(),
                proposal_type: proposal_type.clone(),
                title: title.clone(),
                description,
                call_data,
                deposit,
                status: ProposalStatus::Active,
                created_at: current_block,
                voting_end,
                aye_votes: 0,
                nay_votes: 0,
                abstain_votes: 0,
                total_stake_aye: Zero::zero(),
                total_stake_nay: Zero::zero(),
            };
            
            Proposals::<T>::insert(proposal_id, &proposal);
            NextProposalId::mutate(|id| *id += 1);
            ProposalCountByType::mutate(&proposal_type, |count| *count += 1);
            
            // Update proposer stats
            Members::<T>::mutate(&who, |member_opt| {
                if let Some(member) = member_opt {
                    member.proposals_created += 1;
                }
            });
            
            Self::deposit_event(RawEvent::ProposalCreated(proposal_id, who));
            Ok(())
        }

        /// Vote on a proposal
        #[weight = 10_000]
        pub fn vote(
            origin,
            proposal_id: u32,
            vote_type: VoteType,
            conviction: u8,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            ensure!(!Self::is_paused(), Error::<T>::DaoPaused);
            ensure!(Members::<T>::contains_key(&who), Error::<T>::NotMember);
            ensure!(conviction >= 1 && conviction <= 6, Error::<T>::InvalidConviction);
            
            let mut proposal = Self::proposals(proposal_id).ok_or(Error::<T>::ProposalNotFound)?;
            let current_block = frame_system::Module::<T>::block_number();
            
            ensure!(current_block <= proposal.voting_end, Error::<T>::VotingEnded);
            ensure!(!ProposalVotes::<T>::contains_key(proposal_id, &who), Error::<T>::AlreadyVoted);
            
            let member = Self::members(&who).ok_or(Error::<T>::NotMember)?;
            let vote_stake = member.stake;
            let conviction_multiplier = Self::conviction_multiplier(conviction);
            let weighted_stake = vote_stake.saturating_mul(conviction_multiplier.into());
            
            let vote = Vote {
                voter: who.clone(),
                vote_type: vote_type.clone(),
                stake: vote_stake,
                conviction,
            };
            
            // Update proposal vote counts
            match vote_type {
                VoteType::Aye => {
                    proposal.aye_votes += 1;
                    proposal.total_stake_aye = proposal.total_stake_aye.saturating_add(weighted_stake);
                },
                VoteType::Nay => {
                    proposal.nay_votes += 1;
                    proposal.total_stake_nay = proposal.total_stake_nay.saturating_add(weighted_stake);
                },
                VoteType::Abstain => {
                    proposal.abstain_votes += 1;
                },
            }
            
            ProposalVotes::<T>::insert(proposal_id, &who, &vote);
            Proposals::<T>::insert(proposal_id, &proposal);
            
            // Update member stats
            Members::<T>::mutate(&who, |member_opt| {
                if let Some(member) = member_opt {
                    member.votes_cast += 1;
                }
            });
            
            Self::deposit_event(RawEvent::VoteCast(proposal_id, who, vote_type));
            Ok(())
        }

        /// Execute a proposal that has been approved
        #[weight = 50_000]
        pub fn execute_proposal(
            origin,
            proposal_id: u32,
        ) -> DispatchResult {
            let _who = ensure_signed(origin)?;
            
            ensure!(!Self::is_paused(), Error::<T>::DaoPaused);
            
            let mut proposal = Self::proposals(proposal_id).ok_or(Error::<T>::ProposalNotFound)?;
            let current_block = frame_system::Module::<T>::block_number();
            
            ensure!(current_block > proposal.voting_end, Error::<T>::VotingStillActive);
            ensure!(proposal.status == ProposalStatus::Active, Error::<T>::ProposalAlreadyExecuted);
            
            // Check if proposal passed
            let total_votes = proposal.aye_votes + proposal.nay_votes + proposal.abstain_votes;
            let total_members = Self::total_members();
            let quorum_threshold = Self::quorum_threshold().mul_floor(total_members);
            
            ensure!(total_votes >= quorum_threshold, Error::<T>::QuorumNotReached);
            
            let approval_threshold = T::ApprovalThreshold::get();
            let total_stake_votes = proposal.total_stake_aye + proposal.total_stake_nay;
            let approval_ratio = if total_stake_votes.is_zero() {
                Permill::zero()
            } else {
                Permill::from_rational(proposal.total_stake_aye, total_stake_votes)
            };
            
            if approval_ratio >= approval_threshold {
                proposal.status = ProposalStatus::Approved;
                
                // Execute based on proposal type
                match proposal.proposal_type {
                    ProposalType::LegalFramework => {
                        Self::execute_legal_framework(&proposal)?;
                    },
                    ProposalType::ArbitrationRule => {
                        Self::execute_arbitration_rule(&proposal)?;
                    },
                    ProposalType::TreasurySpend => {
                        Self::execute_treasury_spend(&proposal)?;
                    },
                    _ => {
                        // For other types, just mark as executed
                    }
                }
                
                proposal.status = ProposalStatus::Executed;
                Self::deposit_event(RawEvent::ProposalApproved(proposal_id));
                Self::deposit_event(RawEvent::ProposalExecuted(proposal_id));
                
                // Return deposit to proposer
                T::Currency::unreserve(&proposal.proposer, proposal.deposit);
                
                // Update proposer reputation
                Self::update_reputation(&proposal.proposer, true);
                
            } else {
                proposal.status = ProposalStatus::Rejected;
                Self::deposit_event(RawEvent::ProposalRejected(proposal_id));
                
                // Slash deposit for failed proposals
                T::Currency::slash_reserved(&proposal.proposer, proposal.deposit);
                
                // Update proposer reputation
                Self::update_reputation(&proposal.proposer, false);
            }
            
            Proposals::<T>::insert(proposal_id, &proposal);
            Ok(())
        }

        /// Create a legal framework
        #[weight = 10_000]
        pub fn create_legal_framework(
            origin,
            title: Vec<u8>,
            content: Vec<u8>,
            jurisdiction: Vec<u8>,
        ) -> DispatchResult {
            let _who = ensure_signed(origin)?;
            
            let framework_id = Self::next_framework_id();
            let framework = LegalFramework {
                id: framework_id,
                title: title.clone(),
                content,
                jurisdiction,
                version: 1,
                created_at: frame_system::Module::<T>::block_number().saturated_into::<u32>(),
                last_updated: frame_system::Module::<T>::block_number().saturated_into::<u32>(),
                is_active: true,
            };
            
            LegalFrameworks::insert(framework_id, &framework);
            NextFrameworkId::mutate(|id| *id += 1);
            
            Self::deposit_event(RawEvent::LegalFrameworkCreated(framework_id, title));
            Ok(())
        }

        /// Admin function to pause the DAO
        #[weight = 10_000]
        pub fn pause_dao(origin) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            if let Some(admin) = Self::dao_admin() {
                ensure!(who == admin, Error::<T>::NotAuthorized);
            } else {
                // If no admin set, only sudo can pause
                ensure_root(origin)?;
            }
            
            IsPaused::put(true);
            Self::deposit_event(RawEvent::DaoPaused);
            Ok(())
        }

        /// Admin function to unpause the DAO
        #[weight = 10_000]
        pub fn unpause_dao(origin) -> DispatchResult {
            let who = ensure_signed(origin)?;
            
            if let Some(admin) = Self::dao_admin() {
                ensure!(who == admin, Error::<T>::NotAuthorized);
            } else {
                ensure_root(origin)?;
            }
            
            IsPaused::put(false);
            Self::deposit_event(RawEvent::DaoUnpaused);
            Ok(())
        }
    }
}

impl<T: Config> Module<T> {
    /// Calculate voting power based on role and stake
    fn calculate_voting_power(role: &MemberRole, stake: BalanceOf<T>) -> u32 {
        let base_power = match role {
            MemberRole::LegalExpert => 100,
            MemberRole::TechnicalAdvisor => 80,
            MemberRole::CommunityRepresentative => 60,
            MemberRole::Arbitrator => 90,
            MemberRole::Validator => 70,
        };
        
        // Add stake-based bonus (simplified calculation)
        let stake_bonus = if stake > T::MinimumDeposit::get() {
            20
        } else {
            0
        };
        
        base_power + stake_bonus
    }
    
    /// Get conviction multiplier
    fn conviction_multiplier(conviction: u8) -> u32 {
        match conviction {
            1 => 1,
            2 => 2,
            3 => 4,
            4 => 8,
            5 => 16,
            6 => 32,
            _ => 1,
        }
    }
    
    /// Update member reputation
    fn update_reputation(member: &T::AccountId, positive: bool) {
        ReputationScores::<T>::mutate(member, |score| {
            if positive {
                *score = (*score).saturating_add(5).min(100);
            } else {
                *score = (*score).saturating_sub(3).max(0);
            }
        });
        
        let new_score = Self::reputation_scores(member);
        Self::deposit_event(RawEvent::ReputationUpdated(member.clone(), new_score));
    }
    
    /// Execute legal framework proposal
    fn execute_legal_framework(proposal: &Proposal<T::AccountId, T::BlockNumber, BalanceOf<T>>) -> DispatchResult {
        // Parse call_data to extract framework details
        // This is a simplified implementation
        Self::create_legal_framework(
            frame_system::RawOrigin::Signed(proposal.proposer.clone()).into(),
            proposal.title.clone(),
            proposal.description.clone(),
            b"Global".to_vec(),
        )
    }
    
    /// Execute arbitration rule proposal
    fn execute_arbitration_rule(proposal: &Proposal<T::AccountId, T::BlockNumber, BalanceOf<T>>) -> DispatchResult {
        // Implementation for creating arbitration rules
        Ok(())
    }
    
    /// Execute treasury spend proposal
    fn execute_treasury_spend(proposal: &Proposal<T::AccountId, T::BlockNumber, BalanceOf<T>>) -> DispatchResult {
        // Implementation for treasury spending
        Ok(())
    }
    
    /// Get treasury account ID
    fn treasury_account_id() -> T::AccountId {
        T::ModuleId::get().into_account()
    }
}

/// Runtime tests
#[cfg(test)]
mod tests {
    use super::*;
    use frame_support::{
        assert_ok, assert_noop, impl_outer_origin, parameter_types,
        traits::{OnFinalize, OnInitialize},
        weights::Weight,
    };
    use sp_core::H256;
    use sp_runtime::{
        traits::{BlakeTwo256, IdentityLookup}, testing::Header, Perbill,
    };
    use frame_system as system;

    // Test implementation would go here
}