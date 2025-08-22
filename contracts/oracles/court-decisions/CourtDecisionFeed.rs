use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer, Mint};
use std::collections::HashMap;

declare_id!("CourtDecision1111111111111111111111111111111");

#[program]
pub mod court_decision_feed {
    use super::*;

    /// Initialize the court decision feed
    pub fn initialize(
        ctx: Context<Initialize>,
        feed_authority: Pubkey,
        subscription_price: u64,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        config.authority = ctx.accounts.authority.key();
        config.feed_authority = feed_authority;
        config.subscription_price = subscription_price;
        config.total_decisions = 0;
        config.total_subscribers = 0;
        config.is_paused = false;
        Ok(())
    }

    /// Submit a court decision
    pub fn submit_decision(
        ctx: Context<SubmitDecision>,
        case_number: String,
        court_name: String,
        jurisdiction: String,
        decision_date: i64,
        case_type: CaseType,
        decision_summary: String,
        legal_precedent: bool,
        decision_hash: [u8; 32],
        metadata_uri: String,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        let decision = &mut ctx.accounts.decision;
        
        require!(!config.is_paused, ErrorCode::FeedPaused);
        require!(case_number.len() <= 50, ErrorCode::CaseNumberTooLong);
        require!(court_name.len() <= 100, ErrorCode::CourtNameTooLong);
        require!(decision_summary.len() <= 1000, ErrorCode::SummaryTooLong);
        
        decision.id = config.total_decisions;
        decision.submitter = ctx.accounts.submitter.key();
        decision.case_number = case_number;
        decision.court_name = court_name;
        decision.jurisdiction = jurisdiction;
        decision.decision_date = decision_date;
        decision.submission_time = Clock::get()?.unix_timestamp;
        decision.case_type = case_type;
        decision.decision_summary = decision_summary;
        decision.legal_precedent = legal_precedent;
        decision.decision_hash = decision_hash;
        decision.metadata_uri = metadata_uri;
        decision.status = DecisionStatus::Pending;
        decision.verification_count = 0;
        decision.citation_count = 0;
        decision.impact_score = 0;
        
        config.total_decisions += 1;
        
        emit!(DecisionSubmitted {
            decision_id: decision.id,
            submitter: decision.submitter,
            case_number: decision.case_number.clone(),
            court_name: decision.court_name.clone(),
        });
        
        Ok(())
    }

    /// Verify a court decision
    pub fn verify_decision(
        ctx: Context<VerifyDecision>,
        decision_id: u64,
        verified: bool,
        verification_notes: String,
    ) -> Result<()> {
        let decision = &mut ctx.accounts.decision;
        let verification = &mut ctx.accounts.verification;
        
        require!(decision.id == decision_id, ErrorCode::InvalidDecisionId);
        require!(decision.status == DecisionStatus::Pending, ErrorCode::InvalidDecisionStatus);
        require!(verification_notes.len() <= 500, ErrorCode::VerificationNotesTooLong);
        
        verification.decision_id = decision_id;
        verification.verifier = ctx.accounts.verifier.key();
        verification.verified = verified;
        verification.verification_time = Clock::get()?.unix_timestamp;
        verification.notes = verification_notes;
        
        decision.verification_count += 1;
        
        // Auto-approve if enough verifications
        if decision.verification_count >= 3 {
            decision.status = DecisionStatus::Verified;
            decision.impact_score = calculate_impact_score(decision);
        }
        
        emit!(DecisionVerified {
            decision_id,
            verifier: verification.verifier,
            verified,
        });
        
        Ok(())
    }

    /// Subscribe to court decision feed
    pub fn subscribe_to_feed(
        ctx: Context<SubscribeToFeed>,
        jurisdiction_filter: Vec<String>,
        case_type_filter: Vec<CaseType>,
        subscription_duration: i64,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        let subscription = &mut ctx.accounts.subscription;
        
        require!(!config.is_paused, ErrorCode::FeedPaused);
        require!(subscription_duration >= 30 * 24 * 60 * 60, ErrorCode::SubscriptionTooShort); // Min 30 days
        require!(jurisdiction_filter.len() <= 10, ErrorCode::TooManyJurisdictions);
        require!(case_type_filter.len() <= 20, ErrorCode::TooManyCaseTypes);
        
        // Transfer subscription payment
        let cpi_accounts = Transfer {
            from: ctx.accounts.subscriber_token_account.to_account_info(),
            to: ctx.accounts.treasury_account.to_account_info(),
            authority: ctx.accounts.subscriber.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        
        let subscription_cost = (config.subscription_price * subscription_duration as u64) / (30 * 24 * 60 * 60); // Pro-rated
        token::transfer(cpi_ctx, subscription_cost)?;
        
        subscription.subscriber = ctx.accounts.subscriber.key();
        subscription.start_time = Clock::get()?.unix_timestamp;
        subscription.end_time = Clock::get()?.unix_timestamp + subscription_duration;
        subscription.jurisdiction_filter = jurisdiction_filter;
        subscription.case_type_filter = case_type_filter;
        subscription.active = true;
        subscription.notifications_received = 0;
        subscription.last_accessed = Clock::get()?.unix_timestamp;
        
        config.total_subscribers += 1;
        
        emit!(SubscriptionCreated {
            subscriber: subscription.subscriber,
            start_time: subscription.start_time,
            end_time: subscription.end_time,
        });
        
        Ok(())
    }

    /// Query court decisions with filters
    pub fn query_decisions(
        ctx: Context<QueryDecisions>,
        jurisdiction: Option<String>,
        case_type: Option<CaseType>,
        date_from: Option<i64>,
        date_to: Option<i64>,
        legal_precedent_only: bool,
        limit: u8,
        offset: u64,
    ) -> Result<Vec<PublicDecisionInfo>> {
        let subscription = &ctx.accounts.subscription;
        
        require!(subscription.active, ErrorCode::InactiveSubscription);
        require!(Clock::get()?.unix_timestamp <= subscription.end_time, ErrorCode::SubscriptionExpired);
        require!(limit <= 100, ErrorCode::LimitTooHigh);
        
        // This would typically involve iterating through stored decisions
        // For demonstration, we'll return an empty vector
        // In a real implementation, you'd filter and paginate results
        
        emit!(DecisionQueried {
            subscriber: subscription.subscriber,
            query_time: Clock::get()?.unix_timestamp,
        });
        
        Ok(vec![])
    }

    /// Create citation link between decisions
    pub fn create_citation(
        ctx: Context<CreateCitation>,
        citing_decision_id: u64,
        cited_decision_id: u64,
        citation_type: CitationType,
        relevance_score: u8,
    ) -> Result<()> {
        let citation = &mut ctx.accounts.citation;
        let citing_decision = &mut ctx.accounts.citing_decision;
        let cited_decision = &mut ctx.accounts.cited_decision;
        
        require!(citing_decision.id == citing_decision_id, ErrorCode::InvalidDecisionId);
        require!(cited_decision.id == cited_decision_id, ErrorCode::InvalidDecisionId);
        require!(citing_decision_id != cited_decision_id, ErrorCode::SelfCitation);
        require!(relevance_score <= 100, ErrorCode::InvalidRelevanceScore);
        
        citation.citing_decision_id = citing_decision_id;
        citation.cited_decision_id = cited_decision_id;
        citation.citation_type = citation_type;
        citation.relevance_score = relevance_score;
        citation.created_at = Clock::get()?.unix_timestamp;
        citation.created_by = ctx.accounts.authority.key();
        
        // Update citation counts
        cited_decision.citation_count += 1;
        cited_decision.impact_score = calculate_impact_score(cited_decision);
        
        emit!(CitationCreated {
            citing_decision_id,
            cited_decision_id,
            citation_type,
        });
        
        Ok(())
    }

    /// Add legal analysis to a decision
    pub fn add_legal_analysis(
        ctx: Context<AddLegalAnalysis>,
        decision_id: u64,
        analysis_type: AnalysisType,
        analysis_content: String,
        key_holdings: Vec<String>,
        legal_principles: Vec<String>,
        impact_assessment: String,
    ) -> Result<()> {
        let analysis = &mut ctx.accounts.analysis;
        let decision = &ctx.accounts.decision;
        
        require!(decision.id == decision_id, ErrorCode::InvalidDecisionId);
        require!(decision.status == DecisionStatus::Verified, ErrorCode::DecisionNotVerified);
        require!(analysis_content.len() <= 5000, ErrorCode::AnalysisContentTooLong);
        require!(key_holdings.len() <= 10, ErrorCode::TooManyKeyHoldings);
        require!(legal_principles.len() <= 10, ErrorCode::TooManyLegalPrinciples);
        
        analysis.decision_id = decision_id;
        analysis.analyst = ctx.accounts.analyst.key();
        analysis.analysis_type = analysis_type;
        analysis.analysis_content = analysis_content;
        analysis.key_holdings = key_holdings;
        analysis.legal_principles = legal_principles;
        analysis.impact_assessment = impact_assessment;
        analysis.created_at = Clock::get()?.unix_timestamp;
        analysis.peer_reviews = 0;
        analysis.quality_score = 0;
        
        emit!(AnalysisAdded {
            decision_id,
            analyst: analysis.analyst,
            analysis_type,
        });
        
        Ok(())
    }

    /// Update decision impact score
    pub fn update_impact_score(
        ctx: Context<UpdateImpactScore>,
        decision_id: u64,
    ) -> Result<()> {
        let decision = &mut ctx.accounts.decision;
        
        require!(decision.id == decision_id, ErrorCode::InvalidDecisionId);
        
        decision.impact_score = calculate_impact_score(decision);
        
        emit!(ImpactScoreUpdated {
            decision_id,
            new_score: decision.impact_score,
        });
        
        Ok(())
    }

    /// Pause the feed (admin only)
    pub fn pause_feed(ctx: Context<PauseFeed>) -> Result<()> {
        let config = &mut ctx.accounts.config;
        require!(config.authority == ctx.accounts.authority.key(), ErrorCode::UnauthorizedAdmin);
        
        config.is_paused = true;
        
        emit!(FeedPaused {});
        
        Ok(())
    }

    /// Resume the feed (admin only)
    pub fn resume_feed(ctx: Context<ResumeFeed>) -> Result<()> {
        let config = &mut ctx.accounts.config;
        require!(config.authority == ctx.accounts.authority.key(), ErrorCode::UnauthorizedAdmin);
        
        config.is_paused = false;
        
        emit!(FeedResumed {});
        
        Ok(())
    }
}

// Helper function to calculate impact score
fn calculate_impact_score(decision: &CourtDecision) -> u32 {
    let mut score = 0u32;
    
    // Base score from legal precedent
    if decision.legal_precedent {
        score += 50;
    }
    
    // Score from citations (max 30 points)
    score += std::cmp::min(30, decision.citation_count * 2);
    
    // Score from verification (max 20 points)
    score += std::cmp::min(20, decision.verification_count * 5);
    
    // Adjust based on case type importance
    score += match decision.case_type {
        CaseType::Supreme => 20,
        CaseType::Appellate => 15,
        CaseType::Federal => 10,
        CaseType::State => 5,
        CaseType::Local => 2,
        _ => 0,
    };
    
    std::cmp::min(100, score)
}

// Account structures
#[account]
pub struct FeedConfig {
    pub authority: Pubkey,
    pub feed_authority: Pubkey,
    pub subscription_price: u64,
    pub total_decisions: u64,
    pub total_subscribers: u64,
    pub is_paused: bool,
}

#[account]
pub struct CourtDecision {
    pub id: u64,
    pub submitter: Pubkey,
    pub case_number: String,
    pub court_name: String,
    pub jurisdiction: String,
    pub decision_date: i64,
    pub submission_time: i64,
    pub case_type: CaseType,
    pub decision_summary: String,
    pub legal_precedent: bool,
    pub decision_hash: [u8; 32],
    pub metadata_uri: String,
    pub status: DecisionStatus,
    pub verification_count: u32,
    pub citation_count: u32,
    pub impact_score: u32,
}

#[account]
pub struct DecisionVerification {
    pub decision_id: u64,
    pub verifier: Pubkey,
    pub verified: bool,
    pub verification_time: i64,
    pub notes: String,
}

#[account]
pub struct Subscription {
    pub subscriber: Pubkey,
    pub start_time: i64,
    pub end_time: i64,
    pub jurisdiction_filter: Vec<String>,
    pub case_type_filter: Vec<CaseType>,
    pub active: bool,
    pub notifications_received: u64,
    pub last_accessed: i64,
}

#[account]
pub struct Citation {
    pub citing_decision_id: u64,
    pub cited_decision_id: u64,
    pub citation_type: CitationType,
    pub relevance_score: u8,
    pub created_at: i64,
    pub created_by: Pubkey,
}

#[account]
pub struct LegalAnalysis {
    pub decision_id: u64,
    pub analyst: Pubkey,
    pub analysis_type: AnalysisType,
    pub analysis_content: String,
    pub key_holdings: Vec<String>,
    pub legal_principles: Vec<String>,
    pub impact_assessment: String,
    pub created_at: i64,
    pub peer_reviews: u32,
    pub quality_score: u32,
}

// Enums
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum CaseType {
    Supreme,
    Appellate,
    Federal,
    State,
    Local,
    Administrative,
    Arbitration,
    Mediation,
    International,
    Commercial,
    Criminal,
    Civil,
    Constitutional,
    Tax,
    IP, // Intellectual Property
    Employment,
    Environmental,
    Healthcare,
    Securities,
    Immigration,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum DecisionStatus {
    Pending,
    Verified,
    Disputed,
    Archived,
    Superseded,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum CitationType {
    Relied,
    Distinguished,
    Overruled,
    Followed,
    Considered,
    Cited,
    Questioned,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum AnalysisType {
    Summary,
    DeepDive,
    Comparative,
    Historical,
    PredictiveImpact,
    LegalComment,
    CaseNote,
    Annotation,
}

// Response structures
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct PublicDecisionInfo {
    pub id: u64,
    pub case_number: String,
    pub court_name: String,
    pub jurisdiction: String,
    pub decision_date: i64,
    pub case_type: CaseType,
    pub decision_summary: String,
    pub legal_precedent: bool,
    pub impact_score: u32,
    pub citation_count: u32,
}

// Context structures
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 32 + 8 + 8 + 8 + 1,
        seeds = [b"feed_config"],
        bump
    )]
    pub config: Account<'info, FeedConfig>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(case_number: String, court_name: String, decision_summary: String)]
pub struct SubmitDecision<'info> {
    #[account(mut)]
    pub config: Account<'info, FeedConfig>,
    #[account(
        init,
        payer = submitter,
        space = 8 + 8 + 32 + 4 + case_number.len() + 4 + court_name.len() + 4 + 50 + 8 + 8 + 1 + 4 + decision_summary.len() + 1 + 32 + 4 + 100 + 1 + 4 + 4 + 4,
        seeds = [b"decision", &config.total_decisions.to_le_bytes()],
        bump
    )]
    pub decision: Account<'info, CourtDecision>,
    #[account(mut)]
    pub submitter: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(verification_notes: String)]
pub struct VerifyDecision<'info> {
    #[account(mut)]
    pub decision: Account<'info, CourtDecision>,
    #[account(
        init,
        payer = verifier,
        space = 8 + 8 + 32 + 1 + 8 + 4 + verification_notes.len(),
        seeds = [b"verification", &decision.id.to_le_bytes(), verifier.key().as_ref()],
        bump
    )]
    pub verification: Account<'info, DecisionVerification>,
    #[account(mut)]
    pub verifier: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SubscribeToFeed<'info> {
    #[account(mut)]
    pub config: Account<'info, FeedConfig>,
    #[account(
        init,
        payer = subscriber,
        space = 8 + 32 + 8 + 8 + 4 + 200 + 4 + 100 + 1 + 8 + 8, // Estimated space for filters
        seeds = [b"subscription", subscriber.key().as_ref()],
        bump
    )]
    pub subscription: Account<'info, Subscription>,
    #[account(
        mut,
        constraint = subscriber_token_account.owner == subscriber.key()
    )]
    pub subscriber_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub treasury_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub subscriber: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct QueryDecisions<'info> {
    pub subscription: Account<'info, Subscription>,
}

#[derive(Accounts)]
pub struct CreateCitation<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 8 + 8 + 1 + 1 + 8 + 32,
        seeds = [b"citation", &citing_decision.id.to_le_bytes(), &cited_decision.id.to_le_bytes()],
        bump
    )]
    pub citation: Account<'info, Citation>,
    #[account(mut)]
    pub citing_decision: Account<'info, CourtDecision>,
    #[account(mut)]
    pub cited_decision: Account<'info, CourtDecision>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(analysis_content: String)]
pub struct AddLegalAnalysis<'info> {
    pub decision: Account<'info, CourtDecision>,
    #[account(
        init,
        payer = analyst,
        space = 8 + 8 + 32 + 1 + 4 + analysis_content.len() + 4 + 200 + 4 + 200 + 4 + 500 + 8 + 4 + 4,
        seeds = [b"analysis", &decision.id.to_le_bytes(), analyst.key().as_ref()],
        bump
    )]
    pub analysis: Account<'info, LegalAnalysis>,
    #[account(mut)]
    pub analyst: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateImpactScore<'info> {
    #[account(mut)]
    pub decision: Account<'info, CourtDecision>,
}

#[derive(Accounts)]
pub struct PauseFeed<'info> {
    #[account(mut)]
    pub config: Account<'info, FeedConfig>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResumeFeed<'info> {
    #[account(mut)]
    pub config: Account<'info, FeedConfig>,
    pub authority: Signer<'info>,
}

// Events
#[event]
pub struct DecisionSubmitted {
    pub decision_id: u64,
    pub submitter: Pubkey,
    pub case_number: String,
    pub court_name: String,
}

#[event]
pub struct DecisionVerified {
    pub decision_id: u64,
    pub verifier: Pubkey,
    pub verified: bool,
}

#[event]
pub struct SubscriptionCreated {
    pub subscriber: Pubkey,
    pub start_time: i64,
    pub end_time: i64,
}

#[event]
pub struct DecisionQueried {
    pub subscriber: Pubkey,
    pub query_time: i64,
}

#[event]
pub struct CitationCreated {
    pub citing_decision_id: u64,
    pub cited_decision_id: u64,
    pub citation_type: CitationType,
}

#[event]
pub struct AnalysisAdded {
    pub decision_id: u64,
    pub analyst: Pubkey,
    pub analysis_type: AnalysisType,
}

#[event]
pub struct ImpactScoreUpdated {
    pub decision_id: u64,
    pub new_score: u32,
}

#[event]
pub struct FeedPaused {}

#[event]
pub struct FeedResumed {}

// Error codes
#[error_code]
pub enum ErrorCode {
    #[msg("Feed is currently paused")]
    FeedPaused,
    #[msg("Case number too long")]
    CaseNumberTooLong,
    #[msg("Court name too long")]
    CourtNameTooLong,
    #[msg("Summary too long")]
    SummaryTooLong,
    #[msg("Invalid decision ID")]
    InvalidDecisionId,
    #[msg("Invalid decision status")]
    InvalidDecisionStatus,
    #[msg("Verification notes too long")]
    VerificationNotesTooLong,
    #[msg("Subscription too short")]
    SubscriptionTooShort,
    #[msg("Too many jurisdictions")]
    TooManyJurisdictions,
    #[msg("Too many case types")]
    TooManyCaseTypes,
    #[msg("Inactive subscription")]
    InactiveSubscription,
    #[msg("Subscription expired")]
    SubscriptionExpired,
    #[msg("Limit too high")]
    LimitTooHigh,
    #[msg("Cannot cite yourself")]
    SelfCitation,
    #[msg("Invalid relevance score")]
    InvalidRelevanceScore,
    #[msg("Decision not verified")]
    DecisionNotVerified,
    #[msg("Analysis content too long")]
    AnalysisContentTooLong,
    #[msg("Too many key holdings")]
    TooManyKeyHoldings,
    #[msg("Too many legal principles")]
    TooManyLegalPrinciples,
    #[msg("Unauthorized admin")]
    UnauthorizedAdmin,
}