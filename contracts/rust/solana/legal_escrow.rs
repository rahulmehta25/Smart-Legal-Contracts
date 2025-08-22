use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer, Mint};
use std::collections::BTreeMap;

declare_id!("LegalEscrow1111111111111111111111111111111111");

#[program]
pub mod legal_escrow {
    use super::*;

    /// Initialize the escrow program
    pub fn initialize(ctx: Context<Initialize>, platform_fee: u16) -> Result<()> {
        let escrow_config = &mut ctx.accounts.escrow_config;
        escrow_config.authority = ctx.accounts.authority.key();
        escrow_config.platform_fee_bps = platform_fee;
        escrow_config.total_escrows = 0;
        escrow_config.total_volume = 0;
        escrow_config.is_paused = false;
        Ok(())
    }

    /// Create a new escrow
    pub fn create_escrow(
        ctx: Context<CreateEscrow>,
        amount: u64,
        deadline: i64,
        terms_hash: [u8; 32],
        auto_release_enabled: bool,
        auto_release_delay: i64,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let escrow_config = &mut ctx.accounts.escrow_config;
        
        require!(!escrow_config.is_paused, ErrorCode::ProgramPaused);
        require!(amount > 0, ErrorCode::InvalidAmount);
        require!(deadline > Clock::get()?.unix_timestamp, ErrorCode::InvalidDeadline);
        
        escrow.id = escrow_config.total_escrows;
        escrow.payer = ctx.accounts.payer.key();
        escrow.payee = ctx.accounts.payee.key();
        escrow.arbitrator = ctx.accounts.arbitrator.key();
        escrow.mint = ctx.accounts.mint.key();
        escrow.amount = amount;
        escrow.status = EscrowStatus::Created;
        escrow.created_at = Clock::get()?.unix_timestamp;
        escrow.deadline = deadline;
        escrow.terms_hash = terms_hash;
        escrow.auto_release_enabled = auto_release_enabled;
        escrow.auto_release_delay = auto_release_delay;
        escrow.milestones_count = 0;
        escrow.completed_milestones = 0;
        escrow.dispute_raised = false;
        escrow.released_amount = 0;
        
        escrow_config.total_escrows += 1;
        
        emit!(EscrowCreated {
            escrow_id: escrow.id,
            payer: escrow.payer,
            payee: escrow.payee,
            amount: escrow.amount,
        });
        
        Ok(())
    }

    /// Fund the escrow
    pub fn fund_escrow(ctx: Context<FundEscrow>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        
        require!(escrow.status == EscrowStatus::Created, ErrorCode::InvalidEscrowStatus);
        require!(escrow.payer == ctx.accounts.payer.key(), ErrorCode::UnauthorizedPayer);
        
        // Transfer tokens to escrow vault
        let cpi_accounts = Transfer {
            from: ctx.accounts.payer_token_account.to_account_info(),
            to: ctx.accounts.escrow_vault.to_account_info(),
            authority: ctx.accounts.payer.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        
        token::transfer(cpi_ctx, escrow.amount)?;
        
        escrow.status = EscrowStatus::Funded;
        
        let escrow_config = &mut ctx.accounts.escrow_config;
        escrow_config.total_volume += escrow.amount;
        
        emit!(EscrowFunded {
            escrow_id: escrow.id,
            amount: escrow.amount,
        });
        
        Ok(())
    }

    /// Create a milestone for the escrow
    pub fn create_milestone(
        ctx: Context<CreateMilestone>,
        description: String,
        amount: u64,
        deadline: i64,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let milestone = &mut ctx.accounts.milestone;
        
        require!(
            escrow.payer == ctx.accounts.authority.key() || 
            escrow.payee == ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        require!(escrow.status == EscrowStatus::Funded, ErrorCode::InvalidEscrowStatus);
        require!(amount > 0, ErrorCode::InvalidAmount);
        require!(deadline > Clock::get()?.unix_timestamp, ErrorCode::InvalidDeadline);
        require!(description.len() <= 200, ErrorCode::DescriptionTooLong);
        
        milestone.id = escrow.milestones_count;
        milestone.escrow_id = escrow.id;
        milestone.description = description;
        milestone.amount = amount;
        milestone.status = MilestoneStatus::Pending;
        milestone.deadline = deadline;
        milestone.created_at = Clock::get()?.unix_timestamp;
        milestone.submitted_at = 0;
        milestone.approved_at = 0;
        milestone.deliverable_hash = [0; 32];
        
        escrow.milestones_count += 1;
        
        emit!(MilestoneCreated {
            escrow_id: escrow.id,
            milestone_id: milestone.id,
            amount: milestone.amount,
        });
        
        Ok(())
    }

    /// Submit deliverable for milestone
    pub fn submit_milestone(
        ctx: Context<SubmitMilestone>,
        deliverable_hash: [u8; 32],
    ) -> Result<()> {
        let escrow = &ctx.accounts.escrow;
        let milestone = &mut ctx.accounts.milestone;
        
        require!(escrow.payee == ctx.accounts.payee.key(), ErrorCode::UnauthorizedPayee);
        require!(milestone.status == MilestoneStatus::Pending, ErrorCode::InvalidMilestoneStatus);
        require!(Clock::get()?.unix_timestamp <= milestone.deadline, ErrorCode::MilestoneExpired);
        
        milestone.status = MilestoneStatus::Submitted;
        milestone.deliverable_hash = deliverable_hash;
        milestone.submitted_at = Clock::get()?.unix_timestamp;
        
        emit!(MilestoneSubmitted {
            escrow_id: escrow.id,
            milestone_id: milestone.id,
            deliverable_hash,
        });
        
        Ok(())
    }

    /// Approve milestone and release payment
    pub fn approve_milestone(ctx: Context<ApproveMilestone>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let milestone = &mut ctx.accounts.milestone;
        let escrow_config = &ctx.accounts.escrow_config;
        
        require!(
            escrow.payer == ctx.accounts.authority.key() || 
            escrow.arbitrator == ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        require!(milestone.status == MilestoneStatus::Submitted, ErrorCode::InvalidMilestoneStatus);
        
        milestone.status = MilestoneStatus::Approved;
        milestone.approved_at = Clock::get()?.unix_timestamp;
        escrow.completed_milestones += 1;
        
        // Calculate platform fee
        let platform_fee = (milestone.amount * escrow_config.platform_fee_bps as u64) / 10000;
        let net_amount = milestone.amount - platform_fee;
        
        // Transfer tokens to payee
        let seeds = &[
            b"escrow_vault",
            &escrow.id.to_le_bytes(),
            &[ctx.bumps.escrow_vault],
        ];
        let signer = &[&seeds[..]];
        
        let cpi_accounts = Transfer {
            from: ctx.accounts.escrow_vault.to_account_info(),
            to: ctx.accounts.payee_token_account.to_account_info(),
            authority: ctx.accounts.escrow_vault.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
        
        token::transfer(cpi_ctx, net_amount)?;
        
        // Transfer platform fee if applicable
        if platform_fee > 0 {
            let cpi_accounts_fee = Transfer {
                from: ctx.accounts.escrow_vault.to_account_info(),
                to: ctx.accounts.platform_fee_account.to_account_info(),
                authority: ctx.accounts.escrow_vault.to_account_info(),
            };
            let cpi_ctx_fee = CpiContext::new_with_signer(cpi_program, cpi_accounts_fee, signer);
            token::transfer(cpi_ctx_fee, platform_fee)?;
        }
        
        escrow.released_amount += milestone.amount;
        
        // Check if all milestones completed
        if escrow.completed_milestones == escrow.milestones_count && escrow.milestones_count > 0 {
            escrow.status = EscrowStatus::Completed;
        }
        
        emit!(MilestoneApproved {
            escrow_id: escrow.id,
            milestone_id: milestone.id,
            amount: net_amount,
        });
        
        Ok(())
    }

    /// Raise a dispute
    pub fn raise_dispute(
        ctx: Context<RaiseDispute>,
        reason: String,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let dispute = &mut ctx.accounts.dispute;
        
        require!(
            escrow.payer == ctx.accounts.authority.key() || 
            escrow.payee == ctx.accounts.authority.key(),
            ErrorCode::UnauthorizedAccess
        );
        require!(!escrow.dispute_raised, ErrorCode::DisputeAlreadyRaised);
        require!(reason.len() <= 500, ErrorCode::ReasonTooLong);
        
        dispute.escrow_id = escrow.id;
        dispute.initiator = ctx.accounts.authority.key();
        dispute.reason = reason;
        dispute.status = DisputeStatus::Open;
        dispute.created_at = Clock::get()?.unix_timestamp;
        dispute.resolved_at = 0;
        dispute.resolution = String::new();
        
        escrow.dispute_raised = true;
        escrow.status = EscrowStatus::Disputed;
        
        emit!(DisputeRaised {
            escrow_id: escrow.id,
            initiator: dispute.initiator,
            reason: dispute.reason.clone(),
        });
        
        Ok(())
    }

    /// Resolve dispute (arbitrator only)
    pub fn resolve_dispute(
        ctx: Context<ResolveDispute>,
        resolution: String,
        award_to_payer: u64,
        award_to_payee: u64,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let dispute = &mut ctx.accounts.dispute;
        
        require!(escrow.arbitrator == ctx.accounts.arbitrator.key(), ErrorCode::UnauthorizedArbitrator);
        require!(dispute.status == DisputeStatus::Open, ErrorCode::InvalidDisputeStatus);
        require!(award_to_payer + award_to_payee <= escrow.amount - escrow.released_amount, ErrorCode::ExcessiveAward);
        require!(resolution.len() <= 1000, ErrorCode::ResolutionTooLong);
        
        dispute.status = DisputeStatus::Resolved;
        dispute.resolution = resolution;
        dispute.resolved_at = Clock::get()?.unix_timestamp;
        
        let escrow_config = &ctx.accounts.escrow_config;
        let seeds = &[
            b"escrow_vault",
            &escrow.id.to_le_bytes(),
            &[ctx.bumps.escrow_vault],
        ];
        let signer = &[&seeds[..]];
        
        // Award to payer if applicable
        if award_to_payer > 0 {
            let platform_fee = (award_to_payer * escrow_config.platform_fee_bps as u64) / 10000;
            let net_amount = award_to_payer - platform_fee;
            
            let cpi_accounts = Transfer {
                from: ctx.accounts.escrow_vault.to_account_info(),
                to: ctx.accounts.payer_token_account.to_account_info(),
                authority: ctx.accounts.escrow_vault.to_account_info(),
            };
            let cpi_program = ctx.accounts.token_program.to_account_info();
            let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
            
            token::transfer(cpi_ctx, net_amount)?;
        }
        
        // Award to payee if applicable
        if award_to_payee > 0 {
            let platform_fee = (award_to_payee * escrow_config.platform_fee_bps as u64) / 10000;
            let net_amount = award_to_payee - platform_fee;
            
            let cpi_accounts = Transfer {
                from: ctx.accounts.escrow_vault.to_account_info(),
                to: ctx.accounts.payee_token_account.to_account_info(),
                authority: ctx.accounts.escrow_vault.to_account_info(),
            };
            let cpi_program = ctx.accounts.token_program.to_account_info();
            let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
            
            token::transfer(cpi_ctx, net_amount)?;
        }
        
        escrow.status = EscrowStatus::Resolved;
        escrow.released_amount += award_to_payer + award_to_payee;
        
        emit!(DisputeResolved {
            escrow_id: escrow.id,
            award_to_payer,
            award_to_payee,
        });
        
        Ok(())
    }

    /// Auto-release escrow after delay
    pub fn auto_release_escrow(ctx: Context<AutoReleaseEscrow>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let escrow_config = &ctx.accounts.escrow_config;
        
        require!(escrow.auto_release_enabled, ErrorCode::AutoReleaseDisabled);
        require!(escrow.status == EscrowStatus::Funded, ErrorCode::InvalidEscrowStatus);
        require!(
            Clock::get()?.unix_timestamp >= escrow.created_at + escrow.auto_release_delay,
            ErrorCode::AutoReleaseDelayNotMet
        );
        
        let remaining_amount = escrow.amount - escrow.released_amount;
        let platform_fee = (remaining_amount * escrow_config.platform_fee_bps as u64) / 10000;
        let net_amount = remaining_amount - platform_fee;
        
        // Transfer remaining amount to payee
        let seeds = &[
            b"escrow_vault",
            &escrow.id.to_le_bytes(),
            &[ctx.bumps.escrow_vault],
        ];
        let signer = &[&seeds[..]];
        
        let cpi_accounts = Transfer {
            from: ctx.accounts.escrow_vault.to_account_info(),
            to: ctx.accounts.payee_token_account.to_account_info(),
            authority: ctx.accounts.escrow_vault.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
        
        token::transfer(cpi_ctx, net_amount)?;
        
        escrow.status = EscrowStatus::AutoReleased;
        escrow.released_amount = escrow.amount;
        
        emit!(EscrowAutoReleased {
            escrow_id: escrow.id,
            amount: net_amount,
        });
        
        Ok(())
    }

    /// Emergency pause (admin only)
    pub fn pause_program(ctx: Context<PauseProgram>) -> Result<()> {
        let escrow_config = &mut ctx.accounts.escrow_config;
        require!(escrow_config.authority == ctx.accounts.authority.key(), ErrorCode::UnauthorizedAdmin);
        
        escrow_config.is_paused = true;
        
        emit!(ProgramPaused {});
        
        Ok(())
    }

    /// Resume program (admin only)
    pub fn resume_program(ctx: Context<ResumeProgram>) -> Result<()> {
        let escrow_config = &mut ctx.accounts.escrow_config;
        require!(escrow_config.authority == ctx.accounts.authority.key(), ErrorCode::UnauthorizedAdmin);
        
        escrow_config.is_paused = false;
        
        emit!(ProgramResumed {});
        
        Ok(())
    }
}

// Account structures
#[account]
pub struct EscrowConfig {
    pub authority: Pubkey,
    pub platform_fee_bps: u16,
    pub total_escrows: u64,
    pub total_volume: u64,
    pub is_paused: bool,
}

#[account]
pub struct Escrow {
    pub id: u64,
    pub payer: Pubkey,
    pub payee: Pubkey,
    pub arbitrator: Pubkey,
    pub mint: Pubkey,
    pub amount: u64,
    pub status: EscrowStatus,
    pub created_at: i64,
    pub deadline: i64,
    pub terms_hash: [u8; 32],
    pub auto_release_enabled: bool,
    pub auto_release_delay: i64,
    pub milestones_count: u32,
    pub completed_milestones: u32,
    pub dispute_raised: bool,
    pub released_amount: u64,
}

#[account]
pub struct Milestone {
    pub id: u32,
    pub escrow_id: u64,
    pub description: String,
    pub amount: u64,
    pub status: MilestoneStatus,
    pub deadline: i64,
    pub created_at: i64,
    pub submitted_at: i64,
    pub approved_at: i64,
    pub deliverable_hash: [u8; 32],
}

#[account]
pub struct Dispute {
    pub escrow_id: u64,
    pub initiator: Pubkey,
    pub reason: String,
    pub status: DisputeStatus,
    pub created_at: i64,
    pub resolved_at: i64,
    pub resolution: String,
}

// Enums
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum EscrowStatus {
    Created,
    Funded,
    InProgress,
    Completed,
    Disputed,
    Resolved,
    AutoReleased,
    Cancelled,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum MilestoneStatus {
    Pending,
    Submitted,
    Approved,
    Rejected,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum DisputeStatus {
    Open,
    Resolved,
    Cancelled,
}

// Context structures
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 2 + 8 + 8 + 1,
        seeds = [b"escrow_config"],
        bump
    )]
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(amount: u64)]
pub struct CreateEscrow<'info> {
    #[account(
        init,
        payer = payer,
        space = 8 + 8 + 32 + 32 + 32 + 32 + 8 + 1 + 8 + 8 + 32 + 1 + 8 + 4 + 4 + 1 + 8,
        seeds = [b"escrow", &escrow_config.total_escrows.to_le_bytes()],
        bump
    )]
    pub escrow: Account<'info, Escrow>,
    #[account(mut)]
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(mut)]
    pub payer: Signer<'info>,
    /// CHECK: This is just used for identification
    pub payee: AccountInfo<'info>,
    /// CHECK: This is just used for identification
    pub arbitrator: AccountInfo<'info>,
    pub mint: Account<'info, Mint>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FundEscrow<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(mut)]
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(
        init,
        payer = payer,
        token::mint = escrow.mint,
        token::authority = escrow_vault,
        seeds = [b"escrow_vault", &escrow.id.to_le_bytes()],
        bump
    )]
    pub escrow_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = payer_token_account.mint == escrow.mint,
        constraint = payer_token_account.owner == payer.key()
    )]
    pub payer_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub payer: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(description: String)]
pub struct CreateMilestone<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(
        init,
        payer = authority,
        space = 8 + 4 + 8 + 4 + description.len() + 8 + 1 + 8 + 8 + 8 + 8 + 32,
        seeds = [b"milestone", &escrow.id.to_le_bytes(), &escrow.milestones_count.to_le_bytes()],
        bump
    )]
    pub milestone: Account<'info, Milestone>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SubmitMilestone<'info> {
    pub escrow: Account<'info, Escrow>,
    #[account(
        mut,
        constraint = milestone.escrow_id == escrow.id
    )]
    pub milestone: Account<'info, Milestone>,
    pub payee: Signer<'info>,
}

#[derive(Accounts)]
pub struct ApproveMilestone<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(
        mut,
        constraint = milestone.escrow_id == escrow.id
    )]
    pub milestone: Account<'info, Milestone>,
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(
        mut,
        seeds = [b"escrow_vault", &escrow.id.to_le_bytes()],
        bump
    )]
    pub escrow_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = payee_token_account.mint == escrow.mint
    )]
    pub payee_token_account: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = platform_fee_account.mint == escrow.mint
    )]
    pub platform_fee_account: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
#[instruction(reason: String)]
pub struct RaiseDispute<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(
        init,
        payer = authority,
        space = 8 + 8 + 32 + 4 + reason.len() + 1 + 8 + 8 + 4 + 100, // Extra space for resolution
        seeds = [b"dispute", &escrow.id.to_le_bytes()],
        bump
    )]
    pub dispute: Account<'info, Dispute>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ResolveDispute<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(
        mut,
        constraint = dispute.escrow_id == escrow.id
    )]
    pub dispute: Account<'info, Dispute>,
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(
        mut,
        seeds = [b"escrow_vault", &escrow.id.to_le_bytes()],
        bump
    )]
    pub escrow_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = payer_token_account.mint == escrow.mint
    )]
    pub payer_token_account: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = payee_token_account.mint == escrow.mint
    )]
    pub payee_token_account: Account<'info, TokenAccount>,
    pub arbitrator: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct AutoReleaseEscrow<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    pub escrow_config: Account<'info, EscrowConfig>,
    #[account(
        mut,
        seeds = [b"escrow_vault", &escrow.id.to_le_bytes()],
        bump
    )]
    pub escrow_vault: Account<'info, TokenAccount>,
    #[account(
        mut,
        constraint = payee_token_account.mint == escrow.mint
    )]
    pub payee_token_account: Account<'info, TokenAccount>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct PauseProgram<'info> {
    #[account(mut)]
    pub escrow_config: Account<'info, EscrowConfig>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResumeProgram<'info> {
    #[account(mut)]
    pub escrow_config: Account<'info, EscrowConfig>,
    pub authority: Signer<'info>,
}

// Events
#[event]
pub struct EscrowCreated {
    pub escrow_id: u64,
    pub payer: Pubkey,
    pub payee: Pubkey,
    pub amount: u64,
}

#[event]
pub struct EscrowFunded {
    pub escrow_id: u64,
    pub amount: u64,
}

#[event]
pub struct MilestoneCreated {
    pub escrow_id: u64,
    pub milestone_id: u32,
    pub amount: u64,
}

#[event]
pub struct MilestoneSubmitted {
    pub escrow_id: u64,
    pub milestone_id: u32,
    pub deliverable_hash: [u8; 32],
}

#[event]
pub struct MilestoneApproved {
    pub escrow_id: u64,
    pub milestone_id: u32,
    pub amount: u64,
}

#[event]
pub struct DisputeRaised {
    pub escrow_id: u64,
    pub initiator: Pubkey,
    pub reason: String,
}

#[event]
pub struct DisputeResolved {
    pub escrow_id: u64,
    pub award_to_payer: u64,
    pub award_to_payee: u64,
}

#[event]
pub struct EscrowAutoReleased {
    pub escrow_id: u64,
    pub amount: u64,
}

#[event]
pub struct ProgramPaused {}

#[event]
pub struct ProgramResumed {}

// Error codes
#[error_code]
pub enum ErrorCode {
    #[msg("Program is currently paused")]
    ProgramPaused,
    #[msg("Invalid amount specified")]
    InvalidAmount,
    #[msg("Invalid deadline specified")]
    InvalidDeadline,
    #[msg("Invalid escrow status for this operation")]
    InvalidEscrowStatus,
    #[msg("Unauthorized payer")]
    UnauthorizedPayer,
    #[msg("Unauthorized payee")]
    UnauthorizedPayee,
    #[msg("Unauthorized access")]
    UnauthorizedAccess,
    #[msg("Unauthorized arbitrator")]
    UnauthorizedArbitrator,
    #[msg("Unauthorized admin")]
    UnauthorizedAdmin,
    #[msg("Description too long")]
    DescriptionTooLong,
    #[msg("Invalid milestone status")]
    InvalidMilestoneStatus,
    #[msg("Milestone deadline has expired")]
    MilestoneExpired,
    #[msg("Dispute already raised for this escrow")]
    DisputeAlreadyRaised,
    #[msg("Reason too long")]
    ReasonTooLong,
    #[msg("Invalid dispute status")]
    InvalidDisputeStatus,
    #[msg("Award amount exceeds available balance")]
    ExcessiveAward,
    #[msg("Resolution too long")]
    ResolutionTooLong,
    #[msg("Auto-release is disabled for this escrow")]
    AutoReleaseDisabled,
    #[msg("Auto-release delay not met")]
    AutoReleaseDelayNotMet,
}