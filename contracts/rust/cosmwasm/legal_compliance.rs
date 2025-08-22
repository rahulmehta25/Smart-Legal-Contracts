use cosmwasm_std::{
    entry_point, to_binary, Binary, Deps, DepsMut, Env, MessageInfo, Response, StdResult,
    StdError, Addr, Uint128, BlockInfo, Order, Storage, Api, Querier,
};
use cw_storage_plus::{Item, Map, Bound};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// State management
const CONFIG: Item<Config> = Item::new("config");
const COMPLIANCE_RECORDS: Map<&str, ComplianceRecord> = Map::new("compliance_records");
const REGULATIONS: Map<u64, Regulation> = Map::new("regulations");
const AUDITS: Map<&str, Vec<Audit>> = Map::new("audits");
const RISK_ASSESSMENTS: Map<&str, RiskAssessment> = Map::new("risk_assessments");
const NOTIFICATIONS: Map<u64, ComplianceNotification> = Map::new("notifications");

// Types and structures
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct Config {
    pub admin: Addr,
    pub oracle_address: Option<Addr>,
    pub compliance_fee: Uint128,
    pub audit_threshold: u8,
    pub risk_tolerance: u8,
    pub auto_monitoring: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    PartiallyCompliant,
    UnderReview,
    Expired,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub enum RegulationType {
    Financial,
    DataPrivacy,
    Environmental,
    Labor,
    Healthcare,
    Securities,
    AML,
    KYC,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct ComplianceRecord {
    pub entity_id: String,
    pub entity_name: String,
    pub jurisdiction: String,
    pub status: ComplianceStatus,
    pub risk_level: RiskLevel,
    pub last_audit: Option<u64>,
    pub next_audit_due: u64,
    pub violations: Vec<Violation>,
    pub certifications: Vec<String>,
    pub compliance_score: u8,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct Regulation {
    pub id: u64,
    pub title: String,
    pub regulation_type: RegulationType,
    pub jurisdiction: String,
    pub description: String,
    pub requirements: Vec<Requirement>,
    pub effective_date: u64,
    pub expiry_date: Option<u64>,
    pub is_active: bool,
    pub penalty_amount: Uint128,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct Requirement {
    pub id: String,
    pub description: String,
    pub mandatory: bool,
    pub verification_method: String,
    pub deadline: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct Violation {
    pub regulation_id: u64,
    pub requirement_id: String,
    pub description: String,
    pub severity: RiskLevel,
    pub detected_at: u64,
    pub resolved: bool,
    pub resolution_deadline: u64,
    pub penalty: Uint128,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct Audit {
    pub id: String,
    pub auditor: Addr,
    pub audit_type: String,
    pub scope: Vec<String>,
    pub findings: Vec<AuditFinding>,
    pub overall_score: u8,
    pub recommendations: Vec<String>,
    pub conducted_at: u64,
    pub report_hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct AuditFinding {
    pub category: String,
    pub description: String,
    pub severity: RiskLevel,
    pub regulation_references: Vec<u64>,
    pub remediation_required: bool,
    pub remediation_deadline: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct RiskAssessment {
    pub entity_id: String,
    pub assessed_by: Addr,
    pub overall_risk: RiskLevel,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_measures: Vec<String>,
    pub assessment_date: u64,
    pub next_assessment_due: u64,
    pub confidence_level: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct RiskFactor {
    pub category: String,
    pub description: String,
    pub impact: RiskLevel,
    pub probability: u8, // 0-100
    pub mitigation_status: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct ComplianceNotification {
    pub id: u64,
    pub entity_id: String,
    pub notification_type: String,
    pub message: String,
    pub severity: RiskLevel,
    pub created_at: u64,
    pub read: bool,
    pub action_required: bool,
    pub deadline: Option<u64>,
}

// Messages
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct InstantiateMsg {
    pub admin: String,
    pub compliance_fee: Uint128,
    pub audit_threshold: u8,
    pub risk_tolerance: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecuteMsg {
    UpdateConfig {
        admin: Option<String>,
        compliance_fee: Option<Uint128>,
        audit_threshold: Option<u8>,
        risk_tolerance: Option<u8>,
        auto_monitoring: Option<bool>,
    },
    RegisterEntity {
        entity_id: String,
        entity_name: String,
        jurisdiction: String,
    },
    UpdateComplianceStatus {
        entity_id: String,
        status: ComplianceStatus,
        risk_level: RiskLevel,
    },
    AddRegulation {
        title: String,
        regulation_type: RegulationType,
        jurisdiction: String,
        description: String,
        requirements: Vec<Requirement>,
        effective_date: u64,
        expiry_date: Option<u64>,
        penalty_amount: Uint128,
    },
    ReportViolation {
        entity_id: String,
        regulation_id: u64,
        requirement_id: String,
        description: String,
        severity: RiskLevel,
    },
    ConductAudit {
        entity_id: String,
        audit_type: String,
        scope: Vec<String>,
        findings: Vec<AuditFinding>,
        overall_score: u8,
        recommendations: Vec<String>,
        report_hash: String,
    },
    PerformRiskAssessment {
        entity_id: String,
        risk_factors: Vec<RiskFactor>,
        mitigation_measures: Vec<String>,
        confidence_level: u8,
    },
    ResolveViolation {
        entity_id: String,
        violation_index: usize,
    },
    SendNotification {
        entity_id: String,
        notification_type: String,
        message: String,
        severity: RiskLevel,
        action_required: bool,
        deadline: Option<u64>,
    },
    MarkNotificationRead {
        notification_id: u64,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryMsg {
    GetConfig {},
    GetComplianceRecord { entity_id: String },
    GetRegulation { regulation_id: u64 },
    GetAudits { entity_id: String },
    GetRiskAssessment { entity_id: String },
    GetViolations { entity_id: String },
    GetNotifications { entity_id: String },
    ListRegulationsByJurisdiction { jurisdiction: String },
    ListEntitiesByStatus { status: ComplianceStatus },
    GetComplianceStatistics {},
    CheckCompliance { entity_id: String, regulation_ids: Vec<u64> },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct ComplianceStatistics {
    pub total_entities: u64,
    pub compliant_entities: u64,
    pub non_compliant_entities: u64,
    pub partially_compliant_entities: u64,
    pub entities_under_review: u64,
    pub total_violations: u64,
    pub resolved_violations: u64,
    pub pending_audits: u64,
    pub average_compliance_score: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct ComplianceCheckResult {
    pub entity_id: String,
    pub overall_status: ComplianceStatus,
    pub regulation_results: Vec<RegulationComplianceResult>,
    pub checked_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, JsonSchema)]
pub struct RegulationComplianceResult {
    pub regulation_id: u64,
    pub status: ComplianceStatus,
    pub violations: Vec<String>,
    pub risk_level: RiskLevel,
}

// Entry points
#[entry_point]
pub fn instantiate(
    deps: DepsMut,
    _env: Env,
    info: MessageInfo,
    msg: InstantiateMsg,
) -> StdResult<Response> {
    let admin = deps.api.addr_validate(&msg.admin)?;
    
    let config = Config {
        admin,
        oracle_address: None,
        compliance_fee: msg.compliance_fee,
        audit_threshold: msg.audit_threshold,
        risk_tolerance: msg.risk_tolerance,
        auto_monitoring: false,
    };
    
    CONFIG.save(deps.storage, &config)?;
    
    Ok(Response::new()
        .add_attribute("method", "instantiate")
        .add_attribute("admin", msg.admin)
        .add_attribute("compliance_fee", msg.compliance_fee))
}

#[entry_point]
pub fn execute(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    msg: ExecuteMsg,
) -> StdResult<Response> {
    match msg {
        ExecuteMsg::UpdateConfig {
            admin,
            compliance_fee,
            audit_threshold,
            risk_tolerance,
            auto_monitoring,
        } => execute_update_config(deps, info, admin, compliance_fee, audit_threshold, risk_tolerance, auto_monitoring),
        ExecuteMsg::RegisterEntity {
            entity_id,
            entity_name,
            jurisdiction,
        } => execute_register_entity(deps, env, info, entity_id, entity_name, jurisdiction),
        ExecuteMsg::UpdateComplianceStatus {
            entity_id,
            status,
            risk_level,
        } => execute_update_compliance_status(deps, env, info, entity_id, status, risk_level),
        ExecuteMsg::AddRegulation {
            title,
            regulation_type,
            jurisdiction,
            description,
            requirements,
            effective_date,
            expiry_date,
            penalty_amount,
        } => execute_add_regulation(deps, env, info, title, regulation_type, jurisdiction, description, requirements, effective_date, expiry_date, penalty_amount),
        ExecuteMsg::ReportViolation {
            entity_id,
            regulation_id,
            requirement_id,
            description,
            severity,
        } => execute_report_violation(deps, env, info, entity_id, regulation_id, requirement_id, description, severity),
        ExecuteMsg::ConductAudit {
            entity_id,
            audit_type,
            scope,
            findings,
            overall_score,
            recommendations,
            report_hash,
        } => execute_conduct_audit(deps, env, info, entity_id, audit_type, scope, findings, overall_score, recommendations, report_hash),
        ExecuteMsg::PerformRiskAssessment {
            entity_id,
            risk_factors,
            mitigation_measures,
            confidence_level,
        } => execute_perform_risk_assessment(deps, env, info, entity_id, risk_factors, mitigation_measures, confidence_level),
        ExecuteMsg::ResolveViolation {
            entity_id,
            violation_index,
        } => execute_resolve_violation(deps, env, info, entity_id, violation_index),
        ExecuteMsg::SendNotification {
            entity_id,
            notification_type,
            message,
            severity,
            action_required,
            deadline,
        } => execute_send_notification(deps, env, info, entity_id, notification_type, message, severity, action_required, deadline),
        ExecuteMsg::MarkNotificationRead { notification_id } => execute_mark_notification_read(deps, info, notification_id),
    }
}

// Execute functions
fn execute_update_config(
    deps: DepsMut,
    info: MessageInfo,
    admin: Option<String>,
    compliance_fee: Option<Uint128>,
    audit_threshold: Option<u8>,
    risk_tolerance: Option<u8>,
    auto_monitoring: Option<bool>,
) -> StdResult<Response> {
    let mut config = CONFIG.load(deps.storage)?;
    
    // Only admin can update config
    if info.sender != config.admin {
        return Err(StdError::generic_err("Unauthorized"));
    }
    
    if let Some(admin) = admin {
        config.admin = deps.api.addr_validate(&admin)?;
    }
    if let Some(fee) = compliance_fee {
        config.compliance_fee = fee;
    }
    if let Some(threshold) = audit_threshold {
        config.audit_threshold = threshold;
    }
    if let Some(tolerance) = risk_tolerance {
        config.risk_tolerance = tolerance;
    }
    if let Some(monitoring) = auto_monitoring {
        config.auto_monitoring = monitoring;
    }
    
    CONFIG.save(deps.storage, &config)?;
    
    Ok(Response::new().add_attribute("method", "update_config"))
}

fn execute_register_entity(
    deps: DepsMut,
    env: Env,
    _info: MessageInfo,
    entity_id: String,
    entity_name: String,
    jurisdiction: String,
) -> StdResult<Response> {
    // Check if entity already exists
    if COMPLIANCE_RECORDS.has(deps.storage, &entity_id) {
        return Err(StdError::generic_err("Entity already registered"));
    }
    
    let record = ComplianceRecord {
        entity_id: entity_id.clone(),
        entity_name,
        jurisdiction,
        status: ComplianceStatus::UnderReview,
        risk_level: RiskLevel::Medium,
        last_audit: None,
        next_audit_due: env.block.time.seconds() + 365 * 24 * 60 * 60, // 1 year from now
        violations: vec![],
        certifications: vec![],
        compliance_score: 50, // Starting score
        created_at: env.block.time.seconds(),
        updated_at: env.block.time.seconds(),
    };
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "register_entity")
        .add_attribute("entity_id", entity_id))
}

fn execute_update_compliance_status(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    entity_id: String,
    status: ComplianceStatus,
    risk_level: RiskLevel,
) -> StdResult<Response> {
    let config = CONFIG.load(deps.storage)?;
    
    // Only admin can update status
    if info.sender != config.admin {
        return Err(StdError::generic_err("Unauthorized"));
    }
    
    let mut record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
        .map_err(|_| StdError::generic_err("Entity not found"))?;
    
    record.status = status;
    record.risk_level = risk_level;
    record.updated_at = env.block.time.seconds();
    
    // Update compliance score based on status
    record.compliance_score = match record.status {
        ComplianceStatus::Compliant => 90 + (record.compliance_score / 10),
        ComplianceStatus::PartiallyCompliant => 60 + (record.compliance_score / 5),
        ComplianceStatus::NonCompliant => record.compliance_score.saturating_sub(20),
        ComplianceStatus::UnderReview => record.compliance_score,
        ComplianceStatus::Expired => record.compliance_score.saturating_sub(10),
    };
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "update_compliance_status")
        .add_attribute("entity_id", entity_id)
        .add_attribute("status", format!("{:?}", record.status)))
}

fn execute_add_regulation(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    title: String,
    regulation_type: RegulationType,
    jurisdiction: String,
    description: String,
    requirements: Vec<Requirement>,
    effective_date: u64,
    expiry_date: Option<u64>,
    penalty_amount: Uint128,
) -> StdResult<Response> {
    let config = CONFIG.load(deps.storage)?;
    
    // Only admin can add regulations
    if info.sender != config.admin {
        return Err(StdError::generic_err("Unauthorized"));
    }
    
    // Generate regulation ID
    let regulation_id = env.block.time.seconds();
    
    let regulation = Regulation {
        id: regulation_id,
        title,
        regulation_type,
        jurisdiction,
        description,
        requirements,
        effective_date,
        expiry_date,
        is_active: true,
        penalty_amount,
    };
    
    REGULATIONS.save(deps.storage, regulation_id, &regulation)?;
    
    Ok(Response::new()
        .add_attribute("method", "add_regulation")
        .add_attribute("regulation_id", regulation_id.to_string()))
}

fn execute_report_violation(
    deps: DepsMut,
    env: Env,
    _info: MessageInfo,
    entity_id: String,
    regulation_id: u64,
    requirement_id: String,
    description: String,
    severity: RiskLevel,
) -> StdResult<Response> {
    let mut record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
        .map_err(|_| StdError::generic_err("Entity not found"))?;
    
    let regulation = REGULATIONS.load(deps.storage, regulation_id)
        .map_err(|_| StdError::generic_err("Regulation not found"))?;
    
    let violation = Violation {
        regulation_id,
        requirement_id,
        description,
        severity: severity.clone(),
        detected_at: env.block.time.seconds(),
        resolved: false,
        resolution_deadline: env.block.time.seconds() + 30 * 24 * 60 * 60, // 30 days
        penalty: regulation.penalty_amount,
    };
    
    record.violations.push(violation);
    record.status = ComplianceStatus::NonCompliant;
    record.risk_level = match severity {
        RiskLevel::Critical => RiskLevel::Critical,
        RiskLevel::High => if matches!(record.risk_level, RiskLevel::Critical) { RiskLevel::Critical } else { RiskLevel::High },
        _ => record.risk_level,
    };
    record.updated_at = env.block.time.seconds();
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "report_violation")
        .add_attribute("entity_id", entity_id)
        .add_attribute("regulation_id", regulation_id.to_string()))
}

fn execute_conduct_audit(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    entity_id: String,
    audit_type: String,
    scope: Vec<String>,
    findings: Vec<AuditFinding>,
    overall_score: u8,
    recommendations: Vec<String>,
    report_hash: String,
) -> StdResult<Response> {
    let audit_id = format!("{}_{}", entity_id, env.block.time.seconds());
    
    let audit = Audit {
        id: audit_id.clone(),
        auditor: info.sender,
        audit_type,
        scope,
        findings,
        overall_score,
        recommendations,
        conducted_at: env.block.time.seconds(),
        report_hash,
    };
    
    // Add audit to entity's audit history
    let mut audits = AUDITS.load(deps.storage, &entity_id).unwrap_or_default();
    audits.push(audit);
    AUDITS.save(deps.storage, &entity_id, &audits)?;
    
    // Update entity record with audit information
    let mut record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
        .map_err(|_| StdError::generic_err("Entity not found"))?;
    
    record.last_audit = Some(env.block.time.seconds());
    record.next_audit_due = env.block.time.seconds() + 365 * 24 * 60 * 60; // Next year
    record.compliance_score = overall_score;
    record.updated_at = env.block.time.seconds();
    
    // Update status based on audit score
    record.status = if overall_score >= 80 {
        ComplianceStatus::Compliant
    } else if overall_score >= 60 {
        ComplianceStatus::PartiallyCompliant
    } else {
        ComplianceStatus::NonCompliant
    };
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "conduct_audit")
        .add_attribute("entity_id", entity_id)
        .add_attribute("audit_id", audit_id))
}

fn execute_perform_risk_assessment(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    entity_id: String,
    risk_factors: Vec<RiskFactor>,
    mitigation_measures: Vec<String>,
    confidence_level: u8,
) -> StdResult<Response> {
    // Calculate overall risk based on factors
    let overall_risk = calculate_overall_risk(&risk_factors);
    
    let assessment = RiskAssessment {
        entity_id: entity_id.clone(),
        assessed_by: info.sender,
        overall_risk: overall_risk.clone(),
        risk_factors,
        mitigation_measures,
        assessment_date: env.block.time.seconds(),
        next_assessment_due: env.block.time.seconds() + 180 * 24 * 60 * 60, // 6 months
        confidence_level,
    };
    
    RISK_ASSESSMENTS.save(deps.storage, &entity_id, &assessment)?;
    
    // Update entity record with risk level
    let mut record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
        .map_err(|_| StdError::generic_err("Entity not found"))?;
    
    record.risk_level = overall_risk;
    record.updated_at = env.block.time.seconds();
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "perform_risk_assessment")
        .add_attribute("entity_id", entity_id))
}

fn execute_resolve_violation(
    deps: DepsMut,
    env: Env,
    info: MessageInfo,
    entity_id: String,
    violation_index: usize,
) -> StdResult<Response> {
    let config = CONFIG.load(deps.storage)?;
    
    // Only admin can resolve violations
    if info.sender != config.admin {
        return Err(StdError::generic_err("Unauthorized"));
    }
    
    let mut record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
        .map_err(|_| StdError::generic_err("Entity not found"))?;
    
    if violation_index >= record.violations.len() {
        return Err(StdError::generic_err("Invalid violation index"));
    }
    
    record.violations[violation_index].resolved = true;
    record.updated_at = env.block.time.seconds();
    
    // Check if all violations are resolved
    let all_resolved = record.violations.iter().all(|v| v.resolved);
    if all_resolved && record.violations.len() > 0 {
        record.status = ComplianceStatus::Compliant;
        record.compliance_score = std::cmp::min(100, record.compliance_score + 10);
    }
    
    COMPLIANCE_RECORDS.save(deps.storage, &entity_id, &record)?;
    
    Ok(Response::new()
        .add_attribute("method", "resolve_violation")
        .add_attribute("entity_id", entity_id))
}

fn execute_send_notification(
    deps: DepsMut,
    env: Env,
    _info: MessageInfo,
    entity_id: String,
    notification_type: String,
    message: String,
    severity: RiskLevel,
    action_required: bool,
    deadline: Option<u64>,
) -> StdResult<Response> {
    let notification_id = env.block.time.seconds();
    
    let notification = ComplianceNotification {
        id: notification_id,
        entity_id: entity_id.clone(),
        notification_type,
        message,
        severity,
        created_at: env.block.time.seconds(),
        read: false,
        action_required,
        deadline,
    };
    
    NOTIFICATIONS.save(deps.storage, notification_id, &notification)?;
    
    Ok(Response::new()
        .add_attribute("method", "send_notification")
        .add_attribute("entity_id", entity_id)
        .add_attribute("notification_id", notification_id.to_string()))
}

fn execute_mark_notification_read(
    deps: DepsMut,
    _info: MessageInfo,
    notification_id: u64,
) -> StdResult<Response> {
    let mut notification = NOTIFICATIONS.load(deps.storage, notification_id)
        .map_err(|_| StdError::generic_err("Notification not found"))?;
    
    notification.read = true;
    NOTIFICATIONS.save(deps.storage, notification_id, &notification)?;
    
    Ok(Response::new()
        .add_attribute("method", "mark_notification_read")
        .add_attribute("notification_id", notification_id.to_string()))
}

// Query functions
#[entry_point]
pub fn query(deps: Deps, _env: Env, msg: QueryMsg) -> StdResult<Binary> {
    match msg {
        QueryMsg::GetConfig {} => to_binary(&query_config(deps)?),
        QueryMsg::GetComplianceRecord { entity_id } => to_binary(&query_compliance_record(deps, entity_id)?),
        QueryMsg::GetRegulation { regulation_id } => to_binary(&query_regulation(deps, regulation_id)?),
        QueryMsg::GetAudits { entity_id } => to_binary(&query_audits(deps, entity_id)?),
        QueryMsg::GetRiskAssessment { entity_id } => to_binary(&query_risk_assessment(deps, entity_id)?),
        QueryMsg::GetViolations { entity_id } => to_binary(&query_violations(deps, entity_id)?),
        QueryMsg::GetNotifications { entity_id } => to_binary(&query_notifications(deps, entity_id)?),
        QueryMsg::ListRegulationsByJurisdiction { jurisdiction } => to_binary(&query_regulations_by_jurisdiction(deps, jurisdiction)?),
        QueryMsg::ListEntitiesByStatus { status } => to_binary(&query_entities_by_status(deps, status)?),
        QueryMsg::GetComplianceStatistics {} => to_binary(&query_compliance_statistics(deps)?),
        QueryMsg::CheckCompliance { entity_id, regulation_ids } => to_binary(&query_check_compliance(deps, entity_id, regulation_ids)?),
    }
}

fn query_config(deps: Deps) -> StdResult<Config> {
    CONFIG.load(deps.storage)
}

fn query_compliance_record(deps: Deps, entity_id: String) -> StdResult<ComplianceRecord> {
    COMPLIANCE_RECORDS.load(deps.storage, &entity_id)
}

fn query_regulation(deps: Deps, regulation_id: u64) -> StdResult<Regulation> {
    REGULATIONS.load(deps.storage, regulation_id)
}

fn query_audits(deps: Deps, entity_id: String) -> StdResult<Vec<Audit>> {
    Ok(AUDITS.load(deps.storage, &entity_id).unwrap_or_default())
}

fn query_risk_assessment(deps: Deps, entity_id: String) -> StdResult<Option<RiskAssessment>> {
    Ok(RISK_ASSESSMENTS.may_load(deps.storage, &entity_id)?)
}

fn query_violations(deps: Deps, entity_id: String) -> StdResult<Vec<Violation>> {
    let record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)?;
    Ok(record.violations)
}

fn query_notifications(deps: Deps, entity_id: String) -> StdResult<Vec<ComplianceNotification>> {
    let notifications: StdResult<Vec<_>> = NOTIFICATIONS
        .range(deps.storage, None, None, Order::Descending)
        .filter_map(|item| {
            let (_, notification) = item.ok()?;
            if notification.entity_id == entity_id {
                Some(Ok(notification))
            } else {
                None
            }
        })
        .collect();
    notifications
}

fn query_regulations_by_jurisdiction(deps: Deps, jurisdiction: String) -> StdResult<Vec<Regulation>> {
    let regulations: StdResult<Vec<_>> = REGULATIONS
        .range(deps.storage, None, None, Order::Ascending)
        .filter_map(|item| {
            let (_, regulation) = item.ok()?;
            if regulation.jurisdiction == jurisdiction && regulation.is_active {
                Some(Ok(regulation))
            } else {
                None
            }
        })
        .collect();
    regulations
}

fn query_entities_by_status(deps: Deps, status: ComplianceStatus) -> StdResult<Vec<ComplianceRecord>> {
    let entities: StdResult<Vec<_>> = COMPLIANCE_RECORDS
        .range(deps.storage, None, None, Order::Ascending)
        .filter_map(|item| {
            let (_, record) = item.ok()?;
            if record.status == status {
                Some(Ok(record))
            } else {
                None
            }
        })
        .collect();
    entities
}

fn query_compliance_statistics(deps: Deps) -> StdResult<ComplianceStatistics> {
    let mut stats = ComplianceStatistics {
        total_entities: 0,
        compliant_entities: 0,
        non_compliant_entities: 0,
        partially_compliant_entities: 0,
        entities_under_review: 0,
        total_violations: 0,
        resolved_violations: 0,
        pending_audits: 0,
        average_compliance_score: 0,
    };
    
    let mut total_score = 0u64;
    
    for item in COMPLIANCE_RECORDS.range(deps.storage, None, None, Order::Ascending) {
        let (_, record) = item?;
        stats.total_entities += 1;
        total_score += record.compliance_score as u64;
        
        match record.status {
            ComplianceStatus::Compliant => stats.compliant_entities += 1,
            ComplianceStatus::NonCompliant => stats.non_compliant_entities += 1,
            ComplianceStatus::PartiallyCompliant => stats.partially_compliant_entities += 1,
            ComplianceStatus::UnderReview => stats.entities_under_review += 1,
            _ => {}
        }
        
        stats.total_violations += record.violations.len() as u64;
        stats.resolved_violations += record.violations.iter().filter(|v| v.resolved).count() as u64;
    }
    
    if stats.total_entities > 0 {
        stats.average_compliance_score = (total_score / stats.total_entities) as u8;
    }
    
    Ok(stats)
}

fn query_check_compliance(
    deps: Deps,
    entity_id: String,
    regulation_ids: Vec<u64>,
) -> StdResult<ComplianceCheckResult> {
    let record = COMPLIANCE_RECORDS.load(deps.storage, &entity_id)?;
    let mut regulation_results = Vec::new();
    let mut overall_compliant = true;
    
    for reg_id in regulation_ids {
        let regulation = REGULATIONS.load(deps.storage, reg_id)?;
        
        // Check for violations related to this regulation
        let violations: Vec<String> = record.violations
            .iter()
            .filter(|v| v.regulation_id == reg_id && !v.resolved)
            .map(|v| v.description.clone())
            .collect();
        
        let status = if violations.is_empty() {
            ComplianceStatus::Compliant
        } else {
            overall_compliant = false;
            ComplianceStatus::NonCompliant
        };
        
        let risk_level = if violations.is_empty() {
            RiskLevel::Low
        } else {
            // Determine risk based on violation severity
            record.violations
                .iter()
                .filter(|v| v.regulation_id == reg_id && !v.resolved)
                .map(|v| &v.severity)
                .max()
                .cloned()
                .unwrap_or(RiskLevel::Low)
        };
        
        regulation_results.push(RegulationComplianceResult {
            regulation_id: reg_id,
            status,
            violations,
            risk_level,
        });
    }
    
    let overall_status = if overall_compliant {
        ComplianceStatus::Compliant
    } else {
        record.status
    };
    
    Ok(ComplianceCheckResult {
        entity_id,
        overall_status,
        regulation_results,
        checked_at: deps.api.extern_ref().env.block.time.seconds(),
    })
}

// Helper functions
fn calculate_overall_risk(risk_factors: &[RiskFactor]) -> RiskLevel {
    if risk_factors.is_empty() {
        return RiskLevel::Low;
    }
    
    let critical_count = risk_factors.iter().filter(|f| matches!(f.impact, RiskLevel::Critical)).count();
    let high_count = risk_factors.iter().filter(|f| matches!(f.impact, RiskLevel::High)).count();
    
    if critical_count > 0 {
        RiskLevel::Critical
    } else if high_count > risk_factors.len() / 2 {
        RiskLevel::High
    } else if high_count > 0 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmwasm_std::testing::{mock_dependencies, mock_env, mock_info};
    use cosmwasm_std::{coins, from_binary};

    #[test]
    fn proper_initialization() {
        let mut deps = mock_dependencies();
        
        let msg = InstantiateMsg {
            admin: "admin".to_string(),
            compliance_fee: Uint128::new(1000),
            audit_threshold: 80,
            risk_tolerance: 70,
        };
        let info = mock_info("creator", &coins(1000, "earth"));
        
        let res = instantiate(deps.as_mut(), mock_env(), info, msg).unwrap();
        assert_eq!(0, res.messages.len());
        
        let res = query(deps.as_ref(), mock_env(), QueryMsg::GetConfig {}).unwrap();
        let config: Config = from_binary(&res).unwrap();
        assert_eq!("admin", config.admin);
        assert_eq!(Uint128::new(1000), config.compliance_fee);
    }
    
    #[test]
    fn register_entity() {
        let mut deps = mock_dependencies();
        
        let msg = InstantiateMsg {
            admin: "admin".to_string(),
            compliance_fee: Uint128::new(1000),
            audit_threshold: 80,
            risk_tolerance: 70,
        };
        let info = mock_info("creator", &coins(1000, "earth"));
        instantiate(deps.as_mut(), mock_env(), info, msg).unwrap();
        
        let msg = ExecuteMsg::RegisterEntity {
            entity_id: "entity1".to_string(),
            entity_name: "Test Entity".to_string(),
            jurisdiction: "US".to_string(),
        };
        let info = mock_info("user", &[]);
        let res = execute(deps.as_mut(), mock_env(), info, msg).unwrap();
        assert_eq!(res.attributes[0].value, "register_entity");
        
        let res = query(deps.as_ref(), mock_env(), QueryMsg::GetComplianceRecord {
            entity_id: "entity1".to_string(),
        }).unwrap();
        let record: ComplianceRecord = from_binary(&res).unwrap();
        assert_eq!("Test Entity", record.entity_name);
        assert_eq!("US", record.jurisdiction);
    }
}