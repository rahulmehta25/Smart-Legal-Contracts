/**
 * ValidationEngine.tsx - Contract Validation System
 * 
 * Comprehensive validation engine with risk assessment, compliance checking,
 * conflict detection, and intelligent recommendations.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  RiskAssessment,
  RiskLevel,
  RiskFactor,
  RiskRecommendation,
  ComplianceIssue,
  ContractDraft,
  Clause,
  ClauseCategory,
  UUID,
  ValidationResult,
  ValidationError,
  ValidationWarning
} from './types';

interface ValidationEngineProps {
  assessment?: RiskAssessment | null;
  errors?: string[];
  contract?: ContractDraft;
  clauses?: Clause[];
  onRevalidate?: () => void;
  onFixIssue?: (issueId: string, solution: string) => void;
  showDetailedAnalysis?: boolean;
  enableAutoFix?: boolean;
  readonly?: boolean;
  className?: string;
}

interface EngineState {
  activeTab: 'overview' | 'risks' | 'compliance' | 'conflicts' | 'recommendations' | 'history';
  expandedSections: Set<string>;
  filterLevel: RiskLevel | 'ALL';
  sortBy: 'severity' | 'category' | 'impact' | 'probability';
  showResolvedIssues: boolean;
  validationHistory: ValidationHistoryEntry[];
  autoFixInProgress: Set<string>;
}

interface ValidationHistoryEntry {
  id: UUID;
  timestamp: Date;
  overallRisk: RiskLevel;
  issueCount: number;
  fixesApplied: number;
  summary: string;
}

interface RiskFactorCardProps {
  factor: RiskFactor;
  onExpand: (factorId: string) => void;
  isExpanded: boolean;
  onApplyFix?: (factorId: string, fixId: string) => void;
  readonly?: boolean;
}

interface ComplianceIssueCardProps {
  issue: ComplianceIssue;
  onResolve?: (issueId: string, action: string) => void;
  readonly?: boolean;
}

interface ValidationMetrics {
  totalIssues: number;
  criticalIssues: number;
  highRiskIssues: number;
  complianceIssues: number;
  riskScore: number;
  complianceScore: number;
  improvement: number;
}

// Risk calculation service
const riskCalculationService = {
  calculateOverallRisk: (factors: RiskFactor[]): RiskLevel => {
    if (factors.length === 0) return RiskLevel.LOW;
    
    const riskScores = factors.map(factor => {
      const severityWeight = {
        [RiskLevel.LOW]: 1,
        [RiskLevel.MEDIUM]: 2,
        [RiskLevel.HIGH]: 3,
        [RiskLevel.CRITICAL]: 4
      };
      
      return (severityWeight[factor.severity] * factor.impact * factor.probability) / 10;
    });
    
    const averageScore = riskScores.reduce((sum, score) => sum + score, 0) / riskScores.length;
    
    if (averageScore >= 3) return RiskLevel.CRITICAL;
    if (averageScore >= 2.5) return RiskLevel.HIGH;
    if (averageScore >= 1.5) return RiskLevel.MEDIUM;
    return RiskLevel.LOW;
  },

  calculateComplianceScore: (issues: ComplianceIssue[]): number => {
    if (issues.length === 0) return 100;
    
    const maxPenalty = issues.length * 4; // Max 4 points per issue
    const actualPenalty = issues.reduce((sum, issue) => {
      const severityPenalty = {
        [RiskLevel.LOW]: 1,
        [RiskLevel.MEDIUM]: 2,
        [RiskLevel.HIGH]: 3,
        [RiskLevel.CRITICAL]: 4
      };
      return sum + severityPenalty[issue.severity];
    }, 0);
    
    return Math.max(0, 100 - (actualPenalty / maxPenalty) * 100);
  },

  generateRecommendations: (factors: RiskFactor[], issues: ComplianceIssue[]): RiskRecommendation[] => {
    const recommendations: RiskRecommendation[] = [];
    
    // Risk-based recommendations
    factors.forEach(factor => {
      if (factor.severity === RiskLevel.CRITICAL || factor.severity === RiskLevel.HIGH) {
        recommendations.push({
          priority: factor.severity === RiskLevel.CRITICAL ? 'high' : 'medium',
          category: factor.category,
          recommendation: factor.mitigationSuggestions[0] || 'Review and mitigate this risk factor',
          impactDescription: `Addressing this ${factor.severity.toLowerCase()} risk could reduce overall contract risk`,
          alternativeClauseIds: []
        });
      }
    });
    
    // Compliance-based recommendations
    issues.forEach(issue => {
      if (issue.severity === RiskLevel.HIGH || issue.severity === RiskLevel.CRITICAL) {
        recommendations.push({
          priority: 'high',
          category: 'compliance',
          recommendation: issue.requiredActions[0] || 'Address compliance requirement',
          impactDescription: `Required for ${issue.jurisdiction} compliance`,
          alternativeClauseIds: []
        });
      }
    });
    
    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }
};

const RiskFactorCard: React.FC<RiskFactorCardProps> = ({
  factor,
  onExpand,
  isExpanded,
  onApplyFix,
  readonly = false
}) => {
  const getRiskColor = (risk: RiskLevel): string => {
    switch (risk) {
      case RiskLevel.LOW: return 'green';
      case RiskLevel.MEDIUM: return 'yellow';
      case RiskLevel.HIGH: return 'orange';
      case RiskLevel.CRITICAL: return 'red';
      default: return 'gray';
    }
  };

  const riskScore = (factor.impact * factor.probability).toFixed(1);

  return (
    <div className="risk-factor-card" data-factor-category={factor.category}>
      <div 
        className="factor-header"
        onClick={() => onExpand(`risk-${factor.category}`)}
      >
        <div className="factor-info">
          <h4 className="factor-category">{factor.category}</h4>
          <p className="factor-description">{factor.description}</p>
        </div>
        
        <div className="factor-metrics">
          <div className="risk-badge-container">
            <span className={`risk-badge ${getRiskColor(factor.severity)}`}>
              {factor.severity}
            </span>
            <span className="risk-score">{riskScore}</span>
          </div>
          
          <div className="factor-details">
            <div className="metric">
              <span className="metric-label">Impact:</span>
              <div className="metric-bar">
                <div 
                  className="metric-fill impact"
                  style={{ width: `${factor.impact * 10}%` }}
                ></div>
              </div>
              <span className="metric-value">{factor.impact}/10</span>
            </div>
            
            <div className="metric">
              <span className="metric-label">Probability:</span>
              <div className="metric-bar">
                <div 
                  className="metric-fill probability"
                  style={{ width: `${factor.probability * 10}%` }}
                ></div>
              </div>
              <span className="metric-value">{factor.probability}/10</span>
            </div>
          </div>
        </div>
        
        <button className="expand-btn">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <div className="factor-details-expanded">
          <div className="mitigation-suggestions">
            <h5>Mitigation Suggestions:</h5>
            <ul>
              {factor.mitigationSuggestions.map((suggestion, index) => (
                <li key={index} className="suggestion-item">
                  {suggestion}
                  {onApplyFix && !readonly && (
                    <button
                      onClick={() => onApplyFix(factor.category, `suggestion-${index}`)}
                      className="apply-fix-btn"
                    >
                      Apply Fix
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

const ComplianceIssueCard: React.FC<ComplianceIssueCardProps> = ({
  issue,
  onResolve,
  readonly = false
}) => {
  const getRiskColor = (risk: RiskLevel): string => {
    switch (risk) {
      case RiskLevel.LOW: return 'green';
      case RiskLevel.MEDIUM: return 'yellow';
      case RiskLevel.HIGH: return 'orange';
      case RiskLevel.CRITICAL: return 'red';
      default: return 'gray';
    }
  };

  const hasDeadlines = issue.deadlines && issue.deadlines.length > 0;
  const nextDeadline = hasDeadlines ? issue.deadlines![0] : null;
  const daysUntilDeadline = nextDeadline 
    ? Math.ceil((nextDeadline.getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))
    : null;

  return (
    <div className="compliance-issue-card" data-regulation={issue.regulation}>
      <div className="issue-header">
        <div className="issue-info">
          <h4 className="regulation-name">{issue.regulation}</h4>
          <p className="issue-description">{issue.description}</p>
          <span className="jurisdiction-tag">{issue.jurisdiction}</span>
        </div>
        
        <div className="issue-severity">
          <span className={`severity-badge ${getRiskColor(issue.severity)}`}>
            {issue.severity}
          </span>
          
          {hasDeadlines && daysUntilDeadline !== null && (
            <div className={`deadline-indicator ${daysUntilDeadline <= 30 ? 'urgent' : ''}`}>
              <span className="deadline-label">Due in:</span>
              <span className="deadline-value">{daysUntilDeadline} days</span>
            </div>
          )}
        </div>
      </div>

      <div className="required-actions">
        <h5>Required Actions:</h5>
        <ul>
          {issue.requiredActions.map((action, index) => (
            <li key={index} className="action-item">
              {action}
              {onResolve && !readonly && (
                <button
                  onClick={() => onResolve(`${issue.regulation}-${index}`, action)}
                  className="resolve-action-btn"
                >
                  Mark Resolved
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export const ValidationEngine: React.FC<ValidationEngineProps> = ({
  assessment,
  errors = [],
  contract,
  clauses = [],
  onRevalidate,
  onFixIssue,
  showDetailedAnalysis = true,
  enableAutoFix = false,
  readonly = false,
  className = ''
}) => {
  const [state, setState] = useState<EngineState>({
    activeTab: 'overview',
    expandedSections: new Set(),
    filterLevel: 'ALL',
    sortBy: 'severity',
    showResolvedIssues: false,
    validationHistory: [],
    autoFixInProgress: new Set()
  });

  // Calculate validation metrics
  const metrics = useMemo((): ValidationMetrics => {
    if (!assessment) {
      return {
        totalIssues: errors.length,
        criticalIssues: 0,
        highRiskIssues: 0,
        complianceIssues: 0,
        riskScore: 0,
        complianceScore: 100,
        improvement: 0
      };
    }

    const criticalFactors = assessment.riskFactors.filter(f => f.severity === RiskLevel.CRITICAL);
    const highRiskFactors = assessment.riskFactors.filter(f => f.severity === RiskLevel.HIGH);
    const criticalCompliance = assessment.complianceIssues.filter(i => i.severity === RiskLevel.CRITICAL);

    const riskScore = assessment.riskFactors.length > 0 
      ? assessment.riskFactors.reduce((sum, factor) => {
          const severityWeight = {
            [RiskLevel.LOW]: 25,
            [RiskLevel.MEDIUM]: 50,
            [RiskLevel.HIGH]: 75,
            [RiskLevel.CRITICAL]: 100
          };
          return sum + severityWeight[factor.severity];
        }, 0) / assessment.riskFactors.length
      : 0;

    const complianceScore = riskCalculationService.calculateComplianceScore(assessment.complianceIssues);

    return {
      totalIssues: assessment.riskFactors.length + assessment.complianceIssues.length + errors.length,
      criticalIssues: criticalFactors.length + criticalCompliance.length,
      highRiskIssues: highRiskFactors.length,
      complianceIssues: assessment.complianceIssues.length,
      riskScore: Math.round(riskScore),
      complianceScore: Math.round(complianceScore),
      improvement: 0 // Would be calculated based on history
    };
  }, [assessment, errors]);

  // Filtered risk factors based on current filter
  const filteredRiskFactors = useMemo(() => {
    if (!assessment) return [];
    
    let filtered = assessment.riskFactors;
    
    if (state.filterLevel !== 'ALL') {
      filtered = filtered.filter(factor => factor.severity === state.filterLevel);
    }
    
    // Sort factors
    filtered.sort((a, b) => {
      switch (state.sortBy) {
        case 'severity':
          const severityOrder = { [RiskLevel.CRITICAL]: 4, [RiskLevel.HIGH]: 3, [RiskLevel.MEDIUM]: 2, [RiskLevel.LOW]: 1 };
          return severityOrder[b.severity] - severityOrder[a.severity];
        case 'category':
          return a.category.localeCompare(b.category);
        case 'impact':
          return b.impact - a.impact;
        case 'probability':
          return b.probability - a.probability;
        default:
          return 0;
      }
    });
    
    return filtered;
  }, [assessment, state.filterLevel, state.sortBy]);

  // Handle section expansion
  const handleSectionExpand = useCallback((sectionId: string) => {
    setState(prev => {
      const newExpanded = new Set(prev.expandedSections);
      if (newExpanded.has(sectionId)) {
        newExpanded.delete(sectionId);
      } else {
        newExpanded.add(sectionId);
      }
      return { ...prev, expandedSections: newExpanded };
    });
  }, []);

  // Handle auto-fix application
  const handleAutoFix = useCallback(async (factorId: string, fixId: string) => {
    if (readonly || !enableAutoFix) return;
    
    setState(prev => ({
      ...prev,
      autoFixInProgress: new Set([...prev.autoFixInProgress, `${factorId}-${fixId}`])
    }));
    
    try {
      // Simulate auto-fix application
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      onFixIssue?.(factorId, fixId);
      
      // Add to validation history
      setState(prev => ({
        ...prev,
        autoFixInProgress: new Set([...prev.autoFixInProgress].filter(id => id !== `${factorId}-${fixId}`)),
        validationHistory: [
          {
            id: `fix-${Date.now()}`,
            timestamp: new Date(),
            overallRisk: assessment?.overallRisk || RiskLevel.LOW,
            issueCount: metrics.totalIssues - 1,
            fixesApplied: 1,
            summary: `Auto-fixed ${factorId} issue`
          },
          ...prev.validationHistory.slice(0, 9) // Keep last 10 entries
        ]
      }));
    } catch (error) {
      console.error('Auto-fix failed:', error);
      setState(prev => ({
        ...prev,
        autoFixInProgress: new Set([...prev.autoFixInProgress].filter(id => id !== `${factorId}-${fixId}`))
      }));
    }
  }, [readonly, enableAutoFix, onFixIssue, assessment, metrics.totalIssues]);

  const getOverallRiskColor = (risk: RiskLevel): string => {
    switch (risk) {
      case RiskLevel.LOW: return 'green';
      case RiskLevel.MEDIUM: return 'yellow';
      case RiskLevel.HIGH: return 'orange';
      case RiskLevel.CRITICAL: return 'red';
      default: return 'gray';
    }
  };

  if (!assessment && errors.length === 0) {
    return (
      <div className={`validation-engine empty ${className}`}>
        <div className="empty-state">
          <h3>No Validation Data</h3>
          <p>Run validation to see risk assessment and compliance analysis.</p>
          {onRevalidate && (
            <button
              onClick={onRevalidate}
              className="btn btn-primary"
            >
              Run Validation
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`validation-engine ${className}`} id="validation-engine-main">
      {/* Header */}
      <div className="engine-header" id="engine-header">
        <div className="header-title">
          <h3>Validation Engine</h3>
          {assessment && (
            <div className="overall-risk-indicator">
              <span className="risk-label">Overall Risk:</span>
              <span className={`risk-value ${getOverallRiskColor(assessment.overallRisk)}`}>
                {assessment.overallRisk}
              </span>
            </div>
          )}
        </div>

        <div className="header-actions">
          {onRevalidate && (
            <button
              onClick={onRevalidate}
              className="btn btn-outline"
              disabled={readonly}
            >
              🔄 Revalidate
            </button>
          )}
          
          <button
            onClick={() => setState(prev => ({ ...prev, showResolvedIssues: !prev.showResolvedIssues }))}
            className={`btn btn-outline ${state.showResolvedIssues ? 'active' : ''}`}
          >
            {state.showResolvedIssues ? 'Hide' : 'Show'} Resolved
          </button>
        </div>
      </div>

      {/* Metrics Dashboard */}
      <div className="metrics-dashboard" id="metrics-dashboard">
        <div className="metric-card">
          <div className="metric-value">{metrics.totalIssues}</div>
          <div className="metric-label">Total Issues</div>
        </div>
        
        <div className="metric-card critical">
          <div className="metric-value">{metrics.criticalIssues}</div>
          <div className="metric-label">Critical Issues</div>
        </div>
        
        <div className="metric-card high-risk">
          <div className="metric-value">{metrics.highRiskIssues}</div>
          <div className="metric-label">High Risk</div>
        </div>
        
        <div className="metric-card compliance">
          <div className="metric-value">{metrics.complianceIssues}</div>
          <div className="metric-label">Compliance Issues</div>
        </div>
        
        <div className="metric-card score">
          <div className="metric-value">{100 - metrics.riskScore}%</div>
          <div className="metric-label">Safety Score</div>
        </div>
        
        <div className="metric-card compliance-score">
          <div className="metric-value">{metrics.complianceScore}%</div>
          <div className="metric-label">Compliance Score</div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="engine-tabs" id="engine-tabs">
        {[
          { key: 'overview', label: 'Overview', count: metrics.totalIssues },
          { key: 'risks', label: 'Risk Factors', count: assessment?.riskFactors.length || 0 },
          { key: 'compliance', label: 'Compliance', count: assessment?.complianceIssues.length || 0 },
          { key: 'recommendations', label: 'Recommendations', count: assessment?.recommendations.length || 0 },
          ...(showDetailedAnalysis ? [{ key: 'conflicts', label: 'Conflicts', count: 0 }] : []),
          { key: 'history', label: 'History', count: state.validationHistory.length }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setState(prev => ({ ...prev, activeTab: tab.key as any }))}
            className={`tab ${state.activeTab === tab.key ? 'active' : ''}`}
          >
            {tab.label}
            {tab.count > 0 && <span className="tab-count">({tab.count})</span>}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content" id="tab-content">
        {/* Overview Tab */}
        {state.activeTab === 'overview' && (
          <div className="overview-content">
            {/* Quick Summary */}
            <div className="quick-summary">
              <h4>Validation Summary</h4>
              {assessment ? (
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Assessment Date:</span>
                    <span className="summary-value">
                      {assessment.calculatedAt.toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Contract Sections:</span>
                    <span className="summary-value">
                      {contract?.structure.length || 0}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Total Clauses:</span>
                    <span className="summary-value">
                      {contract?.selectedClauses.length || 0}
                    </span>
                  </div>
                </div>
              ) : (
                <p>No assessment data available.</p>
              )}
            </div>

            {/* Critical Issues */}
            {metrics.criticalIssues > 0 && (
              <div className="critical-issues-section">
                <h4 className="section-title critical">
                  🚨 Critical Issues Requiring Immediate Attention
                </h4>
                <div className="critical-issues-list">
                  {assessment?.riskFactors
                    .filter(factor => factor.severity === RiskLevel.CRITICAL)
                    .slice(0, 3)
                    .map((factor, index) => (
                      <div key={index} className="critical-issue-item">
                        <div className="issue-icon">⚠️</div>
                        <div className="issue-content">
                          <h5>{factor.category}</h5>
                          <p>{factor.description}</p>
                          <div className="issue-actions">
                            {enableAutoFix && !readonly && (
                              <button
                                onClick={() => handleAutoFix(factor.category, 'auto-fix')}
                                className="btn btn-sm btn-primary"
                                disabled={state.autoFixInProgress.has(`${factor.category}-auto-fix`)}
                              >
                                {state.autoFixInProgress.has(`${factor.category}-auto-fix`) 
                                  ? 'Applying Fix...' 
                                  : 'Auto Fix'
                                }
                              </button>
                            )}
                            <button
                              onClick={() => setState(prev => ({ ...prev, activeTab: 'risks' }))}
                              className="btn btn-sm btn-outline"
                            >
                              View Details
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  
                  {assessment?.complianceIssues
                    .filter(issue => issue.severity === RiskLevel.CRITICAL)
                    .slice(0, 2)
                    .map((issue, index) => (
                      <div key={`compliance-${index}`} className="critical-issue-item">
                        <div className="issue-icon">📋</div>
                        <div className="issue-content">
                          <h5>{issue.regulation}</h5>
                          <p>{issue.description}</p>
                          <div className="issue-actions">
                            <button
                              onClick={() => setState(prev => ({ ...prev, activeTab: 'compliance' }))}
                              className="btn btn-sm btn-outline"
                            >
                              View Compliance
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* General Errors */}
            {errors.length > 0 && (
              <div className="general-errors-section">
                <h4 className="section-title">General Validation Errors</h4>
                <div className="errors-list">
                  {errors.map((error, index) => (
                    <div key={index} className="error-item">
                      <span className="error-icon">❌</span>
                      <span className="error-message">{error}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Risk Factors Tab */}
        {state.activeTab === 'risks' && (
          <div className="risks-content">
            <div className="risks-header">
              <div className="filter-controls">
                <label>Filter by Risk Level:</label>
                <select
                  value={state.filterLevel}
                  onChange={(e) => setState(prev => ({ 
                    ...prev, 
                    filterLevel: e.target.value as RiskLevel | 'ALL' 
                  }))}
                >
                  <option value="ALL">All Levels</option>
                  <option value={RiskLevel.CRITICAL}>Critical</option>
                  <option value={RiskLevel.HIGH}>High</option>
                  <option value={RiskLevel.MEDIUM}>Medium</option>
                  <option value={RiskLevel.LOW}>Low</option>
                </select>
              </div>

              <div className="sort-controls">
                <label>Sort by:</label>
                <select
                  value={state.sortBy}
                  onChange={(e) => setState(prev => ({ 
                    ...prev, 
                    sortBy: e.target.value as any 
                  }))}
                >
                  <option value="severity">Severity</option>
                  <option value="category">Category</option>
                  <option value="impact">Impact</option>
                  <option value="probability">Probability</option>
                </select>
              </div>
            </div>

            <div className="risk-factors-list">
              {filteredRiskFactors.map((factor, index) => (
                <RiskFactorCard
                  key={`${factor.category}-${index}`}
                  factor={factor}
                  onExpand={handleSectionExpand}
                  isExpanded={state.expandedSections.has(`risk-${factor.category}`)}
                  onApplyFix={enableAutoFix ? handleAutoFix : undefined}
                  readonly={readonly}
                />
              ))}

              {filteredRiskFactors.length === 0 && (
                <div className="empty-state">
                  <h4>No risk factors found</h4>
                  <p>Try adjusting your filter criteria.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Compliance Tab */}
        {state.activeTab === 'compliance' && (
          <div className="compliance-content">
            <div className="compliance-issues-list">
              {assessment?.complianceIssues.map((issue, index) => (
                <ComplianceIssueCard
                  key={`${issue.regulation}-${index}`}
                  issue={issue}
                  onResolve={onFixIssue}
                  readonly={readonly}
                />
              )) || (
                <div className="empty-state">
                  <h4>No compliance issues detected</h4>
                  <p>Your contract appears to meet all regulatory requirements.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Recommendations Tab */}
        {state.activeTab === 'recommendations' && (
          <div className="recommendations-content">
            <div className="recommendations-list">
              {assessment?.recommendations.map((recommendation, index) => (
                <div key={index} className="recommendation-card">
                  <div className="recommendation-header">
                    <span className={`priority-badge ${recommendation.priority}`}>
                      {recommendation.priority.toUpperCase()}
                    </span>
                    <h4>{recommendation.category}</h4>
                  </div>
                  
                  <div className="recommendation-content">
                    <p className="recommendation-text">{recommendation.recommendation}</p>
                    <p className="impact-description">{recommendation.impactDescription}</p>
                    
                    {recommendation.alternativeClauseIds && recommendation.alternativeClauseIds.length > 0 && (
                      <div className="alternative-clauses">
                        <span>Suggested alternatives: {recommendation.alternativeClauseIds.length} clauses</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="recommendation-actions">
                    <button className="btn btn-primary btn-sm">
                      Apply Recommendation
                    </button>
                    <button className="btn btn-outline btn-sm">
                      View Alternatives
                    </button>
                  </div>
                </div>
              )) || (
                <div className="empty-state">
                  <h4>No recommendations available</h4>
                  <p>Your contract validation did not generate any specific recommendations.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* History Tab */}
        {state.activeTab === 'history' && (
          <div className="history-content">
            <div className="history-list">
              {state.validationHistory.map(entry => (
                <div key={entry.id} className="history-entry">
                  <div className="entry-header">
                    <div className="entry-time">
                      {entry.timestamp.toLocaleString()}
                    </div>
                    <div className={`entry-risk ${getOverallRiskColor(entry.overallRisk)}`}>
                      {entry.overallRisk}
                    </div>
                  </div>
                  
                  <div className="entry-content">
                    <p>{entry.summary}</p>
                    <div className="entry-stats">
                      <span>Issues: {entry.issueCount}</span>
                      <span>Fixes Applied: {entry.fixesApplied}</span>
                    </div>
                  </div>
                </div>
              ))}

              {state.validationHistory.length === 0 && (
                <div className="empty-state">
                  <h4>No validation history</h4>
                  <p>Validation history will appear here as you run validations and apply fixes.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ValidationEngine;