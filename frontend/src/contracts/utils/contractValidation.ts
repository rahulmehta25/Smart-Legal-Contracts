/**
 * contractValidation.ts - Contract Validation Utilities
 * 
 * Comprehensive validation service for contract analysis, risk assessment,
 * and compliance checking.
 */

import {
  ContractDraft,
  RiskAssessment,
  RiskLevel,
  RiskFactor,
  RiskRecommendation,
  ComplianceIssue,
  Clause,
  ClauseCategory,
  GeneratedSection,
  VariableDefinition
} from '../types';

interface ValidationRule {
  id: string;
  name: string;
  category: ClauseCategory | 'GENERAL';
  severity: RiskLevel;
  description: string;
  check: (draft: ContractDraft) => ValidationResult;
}

interface ValidationResult {
  passed: boolean;
  riskFactors: RiskFactor[];
  recommendations: RiskRecommendation[];
  complianceIssues: ComplianceIssue[];
}

interface RiskCalculator {
  calculateClauseRisk: (clause: Clause) => number;
  calculateSectionRisk: (section: GeneratedSection) => number;
  calculateOverallRisk: (factors: RiskFactor[]) => RiskLevel;
  assessVariableRisks: (variables: Record<string, any>, definitions: VariableDefinition[]) => RiskFactor[];
}

// Risk assessment engine
class ContractRiskCalculator implements RiskCalculator {
  private static readonly RISK_WEIGHTS = {
    [RiskLevel.LOW]: 1,
    [RiskLevel.MEDIUM]: 2,
    [RiskLevel.HIGH]: 3,
    [RiskLevel.CRITICAL]: 4
  };

  private static readonly CATEGORY_RISK_MULTIPLIERS = {
    [ClauseCategory.LIABILITY]: 1.5,
    [ClauseCategory.PAYMENT]: 1.3,
    [ClauseCategory.TERMINATION]: 1.2,
    [ClauseCategory.DISPUTE_RESOLUTION]: 1.4,
    [ClauseCategory.INTELLECTUAL_PROPERTY]: 1.3,
    [ClauseCategory.CONFIDENTIALITY]: 1.1,
    [ClauseCategory.COMPLIANCE]: 1.4,
    [ClauseCategory.FORCE_MAJEURE]: 0.8,
    [ClauseCategory.GOVERNING_LAW]: 0.9,
    [ClauseCategory.GENERAL]: 1.0
  };

  calculateClauseRisk(clause: Clause): number {
    const baseRisk = ContractRiskCalculator.RISK_WEIGHTS[clause.riskLevel];
    const categoryMultiplier = ContractRiskCalculator.CATEGORY_RISK_MULTIPLIERS[clause.category] || 1.0;
    const usageSuccessRate = clause.metadata.successRate || 1.0;
    
    // Lower success rate increases risk
    const successRateAdjustment = (2 - usageSuccessRate);
    
    return baseRisk * categoryMultiplier * successRateAdjustment;
  }

  calculateSectionRisk(section: GeneratedSection): number {
    if (section.clauses.length === 0) return 0;
    
    const clauseRisks = section.clauses.map(clause => {
      // Mock clause data for calculation
      const mockClause: Clause = {
        id: clause.clauseId,
        title: clause.title,
        content: clause.content,
        category: this.inferClauseCategory(clause.title, clause.content),
        language: 'en',
        variables: [],
        riskLevel: this.inferRiskLevel(clause.content),
        version: '1.0',
        createdAt: new Date(),
        updatedAt: new Date(),
        isApproved: true,
        tags: [],
        metadata: {
          author: 'system',
          usageCount: 0,
          successRate: 0.9,
          averageNegotiationTime: 0,
          commonModifications: [],
          legalReferences: []
        }
      };
      
      return this.calculateClauseRisk(mockClause);
    });
    
    return clauseRisks.reduce((sum, risk) => sum + risk, 0) / clauseRisks.length;
  }

  calculateOverallRisk(factors: RiskFactor[]): RiskLevel {
    if (factors.length === 0) return RiskLevel.LOW;
    
    const weightedScore = factors.reduce((sum, factor) => {
      const severityWeight = ContractRiskCalculator.RISK_WEIGHTS[factor.severity];
      const impactProbabilityScore = (factor.impact * factor.probability) / 100;
      return sum + (severityWeight * impactProbabilityScore);
    }, 0);
    
    const averageScore = weightedScore / factors.length;
    
    if (averageScore >= 3.0) return RiskLevel.CRITICAL;
    if (averageScore >= 2.0) return RiskLevel.HIGH;
    if (averageScore >= 1.0) return RiskLevel.MEDIUM;
    return RiskLevel.LOW;
  }

  assessVariableRisks(variables: Record<string, any>, definitions: VariableDefinition[]): RiskFactor[] {
    const risks: RiskFactor[] = [];
    
    definitions.forEach(def => {
      const value = variables[def.name];
      
      if (def.required && (value === undefined || value === null || value === '')) {
        risks.push({
          category: 'Missing Required Information',
          description: `Required variable "${def.name}" is not defined`,
          severity: RiskLevel.HIGH,
          impact: 8,
          probability: 10,
          mitigationSuggestions: [
            `Provide a value for ${def.name}`,
            'Review all required contract variables',
            'Use template defaults where appropriate'
          ]
        });
      }
      
      // Type-specific risk assessment
      if (value !== undefined && value !== null) {
        switch (def.type) {
          case 'currency':
            if (typeof value === 'number' && value > 1000000) {
              risks.push({
                category: 'High Financial Exposure',
                description: `High contract value: ${value}`,
                severity: RiskLevel.MEDIUM,
                impact: 7,
                probability: 5,
                mitigationSuggestions: [
                  'Consider liability caps',
                  'Add insurance requirements',
                  'Include payment milestones'
                ]
              });
            }
            break;
            
          case 'date':
            const dateValue = new Date(value);
            const now = new Date();
            const diffDays = (dateValue.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
            
            if (diffDays < 0) {
              risks.push({
                category: 'Past Date Reference',
                description: `Date variable "${def.name}" is in the past`,
                severity: RiskLevel.MEDIUM,
                impact: 6,
                probability: 10,
                mitigationSuggestions: [
                  'Update to current or future date',
                  'Verify all date references',
                  'Consider effective date implications'
                ]
              });
            }
            break;
        }
      }
    });
    
    return risks;
  }

  private inferClauseCategory(title: string, content: string): ClauseCategory {
    const titleLower = title.toLowerCase();
    const contentLower = content.toLowerCase();
    
    if (titleLower.includes('payment') || contentLower.includes('payment') || contentLower.includes('invoice')) {
      return ClauseCategory.PAYMENT;
    }
    if (titleLower.includes('liability') || contentLower.includes('liable') || contentLower.includes('damages')) {
      return ClauseCategory.LIABILITY;
    }
    if (titleLower.includes('termination') || contentLower.includes('terminate') || contentLower.includes('expire')) {
      return ClauseCategory.TERMINATION;
    }
    if (titleLower.includes('confidential') || contentLower.includes('confidential') || contentLower.includes('proprietary')) {
      return ClauseCategory.CONFIDENTIALITY;
    }
    if (titleLower.includes('intellectual') || contentLower.includes('copyright') || contentLower.includes('patent')) {
      return ClauseCategory.INTELLECTUAL_PROPERTY;
    }
    if (titleLower.includes('dispute') || contentLower.includes('arbitration') || contentLower.includes('litigation')) {
      return ClauseCategory.DISPUTE_RESOLUTION;
    }
    if (titleLower.includes('force majeure') || contentLower.includes('force majeure') || contentLower.includes('act of god')) {
      return ClauseCategory.FORCE_MAJEURE;
    }
    if (titleLower.includes('governing') || contentLower.includes('governing law') || contentLower.includes('jurisdiction')) {
      return ClauseCategory.GOVERNING_LAW;
    }
    
    return ClauseCategory.GENERAL;
  }

  private inferRiskLevel(content: string): RiskLevel {
    const contentLower = content.toLowerCase();
    
    // High-risk indicators
    if (contentLower.includes('unlimited liability') || 
        contentLower.includes('no limitation') ||
        contentLower.includes('liquidated damages')) {
      return RiskLevel.HIGH;
    }
    
    // Medium-risk indicators
    if (contentLower.includes('material breach') ||
        contentLower.includes('indemnify') ||
        contentLower.includes('consequential damages')) {
      return RiskLevel.MEDIUM;
    }
    
    // Generally low risk
    return RiskLevel.LOW;
  }
}

// Validation rules engine
class ValidationRulesEngine {
  private static readonly VALIDATION_RULES: ValidationRule[] = [
    {
      id: 'missing-liability-cap',
      name: 'Missing Liability Limitation',
      category: ClauseCategory.LIABILITY,
      severity: RiskLevel.HIGH,
      description: 'Contract should include liability limitation clauses',
      check: (draft: ContractDraft): ValidationResult => {
        const hasLiabilityCap = draft.structure.some(section =>
          section.clauses.some(clause =>
            clause.content.toLowerCase().includes('limitation of liability') ||
            clause.content.toLowerCase().includes('liability cap') ||
            clause.title.toLowerCase().includes('liability')
          )
        );

        if (!hasLiabilityCap) {
          return {
            passed: false,
            riskFactors: [{
              category: 'Liability Exposure',
              description: 'No liability limitation clauses found',
              severity: RiskLevel.HIGH,
              impact: 9,
              probability: 7,
              mitigationSuggestions: [
                'Add liability limitation clause',
                'Consider mutual liability caps',
                'Exclude consequential damages',
                'Add insurance requirements'
              ]
            }],
            recommendations: [{
              priority: 'high',
              category: 'risk-mitigation',
              recommendation: 'Add comprehensive liability limitation clauses',
              impactDescription: 'Protects against excessive financial exposure in case of disputes'
            }],
            complianceIssues: []
          };
        }

        return { passed: true, riskFactors: [], recommendations: [], complianceIssues: [] };
      }
    },
    
    {
      id: 'payment-terms-validation',
      name: 'Payment Terms Analysis',
      category: ClauseCategory.PAYMENT,
      severity: RiskLevel.MEDIUM,
      description: 'Validates payment terms for business risk',
      check: (draft: ContractDraft): ValidationResult => {
        const paymentDays = draft.variables['payment-days'] || draft.variables['payment-terms'];
        const results: ValidationResult = { passed: true, riskFactors: [], recommendations: [], complianceIssues: [] };

        if (typeof paymentDays === 'number' && paymentDays > 60) {
          results.passed = false;
          results.riskFactors.push({
            category: 'Cash Flow Risk',
            description: `Extended payment terms: ${paymentDays} days`,
            severity: RiskLevel.MEDIUM,
            impact: 6,
            probability: 8,
            mitigationSuggestions: [
              'Negotiate shorter payment terms',
              'Add early payment discounts',
              'Include late payment penalties',
              'Request partial payments upfront'
            ]
          });

          results.recommendations.push({
            priority: 'medium',
            category: 'cash-flow',
            recommendation: 'Consider reducing payment terms to improve cash flow',
            impactDescription: 'Shorter payment terms reduce financial risk and improve cash flow'
          });
        }

        return results;
      }
    },

    {
      id: 'termination-clause-check',
      name: 'Termination Provisions',
      category: ClauseCategory.TERMINATION,
      severity: RiskLevel.MEDIUM,
      description: 'Ensures proper termination clauses are included',
      check: (draft: ContractDraft): ValidationResult => {
        const hasTerminationClause = draft.structure.some(section =>
          section.clauses.some(clause =>
            clause.content.toLowerCase().includes('termination') ||
            clause.content.toLowerCase().includes('terminate') ||
            clause.title.toLowerCase().includes('termination')
          )
        );

        if (!hasTerminationClause) {
          return {
            passed: false,
            riskFactors: [{
              category: 'Contract Management',
              description: 'No termination provisions found',
              severity: RiskLevel.MEDIUM,
              impact: 6,
              probability: 9,
              mitigationSuggestions: [
                'Add termination for convenience clause',
                'Include termination for cause provisions',
                'Specify notice requirements',
                'Define post-termination obligations'
              ]
            }],
            recommendations: [{
              priority: 'medium',
              category: 'contract-management',
              recommendation: 'Add comprehensive termination provisions',
              impactDescription: 'Provides flexibility to exit the contract when needed'
            }],
            complianceIssues: []
          };
        }

        return { passed: true, riskFactors: [], recommendations: [], complianceIssues: [] };
      }
    },

    {
      id: 'gdpr-compliance-check',
      name: 'GDPR Compliance',
      category: ClauseCategory.COMPLIANCE,
      severity: RiskLevel.HIGH,
      description: 'Checks for GDPR compliance requirements',
      check: (draft: ContractDraft): ValidationResult => {
        const hasEUParties = draft.parties.some(party => 
          party.contact.address.country === 'EU' || 
          party.jurisdiction === 'EU'
        );

        const hasDataProcessing = draft.structure.some(section =>
          section.clauses.some(clause =>
            clause.content.toLowerCase().includes('data processing') ||
            clause.content.toLowerCase().includes('personal data') ||
            clause.content.toLowerCase().includes('gdpr')
          )
        );

        if (hasEUParties && !hasDataProcessing) {
          return {
            passed: false,
            riskFactors: [{
              category: 'Regulatory Compliance',
              description: 'EU parties detected but no GDPR provisions found',
              severity: RiskLevel.HIGH,
              impact: 8,
              probability: 10,
              mitigationSuggestions: [
                'Add GDPR compliance clause',
                'Include data processing agreement',
                'Specify lawful basis for processing',
                'Add data subject rights provisions'
              ]
            }],
            recommendations: [{
              priority: 'high',
              category: 'compliance',
              recommendation: 'Add GDPR compliance provisions for EU data processing',
              impactDescription: 'Ensures compliance with EU data protection regulations'
            }],
            complianceIssues: [{
              regulation: 'GDPR',
              jurisdiction: 'EU',
              severity: RiskLevel.HIGH,
              description: 'Contract involves EU parties but lacks GDPR compliance provisions',
              requiredActions: [
                'Add data processing agreement',
                'Specify purpose and legal basis',
                'Include data subject rights',
                'Add breach notification procedures'
              ],
              deadlines: [new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)] // 30 days
            }]
          };
        }

        return { passed: true, riskFactors: [], recommendations: [], complianceIssues: [] };
      }
    },

    {
      id: 'intellectual-property-check',
      name: 'Intellectual Property Protection',
      category: ClauseCategory.INTELLECTUAL_PROPERTY,
      severity: RiskLevel.MEDIUM,
      description: 'Validates IP ownership and protection clauses',
      check: (draft: ContractDraft): ValidationResult => {
        const hasIPClauses = draft.structure.some(section =>
          section.clauses.some(clause =>
            clause.content.toLowerCase().includes('intellectual property') ||
            clause.content.toLowerCase().includes('copyright') ||
            clause.content.toLowerCase().includes('patent') ||
            clause.content.toLowerCase().includes('trademark') ||
            clause.title.toLowerCase().includes('intellectual property')
          )
        );

        const isTechContract = draft.variables['industry'] === 'TECHNOLOGY' ||
          draft.structure.some(section => 
            section.title.toLowerCase().includes('software') ||
            section.title.toLowerCase().includes('development')
          );

        if (isTechContract && !hasIPClauses) {
          return {
            passed: false,
            riskFactors: [{
              category: 'Intellectual Property Risk',
              description: 'Technology contract lacks IP protection clauses',
              severity: RiskLevel.MEDIUM,
              impact: 7,
              probability: 8,
              mitigationSuggestions: [
                'Add IP ownership clause',
                'Include work-for-hire provisions',
                'Specify license terms',
                'Add IP indemnification'
              ]
            }],
            recommendations: [{
              priority: 'medium',
              category: 'ip-protection',
              recommendation: 'Add intellectual property ownership and protection clauses',
              impactDescription: 'Protects intellectual property rights and prevents disputes'
            }],
            complianceIssues: []
          };
        }

        return { passed: true, riskFactors: [], recommendations: [], complianceIssues: [] };
      }
    }
  ];

  static runAllValidations(draft: ContractDraft): ValidationResult {
    const allResults = this.VALIDATION_RULES.map(rule => rule.check(draft));
    
    return {
      passed: allResults.every(result => result.passed),
      riskFactors: allResults.flatMap(result => result.riskFactors),
      recommendations: allResults.flatMap(result => result.recommendations),
      complianceIssues: allResults.flatMap(result => result.complianceIssues)
    };
  }

  static runSpecificValidation(draft: ContractDraft, ruleId: string): ValidationResult | null {
    const rule = this.VALIDATION_RULES.find(r => r.id === ruleId);
    return rule ? rule.check(draft) : null;
  }

  static getAvailableRules(): ValidationRule[] {
    return [...this.VALIDATION_RULES];
  }
}

// Main validation service
export class ContractValidationService {
  private riskCalculator: ContractRiskCalculator;

  constructor() {
    this.riskCalculator = new ContractRiskCalculator();
  }

  async validateContract(draft: ContractDraft): Promise<RiskAssessment> {
    // Run all validation rules
    const validationResults = ValidationRulesEngine.runAllValidations(draft);
    
    // Assess variable risks
    const variableDefinitions = this.extractVariableDefinitions(draft);
    const variableRisks = this.riskCalculator.assessVariableRisks(draft.variables, variableDefinitions);
    
    // Combine all risk factors
    const allRiskFactors = [...validationResults.riskFactors, ...variableRisks];
    
    // Calculate overall risk
    const overallRisk = this.riskCalculator.calculateOverallRisk(allRiskFactors);
    
    // Generate additional recommendations based on contract analysis
    const additionalRecommendations = this.generateContextualRecommendations(draft, allRiskFactors);
    
    return {
      overallRisk,
      riskFactors: allRiskFactors,
      recommendations: [...validationResults.recommendations, ...additionalRecommendations],
      complianceIssues: validationResults.complianceIssues,
      calculatedAt: new Date()
    };
  }

  private extractVariableDefinitions(draft: ContractDraft): VariableDefinition[] {
    // In a real implementation, this would extract variable definitions from template or clause metadata
    return [
      {
        id: 'payment-days',
        name: 'payment-days',
        type: 'number',
        required: true,
        validation: [
          { type: 'range', value: { min: 1, max: 120 }, message: 'Payment terms should be between 1 and 120 days' }
        ]
      },
      {
        id: 'contract-value',
        name: 'contract-value',
        type: 'currency',
        required: false
      },
      {
        id: 'effective-date',
        name: 'effective-date',
        type: 'date',
        required: true
      }
    ];
  }

  private generateContextualRecommendations(draft: ContractDraft, riskFactors: RiskFactor[]): RiskRecommendation[] {
    const recommendations: RiskRecommendation[] = [];
    
    // High-value contract recommendations
    const contractValue = draft.variables['contract-value'];
    if (contractValue && contractValue > 100000) {
      recommendations.push({
        priority: 'high',
        category: 'risk-management',
        recommendation: 'Consider additional risk management measures for high-value contract',
        impactDescription: 'High-value contracts require enhanced risk mitigation strategies',
        alternativeClauseIds: ['insurance-clause', 'escrow-clause', 'milestone-payments']
      });
    }
    
    // Multi-jurisdiction recommendations
    const jurisdictions = new Set(draft.parties.flatMap(party => party.jurisdiction ? [party.jurisdiction] : []));
    if (jurisdictions.size > 1) {
      recommendations.push({
        priority: 'medium',
        category: 'legal-complexity',
        recommendation: 'Clarify governing law and jurisdiction for multi-jurisdictional contract',
        impactDescription: 'Multiple jurisdictions can create legal complexity and enforcement challenges'
      });
    }
    
    // Critical risk factor recommendations
    const criticalRisks = riskFactors.filter(factor => factor.severity === RiskLevel.CRITICAL);
    if (criticalRisks.length > 0) {
      recommendations.push({
        priority: 'high',
        category: 'critical-risk',
        recommendation: 'Address all critical risk factors before contract execution',
        impactDescription: 'Critical risks can result in significant financial or legal exposure'
      });
    }
    
    return recommendations;
  }

  async runQuickValidation(draft: ContractDraft): Promise<{ riskLevel: RiskLevel; criticalIssues: number }> {
    const validationResults = ValidationRulesEngine.runAllValidations(draft);
    const criticalIssues = validationResults.riskFactors.filter(factor => 
      factor.severity === RiskLevel.CRITICAL || factor.severity === RiskLevel.HIGH
    ).length;
    
    const overallRisk = this.riskCalculator.calculateOverallRisk(validationResults.riskFactors);
    
    return {
      riskLevel: overallRisk,
      criticalIssues
    };
  }

  getValidationRules(): ValidationRule[] {
    return ValidationRulesEngine.getAvailableRules();
  }

  async validateSpecificAspect(draft: ContractDraft, aspect: string): Promise<ValidationResult | null> {
    return ValidationRulesEngine.runSpecificValidation(draft, aspect);
  }
}

// Export the validation service
export const contractValidationService = new ContractValidationService();