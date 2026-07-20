/**
 * useContractBuilder.ts - Contract Builder Hook
 * 
 * Custom hook for managing contract building state, operations, and API interactions.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  ContractDraft,
  RiskAssessment,
  ExportConfiguration,
  UseContractBuilderReturn,
  UUID
} from '../contracts/types';

interface ContractBuilderAPI {
  getDraft: (id: UUID) => Promise<ContractDraft>;
  saveDraft: (draft: Partial<ContractDraft>) => Promise<void>;
  validateContract: (draft: ContractDraft) => Promise<RiskAssessment>;
  generateContract: (draft: ContractDraft) => Promise<string>;
  exportContract: (draft: ContractDraft, config: ExportConfiguration) => Promise<Blob>;
}

// Mock API implementation
const mockAPI: ContractBuilderAPI = {
  getDraft: async (id: UUID): Promise<ContractDraft> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const mockDraft: ContractDraft = {
      id,
      templateId: 'template-1',
      name: 'Service Agreement Draft',
      parties: [
        {
          id: 'party-1',
          name: 'Acme Corporation',
          type: 'company',
          role: 'Service Provider',
          contact: {
            address: {
              street: '123 Business St',
              city: 'San Francisco',
              state: 'CA',
              postalCode: '94105',
              country: 'US'
            },
            email: 'legal@acme.com',
            phone: '+1-555-0123'
          },
          signatoryInfo: {
            name: 'John Smith',
            title: 'CEO',
            email: 'john.smith@acme.com',
            signatureRequired: true,
            signatureType: 'digital'
          }
        },
        {
          id: 'party-2',
          name: 'Client Corp',
          type: 'company',
          role: 'Client',
          contact: {
            address: {
              street: '456 Client Ave',
              city: 'New York',
              state: 'NY',
              postalCode: '10001',
              country: 'US'
            },
            email: 'legal@client.com',
            phone: '+1-555-0456'
          },
          signatoryInfo: {
            name: 'Jane Doe',
            title: 'Legal Director',
            email: 'jane.doe@client.com',
            signatureRequired: true,
            signatureType: 'digital'
          }
        }
      ],
      variables: {
        'contract-title': 'Professional Services Agreement',
        'effective-date': new Date().toISOString(),
        'contract-value': 50000,
        'payment-terms': '30 days',
        'governing-law': 'California'
      },
      selectedClauses: ['clause-1', 'clause-2', 'clause-3'],
      customClauses: [],
      structure: [
        {
          id: 'section-1',
          title: 'Definitions',
          order: 1,
          content: 'Terms and definitions used throughout this agreement.',
          clauses: [
            {
              id: 'clause-def-1',
              clauseId: 'clause-1',
              title: 'Services',
              content: 'Services means the professional services to be provided by Provider as described in Exhibit A.',
              order: 1,
              numbering: '1.1',
              variables: {},
              isCustom: false,
              modifications: []
            }
          ],
          numbering: '1'
        },
        {
          id: 'section-2',
          title: 'Services',
          order: 2,
          content: 'Description of services to be provided.',
          clauses: [
            {
              id: 'clause-svc-1',
              clauseId: 'clause-2',
              title: 'Service Description',
              content: 'Provider shall provide the Services described in Exhibit A in accordance with the terms of this Agreement.',
              order: 1,
              numbering: '2.1',
              variables: {},
              isCustom: false,
              modifications: []
            }
          ],
          numbering: '2'
        }
      ],
      status: 'DRAFT' as const,
      version: 1,
      language: 'en',
      jurisdiction: 'US-CA',
      createdBy: 'user-1',
      createdAt: new Date(Date.now() - 86400000), // 1 day ago
      updatedAt: new Date(),
      metadata: {
        totalClauses: 3,
        riskScore: 25,
        complianceScore: 85,
        estimatedValue: 50000,
        currency: 'USD'
      }
    };
    
    return mockDraft;
  },

  saveDraft: async (draft: Partial<ContractDraft>): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log('Saving draft:', draft);
  },

  validateContract: async (draft: ContractDraft): Promise<RiskAssessment> => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return {
      overallRisk: 'MEDIUM' as const,
      riskFactors: [
        {
          category: 'Payment Terms',
          description: 'Payment terms exceed recommended 30-day limit',
          severity: 'MEDIUM' as const,
          impact: 6,
          probability: 4,
          mitigationSuggestions: [
            'Consider reducing payment terms to 30 days',
            'Add late payment penalties',
            'Require partial payment upfront'
          ]
        },
        {
          category: 'Liability',
          description: 'No liability cap specified',
          severity: 'HIGH' as const,
          impact: 8,
          probability: 3,
          mitigationSuggestions: [
            'Add liability limitation clause',
            'Consider mutual indemnification',
            'Specify excluded damages'
          ]
        }
      ],
      recommendations: [
        {
          priority: 'high',
          category: 'risk-mitigation',
          recommendation: 'Add liability limitation clause to reduce financial exposure',
          impactDescription: 'This will significantly reduce potential liability in case of disputes',
          alternativeClauseIds: ['liability-cap-1', 'liability-cap-2']
        }
      ],
      complianceIssues: [
        {
          regulation: 'GDPR',
          jurisdiction: 'EU',
          severity: 'LOW' as const,
          description: 'Data processing terms should be more specific for EU clients',
          requiredActions: [
            'Add GDPR compliance clause',
            'Specify data processing purposes',
            'Include data subject rights'
          ]
        }
      ],
      calculatedAt: new Date()
    };
  },

  generateContract: async (draft: ContractDraft): Promise<string> => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    return `
# ${draft.variables['contract-title'] || draft.name}

**Effective Date:** ${draft.variables['effective-date'] || 'Not specified'}
**Parties:** ${draft.parties.map(p => p.name).join(' and ')}
**Governing Law:** ${draft.variables['governing-law'] || 'Not specified'}

${draft.structure.map(section => 
  `## ${section.numbering}. ${section.title}\n\n${section.clauses.map(clause => 
    `### ${clause.numbering} ${clause.title}\n\n${clause.content}`
  ).join('\n\n')}`
).join('\n\n')}
`;
  },

  exportContract: async (draft: ContractDraft, config: ExportConfiguration): Promise<Blob> => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const content = await mockAPI.generateContract(draft);
    
    switch (config.format) {
      case 'PDF':
        // In real implementation, this would generate a PDF
        return new Blob([content], { type: 'application/pdf' });
      case 'DOCX':
        // In real implementation, this would generate a DOCX
        return new Blob([content], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
      case 'HTML':
        const htmlContent = content.replace(/\n/g, '<br>').replace(/^# (.+)$/gm, '<h1>$1</h1>').replace(/^## (.+)$/gm, '<h2>$1</h2>');
        return new Blob([htmlContent], { type: 'text/html' });
      default:
        return new Blob([content], { type: 'text/plain' });
    }
  }
};

export const useContractBuilder = (initialContractId?: UUID): UseContractBuilderReturn => {
  const [draft, setDraft] = useState<ContractDraft | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load draft on mount or when contractId changes
  const loadDraft = useCallback(async (contractId: UUID) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const loadedDraft = await mockAPI.getDraft(contractId);
      setDraft(loadedDraft);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load contract draft';
      setError(errorMessage);
      console.error('Error loading draft:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save draft changes
  const saveDraft = useCallback(async (updates: Partial<ContractDraft>) => {
    if (!draft) {
      setError('No draft loaded to save');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.saveDraft(updates);
      setDraft(prev => prev ? { ...prev, ...updates, updatedAt: new Date() } : null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save draft';
      setError(errorMessage);
      console.error('Error saving draft:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [draft]);

  // Generate contract content
  const generateContract = useCallback(async (): Promise<string> => {
    if (!draft) {
      throw new Error('No draft loaded to generate');
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const content = await mockAPI.generateContract(draft);
      return content;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate contract';
      setError(errorMessage);
      console.error('Error generating contract:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [draft]);

  // Validate contract
  const validateContract = useCallback(async (): Promise<RiskAssessment> => {
    if (!draft) {
      throw new Error('No draft loaded to validate');
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const assessment = await mockAPI.validateContract(draft);
      return assessment;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to validate contract';
      setError(errorMessage);
      console.error('Error validating contract:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [draft]);

  // Export contract
  const exportContract = useCallback(async (config: ExportConfiguration): Promise<Blob> => {
    if (!draft) {
      throw new Error('No draft loaded to export');
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const blob = await mockAPI.exportContract(draft, config);
      return blob;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to export contract';
      setError(errorMessage);
      console.error('Error exporting contract:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [draft]);

  // Load initial draft if provided
  useEffect(() => {
    if (initialContractId) {
      loadDraft(initialContractId);
    }
  }, [initialContractId, loadDraft]);

  // Auto-save functionality (optional)
  useEffect(() => {
    if (draft && draft.updatedAt) {
      const autoSaveInterval = setInterval(() => {
        // Auto-save every 30 seconds if there are unsaved changes
        const lastUpdate = new Date(draft.updatedAt).getTime();
        const now = new Date().getTime();
        const timeSinceUpdate = now - lastUpdate;
        
        if (timeSinceUpdate > 30000) { // 30 seconds
          saveDraft(draft).catch(console.error);
        }
      }, 30000);

      return () => clearInterval(autoSaveInterval);
    }
  }, [draft, saveDraft]);

  return {
    draft,
    isLoading,
    error,
    saveDraft,
    loadDraft,
    generateContract,
    validateContract,
    exportContract
  };
};