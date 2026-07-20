/**
 * useClauseLibrary.ts - Clause Library Hook
 * 
 * Custom hook for managing clause library operations, search, and CRUD functionality.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  Clause,
  ClauseCategory,
  RiskLevel,
  IndustryType,
  LanguageCode,
  SearchFilters,
  UseClauseLibraryReturn,
  UUID
} from '../contracts/types';

interface ClauseLibraryAPI {
  searchClauses: (filters: SearchFilters) => Promise<Clause[]>;
  getClause: (id: UUID) => Promise<Clause>;
  addClause: (clause: Omit<Clause, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  updateClause: (id: UUID, updates: Partial<Clause>) => Promise<void>;
  deleteClause: (id: UUID) => Promise<void>;
  getAlternatives: (clauseId: UUID) => Promise<Clause[]>;
}

// Mock clause library data
const mockClauses: Clause[] = [
  {
    id: 'clause-1',
    title: 'Service Level Agreement',
    content: 'Provider shall maintain a service level of ninety-nine point nine percent (99.9%) uptime for all Services during each calendar month.',
    category: ClauseCategory.GENERAL,
    subcategory: 'Performance Standards',
    language: 'en',
    variables: [
      {
        id: 'sla-percentage',
        name: 'SLA Percentage',
        type: 'percentage',
        required: true,
        defaultValue: 99.9,
        validation: [
          { type: 'range', value: { min: 90, max: 100 }, message: 'SLA must be between 90% and 100%' }
        ]
      }
    ],
    riskLevel: RiskLevel.MEDIUM,
    version: '1.2',
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-03-10'),
    isApproved: true,
    approvedBy: 'legal-team-1',
    approvedAt: new Date('2024-01-20'),
    tags: ['sla', 'performance', 'uptime', 'service-level'],
    metadata: {
      author: 'legal-team-1',
      reviewedBy: ['senior-counsel-1', 'risk-manager-1'],
      usageCount: 156,
      successRate: 0.94,
      averageNegotiationTime: 2.5,
      commonModifications: [
        'Adjusted SLA percentage',
        'Added service credits for downtime',
        'Modified measurement period'
      ],
      legalReferences: ['Contract Law § 2.4', 'Industry Standard SLA-001']
    },
    jurisdiction: ['US', 'CA', 'UK'],
    industrySpecific: [IndustryType.TECHNOLOGY]
  },
  {
    id: 'clause-2',
    title: 'Payment Terms - Net 30',
    content: 'Payment for Services shall be due within thirty (30) days of the invoice date. Late payments shall incur a charge of one and one-half percent (1.5%) per month.',
    category: ClauseCategory.PAYMENT,
    language: 'en',
    variables: [
      {
        id: 'payment-days',
        name: 'Payment Days',
        type: 'number',
        required: true,
        defaultValue: 30,
        validation: [
          { type: 'range', value: { min: 1, max: 120 }, message: 'Payment terms must be between 1 and 120 days' }
        ]
      },
      {
        id: 'late-fee-rate',
        name: 'Late Fee Rate',
        type: 'percentage',
        required: false,
        defaultValue: 1.5,
        validation: [
          { type: 'range', value: { min: 0, max: 10 }, message: 'Late fee rate must be between 0% and 10%' }
        ]
      }
    ],
    riskLevel: RiskLevel.LOW,
    version: '2.1',
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date('2024-02-15'),
    isApproved: true,
    approvedBy: 'finance-team-1',
    approvedAt: new Date('2024-02-05'),
    tags: ['payment', 'net-30', 'late-fees', 'invoicing'],
    metadata: {
      author: 'finance-team-1',
      reviewedBy: ['legal-team-1'],
      usageCount: 203,
      successRate: 0.98,
      averageNegotiationTime: 1.2,
      commonModifications: [
        'Changed payment period',
        'Adjusted late fee percentage',
        'Added early payment discount'
      ],
      legalReferences: ['Commercial Code § 3.2']
    },
    jurisdiction: ['US'],
    industrySpecific: []
  },
  {
    id: 'clause-3',
    title: 'Limitation of Liability',
    content: 'IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION, LOSS OF PROFITS, DATA, USE, GOODWILL, OR OTHER INTANGIBLE LOSSES.',
    category: ClauseCategory.LIABILITY,
    language: 'en',
    variables: [
      {
        id: 'liability-cap-amount',
        name: 'Liability Cap Amount',
        type: 'currency',
        required: false,
        description: 'Maximum liability amount (leave empty for no cap)'
      }
    ],
    riskLevel: RiskLevel.HIGH,
    version: '1.0',
    createdAt: new Date('2024-01-10'),
    updatedAt: new Date('2024-01-10'),
    isApproved: true,
    approvedBy: 'senior-counsel-1',
    approvedAt: new Date('2024-01-12'),
    tags: ['liability', 'limitation', 'damages', 'risk-mitigation'],
    metadata: {
      author: 'senior-counsel-1',
      reviewedBy: ['risk-manager-1', 'insurance-team-1'],
      usageCount: 89,
      successRate: 0.76,
      averageNegotiationTime: 4.8,
      commonModifications: [
        'Added liability cap amount',
        'Excluded certain types of damages',
        'Modified to mutual limitation'
      ],
      legalReferences: ['Tort Law § 5.1', 'Contract Damages Rule 102']
    },
    jurisdiction: ['US', 'UK'],
    industrySpecific: [IndustryType.TECHNOLOGY, IndustryType.FINANCE]
  },
  {
    id: 'clause-4',
    title: 'Confidentiality',
    content: 'Each party acknowledges that it may have access to Confidential Information of the other party. Each party agrees to maintain in confidence all Confidential Information received from the other party.',
    category: ClauseCategory.CONFIDENTIALITY,
    language: 'en',
    variables: [
      {
        id: 'confidentiality-period',
        name: 'Confidentiality Period (years)',
        type: 'number',
        required: true,
        defaultValue: 5,
        validation: [
          { type: 'range', value: { min: 1, max: 10 }, message: 'Confidentiality period must be between 1 and 10 years' }
        ]
      }
    ],
    riskLevel: RiskLevel.MEDIUM,
    version: '1.3',
    createdAt: new Date('2024-01-05'),
    updatedAt: new Date('2024-02-20'),
    isApproved: true,
    approvedBy: 'legal-team-1',
    approvedAt: new Date('2024-01-08'),
    tags: ['confidentiality', 'nda', 'information-protection', 'trade-secrets'],
    metadata: {
      author: 'legal-team-1',
      reviewedBy: ['senior-counsel-1'],
      usageCount: 178,
      successRate: 0.91,
      averageNegotiationTime: 2.1,
      commonModifications: [
        'Adjusted confidentiality period',
        'Added specific exceptions',
        'Modified definition of Confidential Information'
      ],
      legalReferences: ['Trade Secrets Act § 1.1', 'Privacy Law § 4.2']
    },
    jurisdiction: ['US', 'EU'],
    industrySpecific: []
  },
  {
    id: 'clause-5',
    title: 'Force Majeure',
    content: 'Neither party shall be liable for any delay or failure to perform its obligations under this Agreement if such delay or failure results from circumstances beyond its reasonable control.',
    category: ClauseCategory.FORCE_MAJEURE,
    language: 'en',
    variables: [
      {
        id: 'notice-period-days',
        name: 'Notice Period (days)',
        type: 'number',
        required: true,
        defaultValue: 10,
        validation: [
          { type: 'range', value: { min: 1, max: 30 }, message: 'Notice period must be between 1 and 30 days' }
        ]
      }
    ],
    riskLevel: RiskLevel.LOW,
    version: '2.0',
    createdAt: new Date('2024-03-01'),
    updatedAt: new Date('2024-03-01'),
    isApproved: true,
    approvedBy: 'legal-team-1',
    approvedAt: new Date('2024-03-03'),
    tags: ['force-majeure', 'acts-of-god', 'unforeseeable-events', 'performance-excuse'],
    metadata: {
      author: 'legal-team-1',
      reviewedBy: ['senior-counsel-1'],
      usageCount: 67,
      successRate: 0.89,
      averageNegotiationTime: 1.8,
      commonModifications: [
        'Added pandemic-specific language',
        'Modified notice requirements',
        'Expanded definition of force majeure events'
      ],
      legalReferences: ['Contract Law § 7.3', 'Force Majeure Doctrine']
    },
    jurisdiction: ['US', 'UK', 'CA'],
    industrySpecific: []
  }
];

// Mock API implementation
const mockAPI: ClauseLibraryAPI = {
  searchClauses: async (filters: SearchFilters): Promise<Clause[]> => {
    await new Promise(resolve => setTimeout(resolve, 800));
    
    let filtered = [...mockClauses];
    
    // Apply keyword filter
    if (filters.keywords && filters.keywords.length > 0) {
      const keywords = filters.keywords.map(k => k.toLowerCase());
      filtered = filtered.filter(clause =>
        keywords.some(keyword =>
          clause.title.toLowerCase().includes(keyword) ||
          clause.content.toLowerCase().includes(keyword) ||
          clause.tags.some(tag => tag.toLowerCase().includes(keyword))
        )
      );
    }
    
    // Apply category filter
    if (filters.categories && filters.categories.length > 0) {
      filtered = filtered.filter(clause => filters.categories!.includes(clause.category));
    }
    
    // Apply risk level filter
    if (filters.riskLevels && filters.riskLevels.length > 0) {
      filtered = filtered.filter(clause => filters.riskLevels!.includes(clause.riskLevel));
    }
    
    // Apply industry filter
    if (filters.industries && filters.industries.length > 0) {
      filtered = filtered.filter(clause =>
        !clause.industrySpecific ||
        clause.industrySpecific.length === 0 ||
        clause.industrySpecific.some(industry => filters.industries!.includes(industry))
      );
    }
    
    // Apply language filter
    if (filters.languages && filters.languages.length > 0) {
      filtered = filtered.filter(clause => filters.languages!.includes(clause.language));
    }
    
    // Apply jurisdiction filter
    if (filters.jurisdiction && filters.jurisdiction.length > 0) {
      filtered = filtered.filter(clause =>
        !clause.jurisdiction ||
        clause.jurisdiction.some(j => filters.jurisdiction!.includes(j))
      );
    }
    
    // Apply approval status filter
    if (filters.approvalStatus !== undefined) {
      filtered = filtered.filter(clause => clause.isApproved === filters.approvalStatus);
    }
    
    return filtered;
  },

  getClause: async (id: UUID): Promise<Clause> => {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const clause = mockClauses.find(c => c.id === id);
    if (!clause) {
      throw new Error(`Clause with id ${id} not found`);
    }
    
    return clause;
  },

  addClause: async (clause: Omit<Clause, 'id' | 'createdAt' | 'updatedAt'>): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const newClause: Clause = {
      ...clause,
      id: `clause-${Date.now()}`,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    mockClauses.push(newClause);
    console.log('Added new clause:', newClause);
  },

  updateClause: async (id: UUID, updates: Partial<Clause>): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const index = mockClauses.findIndex(c => c.id === id);
    if (index === -1) {
      throw new Error(`Clause with id ${id} not found`);
    }
    
    mockClauses[index] = {
      ...mockClauses[index],
      ...updates,
      updatedAt: new Date()
    };
    
    console.log('Updated clause:', mockClauses[index]);
  },

  deleteClause: async (id: UUID): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const index = mockClauses.findIndex(c => c.id === id);
    if (index === -1) {
      throw new Error(`Clause with id ${id} not found`);
    }
    
    mockClauses.splice(index, 1);
    console.log('Deleted clause with id:', id);
  },

  getAlternatives: async (clauseId: UUID): Promise<Clause[]> => {
    await new Promise(resolve => setTimeout(resolve, 600));
    
    const originalClause = mockClauses.find(c => c.id === clauseId);
    if (!originalClause) {
      return [];
    }
    
    // Find clauses with same category but different risk levels or content
    return mockClauses.filter(clause =>
      clause.id !== clauseId &&
      clause.category === originalClause.category &&
      (clause.riskLevel !== originalClause.riskLevel || 
       clause.subcategory !== originalClause.subcategory)
    );
  }
};

export const useClauseLibrary = (): UseClauseLibraryReturn => {
  const [clauses, setClauses] = useState<Clause[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial clauses
  useEffect(() => {
    const loadInitialClauses = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const initialClauses = await mockAPI.searchClauses({});
        setClauses(initialClauses);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load clauses';
        setError(errorMessage);
        console.error('Error loading initial clauses:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialClauses();
  }, []);

  // Search clauses with filters
  const searchClauses = useCallback(async (filters: SearchFilters) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const results = await mockAPI.searchClauses(filters);
      setClauses(results);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to search clauses';
      setError(errorMessage);
      console.error('Error searching clauses:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Add new clause
  const addClause = useCallback(async (clause: Omit<Clause, 'id' | 'createdAt' | 'updatedAt'>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.addClause(clause);
      // Refresh clauses list
      const updatedClauses = await mockAPI.searchClauses({});
      setClauses(updatedClauses);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to add clause';
      setError(errorMessage);
      console.error('Error adding clause:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update existing clause
  const updateClause = useCallback(async (id: UUID, updates: Partial<Clause>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.updateClause(id, updates);
      // Update clause in local state
      setClauses(prev => prev.map(clause =>
        clause.id === id ? { ...clause, ...updates, updatedAt: new Date() } : clause
      ));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update clause';
      setError(errorMessage);
      console.error('Error updating clause:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Delete clause
  const deleteClause = useCallback(async (id: UUID) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.deleteClause(id);
      // Remove clause from local state
      setClauses(prev => prev.filter(clause => clause.id !== id));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete clause';
      setError(errorMessage);
      console.error('Error deleting clause:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Get alternative clauses
  const getAlternatives = useCallback(async (clauseId: UUID): Promise<Clause[]> => {
    setError(null);
    
    try {
      const alternatives = await mockAPI.getAlternatives(clauseId);
      return alternatives;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get alternatives';
      setError(errorMessage);
      console.error('Error getting alternatives:', err);
      return [];
    }
  }, []);

  return {
    clauses,
    isLoading,
    error,
    searchClauses,
    addClause,
    updateClause,
    deleteClause,
    getAlternatives
  };
};