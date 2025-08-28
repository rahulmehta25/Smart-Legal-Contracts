/**
 * useTemplateEngine.ts - Template Engine Hook
 * 
 * Custom hook for managing contract templates, validation, and CRUD operations.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  ContractTemplate,
  IndustryType,
  LanguageCode,
  UseTemplateEngineReturn,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  UUID
} from '../contracts/types';

interface TemplateEngineAPI {
  getTemplates: () => Promise<ContractTemplate[]>;
  getTemplate: (id: UUID) => Promise<ContractTemplate>;
  saveTemplate: (template: Partial<ContractTemplate>) => Promise<void>;
  cloneTemplate: (id: UUID, newName: string) => Promise<UUID>;
  deleteTemplate: (id: UUID) => Promise<void>;
  validateTemplate: (template: ContractTemplate) => ValidationResult;
}

// Mock template data
const mockTemplates: ContractTemplate[] = [
  {
    id: 'template-1',
    name: 'Professional Services Agreement',
    description: 'Comprehensive template for professional services contracts with customizable terms.',
    industry: IndustryType.TECHNOLOGY,
    jurisdiction: ['US', 'CA'],
    language: 'en',
    version: '2.1',
    structure: [
      {
        id: 'section-1',
        title: 'Definitions',
        order: 1,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-1',
            isRequired: true,
            order: 1
          }
        ]
      },
      {
        id: 'section-2',
        title: 'Services',
        order: 2,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-2',
            isRequired: true,
            order: 1
          },
          {
            clauseId: 'clause-3',
            isRequired: false,
            order: 2
          }
        ]
      }
    ],
    globalVariables: [
      {
        id: 'contract-title',
        name: 'Contract Title',
        type: 'string',
        required: true,
        defaultValue: 'Professional Services Agreement'
      },
      {
        id: 'service-period',
        name: 'Service Period (months)',
        type: 'number',
        required: true,
        defaultValue: 12
      }
    ],
    conditionalLogic: [
      {
        id: 'condition-1',
        condition: 'service-period > 12',
        action: 'require',
        target: 'clause-renewal'
      }
    ],
    requiredClauses: ['clause-1', 'clause-2'],
    optionalClauses: ['clause-3', 'clause-4'],
    metadata: {
      usageCount: 245,
      averageCompletionTime: 1800, // 30 minutes
      successRate: 0.92,
      commonModifications: [
        'Adjusted payment terms',
        'Modified liability clauses',
        'Added confidentiality provisions'
      ],
      userRatings: [5, 4, 5, 4, 5, 3, 4, 5],
      averageRating: 4.4
    },
    isPublic: true,
    createdBy: 'template-team-1',
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-03-10')
  },
  {
    id: 'template-2',
    name: 'Non-Disclosure Agreement',
    description: 'Standard NDA template suitable for various business contexts.',
    industry: IndustryType.GENERIC,
    jurisdiction: ['US', 'UK', 'EU'],
    language: 'en',
    version: '1.5',
    structure: [
      {
        id: 'section-nda-1',
        title: 'Confidential Information',
        order: 1,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-conf-1',
            isRequired: true,
            order: 1
          }
        ]
      },
      {
        id: 'section-nda-2',
        title: 'Obligations',
        order: 2,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-conf-2',
            isRequired: true,
            order: 1
          }
        ]
      }
    ],
    globalVariables: [
      {
        id: 'confidentiality-period',
        name: 'Confidentiality Period (years)',
        type: 'number',
        required: true,
        defaultValue: 5
      },
      {
        id: 'mutual-nda',
        name: 'Mutual NDA',
        type: 'boolean',
        required: true,
        defaultValue: true
      }
    ],
    conditionalLogic: [],
    requiredClauses: ['clause-conf-1', 'clause-conf-2'],
    optionalClauses: ['clause-conf-3'],
    metadata: {
      usageCount: 189,
      averageCompletionTime: 900, // 15 minutes
      successRate: 0.96,
      commonModifications: [
        'Changed confidentiality period',
        'Modified definition of confidential information',
        'Added specific exceptions'
      ],
      userRatings: [5, 5, 4, 5, 4, 5, 5, 4],
      averageRating: 4.6
    },
    isPublic: true,
    createdBy: 'legal-team-1',
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date('2024-02-28')
  },
  {
    id: 'template-3',
    name: 'SaaS License Agreement',
    description: 'Software as a Service licensing template with subscription terms.',
    industry: IndustryType.TECHNOLOGY,
    jurisdiction: ['US'],
    language: 'en',
    version: '3.0',
    structure: [
      {
        id: 'section-saas-1',
        title: 'License Grant',
        order: 1,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-license-1',
            isRequired: true,
            order: 1
          }
        ]
      },
      {
        id: 'section-saas-2',
        title: 'Subscription Terms',
        order: 2,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-subscription-1',
            isRequired: true,
            order: 1
          }
        ]
      },
      {
        id: 'section-saas-3',
        title: 'Data Security',
        order: 3,
        isRequired: true,
        clauses: [
          {
            clauseId: 'clause-security-1',
            isRequired: true,
            order: 1
          }
        ]
      }
    ],
    globalVariables: [
      {
        id: 'subscription-tier',
        name: 'Subscription Tier',
        type: 'string',
        required: true,
        defaultValue: 'Professional'
      },
      {
        id: 'monthly-fee',
        name: 'Monthly Fee',
        type: 'currency',
        required: true,
        defaultValue: 99.00
      }
    ],
    conditionalLogic: [
      {
        id: 'enterprise-condition',
        condition: 'subscription-tier === "Enterprise"',
        action: 'include',
        target: 'clause-enterprise-support'
      }
    ],
    requiredClauses: ['clause-license-1', 'clause-subscription-1', 'clause-security-1'],
    optionalClauses: ['clause-enterprise-support', 'clause-integration'],
    metadata: {
      usageCount: 78,
      averageCompletionTime: 2100, // 35 minutes
      successRate: 0.89,
      commonModifications: [
        'Adjusted pricing structure',
        'Modified data retention policies',
        'Added integration requirements'
      ],
      userRatings: [4, 5, 4, 3, 4, 5, 4],
      averageRating: 4.1
    },
    isPublic: true,
    createdBy: 'product-team-1',
    createdAt: new Date('2024-01-30'),
    updatedAt: new Date('2024-03-15')
  }
];

// Template validation service
const templateValidationService = {
  validateTemplate: (template: ContractTemplate): ValidationResult => {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Required field validation
    if (!template.name || template.name.trim().length === 0) {
      errors.push({
        field: 'name',
        message: 'Template name is required',
        code: 'REQUIRED_FIELD'
      });
    }

    if (!template.description || template.description.trim().length === 0) {
      errors.push({
        field: 'description',
        message: 'Template description is required',
        code: 'REQUIRED_FIELD'
      });
    }

    if (!template.structure || template.structure.length === 0) {
      errors.push({
        field: 'structure',
        message: 'Template must have at least one section',
        code: 'MINIMUM_STRUCTURE'
      });
    }

    // Structure validation
    if (template.structure) {
      const sectionOrders = template.structure.map(s => s.order);
      const uniqueOrders = new Set(sectionOrders);
      if (sectionOrders.length !== uniqueOrders.size) {
        errors.push({
          field: 'structure',
          message: 'Section order numbers must be unique',
          code: 'DUPLICATE_ORDER'
        });
      }

      // Check for required sections
      const hasRequiredSections = template.structure.some(s => s.isRequired);
      if (!hasRequiredSections) {
        warnings.push({
          field: 'structure',
          message: 'Template should have at least one required section',
          suggestion: 'Mark critical sections as required'
        });
      }

      // Validate clause references
      template.structure.forEach(section => {
        section.clauses.forEach(clause => {
          if (!clause.clauseId) {
            errors.push({
              field: 'structure',
              message: `Invalid clause reference in section "${section.title}"`,
              code: 'INVALID_CLAUSE_REFERENCE'
            });
          }
        });
      });
    }

    // Variable validation
    if (template.globalVariables) {
      const variableNames = template.globalVariables.map(v => v.name);
      const uniqueNames = new Set(variableNames);
      if (variableNames.length !== uniqueNames.size) {
        errors.push({
          field: 'globalVariables',
          message: 'Variable names must be unique',
          code: 'DUPLICATE_VARIABLE_NAMES'
        });
      }

      // Check for missing required variables
      const hasRequiredVariables = template.globalVariables.some(v => v.required);
      if (!hasRequiredVariables) {
        warnings.push({
          field: 'globalVariables',
          message: 'Consider adding required variables for better template validation',
          suggestion: 'Mark essential variables as required'
        });
      }
    }

    // Conditional logic validation
    if (template.conditionalLogic) {
      template.conditionalLogic.forEach(rule => {
        if (!rule.condition || rule.condition.trim().length === 0) {
          errors.push({
            field: 'conditionalLogic',
            message: 'Conditional rule must have a valid condition',
            code: 'INVALID_CONDITION'
          });
        }

        if (!rule.target) {
          errors.push({
            field: 'conditionalLogic',
            message: 'Conditional rule must specify a target',
            code: 'MISSING_TARGET'
          });
        }
      });
    }

    // Jurisdiction validation
    if (!template.jurisdiction || template.jurisdiction.length === 0) {
      warnings.push({
        field: 'jurisdiction',
        message: 'Specify applicable jurisdictions for better template targeting',
        suggestion: 'Add relevant jurisdictions'
      });
    }

    // Industry validation
    if (template.industry === IndustryType.GENERIC && template.structure.length > 5) {
      warnings.push({
        field: 'industry',
        message: 'Complex templates should specify a target industry',
        suggestion: 'Consider targeting a specific industry for better relevance'
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }
};

// Mock API implementation
const mockAPI: TemplateEngineAPI = {
  getTemplates: async (): Promise<ContractTemplate[]> => {
    await new Promise(resolve => setTimeout(resolve, 600));
    return [...mockTemplates];
  },

  getTemplate: async (id: UUID): Promise<ContractTemplate> => {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const template = mockTemplates.find(t => t.id === id);
    if (!template) {
      throw new Error(`Template with id ${id} not found`);
    }
    
    return template;
  },

  saveTemplate: async (template: Partial<ContractTemplate>): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (template.id) {
      // Update existing template
      const index = mockTemplates.findIndex(t => t.id === template.id);
      if (index !== -1) {
        mockTemplates[index] = {
          ...mockTemplates[index],
          ...template,
          updatedAt: new Date()
        } as ContractTemplate;
      }
    } else {
      // Create new template
      const newTemplate: ContractTemplate = {
        id: `template-${Date.now()}`,
        name: template.name || 'New Template',
        description: template.description || '',
        industry: template.industry || IndustryType.GENERIC,
        jurisdiction: template.jurisdiction || ['US'],
        language: template.language || 'en',
        version: '1.0',
        structure: template.structure || [],
        globalVariables: template.globalVariables || [],
        conditionalLogic: template.conditionalLogic || [],
        requiredClauses: template.requiredClauses || [],
        optionalClauses: template.optionalClauses || [],
        metadata: {
          usageCount: 0,
          averageCompletionTime: 0,
          successRate: 0,
          commonModifications: [],
          userRatings: [],
          averageRating: 0
        },
        isPublic: template.isPublic || false,
        createdBy: 'current-user',
        createdAt: new Date(),
        updatedAt: new Date()
      };
      
      mockTemplates.push(newTemplate);
    }
    
    console.log('Saved template:', template);
  },

  cloneTemplate: async (id: UUID, newName: string): Promise<UUID> => {
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const originalTemplate = mockTemplates.find(t => t.id === id);
    if (!originalTemplate) {
      throw new Error(`Template with id ${id} not found`);
    }
    
    const clonedTemplate: ContractTemplate = {
      ...originalTemplate,
      id: `template-${Date.now()}`,
      name: newName,
      version: '1.0',
      metadata: {
        ...originalTemplate.metadata,
        usageCount: 0,
        userRatings: [],
        averageRating: 0
      },
      isPublic: false,
      createdBy: 'current-user',
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    mockTemplates.push(clonedTemplate);
    console.log('Cloned template:', clonedTemplate);
    
    return clonedTemplate.id;
  },

  deleteTemplate: async (id: UUID): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const index = mockTemplates.findIndex(t => t.id === id);
    if (index === -1) {
      throw new Error(`Template with id ${id} not found`);
    }
    
    mockTemplates.splice(index, 1);
    console.log('Deleted template with id:', id);
  },

  validateTemplate: (template: ContractTemplate): ValidationResult => {
    return templateValidationService.validateTemplate(template);
  }
};

export const useTemplateEngine = (): UseTemplateEngineReturn => {
  const [templates, setTemplates] = useState<ContractTemplate[]>([]);
  const [currentTemplate, setCurrentTemplate] = useState<ContractTemplate | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load templates on mount
  useEffect(() => {
    const loadTemplates = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const loadedTemplates = await mockAPI.getTemplates();
        setTemplates(loadedTemplates);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load templates';
        setError(errorMessage);
        console.error('Error loading templates:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadTemplates();
  }, []);

  // Load specific template
  const loadTemplate = useCallback(async (id: UUID) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const template = await mockAPI.getTemplate(id);
      setCurrentTemplate(template);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load template';
      setError(errorMessage);
      console.error('Error loading template:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save template
  const saveTemplate = useCallback(async (template: Partial<ContractTemplate>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.saveTemplate(template);
      
      // Refresh templates list
      const updatedTemplates = await mockAPI.getTemplates();
      setTemplates(updatedTemplates);
      
      // Update current template if it was being edited
      if (template.id && currentTemplate?.id === template.id) {
        setCurrentTemplate(prev => prev ? { ...prev, ...template } as ContractTemplate : null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save template';
      setError(errorMessage);
      console.error('Error saving template:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentTemplate]);

  // Clone template
  const cloneTemplate = useCallback(async (id: UUID, newName: string): Promise<UUID> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const newTemplateId = await mockAPI.cloneTemplate(id, newName);
      
      // Refresh templates list
      const updatedTemplates = await mockAPI.getTemplates();
      setTemplates(updatedTemplates);
      
      return newTemplateId;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to clone template';
      setError(errorMessage);
      console.error('Error cloning template:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Delete template
  const deleteTemplate = useCallback(async (id: UUID) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await mockAPI.deleteTemplate(id);
      
      // Remove from local state
      setTemplates(prev => prev.filter(template => template.id !== id));
      
      // Clear current template if it was deleted
      if (currentTemplate?.id === id) {
        setCurrentTemplate(null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete template';
      setError(errorMessage);
      console.error('Error deleting template:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentTemplate]);

  // Validate template
  const validateTemplate = useCallback((template: ContractTemplate): ValidationResult => {
    return mockAPI.validateTemplate(template);
  }, []);

  return {
    templates,
    currentTemplate,
    isLoading,
    error,
    loadTemplate,
    saveTemplate,
    cloneTemplate,
    deleteTemplate,
    validateTemplate
  };
};