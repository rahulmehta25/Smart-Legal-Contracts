/**
 * contracts/index.ts - Contract Generation System Exports
 * 
 * Main export file for the automated contract generation system.
 * Provides clean imports for all components, hooks, utilities, and types.
 */

// Core Components
export { default as ContractBuilder } from './ContractBuilder';
export { default as ClauseLibrary } from './ClauseLibrary';
export { default as TemplateEngine } from './TemplateEngine';
export { default as VariableManager } from './VariableManager';
export { default as ValidationEngine } from './ValidationEngine';
export { default as NegotiationTracker } from './NegotiationTracker';

// Hooks
export { useContractBuilder } from '../hooks/useContractBuilder';
export { useClauseLibrary } from '../hooks/useClauseLibrary';
export { useTemplateEngine } from '../hooks/useTemplateEngine';
export { useDebounce, useDebouncedCallback } from '../hooks/useDebounce';

// Utilities
export { 
  DocumentGenerator,
  ExportProcessor,
  documentGeneratorService 
} from './utils/documentGenerator';

export {
  ContractValidationService,
  contractValidationService
} from './utils/contractValidation';

// Types - Core contract types
export type {
  ContractDraft,
  ContractTemplate,
  Clause,
  GeneratedSection,
  GeneratedClause,
  ContractParty,
  ContractStatus
} from './types';

// Types - Variable and validation types
export type {
  VariableDefinition,
  ValidationRule,
  RiskAssessment,
  RiskLevel,
  RiskFactor,
  RiskRecommendation,
  ComplianceIssue
} from './types';

// Types - Template and structure types
export type {
  TemplateSection,
  TemplateClause,
  ConditionalRule,
  ClauseCategory,
  IndustryType,
  LanguageCode,
  NumberingStyle
} from './types';

// Types - Negotiation and collaboration types
export type {
  NegotiationSession,
  NegotiationParticipant,
  ContractChange,
  NegotiationComment,
  ChangeImpact
} from './types';

// Types - Export and document types
export type {
  ExportConfiguration,
  DocumentFormat,
  DocumentStyling,
  HeaderFooterConfig,
  WatermarkConfig
} from './types';

// Types - Search and filtering types
export type {
  SearchFilters,
  SearchResult,
  SearchFacet,
  FacetValue
} from './types';

// Types - API and response types
export type {
  APIResponse,
  APIError,
  ResponseMetadata,
  UUID,
  Timestamp
} from './types';

// Types - Hook return types
export type {
  UseContractBuilderReturn,
  UseClauseLibraryReturn,
  UseTemplateEngineReturn,
  ValidationResult,
  ValidationError,
  ValidationWarning
} from './types';

// Re-export enums for easy access
export {
  RiskLevel,
  ClauseCategory,
  IndustryType,
  ContractStatus,
  NumberingStyle,
  AuditAction
} from './types';