/**
 * Comprehensive TypeScript type definitions for the automated contract generation system
 * Provides strong typing for all contract components, templates, and workflows
 */

// Base types
export type UUID = string;
export type Timestamp = Date;
export type LanguageCode = 'en' | 'es' | 'fr' | 'de' | 'zh' | 'ja' | 'ar' | 'pt' | 'it' | 'ru';
export type DocumentFormat = 'DOCX' | 'PDF' | 'HTML' | 'TXT' | 'JSON';

// Risk and compliance types
export enum RiskLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export enum ClauseCategory {
  GENERAL = 'GENERAL',
  PAYMENT = 'PAYMENT',
  LIABILITY = 'LIABILITY',
  TERMINATION = 'TERMINATION',
  INTELLECTUAL_PROPERTY = 'INTELLECTUAL_PROPERTY',
  CONFIDENTIALITY = 'CONFIDENTIALITY',
  DISPUTE_RESOLUTION = 'DISPUTE_RESOLUTION',
  COMPLIANCE = 'COMPLIANCE',
  FORCE_MAJEURE = 'FORCE_MAJEURE',
  GOVERNING_LAW = 'GOVERNING_LAW'
}

export enum IndustryType {
  TECHNOLOGY = 'TECHNOLOGY',
  HEALTHCARE = 'HEALTHCARE',
  FINANCE = 'FINANCE',
  REAL_ESTATE = 'REAL_ESTATE',
  MANUFACTURING = 'MANUFACTURING',
  RETAIL = 'RETAIL',
  ENERGY = 'ENERGY',
  EDUCATION = 'EDUCATION',
  GOVERNMENT = 'GOVERNMENT',
  GENERIC = 'GENERIC'
}

// Variable types for template substitution
export interface VariableDefinition {
  id: UUID;
  name: string;
  type: 'string' | 'number' | 'date' | 'boolean' | 'currency' | 'percentage' | 'list' | 'object';
  required: boolean;
  defaultValue?: any;
  validation?: ValidationRule[];
  description?: string;
  placeholder?: string;
  dependencies?: UUID[];
}

export interface ValidationRule {
  type: 'required' | 'minLength' | 'maxLength' | 'pattern' | 'range' | 'custom';
  value?: any;
  message: string;
  customValidator?: (value: any) => boolean;
}

// Clause types
export interface Clause {
  id: UUID;
  title: string;
  content: string;
  category: ClauseCategory;
  subcategory?: string;
  language: LanguageCode;
  variables: VariableDefinition[];
  riskLevel: RiskLevel;
  version: string;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  isApproved: boolean;
  approvedBy?: UUID;
  approvedAt?: Timestamp;
  tags: string[];
  metadata: ClauseMetadata;
  alternativeVersions?: UUID[];
  dependencies?: ClauseDependency[];
  conflictsWith?: UUID[];
  jurisdiction?: string[];
  industrySpecific?: IndustryType[];
}

export interface ClauseMetadata {
  author: UUID;
  reviewedBy?: UUID[];
  usageCount: number;
  successRate: number;
  averageNegotiationTime: number;
  commonModifications: string[];
  legalReferences: string[];
  precedentCases?: string[];
}

export interface ClauseDependency {
  dependsOn: UUID;
  dependencyType: 'requires' | 'suggests' | 'conflicts' | 'mutually_exclusive';
  description?: string;
}

// Template types
export interface ContractTemplate {
  id: UUID;
  name: string;
  description: string;
  industry: IndustryType;
  jurisdiction: string[];
  language: LanguageCode;
  version: string;
  structure: TemplateSection[];
  globalVariables: VariableDefinition[];
  conditionalLogic: ConditionalRule[];
  requiredClauses: UUID[];
  optionalClauses: UUID[];
  metadata: TemplateMetadata;
  isPublic: boolean;
  createdBy: UUID;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

export interface TemplateSection {
  id: UUID;
  title: string;
  order: number;
  isRequired: boolean;
  clauses: TemplateClause[];
  conditionalRules?: ConditionalRule[];
  numbering?: NumberingStyle;
}

export interface TemplateClause {
  clauseId: UUID;
  isRequired: boolean;
  order: number;
  customizations?: ClauseCustomization[];
  conditionalRules?: ConditionalRule[];
}

export interface ClauseCustomization {
  variableId: UUID;
  defaultValue?: any;
  isLocked?: boolean;
  customValidation?: ValidationRule[];
}

export interface ConditionalRule {
  id: UUID;
  condition: string; // JavaScript expression
  action: 'include' | 'exclude' | 'require' | 'modify';
  target: UUID; // clause or section ID
  parameters?: Record<string, any>;
}

export interface TemplateMetadata {
  usageCount: number;
  averageCompletionTime: number;
  successRate: number;
  commonModifications: string[];
  userRatings: number[];
  averageRating: number;
}

export enum NumberingStyle {
  NUMERIC = 'NUMERIC',
  ALPHABETIC = 'ALPHABETIC',
  ROMAN = 'ROMAN',
  DECIMAL = 'DECIMAL',
  CUSTOM = 'CUSTOM'
}

// Contract generation types
export interface ContractDraft {
  id: UUID;
  templateId: UUID;
  name: string;
  parties: ContractParty[];
  variables: Record<string, any>;
  selectedClauses: UUID[];
  customClauses: CustomClause[];
  structure: GeneratedSection[];
  status: ContractStatus;
  version: number;
  language: LanguageCode;
  jurisdiction: string;
  createdBy: UUID;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  metadata: ContractMetadata;
}

export interface ContractParty {
  id: UUID;
  name: string;
  type: 'individual' | 'company' | 'government' | 'organization';
  role: string;
  contact: ContactInformation;
  signatoryInfo: SignatoryInfo;
  jurisdiction?: string;
}

export interface ContactInformation {
  address: Address;
  email?: string;
  phone?: string;
  website?: string;
}

export interface Address {
  street: string;
  city: string;
  state?: string;
  postalCode: string;
  country: string;
}

export interface SignatoryInfo {
  name: string;
  title: string;
  email: string;
  signatureRequired: boolean;
  signatureType: 'wet' | 'digital' | 'electronic';
  signedAt?: Timestamp;
  signatureData?: string;
}

export interface CustomClause {
  id: UUID;
  title: string;
  content: string;
  order: number;
  category: ClauseCategory;
  variables: VariableDefinition[];
  isApproved: boolean;
  riskAssessment?: RiskAssessment;
}

export interface GeneratedSection {
  id: UUID;
  title: string;
  order: number;
  content: string;
  clauses: GeneratedClause[];
  numbering: string;
  pageBreakBefore?: boolean;
  pageBreakAfter?: boolean;
}

export interface GeneratedClause {
  id: UUID;
  clauseId: UUID;
  title: string;
  content: string;
  order: number;
  numbering: string;
  variables: Record<string, any>;
  isCustom: boolean;
  modifications: ClauseModification[];
}

export interface ClauseModification {
  id: UUID;
  type: 'addition' | 'deletion' | 'substitution' | 'reordering';
  originalContent?: string;
  modifiedContent: string;
  reason?: string;
  modifiedBy: UUID;
  modifiedAt: Timestamp;
  isApproved: boolean;
  approvedBy?: UUID;
}

export enum ContractStatus {
  DRAFT = 'DRAFT',
  REVIEW = 'REVIEW',
  NEGOTIATION = 'NEGOTIATION',
  APPROVAL = 'APPROVAL',
  EXECUTION = 'EXECUTION',
  ACTIVE = 'ACTIVE',
  EXPIRED = 'EXPIRED',
  TERMINATED = 'TERMINATED',
  ARCHIVED = 'ARCHIVED'
}

export interface ContractMetadata {
  totalClauses: number;
  riskScore: number;
  complianceScore: number;
  estimatedValue?: number;
  currency?: string;
  effectiveDate?: Timestamp;
  expirationDate?: Timestamp;
  autoRenewal?: boolean;
  renewalPeriod?: number;
  noticePeriod?: number;
}

// Validation and risk assessment
export interface RiskAssessment {
  overallRisk: RiskLevel;
  riskFactors: RiskFactor[];
  recommendations: RiskRecommendation[];
  complianceIssues: ComplianceIssue[];
  calculatedAt: Timestamp;
}

export interface RiskFactor {
  category: string;
  description: string;
  severity: RiskLevel;
  impact: number;
  probability: number;
  mitigationSuggestions: string[];
}

export interface RiskRecommendation {
  priority: 'high' | 'medium' | 'low';
  category: string;
  recommendation: string;
  impactDescription: string;
  alternativeClauseIds?: UUID[];
}

export interface ComplianceIssue {
  regulation: string;
  jurisdiction: string;
  severity: RiskLevel;
  description: string;
  requiredActions: string[];
  deadlines?: Timestamp[];
}

// Change tracking and negotiation
export interface NegotiationSession {
  id: UUID;
  contractId: UUID;
  participants: NegotiationParticipant[];
  changes: ContractChange[];
  comments: NegotiationComment[];
  status: 'active' | 'paused' | 'completed' | 'cancelled';
  startedAt: Timestamp;
  completedAt?: Timestamp;
  currentRound: number;
  deadlines: NegotiationDeadline[];
}

export interface NegotiationParticipant {
  userId: UUID;
  partyId: UUID;
  role: 'primary' | 'advisor' | 'observer';
  permissions: NegotiationPermission[];
  lastActive?: Timestamp;
}

export interface NegotiationPermission {
  action: 'view' | 'comment' | 'edit' | 'approve' | 'reject';
  scope: 'all' | 'specific_clauses' | 'specific_sections';
  targets?: UUID[];
}

export interface ContractChange {
  id: UUID;
  changeType: 'clause_addition' | 'clause_removal' | 'clause_modification' | 'variable_change' | 'reordering';
  targetId: UUID; // clause, section, or variable ID
  previousValue?: any;
  newValue: any;
  proposedBy: UUID;
  proposedAt: Timestamp;
  status: 'proposed' | 'accepted' | 'rejected' | 'pending';
  reviewedBy?: UUID[];
  reason?: string;
  impact: ChangeImpact;
  relatedChanges?: UUID[];
}

export interface ChangeImpact {
  riskChange: number;
  affectedClauses: UUID[];
  complianceImpact: ComplianceImpact[];
  costImplication?: number;
  timeImplication?: number;
}

export interface ComplianceImpact {
  regulation: string;
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
}

export interface NegotiationComment {
  id: UUID;
  targetId: UUID; // clause, section, or change ID
  targetType: 'clause' | 'section' | 'change' | 'contract';
  content: string;
  author: UUID;
  createdAt: Timestamp;
  parentCommentId?: UUID;
  isResolved: boolean;
  resolvedBy?: UUID;
  resolvedAt?: Timestamp;
  attachments?: CommentAttachment[];
}

export interface CommentAttachment {
  id: UUID;
  filename: string;
  fileType: string;
  fileSize: number;
  url: string;
  uploadedAt: Timestamp;
}

export interface NegotiationDeadline {
  id: UUID;
  description: string;
  dueDate: Timestamp;
  assignedTo: UUID[];
  status: 'upcoming' | 'overdue' | 'completed';
  priority: 'high' | 'medium' | 'low';
}

// Export and document generation
export interface ExportConfiguration {
  format: DocumentFormat;
  includeComments: boolean;
  includeTrackChanges: boolean;
  includeMetadata: boolean;
  watermark?: WatermarkConfig;
  headerFooter?: HeaderFooterConfig;
  styling: DocumentStyling;
  sections: ExportSection[];
}

export interface WatermarkConfig {
  text: string;
  opacity: number;
  rotation: number;
  fontSize: number;
  color: string;
}

export interface HeaderFooterConfig {
  header?: {
    left?: string;
    center?: string;
    right?: string;
  };
  footer?: {
    left?: string;
    center?: string;
    right?: string;
  };
  includePageNumbers: boolean;
  includeDate: boolean;
  includeTotalPages: boolean;
}

export interface DocumentStyling {
  fontFamily: string;
  fontSize: number;
  lineSpacing: number;
  margins: DocumentMargins;
  headingStyles: HeadingStyle[];
  pageSize: 'A4' | 'US_LETTER' | 'A3' | 'LEGAL';
  orientation: 'portrait' | 'landscape';
}

export interface DocumentMargins {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

export interface HeadingStyle {
  level: number;
  fontFamily?: string;
  fontSize: number;
  bold: boolean;
  italic: boolean;
  color?: string;
  spacing: {
    before: number;
    after: number;
  };
}

export interface ExportSection {
  type: 'title_page' | 'toc' | 'contract_body' | 'signature_page' | 'appendix';
  include: boolean;
  customContent?: string;
}

// Blockchain and audit trail
export interface AuditTrailEntry {
  id: UUID;
  contractId: UUID;
  action: AuditAction;
  actorId: UUID;
  timestamp: Timestamp;
  details: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
  blockchainHash?: string;
  previousHash?: string;
}

export enum AuditAction {
  CREATED = 'CREATED',
  VIEWED = 'VIEWED',
  MODIFIED = 'MODIFIED',
  APPROVED = 'APPROVED',
  REJECTED = 'REJECTED',
  SIGNED = 'SIGNED',
  EXPORTED = 'EXPORTED',
  SHARED = 'SHARED',
  ARCHIVED = 'ARCHIVED',
  DELETED = 'DELETED'
}

// Search and filtering
export interface SearchFilters {
  keywords?: string[];
  categories?: ClauseCategory[];
  riskLevels?: RiskLevel[];
  industries?: IndustryType[];
  languages?: LanguageCode[];
  dateRange?: {
    start: Timestamp;
    end: Timestamp;
  };
  authors?: UUID[];
  tags?: string[];
  approvalStatus?: boolean;
  jurisdiction?: string[];
}

export interface SearchResult<T> {
  items: T[];
  totalCount: number;
  currentPage: number;
  totalPages: number;
  hasMore: boolean;
  facets: SearchFacet[];
}

export interface SearchFacet {
  field: string;
  values: FacetValue[];
}

export interface FacetValue {
  value: string;
  count: number;
  selected: boolean;
}

// API response types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: APIError;
  metadata?: ResponseMetadata;
}

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: Timestamp;
}

export interface ResponseMetadata {
  requestId: UUID;
  processingTime: number;
  version: string;
  rateLimit?: {
    remaining: number;
    resetAt: Timestamp;
  };
}

// Hook return types
export interface UseContractBuilderReturn {
  draft: ContractDraft | null;
  isLoading: boolean;
  error: string | null;
  saveDraft: (draft: Partial<ContractDraft>) => Promise<void>;
  loadDraft: (id: UUID) => Promise<void>;
  generateContract: () => Promise<string>;
  validateContract: () => Promise<RiskAssessment>;
  exportContract: (config: ExportConfiguration) => Promise<Blob>;
}

export interface UseClauseLibraryReturn {
  clauses: Clause[];
  isLoading: boolean;
  error: string | null;
  searchClauses: (filters: SearchFilters) => Promise<void>;
  addClause: (clause: Omit<Clause, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  updateClause: (id: UUID, updates: Partial<Clause>) => Promise<void>;
  deleteClause: (id: UUID) => Promise<void>;
  getAlternatives: (clauseId: UUID) => Promise<Clause[]>;
}

export interface UseTemplateEngineReturn {
  templates: ContractTemplate[];
  currentTemplate: ContractTemplate | null;
  isLoading: boolean;
  error: string | null;
  loadTemplate: (id: UUID) => Promise<void>;
  saveTemplate: (template: Partial<ContractTemplate>) => Promise<void>;
  cloneTemplate: (id: UUID, newName: string) => Promise<UUID>;
  deleteTemplate: (id: UUID) => Promise<void>;
  validateTemplate: (template: ContractTemplate) => ValidationResult[];
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface ValidationWarning {
  field: string;
  message: string;
  suggestion?: string;
}