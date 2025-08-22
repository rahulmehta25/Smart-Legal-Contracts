/**
 * TemplateEngine.tsx - Template Management System
 * 
 * Comprehensive template management with industry-specific templates,
 * conditional logic, multi-language support, and custom template creation.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import {
  ContractTemplate,
  TemplateSection,
  TemplateClause,
  ConditionalRule,
  IndustryType,
  LanguageCode,
  ClauseCategory,
  VariableDefinition,
  NumberingStyle,
  UUID,
  TemplateMetadata
} from './types';
import { useTemplateEngine } from '../hooks/useTemplateEngine';
import { VariableManager } from './VariableManager';

interface TemplateEngineProps {
  onTemplateSelect?: (template: ContractTemplate) => void;
  industry?: IndustryType;
  language?: LanguageCode;
  jurisdiction?: string;
  readonly?: boolean;
  showCreationTools?: boolean;
  enableTemplateBuilder?: boolean;
  className?: string;
}

interface EngineState {
  activeView: 'browse' | 'create' | 'edit' | 'preview';
  selectedTemplate: ContractTemplate | null;
  currentTemplate: Partial<ContractTemplate>;
  searchTerm: string;
  industryFilter: IndustryType | 'ALL';
  languageFilter: LanguageCode | 'ALL';
  showAdvancedFilters: boolean;
  templateBuilder: {
    sections: TemplateSection[];
    globalVariables: VariableDefinition[];
    conditionalRules: ConditionalRule[];
    selectedSectionId: UUID | null;
    draggedItem: any;
  };
  validation: {
    errors: string[];
    warnings: string[];
    isValid: boolean;
  };
}

interface TemplateCardProps {
  template: ContractTemplate;
  onSelect: (template: ContractTemplate) => void;
  onEdit?: (template: ContractTemplate) => void;
  onDelete?: (templateId: UUID) => void;
  onClone?: (templateId: UUID) => void;
  readonly?: boolean;
}

const TemplateCard: React.FC<TemplateCardProps> = ({
  template,
  onSelect,
  onEdit,
  onDelete,
  onClone,
  readonly = false
}) => {
  const formatMetadata = (metadata: TemplateMetadata): string => {
    return `Used ${metadata.usageCount} times • ${Math.round(metadata.averageRating * 10) / 10}★ • ${Math.round(metadata.averageCompletionTime / 60)} min avg`;
  };

  return (
    <div 
      className="template-card"
      data-template-id={template.id}
      onClick={() => onSelect(template)}
    >
      <div className="template-header">
        <div className="template-title-section">
          <h3 className="template-name">{template.name}</h3>
          <p className="template-description">{template.description}</p>
        </div>
        
        <div className="template-badges">
          <span className="industry-badge">{template.industry}</span>
          <span className="language-badge">{template.language}</span>
          {template.isPublic && <span className="public-badge">Public</span>}
        </div>
      </div>

      <div className="template-content">
        <div className="template-stats">
          <div className="stat-item">
            <span className="stat-label">Sections:</span>
            <span className="stat-value">{template.structure.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Required Clauses:</span>
            <span className="stat-value">{template.requiredClauses.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Optional Clauses:</span>
            <span className="stat-value">{template.optionalClauses.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Variables:</span>
            <span className="stat-value">{template.globalVariables.length}</span>
          </div>
        </div>

        <div className="template-metadata">
          <small>{formatMetadata(template.metadata)}</small>
        </div>

        <div className="jurisdiction-list">
          <span className="jurisdiction-label">Jurisdictions:</span>
          <div className="jurisdiction-tags">
            {template.jurisdiction.slice(0, 3).map(j => (
              <span key={j} className="jurisdiction-tag">{j}</span>
            ))}
            {template.jurisdiction.length > 3 && (
              <span className="jurisdiction-tag more">
                +{template.jurisdiction.length - 3}
              </span>
            )}
          </div>
        </div>
      </div>

      {!readonly && (
        <div className="template-actions">
          {onClone && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClone(template.id);
              }}
              className="btn-icon"
              title="Clone Template"
            >
              📋
            </button>
          )}
          {onEdit && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onEdit(template);
              }}
              className="btn-icon"
              title="Edit Template"
            >
              ✏️
            </button>
          )}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(template.id);
              }}
              className="btn-icon danger"
              title="Delete Template"
            >
              🗑️
            </button>
          )}
        </div>
      )}
    </div>
  );
};

interface TemplateBuilderProps {
  template: Partial<ContractTemplate>;
  onTemplateChange: (template: Partial<ContractTemplate>) => void;
  readonly?: boolean;
}

const TemplateBuilder: React.FC<TemplateBuilderProps> = ({
  template,
  onTemplateChange,
  readonly = false
}) => {
  const [builderState, setBuilderState] = useState({
    selectedSectionId: null as UUID | null,
    showConditionalEditor: false,
    showVariableEditor: false
  });

  const handleSectionDragEnd = useCallback((result: DropResult) => {
    if (!result.destination || readonly) return;

    const sections = [...(template.structure || [])];
    const [reorderedSection] = sections.splice(result.source.index, 1);
    sections.splice(result.destination.index, 0, reorderedSection);

    // Update order numbers
    sections.forEach((section, index) => {
      section.order = index;
    });

    onTemplateChange({
      ...template,
      structure: sections
    });
  }, [template, onTemplateChange, readonly]);

  const addSection = useCallback(() => {
    if (readonly) return;

    const newSection: TemplateSection = {
      id: `section-${Date.now()}`,
      title: 'New Section',
      order: (template.structure?.length || 0),
      isRequired: true,
      clauses: [],
      numbering: NumberingStyle.NUMERIC
    };

    onTemplateChange({
      ...template,
      structure: [...(template.structure || []), newSection]
    });
  }, [template, onTemplateChange, readonly]);

  const updateSection = useCallback((sectionId: UUID, updates: Partial<TemplateSection>) => {
    if (readonly) return;

    const sections = (template.structure || []).map(section =>
      section.id === sectionId ? { ...section, ...updates } : section
    );

    onTemplateChange({
      ...template,
      structure: sections
    });
  }, [template, onTemplateChange, readonly]);

  const deleteSection = useCallback((sectionId: UUID) => {
    if (readonly) return;

    const sections = (template.structure || []).filter(section => section.id !== sectionId);
    
    onTemplateChange({
      ...template,
      structure: sections
    });
  }, [template, onTemplateChange, readonly]);

  return (
    <div className="template-builder" id="template-builder">
      <div className="builder-header">
        <h3>Template Builder</h3>
        <div className="builder-actions">
          <button
            onClick={() => setBuilderState(prev => ({ 
              ...prev, 
              showVariableEditor: !prev.showVariableEditor 
            }))}
            className="btn btn-outline"
            disabled={readonly}
          >
            Global Variables
          </button>
          <button
            onClick={() => setBuilderState(prev => ({ 
              ...prev, 
              showConditionalEditor: !prev.showConditionalEditor 
            }))}
            className="btn btn-outline"
            disabled={readonly}
          >
            Conditional Logic
          </button>
        </div>
      </div>

      {/* Template Basic Info */}
      <div className="template-info-editor">
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="template-name">Template Name:</label>
            <input
              id="template-name"
              type="text"
              value={template.name || ''}
              onChange={(e) => onTemplateChange({ ...template, name: e.target.value })}
              placeholder="Enter template name..."
              readOnly={readonly}
            />
          </div>
          <div className="form-group">
            <label htmlFor="template-industry">Industry:</label>
            <select
              id="template-industry"
              value={template.industry || IndustryType.GENERIC}
              onChange={(e) => onTemplateChange({ 
                ...template, 
                industry: e.target.value as IndustryType 
              })}
              disabled={readonly}
            >
              {Object.values(IndustryType).map(industry => (
                <option key={industry} value={industry}>
                  {industry.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="template-language">Language:</label>
            <select
              id="template-language"
              value={template.language || 'en'}
              onChange={(e) => onTemplateChange({ 
                ...template, 
                language: e.target.value as LanguageCode 
              })}
              disabled={readonly}
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="zh">Chinese</option>
              <option value="ja">Japanese</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="template-public">Public Template:</label>
            <input
              id="template-public"
              type="checkbox"
              checked={template.isPublic || false}
              onChange={(e) => onTemplateChange({ 
                ...template, 
                isPublic: e.target.checked 
              })}
              disabled={readonly}
            />
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="template-description">Description:</label>
          <textarea
            id="template-description"
            value={template.description || ''}
            onChange={(e) => onTemplateChange({ ...template, description: e.target.value })}
            placeholder="Describe the purpose and use cases for this template..."
            rows={3}
            readOnly={readonly}
          />
        </div>
      </div>

      {/* Global Variables Editor */}
      {builderState.showVariableEditor && (
        <div className="variables-editor">
          <h4>Global Variables</h4>
          <VariableManager
            variables={template.globalVariables?.reduce((acc, v) => ({ ...acc, [v.name]: v.defaultValue }), {}) || {}}
            variableDefinitions={template.globalVariables || []}
            onVariableUpdate={(variables) => {
              // Convert variables back to definitions
              const definitions = Object.entries(variables).map(([name, value]) => ({
                id: `var-${name}`,
                name,
                type: typeof value as any,
                required: false,
                defaultValue: value
              }));
              onTemplateChange({
                ...template,
                globalVariables: definitions
              });
            }}
            readonly={readonly}
          />
        </div>
      )}

      {/* Sections Editor */}
      <div className="sections-editor">
        <div className="sections-header">
          <h4>Template Structure</h4>
          <button
            onClick={addSection}
            className="btn btn-outline"
            disabled={readonly}
          >
            + Add Section
          </button>
        </div>

        <DragDropContext onDragEnd={handleSectionDragEnd}>
          <Droppable droppableId="template-sections">
            {(provided) => (
              <div
                ref={provided.innerRef}
                {...provided.droppableProps}
                className="sections-list"
              >
                {(template.structure || []).map((section, index) => (
                  <Draggable
                    key={section.id}
                    draggableId={section.id}
                    index={index}
                    isDragDisabled={readonly}
                  >
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        className="section-editor-item"
                        data-section-id={section.id}
                      >
                        <div className="section-header">
                          <div
                            {...provided.dragHandleProps}
                            className="drag-handle"
                          >
                            ⋮⋮
                          </div>
                          
                          <div className="section-info">
                            <input
                              type="text"
                              value={section.title}
                              onChange={(e) => updateSection(section.id, { title: e.target.value })}
                              className="section-title-input"
                              placeholder="Section title..."
                              readOnly={readonly}
                            />
                            
                            <div className="section-options">
                              <label className="checkbox-label">
                                <input
                                  type="checkbox"
                                  checked={section.isRequired}
                                  onChange={(e) => updateSection(section.id, { isRequired: e.target.checked })}
                                  disabled={readonly}
                                />
                                Required
                              </label>
                              
                              <select
                                value={section.numbering || NumberingStyle.NUMERIC}
                                onChange={(e) => updateSection(section.id, { 
                                  numbering: e.target.value as NumberingStyle 
                                })}
                                disabled={readonly}
                                className="numbering-select"
                              >
                                <option value={NumberingStyle.NUMERIC}>1, 2, 3...</option>
                                <option value={NumberingStyle.ALPHABETIC}>A, B, C...</option>
                                <option value={NumberingStyle.ROMAN}>I, II, III...</option>
                                <option value={NumberingStyle.DECIMAL}>1.1, 1.2, 1.3...</option>
                              </select>
                            </div>
                          </div>
                          
                          <div className="section-actions">
                            <span className="clause-count">
                              {section.clauses.length} clause{section.clauses.length !== 1 ? 's' : ''}
                            </span>
                            <button
                              onClick={() => setBuilderState(prev => ({ 
                                ...prev, 
                                selectedSectionId: prev.selectedSectionId === section.id ? null : section.id 
                              }))}
                              className="btn-icon"
                              title="Edit Clauses"
                            >
                              📝
                            </button>
                            <button
                              onClick={() => deleteSection(section.id)}
                              className="btn-icon danger"
                              title="Delete Section"
                              disabled={readonly}
                            >
                              🗑️
                            </button>
                          </div>
                        </div>

                        {builderState.selectedSectionId === section.id && (
                          <div className="section-clauses-editor">
                            <h5>Section Clauses</h5>
                            <div className="clauses-list">
                              {section.clauses.map(clause => (
                                <div key={clause.clauseId} className="clause-item">
                                  <span className="clause-info">
                                    Clause {clause.clauseId} - Order: {clause.order}
                                    {clause.isRequired && <span className="required-badge">Required</span>}
                                  </span>
                                  <div className="clause-actions">
                                    <button className="btn-icon" title="Configure">⚙️</button>
                                    <button className="btn-icon danger" title="Remove">×</button>
                                  </div>
                                </div>
                              ))}
                              
                              {!readonly && (
                                <button
                                  className="btn btn-outline btn-sm"
                                  onClick={() => {
                                    // Add clause to section
                                    console.log('Adding clause to section:', section.id);
                                  }}
                                >
                                  + Add Clause
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>

      {/* Conditional Logic Editor */}
      {builderState.showConditionalEditor && (
        <div className="conditional-editor">
          <h4>Conditional Logic Rules</h4>
          <div className="rules-list">
            {(template.conditionalLogic || []).map(rule => (
              <div key={rule.id} className="rule-item">
                <div className="rule-condition">
                  <strong>If:</strong> {rule.condition}
                </div>
                <div className="rule-action">
                  <strong>Then:</strong> {rule.action} {rule.target}
                </div>
                <div className="rule-actions">
                  <button className="btn-icon" title="Edit Rule">✏️</button>
                  <button className="btn-icon danger" title="Delete Rule">🗑️</button>
                </div>
              </div>
            ))}
            
            {!readonly && (
              <button
                className="btn btn-outline"
                onClick={() => {
                  // Add new conditional rule
                  console.log('Adding conditional rule');
                }}
              >
                + Add Conditional Rule
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const TemplateEngine: React.FC<TemplateEngineProps> = ({
  onTemplateSelect,
  industry = IndustryType.GENERIC,
  language = 'en',
  jurisdiction = 'US',
  readonly = false,
  showCreationTools = true,
  enableTemplateBuilder = true,
  className = ''
}) => {
  const {
    templates,
    currentTemplate,
    isLoading,
    error,
    loadTemplate,
    saveTemplate,
    cloneTemplate,
    deleteTemplate,
    validateTemplate
  } = useTemplateEngine();

  const [state, setState] = useState<EngineState>({
    activeView: 'browse',
    selectedTemplate: null,
    currentTemplate: {},
    searchTerm: '',
    industryFilter: industry === IndustryType.GENERIC ? 'ALL' : industry,
    languageFilter: language === 'en' ? 'ALL' : language,
    showAdvancedFilters: false,
    templateBuilder: {
      sections: [],
      globalVariables: [],
      conditionalRules: [],
      selectedSectionId: null,
      draggedItem: null
    },
    validation: {
      errors: [],
      warnings: [],
      isValid: true
    }
  });

  // Filter templates based on current filters
  const filteredTemplates = useMemo(() => {
    let filtered = templates;

    // Apply search term
    if (state.searchTerm) {
      const searchLower = state.searchTerm.toLowerCase();
      filtered = filtered.filter(template =>
        template.name.toLowerCase().includes(searchLower) ||
        template.description.toLowerCase().includes(searchLower) ||
        template.industry.toLowerCase().includes(searchLower)
      );
    }

    // Apply industry filter
    if (state.industryFilter !== 'ALL') {
      filtered = filtered.filter(template => template.industry === state.industryFilter);
    }

    // Apply language filter
    if (state.languageFilter !== 'ALL') {
      filtered = filtered.filter(template => template.language === state.languageFilter);
    }

    // Apply jurisdiction filter
    if (jurisdiction && jurisdiction !== 'ALL') {
      filtered = filtered.filter(template => 
        template.jurisdiction.includes(jurisdiction)
      );
    }

    return filtered;
  }, [templates, state.searchTerm, state.industryFilter, state.languageFilter, jurisdiction]);

  // Handle template selection
  const handleTemplateSelect = useCallback((template: ContractTemplate) => {
    setState(prev => ({ ...prev, selectedTemplate: template }));
    onTemplateSelect?.(template);
  }, [onTemplateSelect]);

  // Handle template editing
  const handleTemplateEdit = useCallback((template: ContractTemplate) => {
    setState(prev => ({
      ...prev,
      activeView: 'edit',
      currentTemplate: template
    }));
  }, []);

  // Handle template creation
  const handleCreateNew = useCallback(() => {
    const newTemplate: Partial<ContractTemplate> = {
      name: '',
      description: '',
      industry: industry,
      language: language,
      jurisdiction: [jurisdiction],
      structure: [],
      globalVariables: [],
      conditionalLogic: [],
      requiredClauses: [],
      optionalClauses: [],
      isPublic: false
    };

    setState(prev => ({
      ...prev,
      activeView: 'create',
      currentTemplate: newTemplate
    }));
  }, [industry, language, jurisdiction]);

  // Handle template save
  const handleTemplateSave = useCallback(async () => {
    try {
      const validation = validateTemplate(state.currentTemplate as ContractTemplate);
      
      if (!validation.isValid) {
        setState(prev => ({
          ...prev,
          validation: {
            errors: validation.errors.map(e => e.message),
            warnings: validation.warnings.map(w => w.message),
            isValid: false
          }
        }));
        return;
      }

      await saveTemplate(state.currentTemplate);
      setState(prev => ({ ...prev, activeView: 'browse' }));
    } catch (error) {
      console.error('Failed to save template:', error);
    }
  }, [state.currentTemplate, saveTemplate, validateTemplate]);

  // Handle template clone
  const handleTemplateClone = useCallback(async (templateId: UUID) => {
    try {
      const newTemplateId = await cloneTemplate(templateId, 'Copy of Template');
      console.log('Template cloned:', newTemplateId);
    } catch (error) {
      console.error('Failed to clone template:', error);
    }
  }, [cloneTemplate]);

  if (isLoading) {
    return (
      <div className={`template-engine loading ${className}`}>
        <div className="loading-content">
          <div className="spinner"></div>
          <p>Loading templates...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`template-engine error ${className}`}>
        <div className="error-content">
          <h3>Error Loading Templates</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`template-engine ${className}`} id="template-engine-main">
      {/* Header */}
      <div className="engine-header" id="engine-header">
        <div className="header-title">
          <h2>Template Engine</h2>
          <div className="view-tabs">
            <button
              onClick={() => setState(prev => ({ ...prev, activeView: 'browse' }))}
              className={`tab ${state.activeView === 'browse' ? 'active' : ''}`}
            >
              Browse Templates
            </button>
            {showCreationTools && (
              <button
                onClick={() => setState(prev => ({ ...prev, activeView: 'create' }))}
                className={`tab ${state.activeView === 'create' ? 'active' : ''}`}
                disabled={readonly}
              >
                Create Template
              </button>
            )}
            {state.activeView === 'edit' && (
              <button className="tab active">
                Edit Template
              </button>
            )}
          </div>
        </div>

        <div className="header-actions">
          {showCreationTools && (
            <button
              onClick={handleCreateNew}
              className="btn btn-primary"
              disabled={readonly}
            >
              + New Template
            </button>
          )}
        </div>
      </div>

      {/* Browse View */}
      {state.activeView === 'browse' && (
        <div className="browse-view" id="browse-view">
          {/* Search and Filters */}
          <div className="search-section">
            <div className="search-bar">
              <input
                type="text"
                placeholder="Search templates by name, description, or industry..."
                value={state.searchTerm}
                onChange={(e) => setState(prev => ({ ...prev, searchTerm: e.target.value }))}
                className="search-input"
              />
              <button
                onClick={() => setState(prev => ({ 
                  ...prev, 
                  showAdvancedFilters: !prev.showAdvancedFilters 
                }))}
                className="filter-toggle-btn"
              >
                🔽 Filters
              </button>
            </div>

            {state.showAdvancedFilters && (
              <div className="advanced-filters">
                <div className="filter-group">
                  <label>Industry:</label>
                  <select
                    value={state.industryFilter}
                    onChange={(e) => setState(prev => ({ 
                      ...prev, 
                      industryFilter: e.target.value as IndustryType | 'ALL' 
                    }))}
                  >
                    <option value="ALL">All Industries</option>
                    {Object.values(IndustryType).map(ind => (
                      <option key={ind} value={ind}>
                        {ind.replace(/_/g, ' ')}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="filter-group">
                  <label>Language:</label>
                  <select
                    value={state.languageFilter}
                    onChange={(e) => setState(prev => ({ 
                      ...prev, 
                      languageFilter: e.target.value as LanguageCode | 'ALL' 
                    }))}
                  >
                    <option value="ALL">All Languages</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="zh">Chinese</option>
                    <option value="ja">Japanese</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Templates Grid */}
          <div className="templates-grid" id="templates-grid">
            {filteredTemplates.map(template => (
              <TemplateCard
                key={template.id}
                template={template}
                onSelect={handleTemplateSelect}
                onEdit={showCreationTools ? handleTemplateEdit : undefined}
                onDelete={showCreationTools ? deleteTemplate : undefined}
                onClone={showCreationTools ? handleTemplateClone : undefined}
                readonly={readonly}
              />
            ))}

            {filteredTemplates.length === 0 && (
              <div className="empty-state">
                <h3>No templates found</h3>
                <p>Try adjusting your search criteria or create a new template.</p>
                {showCreationTools && (
                  <button
                    onClick={handleCreateNew}
                    className="btn btn-primary"
                    disabled={readonly}
                  >
                    Create First Template
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create/Edit View */}
      {(state.activeView === 'create' || state.activeView === 'edit') && enableTemplateBuilder && (
        <div className="create-edit-view" id="create-edit-view">
          <div className="view-header">
            <h3>{state.activeView === 'create' ? 'Create New Template' : 'Edit Template'}</h3>
            <div className="view-actions">
              <button
                onClick={() => setState(prev => ({ ...prev, activeView: 'browse' }))}
                className="btn btn-outline"
              >
                Cancel
              </button>
              <button
                onClick={handleTemplateSave}
                className="btn btn-primary"
                disabled={readonly}
              >
                Save Template
              </button>
            </div>
          </div>

          {/* Validation Messages */}
          {(!state.validation.isValid || state.validation.warnings.length > 0) && (
            <div className="validation-messages">
              {state.validation.errors.map((error, index) => (
                <div key={index} className="validation-error">
                  ❌ {error}
                </div>
              ))}
              {state.validation.warnings.map((warning, index) => (
                <div key={index} className="validation-warning">
                  ⚠️ {warning}
                </div>
              ))}
            </div>
          )}

          <TemplateBuilder
            template={state.currentTemplate}
            onTemplateChange={(template) => setState(prev => ({ 
              ...prev, 
              currentTemplate: template 
            }))}
            readonly={readonly}
          />
        </div>
      )}

      {/* Preview Modal */}
      {state.selectedTemplate && state.activeView === 'browse' && (
        <div className="modal-overlay" id="template-preview-modal">
          <div className="modal large">
            <div className="modal-header">
              <h3>{state.selectedTemplate.name}</h3>
              <button
                onClick={() => setState(prev => ({ ...prev, selectedTemplate: null }))}
                className="modal-close"
              >
                ×
              </button>
            </div>

            <div className="modal-content">
              <div className="template-preview">
                <div className="preview-metadata">
                  <div className="metadata-grid">
                    <div className="metadata-item">
                      <label>Industry:</label>
                      <span>{state.selectedTemplate.industry}</span>
                    </div>
                    <div className="metadata-item">
                      <label>Language:</label>
                      <span>{state.selectedTemplate.language}</span>
                    </div>
                    <div className="metadata-item">
                      <label>Version:</label>
                      <span>{state.selectedTemplate.version}</span>
                    </div>
                    <div className="metadata-item">
                      <label>Jurisdictions:</label>
                      <span>{state.selectedTemplate.jurisdiction.join(', ')}</span>
                    </div>
                  </div>
                </div>

                <div className="preview-structure">
                  <h4>Template Structure</h4>
                  <div className="structure-tree">
                    {state.selectedTemplate.structure.map(section => (
                      <div key={section.id} className="structure-section">
                        <div className="section-info">
                          <span className="section-title">{section.title}</span>
                          <span className="section-clauses">
                            ({section.clauses.length} clauses)
                          </span>
                          {section.isRequired && (
                            <span className="required-badge">Required</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {state.selectedTemplate.globalVariables.length > 0 && (
                  <div className="preview-variables">
                    <h4>Global Variables</h4>
                    <div className="variables-grid">
                      {state.selectedTemplate.globalVariables.map(variable => (
                        <div key={variable.id} className="variable-item">
                          <span className="variable-name">{variable.name}</span>
                          <span className="variable-type">({variable.type})</span>
                          {variable.required && <span className="required">*</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="modal-footer">
              <button
                onClick={() => {
                  handleTemplateSelect(state.selectedTemplate!);
                  setState(prev => ({ ...prev, selectedTemplate: null }));
                }}
                className="btn btn-primary"
              >
                Use This Template
              </button>
              <button
                onClick={() => setState(prev => ({ ...prev, selectedTemplate: null }))}
                className="btn btn-outline"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TemplateEngine;