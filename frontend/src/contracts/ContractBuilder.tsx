/**
 * ContractBuilder.tsx - Visual Contract Builder Component
 * 
 * Main component for building contracts with drag-and-drop interface,
 * template selection, clause management, and real-time preview.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import {
  ContractDraft,
  ContractTemplate,
  Clause,
  GeneratedSection,
  ContractStatus,
  IndustryType,
  LanguageCode,
  ContractParty,
  RiskAssessment,
  ExportConfiguration,
  DocumentFormat,
  UUID
} from './types';
import { useContractBuilder } from '../hooks/useContractBuilder';
import { ClauseLibrary } from './ClauseLibrary';
import { TemplateEngine } from './TemplateEngine';
import { VariableManager } from './VariableManager';
import { ValidationEngine } from './ValidationEngine';
import { NegotiationTracker } from './NegotiationTracker';

interface ContractBuilderProps {
  contractId?: UUID;
  templateId?: UUID;
  readOnly?: boolean;
  onSave?: (draft: ContractDraft) => void;
  onExport?: (format: DocumentFormat) => void;
  onValidate?: (assessment: RiskAssessment) => void;
  className?: string;
}

interface BuilderState {
  activeTab: 'structure' | 'clauses' | 'variables' | 'preview' | 'validation' | 'negotiation';
  isTemplateModalOpen: boolean;
  isClauseLibraryOpen: boolean;
  showVariablePanel: boolean;
  showExportOptions: boolean;
  previewMode: 'edit' | 'readonly' | 'comparison';
  sidebarCollapsed: boolean;
}

export const ContractBuilder: React.FC<ContractBuilderProps> = ({
  contractId,
  templateId,
  readOnly = false,
  onSave,
  onExport,
  onValidate,
  className = ''
}) => {
  // Main hook for contract building functionality
  const {
    draft,
    isLoading,
    error,
    saveDraft,
    loadDraft,
    generateContract,
    validateContract,
    exportContract
  } = useContractBuilder(contractId);

  // UI state management
  const [state, setState] = useState<BuilderState>({
    activeTab: 'structure',
    isTemplateModalOpen: !contractId && !templateId,
    isClauseLibraryOpen: false,
    showVariablePanel: false,
    showExportOptions: false,
    previewMode: 'edit',
    sidebarCollapsed: false
  });

  // Contract structure state
  const [sections, setSections] = useState<GeneratedSection[]>([]);
  const [selectedClauseIds, setSelectedClauseIds] = useState<UUID[]>([]);
  const [contractMetadata, setContractMetadata] = useState({
    title: '',
    description: '',
    parties: [] as ContractParty[],
    industry: IndustryType.GENERIC,
    jurisdiction: 'US',
    language: 'en' as LanguageCode
  });

  // Validation state
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Load contract draft on mount or when contractId changes
  useEffect(() => {
    if (contractId) {
      loadDraft(contractId);
    } else if (templateId) {
      // Load template and create new draft
      initializeFromTemplate(templateId);
    }
  }, [contractId, templateId, loadDraft]);

  // Update local state when draft changes
  useEffect(() => {
    if (draft) {
      setSections(draft.structure);
      setSelectedClauseIds(draft.selectedClauses);
      setContractMetadata({
        title: draft.name,
        description: draft.metadata?.description || '',
        parties: draft.parties,
        industry: draft.metadata?.industry || IndustryType.GENERIC,
        jurisdiction: draft.jurisdiction,
        language: draft.language
      });
    }
  }, [draft]);

  // Initialize contract from template
  const initializeFromTemplate = useCallback(async (templateId: UUID) => {
    try {
      // This would typically load template and create initial draft
      // Implementation would depend on your API structure
      console.log('Initializing from template:', templateId);
    } catch (error) {
      console.error('Failed to initialize from template:', error);
    }
  }, []);

  // Handle drag and drop for clauses
  const handleDragEnd = useCallback((result: DropResult) => {
    const { destination, source, draggableId } = result;
    
    if (!destination) return;
    if (destination.droppableId === source.droppableId && 
        destination.index === source.index) return;

    // Handle moving clauses between sections or reordering within sections
    if (destination.droppableId.startsWith('section-')) {
      const sectionId = destination.droppableId.replace('section-', '');
      const newSections = [...sections];
      const targetSection = newSections.find(s => s.id === sectionId);
      
      if (targetSection) {
        // Remove clause from source
        if (source.droppableId === 'clause-library') {
          // Adding new clause from library
          const clauseId = draggableId;
          targetSection.clauses.splice(destination.index, 0, {
            id: `generated-${Date.now()}`,
            clauseId,
            title: '',
            content: '',
            order: destination.index,
            numbering: `${targetSection.order}.${destination.index + 1}`,
            variables: {},
            isCustom: false,
            modifications: []
          });
        } else {
          // Moving between sections or reordering
          // Implementation for moving clauses between sections
        }
        
        setSections(newSections);
      }
    }
  }, [sections]);

  // Handle contract validation
  const handleValidation = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, activeTab: 'validation' }));
      const assessment = await validateContract();
      setRiskAssessment(assessment);
      onValidate?.(assessment);
    } catch (error) {
      console.error('Validation failed:', error);
      setValidationErrors(['Validation failed. Please check your contract structure.']);
    }
  }, [validateContract, onValidate]);

  // Handle contract export
  const handleExport = useCallback(async (format: DocumentFormat, config?: ExportConfiguration) => {
    try {
      const defaultConfig: ExportConfiguration = {
        format,
        includeComments: false,
        includeTrackChanges: true,
        includeMetadata: true,
        styling: {
          fontFamily: 'Times New Roman',
          fontSize: 12,
          lineSpacing: 1.15,
          margins: { top: 1, bottom: 1, left: 1, right: 1 },
          headingStyles: [
            { level: 1, fontSize: 16, bold: true, italic: false, spacing: { before: 12, after: 6 } },
            { level: 2, fontSize: 14, bold: true, italic: false, spacing: { before: 12, after: 6 } },
            { level: 3, fontSize: 12, bold: true, italic: false, spacing: { before: 6, after: 3 } }
          ],
          pageSize: 'US_LETTER',
          orientation: 'portrait'
        },
        sections: [
          { type: 'title_page', include: true },
          { type: 'toc', include: true },
          { type: 'contract_body', include: true },
          { type: 'signature_page', include: true }
        ]
      };

      const blob = await exportContract(config || defaultConfig);
      
      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${contractMetadata.title || 'contract'}.${format.toLowerCase()}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      onExport?.(format);
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [exportContract, contractMetadata.title, onExport]);

  // Handle saving draft
  const handleSave = useCallback(async () => {
    if (!draft) return;
    
    const updatedDraft: Partial<ContractDraft> = {
      ...draft,
      name: contractMetadata.title,
      parties: contractMetadata.parties,
      structure: sections,
      selectedClauses: selectedClauseIds,
      language: contractMetadata.language,
      jurisdiction: contractMetadata.jurisdiction,
      updatedAt: new Date()
    };

    try {
      await saveDraft(updatedDraft);
      onSave?.(updatedDraft as ContractDraft);
    } catch (error) {
      console.error('Failed to save draft:', error);
    }
  }, [draft, contractMetadata, sections, selectedClauseIds, saveDraft, onSave]);

  // Memoized contract preview
  const contractPreview = useMemo(() => {
    if (!sections.length) return '';
    
    return sections
      .sort((a, b) => a.order - b.order)
      .map(section => {
        const sectionContent = section.clauses
          .sort((a, b) => a.order - b.order)
          .map(clause => `${clause.numbering} ${clause.title}\n${clause.content}`)
          .join('\n\n');
        
        return `${section.numbering} ${section.title}\n\n${sectionContent}`;
      })
      .join('\n\n');
  }, [sections]);

  if (isLoading) {
    return (
      <div className={`contract-builder loading ${className}`}>
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading contract builder...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`contract-builder error ${className}`}>
        <div className="error-message">
          <h3>Error Loading Contract Builder</h3>
          <p>{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="btn btn-primary"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`contract-builder ${className}`} id="contract-builder-main">
      {/* Header */}
      <div className="builder-header" id="builder-header">
        <div className="header-left">
          <h1>{contractMetadata.title || 'New Contract'}</h1>
          <div className="status-indicator">
            Status: {draft?.status || ContractStatus.DRAFT}
          </div>
        </div>
        
        <div className="header-actions">
          <button
            onClick={() => setState(prev => ({ ...prev, showVariablePanel: !prev.showVariablePanel }))}
            className="btn btn-outline"
            disabled={readOnly}
          >
            Variables
          </button>
          
          <button
            onClick={handleValidation}
            className="btn btn-outline"
            disabled={readOnly}
          >
            Validate
          </button>
          
          <button
            onClick={() => setState(prev => ({ ...prev, showExportOptions: !prev.showExportOptions }))}
            className="btn btn-outline"
          >
            Export
          </button>
          
          <button
            onClick={handleSave}
            className="btn btn-primary"
            disabled={readOnly}
          >
            Save
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="builder-content" id="builder-content">
        {/* Sidebar */}
        <div className={`builder-sidebar ${state.sidebarCollapsed ? 'collapsed' : ''}`} id="builder-sidebar">
          <div className="sidebar-tabs">
            <button
              onClick={() => setState(prev => ({ ...prev, activeTab: 'structure' }))}
              className={`tab ${state.activeTab === 'structure' ? 'active' : ''}`}
            >
              Structure
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, activeTab: 'clauses' }))}
              className={`tab ${state.activeTab === 'clauses' ? 'active' : ''}`}
            >
              Clauses
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, activeTab: 'variables' }))}
              className={`tab ${state.activeTab === 'variables' ? 'active' : ''}`}
            >
              Variables
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, activeTab: 'validation' }))}
              className={`tab ${state.activeTab === 'validation' ? 'active' : ''}`}
            >
              Validation
            </button>
          </div>

          <div className="sidebar-content">
            {state.activeTab === 'structure' && (
              <div className="structure-panel" id="structure-panel">
                <div className="panel-header">
                  <h3>Document Structure</h3>
                  <button
                    onClick={() => setState(prev => ({ ...prev, isTemplateModalOpen: true }))}
                    className="btn btn-sm btn-outline"
                    disabled={readOnly}
                  >
                    Load Template
                  </button>
                </div>
                
                <div className="section-list">
                  {sections.map((section, index) => (
                    <div key={section.id} className="section-item" data-section-id={section.id}>
                      <div className="section-header">
                        <span className="section-number">{section.numbering}</span>
                        <span className="section-title">{section.title}</span>
                        <div className="section-actions">
                          <button className="btn-icon" title="Edit Section">📝</button>
                          <button className="btn-icon" title="Delete Section">🗑️</button>
                        </div>
                      </div>
                      <div className="clause-count">
                        {section.clauses.length} clause{section.clauses.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  ))}
                </div>
                
                <button
                  onClick={() => {/* Add new section */}}
                  className="btn btn-outline btn-block"
                  disabled={readOnly}
                >
                  + Add Section
                </button>
              </div>
            )}

            {state.activeTab === 'clauses' && (
              <ClauseLibrary
                onClauseSelect={(clause) => {/* Handle clause selection */}}
                selectedClauses={selectedClauseIds}
                readonly={readOnly}
              />
            )}

            {state.activeTab === 'variables' && (
              <VariableManager
                variables={draft?.variables || {}}
                onVariableUpdate={(variables) => {/* Handle variable updates */}}
                readonly={readOnly}
              />
            )}

            {state.activeTab === 'validation' && (
              <ValidationEngine
                assessment={riskAssessment}
                errors={validationErrors}
                onRevalidate={handleValidation}
              />
            )}
          </div>
        </div>

        {/* Main Editor */}
        <div className="builder-main" id="builder-main">
          <div className="editor-tabs">
            <button
              onClick={() => setState(prev => ({ ...prev, previewMode: 'edit' }))}
              className={`tab ${state.previewMode === 'edit' ? 'active' : ''}`}
            >
              Edit
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, previewMode: 'readonly' }))}
              className={`tab ${state.previewMode === 'readonly' ? 'active' : ''}`}
            >
              Preview
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, previewMode: 'comparison' }))}
              className={`tab ${state.previewMode === 'comparison' ? 'active' : ''}`}
            >
              Compare
            </button>
          </div>

          <DragDropContext onDragEnd={handleDragEnd}>
            <div className="contract-editor" id="contract-editor">
              {state.previewMode === 'edit' && (
                <Droppable droppableId="contract-sections">
                  {(provided) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.droppableProps}
                      className="sections-container"
                    >
                      {sections.map((section, index) => (
                        <Draggable
                          key={section.id}
                          draggableId={section.id}
                          index={index}
                          isDragDisabled={readOnly}
                        >
                          {(provided) => (
                            <div
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              className="section-editor"
                              data-section-id={section.id}
                            >
                              <div
                                {...provided.dragHandleProps}
                                className="section-drag-handle"
                              >
                                ⋮⋮
                              </div>
                              
                              <div className="section-content">
                                <h3 className="section-title-editor">
                                  {section.numbering} {section.title}
                                </h3>
                                
                                <Droppable droppableId={`section-${section.id}`}>
                                  {(provided) => (
                                    <div
                                      ref={provided.innerRef}
                                      {...provided.droppableProps}
                                      className="clauses-container"
                                    >
                                      {section.clauses.map((clause, clauseIndex) => (
                                        <Draggable
                                          key={clause.id}
                                          draggableId={clause.id}
                                          index={clauseIndex}
                                          isDragDisabled={readOnly}
                                        >
                                          {(provided) => (
                                            <div
                                              ref={provided.innerRef}
                                              {...provided.draggableProps}
                                              {...provided.dragHandleProps}
                                              className="clause-editor"
                                              data-clause-id={clause.id}
                                            >
                                              <div className="clause-header">
                                                <span className="clause-number">
                                                  {clause.numbering}
                                                </span>
                                                <input
                                                  type="text"
                                                  value={clause.title}
                                                  onChange={(e) => {/* Update clause title */}}
                                                  className="clause-title-input"
                                                  placeholder="Clause title..."
                                                  readOnly={readOnly}
                                                />
                                                <div className="clause-actions">
                                                  <button className="btn-icon" title="Edit Variables">
                                                    🔧
                                                  </button>
                                                  <button className="btn-icon" title="Delete Clause">
                                                    🗑️
                                                  </button>
                                                </div>
                                              </div>
                                              
                                              <textarea
                                                value={clause.content}
                                                onChange={(e) => {/* Update clause content */}}
                                                className="clause-content-editor"
                                                placeholder="Clause content..."
                                                rows={4}
                                                readOnly={readOnly}
                                              />
                                            </div>
                                          )}
                                        </Draggable>
                                      ))}
                                      {provided.placeholder}
                                      
                                      {!readOnly && (
                                        <div className="clause-drop-zone">
                                          Drop clauses here or click to add
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </Droppable>
                              </div>
                            </div>
                          )}
                        </Draggable>
                      ))}
                      {provided.placeholder}
                    </div>
                  )}
                </Droppable>
              )}

              {state.previewMode === 'readonly' && (
                <div className="contract-preview" id="contract-preview">
                  <div className="preview-header">
                    <h2>{contractMetadata.title}</h2>
                    <div className="preview-metadata">
                      <span>Language: {contractMetadata.language}</span>
                      <span>Jurisdiction: {contractMetadata.jurisdiction}</span>
                      <span>Industry: {contractMetadata.industry}</span>
                    </div>
                  </div>
                  
                  <div className="preview-content">
                    <pre>{contractPreview}</pre>
                  </div>
                </div>
              )}

              {state.previewMode === 'comparison' && (
                <NegotiationTracker
                  contractId={draft?.id}
                  readonly={readOnly}
                />
              )}
            </div>
          </DragDropContext>
        </div>
      </div>

      {/* Modals and Overlays */}
      {state.isTemplateModalOpen && (
        <div className="modal-overlay" id="template-modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Select Contract Template</h3>
              <button
                onClick={() => setState(prev => ({ ...prev, isTemplateModalOpen: false }))}
                className="modal-close"
              >
                ×
              </button>
            </div>
            <div className="modal-content">
              <TemplateEngine
                onTemplateSelect={(template) => {
                  initializeFromTemplate(template.id);
                  setState(prev => ({ ...prev, isTemplateModalOpen: false }));
                }}
                industry={contractMetadata.industry}
                language={contractMetadata.language}
              />
            </div>
          </div>
        </div>
      )}

      {state.showExportOptions && (
        <div className="modal-overlay" id="export-modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Export Contract</h3>
              <button
                onClick={() => setState(prev => ({ ...prev, showExportOptions: false }))}
                className="modal-close"
              >
                ×
              </button>
            </div>
            <div className="modal-content">
              <div className="export-options">
                <button
                  onClick={() => handleExport('PDF')}
                  className="export-btn"
                >
                  Export as PDF
                </button>
                <button
                  onClick={() => handleExport('DOCX')}
                  className="export-btn"
                >
                  Export as DOCX
                </button>
                <button
                  onClick={() => handleExport('HTML')}
                  className="export-btn"
                >
                  Export as HTML
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContractBuilder;