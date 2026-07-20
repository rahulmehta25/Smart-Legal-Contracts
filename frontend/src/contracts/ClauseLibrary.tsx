/**
 * ClauseLibrary.tsx - Searchable Clause Library Component
 * 
 * Advanced clause management system with 500+ pre-approved clauses,
 * intelligent search, categorization, risk ratings, and version control.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import {
  Clause,
  ClauseCategory,
  RiskLevel,
  IndustryType,
  LanguageCode,
  SearchFilters,
  SearchResult,
  UUID,
  ClauseDependency,
  ClauseMetadata
} from './types';
import { useClauseLibrary } from '../hooks/useClauseLibrary';
import { useDebounce } from '../hooks/useDebounce';

interface ClauseLibraryProps {
  onClauseSelect?: (clause: Clause) => void;
  selectedClauses?: UUID[];
  readonly?: boolean;
  industryFilter?: IndustryType;
  languageFilter?: LanguageCode;
  jurisdictionFilter?: string;
  showAdvancedSearch?: boolean;
  enableBulkOperations?: boolean;
  className?: string;
}

interface LibraryState {
  searchTerm: string;
  activeCategory: ClauseCategory | 'ALL';
  activeFilters: SearchFilters;
  selectedClauseIds: Set<UUID>;
  viewMode: 'grid' | 'list' | 'detailed';
  sortBy: 'relevance' | 'title' | 'category' | 'risk' | 'usage' | 'updated';
  sortOrder: 'asc' | 'desc';
  showFilters: boolean;
  showPreview: boolean;
  previewClause: Clause | null;
  bulkSelectMode: boolean;
}

interface ClauseCardProps {
  clause: Clause;
  isSelected: boolean;
  isDragDisabled: boolean;
  onSelect: (clause: Clause) => void;
  onPreview: (clause: Clause) => void;
  onEdit?: (clause: Clause) => void;
  onDelete?: (clauseId: UUID) => void;
  showMetadata: boolean;
  viewMode: 'grid' | 'list' | 'detailed';
}

const ClauseCard: React.FC<ClauseCardProps> = ({
  clause,
  isSelected,
  isDragDisabled,
  onSelect,
  onPreview,
  onEdit,
  onDelete,
  showMetadata,
  viewMode
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

  const formatUsageStats = (metadata: ClauseMetadata): string => {
    return `Used ${metadata.usageCount} times • ${Math.round(metadata.successRate * 100)}% success rate`;
  };

  return (
    <div 
      className={`clause-card ${viewMode} ${isSelected ? 'selected' : ''}`}
      data-clause-id={clause.id}
      onClick={() => onSelect(clause)}
    >
      <div className="clause-header">
        <div className="clause-title-section">
          <h4 className="clause-title" title={clause.title}>
            {clause.title}
          </h4>
          <div className="clause-badges">
            <span className={`risk-badge ${getRiskColor(clause.riskLevel)}`}>
              {clause.riskLevel}
            </span>
            <span className="category-badge">
              {clause.category}
            </span>
            {clause.isApproved && (
              <span className="approval-badge">✓ Approved</span>
            )}
            {clause.industrySpecific?.length && (
              <span className="industry-badge">
                {clause.industrySpecific[0]}
              </span>
            )}
          </div>
        </div>
        
        <div className="clause-actions">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onPreview(clause);
            }}
            className="btn-icon"
            title="Preview Clause"
          >
            👁️
          </button>
          {onEdit && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onEdit(clause);
              }}
              className="btn-icon"
              title="Edit Clause"
            >
              ✏️
            </button>
          )}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(clause.id);
              }}
              className="btn-icon danger"
              title="Delete Clause"
            >
              🗑️
            </button>
          )}
        </div>
      </div>

      {viewMode !== 'grid' && (
        <div className="clause-content-preview">
          <p className="clause-excerpt">
            {clause.content.substring(0, 200)}
            {clause.content.length > 200 && '...'}
          </p>
        </div>
      )}

      {showMetadata && (
        <div className="clause-metadata">
          <div className="metadata-row">
            <span className="metadata-label">Language:</span>
            <span className="metadata-value">{clause.language}</span>
          </div>
          <div className="metadata-row">
            <span className="metadata-label">Version:</span>
            <span className="metadata-value">{clause.version}</span>
          </div>
          <div className="metadata-row">
            <span className="metadata-label">Usage:</span>
            <span className="metadata-value">
              {formatUsageStats(clause.metadata)}
            </span>
          </div>
          <div className="metadata-row">
            <span className="metadata-label">Tags:</span>
            <div className="tags-container">
              {clause.tags.slice(0, 3).map(tag => (
                <span key={tag} className="tag">
                  {tag}
                </span>
              ))}
              {clause.tags.length > 3 && (
                <span className="tag more">+{clause.tags.length - 3}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {clause.dependencies && clause.dependencies.length > 0 && (
        <div className="clause-dependencies">
          <span className="dependencies-label">Dependencies:</span>
          {clause.dependencies.map(dep => (
            <span 
              key={dep.dependsOn}
              className={`dependency-badge ${dep.dependencyType}`}
            >
              {dep.dependencyType}
            </span>
          ))}
        </div>
      )}

      {!isDragDisabled && (
        <div className="drag-handle" title="Drag to add to contract">
          ⋮⋮
        </div>
      )}
    </div>
  );
};

export const ClauseLibrary: React.FC<ClauseLibraryProps> = ({
  onClauseSelect,
  selectedClauses = [],
  readonly = false,
  industryFilter,
  languageFilter,
  jurisdictionFilter,
  showAdvancedSearch = true,
  enableBulkOperations = false,
  className = ''
}) => {
  const {
    clauses,
    isLoading,
    error,
    searchClauses,
    addClause,
    updateClause,
    deleteClause,
    getAlternatives
  } = useClauseLibrary();

  const [state, setState] = useState<LibraryState>({
    searchTerm: '',
    activeCategory: 'ALL',
    activeFilters: {
      keywords: [],
      categories: [],
      riskLevels: [],
      industries: industryFilter ? [industryFilter] : [],
      languages: languageFilter ? [languageFilter] : [],
      jurisdiction: jurisdictionFilter ? [jurisdictionFilter] : [],
      approvalStatus: true
    },
    selectedClauseIds: new Set(),
    viewMode: 'list',
    sortBy: 'relevance',
    sortOrder: 'desc',
    showFilters: false,
    showPreview: false,
    previewClause: null,
    bulkSelectMode: false
  });

  // Debounced search term
  const debouncedSearchTerm = useDebounce(state.searchTerm, 300);

  // Categories for navigation
  const categories = useMemo(() => {
    return Object.values(ClauseCategory).map(category => ({
      key: category,
      label: category.replace(/_/g, ' '),
      count: clauses.filter(clause => clause.category === category).length
    }));
  }, [clauses]);

  // Filtered and sorted clauses
  const filteredClauses = useMemo(() => {
    let filtered = clauses;

    // Apply category filter
    if (state.activeCategory !== 'ALL') {
      filtered = filtered.filter(clause => clause.category === state.activeCategory);
    }

    // Apply search term
    if (debouncedSearchTerm) {
      const searchLower = debouncedSearchTerm.toLowerCase();
      filtered = filtered.filter(clause =>
        clause.title.toLowerCase().includes(searchLower) ||
        clause.content.toLowerCase().includes(searchLower) ||
        clause.tags.some(tag => tag.toLowerCase().includes(searchLower))
      );
    }

    // Apply advanced filters
    const { activeFilters } = state;
    if (activeFilters.riskLevels && activeFilters.riskLevels.length > 0) {
      filtered = filtered.filter(clause => 
        activeFilters.riskLevels!.includes(clause.riskLevel)
      );
    }

    if (activeFilters.industries && activeFilters.industries.length > 0) {
      filtered = filtered.filter(clause =>
        !clause.industrySpecific ||
        clause.industrySpecific.some(industry => 
          activeFilters.industries!.includes(industry)
        )
      );
    }

    if (activeFilters.languages && activeFilters.languages.length > 0) {
      filtered = filtered.filter(clause =>
        activeFilters.languages!.includes(clause.language)
      );
    }

    if (activeFilters.jurisdiction && activeFilters.jurisdiction.length > 0) {
      filtered = filtered.filter(clause =>
        !clause.jurisdiction ||
        clause.jurisdiction.some(j => 
          activeFilters.jurisdiction!.includes(j)
        )
      );
    }

    if (activeFilters.approvalStatus !== undefined) {
      filtered = filtered.filter(clause => 
        clause.isApproved === activeFilters.approvalStatus
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (state.sortBy) {
        case 'title':
          comparison = a.title.localeCompare(b.title);
          break;
        case 'category':
          comparison = a.category.localeCompare(b.category);
          break;
        case 'risk':
          const riskOrder = { [RiskLevel.LOW]: 1, [RiskLevel.MEDIUM]: 2, [RiskLevel.HIGH]: 3, [RiskLevel.CRITICAL]: 4 };
          comparison = riskOrder[a.riskLevel] - riskOrder[b.riskLevel];
          break;
        case 'usage':
          comparison = a.metadata.usageCount - b.metadata.usageCount;
          break;
        case 'updated':
          comparison = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
          break;
        default: // relevance
          comparison = a.metadata.successRate - b.metadata.successRate;
      }
      
      return state.sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [clauses, debouncedSearchTerm, state.activeCategory, state.activeFilters, state.sortBy, state.sortOrder]);

  // Search effect
  useEffect(() => {
    if (debouncedSearchTerm || Object.keys(state.activeFilters).length > 0) {
      const filters: SearchFilters = {
        ...state.activeFilters,
        keywords: debouncedSearchTerm ? [debouncedSearchTerm] : undefined
      };
      searchClauses(filters);
    }
  }, [debouncedSearchTerm, state.activeFilters, searchClauses]);

  // Handle clause selection
  const handleClauseSelect = useCallback((clause: Clause) => {
    if (state.bulkSelectMode) {
      setState(prev => {
        const newSelected = new Set(prev.selectedClauseIds);
        if (newSelected.has(clause.id)) {
          newSelected.delete(clause.id);
        } else {
          newSelected.add(clause.id);
        }
        return { ...prev, selectedClauseIds: newSelected };
      });
    } else {
      onClauseSelect?.(clause);
    }
  }, [state.bulkSelectMode, onClauseSelect]);

  // Handle clause preview
  const handleClausePreview = useCallback((clause: Clause) => {
    setState(prev => ({
      ...prev,
      showPreview: true,
      previewClause: clause
    }));
  }, []);

  // Handle bulk operations
  const handleBulkOperation = useCallback(async (operation: 'delete' | 'approve' | 'export') => {
    const selectedIds = Array.from(state.selectedClauseIds);
    
    try {
      switch (operation) {
        case 'delete':
          await Promise.all(selectedIds.map(id => deleteClause(id)));
          break;
        case 'approve':
          await Promise.all(selectedIds.map(id => 
            updateClause(id, { isApproved: true, approvedAt: new Date() })
          ));
          break;
        case 'export':
          // Implementation for bulk export
          console.log('Exporting clauses:', selectedIds);
          break;
      }
      
      setState(prev => ({ 
        ...prev, 
        selectedClauseIds: new Set(),
        bulkSelectMode: false 
      }));
    } catch (error) {
      console.error('Bulk operation failed:', error);
    }
  }, [state.selectedClauseIds, deleteClause, updateClause]);

  if (isLoading) {
    return (
      <div className={`clause-library loading ${className}`}>
        <div className="loading-content">
          <div className="spinner"></div>
          <p>Loading clause library...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`clause-library error ${className}`}>
        <div className="error-content">
          <h3>Error Loading Clauses</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`clause-library ${className}`} id="clause-library-main">
      {/* Header */}
      <div className="library-header" id="library-header">
        <div className="header-title">
          <h2>Clause Library</h2>
          <span className="clause-count">{filteredClauses.length} clauses</span>
        </div>
        
        <div className="header-actions">
          {enableBulkOperations && (
            <button
              onClick={() => setState(prev => ({ 
                ...prev, 
                bulkSelectMode: !prev.bulkSelectMode,
                selectedClauseIds: new Set()
              }))}
              className={`btn btn-outline ${state.bulkSelectMode ? 'active' : ''}`}
            >
              Bulk Select
            </button>
          )}
          
          <button
            onClick={() => setState(prev => ({ ...prev, showFilters: !prev.showFilters }))}
            className={`btn btn-outline ${state.showFilters ? 'active' : ''}`}
          >
            Filters
          </button>
          
          <div className="view-mode-selector">
            <button
              onClick={() => setState(prev => ({ ...prev, viewMode: 'grid' }))}
              className={`btn-icon ${state.viewMode === 'grid' ? 'active' : ''}`}
              title="Grid View"
            >
              ⊞
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, viewMode: 'list' }))}
              className={`btn-icon ${state.viewMode === 'list' ? 'active' : ''}`}
              title="List View"
            >
              ☰
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, viewMode: 'detailed' }))}
              className={`btn-icon ${state.viewMode === 'detailed' ? 'active' : ''}`}
              title="Detailed View"
            >
              📋
            </button>
          </div>
        </div>
      </div>

      {/* Search Bar */}
      <div className="search-section" id="search-section">
        <div className="search-input-container">
          <input
            type="text"
            placeholder="Search clauses by title, content, or tags..."
            value={state.searchTerm}
            onChange={(e) => setState(prev => ({ ...prev, searchTerm: e.target.value }))}
            className="search-input"
          />
          <button className="search-btn">🔍</button>
        </div>
        
        <div className="sort-controls">
          <select
            value={state.sortBy}
            onChange={(e) => setState(prev => ({ 
              ...prev, 
              sortBy: e.target.value as any 
            }))}
            className="sort-select"
          >
            <option value="relevance">Sort by Relevance</option>
            <option value="title">Sort by Title</option>
            <option value="category">Sort by Category</option>
            <option value="risk">Sort by Risk Level</option>
            <option value="usage">Sort by Usage</option>
            <option value="updated">Sort by Last Updated</option>
          </select>
          
          <button
            onClick={() => setState(prev => ({ 
              ...prev, 
              sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc' 
            }))}
            className="sort-order-btn"
            title={`Sort ${state.sortOrder === 'asc' ? 'Descending' : 'Ascending'}`}
          >
            {state.sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Bulk Operations Bar */}
      {state.bulkSelectMode && state.selectedClauseIds.size > 0 && (
        <div className="bulk-operations-bar" id="bulk-operations-bar">
          <span className="selected-count">
            {state.selectedClauseIds.size} clause{state.selectedClauseIds.size !== 1 ? 's' : ''} selected
          </span>
          
          <div className="bulk-actions">
            <button
              onClick={() => handleBulkOperation('approve')}
              className="btn btn-outline"
            >
              Approve Selected
            </button>
            <button
              onClick={() => handleBulkOperation('export')}
              className="btn btn-outline"
            >
              Export Selected
            </button>
            <button
              onClick={() => handleBulkOperation('delete')}
              className="btn btn-outline danger"
            >
              Delete Selected
            </button>
          </div>
        </div>
      )}

      {/* Filters Panel */}
      {state.showFilters && showAdvancedSearch && (
        <div className="filters-panel" id="filters-panel">
          <div className="filter-group">
            <label>Risk Levels:</label>
            <div className="checkbox-group">
              {Object.values(RiskLevel).map(level => (
                <label key={level} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={state.activeFilters.riskLevels?.includes(level) || false}
                    onChange={(e) => {
                      setState(prev => {
                        const riskLevels = prev.activeFilters.riskLevels || [];
                        return {
                          ...prev,
                          activeFilters: {
                            ...prev.activeFilters,
                            riskLevels: e.target.checked
                              ? [...riskLevels, level]
                              : riskLevels.filter(r => r !== level)
                          }
                        };
                      });
                    }}
                  />
                  {level}
                </label>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label>Industries:</label>
            <div className="checkbox-group">
              {Object.values(IndustryType).slice(0, 6).map(industry => (
                <label key={industry} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={state.activeFilters.industries?.includes(industry) || false}
                    onChange={(e) => {
                      setState(prev => {
                        const industries = prev.activeFilters.industries || [];
                        return {
                          ...prev,
                          activeFilters: {
                            ...prev.activeFilters,
                            industries: e.target.checked
                              ? [...industries, industry]
                              : industries.filter(i => i !== industry)
                          }
                        };
                      });
                    }}
                  />
                  {industry}
                </label>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label>Languages:</label>
            <select
              multiple
              value={state.activeFilters.languages || []}
              onChange={(e) => {
                const languages = Array.from(e.target.selectedOptions, option => option.value as LanguageCode);
                setState(prev => ({
                  ...prev,
                  activeFilters: { ...prev.activeFilters, languages }
                }));
              }}
              className="multi-select"
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="zh">Chinese</option>
              <option value="ja">Japanese</option>
            </select>
          </div>

          <div className="filter-actions">
            <button
              onClick={() => setState(prev => ({
                ...prev,
                activeFilters: {
                  keywords: [],
                  categories: [],
                  riskLevels: [],
                  industries: [],
                  languages: [],
                  jurisdiction: [],
                  approvalStatus: true
                }
              }))}
              className="btn btn-outline"
            >
              Clear Filters
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="library-content" id="library-content">
        {/* Category Navigation */}
        <div className="category-nav" id="category-nav">
          <button
            onClick={() => setState(prev => ({ ...prev, activeCategory: 'ALL' }))}
            className={`category-btn ${state.activeCategory === 'ALL' ? 'active' : ''}`}
          >
            All Categories
          </button>
          
          {categories.map(category => (
            <button
              key={category.key}
              onClick={() => setState(prev => ({ ...prev, activeCategory: category.key }))}
              className={`category-btn ${state.activeCategory === category.key ? 'active' : ''}`}
            >
              {category.label}
              <span className="category-count">({category.count})</span>
            </button>
          ))}
        </div>

        {/* Clauses Grid/List */}
        <DragDropContext onDragEnd={() => {}}>
          <Droppable droppableId="clause-library" isDropDisabled={readonly}>
            {(provided) => (
              <div
                ref={provided.innerRef}
                {...provided.droppableProps}
                className={`clauses-container ${state.viewMode}`}
                id="clauses-container"
              >
                {filteredClauses.map((clause, index) => (
                  <Draggable
                    key={clause.id}
                    draggableId={clause.id}
                    index={index}
                    isDragDisabled={readonly}
                  >
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                      >
                        <ClauseCard
                          clause={clause}
                          isSelected={state.selectedClauseIds.has(clause.id) || selectedClauses.includes(clause.id)}
                          isDragDisabled={readonly}
                          onSelect={handleClauseSelect}
                          onPreview={handleClausePreview}
                          onEdit={readonly ? undefined : updateClause}
                          onDelete={readonly ? undefined : deleteClause}
                          showMetadata={state.viewMode === 'detailed'}
                          viewMode={state.viewMode}
                        />
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
                
                {filteredClauses.length === 0 && (
                  <div className="empty-state">
                    <h3>No clauses found</h3>
                    <p>Try adjusting your search terms or filters.</p>
                  </div>
                )}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>

      {/* Preview Modal */}
      {state.showPreview && state.previewClause && (
        <div className="modal-overlay" id="clause-preview-modal">
          <div className="modal large">
            <div className="modal-header">
              <h3>{state.previewClause.title}</h3>
              <button
                onClick={() => setState(prev => ({ 
                  ...prev, 
                  showPreview: false, 
                  previewClause: null 
                }))}
                className="modal-close"
              >
                ×
              </button>
            </div>
            
            <div className="modal-content">
              <div className="clause-preview-content">
                <div className="preview-metadata">
                  <div className="metadata-grid">
                    <div className="metadata-item">
                      <label>Category:</label>
                      <span>{state.previewClause.category}</span>
                    </div>
                    <div className="metadata-item">
                      <label>Risk Level:</label>
                      <span className={`risk-badge ${getRiskColor(state.previewClause.riskLevel)}`}>
                        {state.previewClause.riskLevel}
                      </span>
                    </div>
                    <div className="metadata-item">
                      <label>Language:</label>
                      <span>{state.previewClause.language}</span>
                    </div>
                    <div className="metadata-item">
                      <label>Version:</label>
                      <span>{state.previewClause.version}</span>
                    </div>
                  </div>
                </div>
                
                <div className="preview-content">
                  <h4>Clause Content:</h4>
                  <div className="clause-text">
                    {state.previewClause.content}
                  </div>
                </div>
                
                {state.previewClause.variables.length > 0 && (
                  <div className="preview-variables">
                    <h4>Variables:</h4>
                    <div className="variables-list">
                      {state.previewClause.variables.map(variable => (
                        <div key={variable.id} className="variable-item">
                          <span className="variable-name">{variable.name}</span>
                          <span className="variable-type">({variable.type})</span>
                          {variable.required && <span className="required">*</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {state.previewClause.tags.length > 0 && (
                  <div className="preview-tags">
                    <h4>Tags:</h4>
                    <div className="tags-container">
                      {state.previewClause.tags.map(tag => (
                        <span key={tag} className="tag">{tag}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <div className="modal-footer">
              <button
                onClick={() => {
                  onClauseSelect?.(state.previewClause!);
                  setState(prev => ({ 
                    ...prev, 
                    showPreview: false, 
                    previewClause: null 
                  }));
                }}
                className="btn btn-primary"
                disabled={readonly}
              >
                Add to Contract
              </button>
              <button
                onClick={() => setState(prev => ({ 
                  ...prev, 
                  showPreview: false, 
                  previewClause: null 
                }))}
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

// Helper function for risk color (define outside component to avoid recreation)
const getRiskColor = (risk: RiskLevel): string => {
  switch (risk) {
    case RiskLevel.LOW: return 'green';
    case RiskLevel.MEDIUM: return 'yellow';
    case RiskLevel.HIGH: return 'orange';
    case RiskLevel.CRITICAL: return 'red';
    default: return 'gray';
  }
};

export default ClauseLibrary;