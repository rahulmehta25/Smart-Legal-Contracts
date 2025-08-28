/**
 * VariableManager.tsx - Variable Substitution Engine
 * 
 * Advanced variable management system with type validation, conditional logic,
 * auto-completion, and dynamic dependency resolution.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  VariableDefinition,
  ValidationRule,
  UUID,
  ContractParty,
  Address,
  ContactInformation
} from './types';

interface VariableManagerProps {
  variables: Record<string, any>;
  variableDefinitions?: VariableDefinition[];
  onVariableUpdate: (variables: Record<string, any>) => void;
  readonly?: boolean;
  showAdvancedFeatures?: boolean;
  enableAutoComplete?: boolean;
  contractParties?: ContractParty[];
  className?: string;
}

interface ManagerState {
  activeCategory: 'all' | 'required' | 'optional' | 'parties' | 'dates' | 'financial' | 'custom';
  searchTerm: string;
  validationErrors: Record<string, string[]>;
  showValidationPanel: boolean;
  showPreviewPanel: boolean;
  editingVariable: UUID | null;
  bulkEditMode: boolean;
  selectedVariables: Set<UUID>;
  autoCompleteCache: Record<string, string[]>;
}

interface VariableEditorProps {
  variable: VariableDefinition;
  value: any;
  onValueChange: (value: any) => void;
  onDefinitionChange?: (definition: VariableDefinition) => void;
  readonly?: boolean;
  validationErrors?: string[];
  autoCompleteSuggestions?: string[];
}

interface ValidatorService {
  validateVariable: (value: any, definition: VariableDefinition) => string[];
  validateAllVariables: (variables: Record<string, any>, definitions: VariableDefinition[]) => Record<string, string[]>;
}

// Variable validation service
const validatorService: ValidatorService = {
  validateVariable: (value: any, definition: VariableDefinition): string[] => {
    const errors: string[] = [];
    
    // Required validation
    if (definition.required && (value === undefined || value === null || value === '')) {
      errors.push(`${definition.name} is required`);
      return errors; // Skip other validations if required field is empty
    }
    
    // Skip validation if value is empty and not required
    if (!definition.required && (value === undefined || value === null || value === '')) {
      return errors;
    }
    
    // Type validation
    switch (definition.type) {
      case 'string':
        if (typeof value !== 'string') {
          errors.push(`${definition.name} must be a string`);
        }
        break;
      case 'number':
        if (typeof value !== 'number' || isNaN(value)) {
          errors.push(`${definition.name} must be a valid number`);
        }
        break;
      case 'date':
        if (!(value instanceof Date) && isNaN(Date.parse(value))) {
          errors.push(`${definition.name} must be a valid date`);
        }
        break;
      case 'boolean':
        if (typeof value !== 'boolean') {
          errors.push(`${definition.name} must be true or false`);
        }
        break;
      case 'currency':
        if (typeof value !== 'number' || value < 0) {
          errors.push(`${definition.name} must be a positive number`);
        }
        break;
      case 'percentage':
        if (typeof value !== 'number' || value < 0 || value > 100) {
          errors.push(`${definition.name} must be between 0 and 100`);
        }
        break;
    }
    
    // Custom validation rules
    if (definition.validation) {
      for (const rule of definition.validation) {
        switch (rule.type) {
          case 'minLength':
            if (typeof value === 'string' && value.length < rule.value) {
              errors.push(rule.message);
            }
            break;
          case 'maxLength':
            if (typeof value === 'string' && value.length > rule.value) {
              errors.push(rule.message);
            }
            break;
          case 'pattern':
            if (typeof value === 'string' && !new RegExp(rule.value).test(value)) {
              errors.push(rule.message);
            }
            break;
          case 'range':
            if (typeof value === 'number' && (value < rule.value.min || value > rule.value.max)) {
              errors.push(rule.message);
            }
            break;
          case 'custom':
            if (rule.customValidator && !rule.customValidator(value)) {
              errors.push(rule.message);
            }
            break;
        }
      }
    }
    
    return errors;
  },
  
  validateAllVariables: (variables: Record<string, any>, definitions: VariableDefinition[]): Record<string, string[]> => {
    const allErrors: Record<string, string[]> = {};
    
    for (const definition of definitions) {
      const value = variables[definition.name];
      const errors = validatorService.validateVariable(value, definition);
      if (errors.length > 0) {
        allErrors[definition.name] = errors;
      }
    }
    
    return allErrors;
  }
};

const VariableEditor: React.FC<VariableEditorProps> = ({
  variable,
  value,
  onValueChange,
  onDefinitionChange,
  readonly = false,
  validationErrors = [],
  autoCompleteSuggestions = []
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleValueChange = useCallback((newValue: any) => {
    setLocalValue(newValue);
    onValueChange(newValue);
  }, [onValueChange]);

  const renderInput = () => {
    const commonProps = {
      id: `variable-${variable.id}`,
      value: localValue || '',
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        handleValueChange(e.target.value);
      },
      placeholder: variable.placeholder || `Enter ${variable.name}...`,
      disabled: readonly,
      className: `variable-input ${validationErrors.length > 0 ? 'error' : ''}`
    };

    switch (variable.type) {
      case 'string':
        if (variable.validation?.some(rule => rule.type === 'maxLength' && rule.value > 100)) {
          return (
            <textarea
              {...commonProps}
              rows={4}
              maxLength={variable.validation?.find(rule => rule.type === 'maxLength')?.value}
            />
          );
        }
        return (
          <div className="input-with-suggestions">
            <input
              {...commonProps}
              type="text"
              onFocus={() => setShowSuggestions(autoCompleteSuggestions.length > 0)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            />
            {showSuggestions && autoCompleteSuggestions.length > 0 && (
              <div className="suggestions-dropdown">
                {autoCompleteSuggestions
                  .filter(suggestion => 
                    suggestion.toLowerCase().includes((localValue || '').toLowerCase())
                  )
                  .slice(0, 5)
                  .map((suggestion, index) => (
                    <div
                      key={index}
                      className="suggestion-item"
                      onClick={() => {
                        handleValueChange(suggestion);
                        setShowSuggestions(false);
                      }}
                    >
                      {suggestion}
                    </div>
                  ))}
              </div>
            )}
          </div>
        );

      case 'number':
        return (
          <input
            {...commonProps}
            type="number"
            step="any"
            min={variable.validation?.find(rule => rule.type === 'range')?.value?.min}
            max={variable.validation?.find(rule => rule.type === 'range')?.value?.max}
          />
        );

      case 'currency':
        return (
          <div className="currency-input">
            <span className="currency-symbol">$</span>
            <input
              {...commonProps}
              type="number"
              step="0.01"
              min="0"
            />
          </div>
        );

      case 'percentage':
        return (
          <div className="percentage-input">
            <input
              {...commonProps}
              type="number"
              step="0.1"
              min="0"
              max="100"
            />
            <span className="percentage-symbol">%</span>
          </div>
        );

      case 'date':
        return (
          <input
            {...commonProps}
            type="date"
            value={localValue ? new Date(localValue).toISOString().split('T')[0] : ''}
            onChange={(e) => handleValueChange(new Date(e.target.value))}
          />
        );

      case 'boolean':
        return (
          <div className="boolean-input">
            <label className="radio-label">
              <input
                type="radio"
                name={`variable-${variable.id}`}
                checked={localValue === true}
                onChange={() => handleValueChange(true)}
                disabled={readonly}
              />
              Yes
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name={`variable-${variable.id}`}
                checked={localValue === false}
                onChange={() => handleValueChange(false)}
                disabled={readonly}
              />
              No
            </label>
          </div>
        );

      case 'list':
        return (
          <select {...commonProps} multiple={variable.validation?.some(rule => rule.type === 'multiple')}>
            {variable.validation
              ?.find(rule => rule.type === 'options')
              ?.value?.map((option: string) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
          </select>
        );

      default:
        return <input {...commonProps} type="text" />;
    }
  };

  return (
    <div className="variable-editor" data-variable-id={variable.id}>
      <div className="variable-header">
        <label htmlFor={`variable-${variable.id}`} className="variable-label">
          {variable.name}
          {variable.required && <span className="required-indicator">*</span>}
        </label>
        
        {variable.description && (
          <div className="variable-description" title={variable.description}>
            ℹ️
          </div>
        )}
        
        {onDefinitionChange && (
          <button
            className="variable-edit-btn"
            onClick={() => {/* Open variable definition editor */}}
            disabled={readonly}
            title="Edit Variable Definition"
          >
            ⚙️
          </button>
        )}
      </div>

      <div className="variable-input-container">
        {renderInput()}
      </div>

      {validationErrors.length > 0 && (
        <div className="variable-errors">
          {validationErrors.map((error, index) => (
            <div key={index} className="error-message">
              {error}
            </div>
          ))}
        </div>
      )}

      {variable.dependencies && variable.dependencies.length > 0 && (
        <div className="variable-dependencies">
          <small>Depends on: {variable.dependencies.join(', ')}</small>
        </div>
      )}
    </div>
  );
};

export const VariableManager: React.FC<VariableManagerProps> = ({
  variables,
  variableDefinitions = [],
  onVariableUpdate,
  readonly = false,
  showAdvancedFeatures = true,
  enableAutoComplete = true,
  contractParties = [],
  className = ''
}) => {
  const [state, setState] = useState<ManagerState>({
    activeCategory: 'all',
    searchTerm: '',
    validationErrors: {},
    showValidationPanel: false,
    showPreviewPanel: false,
    editingVariable: null,
    bulkEditMode: false,
    selectedVariables: new Set(),
    autoCompleteCache: {}
  });

  // Enhanced variable definitions with common contract variables
  const enhancedDefinitions = useMemo((): VariableDefinition[] => {
    const commonVariables: VariableDefinition[] = [
      {
        id: 'contract-title',
        name: 'Contract Title',
        type: 'string',
        required: true,
        placeholder: 'Enter contract title...',
        validation: [
          { type: 'required', message: 'Contract title is required' },
          { type: 'minLength', value: 5, message: 'Title must be at least 5 characters' },
          { type: 'maxLength', value: 200, message: 'Title must not exceed 200 characters' }
        ]
      },
      {
        id: 'effective-date',
        name: 'Effective Date',
        type: 'date',
        required: true,
        defaultValue: new Date()
      },
      {
        id: 'expiration-date',
        name: 'Expiration Date',
        type: 'date',
        required: false
      },
      {
        id: 'contract-value',
        name: 'Contract Value',
        type: 'currency',
        required: false,
        validation: [
          { type: 'range', value: { min: 0, max: 1000000000 }, message: 'Value must be positive' }
        ]
      },
      {
        id: 'governing-law',
        name: 'Governing Law',
        type: 'string',
        required: true,
        defaultValue: 'United States',
        validation: [
          { type: 'required', message: 'Governing law must be specified' }
        ]
      },
      {
        id: 'notice-period',
        name: 'Notice Period (days)',
        type: 'number',
        required: false,
        defaultValue: 30,
        validation: [
          { type: 'range', value: { min: 1, max: 365 }, message: 'Notice period must be between 1 and 365 days' }
        ]
      }
    ];

    // Add party-specific variables
    const partyVariables: VariableDefinition[] = contractParties.flatMap((party, index) => [
      {
        id: `party-${index}-name`,
        name: `${party.role} Name`,
        type: 'string',
        required: true,
        defaultValue: party.name
      },
      {
        id: `party-${index}-address`,
        name: `${party.role} Address`,
        type: 'string',
        required: true,
        defaultValue: `${party.contact.address.street}, ${party.contact.address.city}, ${party.contact.address.state} ${party.contact.address.postalCode}`
      },
      {
        id: `party-${index}-email`,
        name: `${party.role} Email`,
        type: 'string',
        required: false,
        defaultValue: party.contact.email
      }
    ]);

    return [...commonVariables, ...partyVariables, ...variableDefinitions];
  }, [variableDefinitions, contractParties]);

  // Categorized variables
  const categorizedVariables = useMemo(() => {
    const categories = {
      required: enhancedDefinitions.filter(def => def.required),
      optional: enhancedDefinitions.filter(def => !def.required),
      parties: enhancedDefinitions.filter(def => def.name.toLowerCase().includes('party') || def.name.toLowerCase().includes('name') || def.name.toLowerCase().includes('address')),
      dates: enhancedDefinitions.filter(def => def.type === 'date'),
      financial: enhancedDefinitions.filter(def => def.type === 'currency' || def.type === 'percentage' || def.name.toLowerCase().includes('value') || def.name.toLowerCase().includes('amount')),
      custom: enhancedDefinitions.filter(def => !variableDefinitions.some(vd => vd.id === def.id))
    };

    return categories;
  }, [enhancedDefinitions, variableDefinitions]);

  // Filtered variables based on active category and search
  const filteredVariables = useMemo(() => {
    let filtered = state.activeCategory === 'all' 
      ? enhancedDefinitions 
      : categorizedVariables[state.activeCategory] || [];

    if (state.searchTerm) {
      const searchLower = state.searchTerm.toLowerCase();
      filtered = filtered.filter(def =>
        def.name.toLowerCase().includes(searchLower) ||
        def.description?.toLowerCase().includes(searchLower)
      );
    }

    return filtered;
  }, [enhancedDefinitions, categorizedVariables, state.activeCategory, state.searchTerm]);

  // Validation effect
  useEffect(() => {
    const errors = validatorService.validateAllVariables(variables, enhancedDefinitions);
    setState(prev => ({ ...prev, validationErrors: errors }));
  }, [variables, enhancedDefinitions]);

  // Auto-complete suggestions
  const getAutoCompleteSuggestions = useCallback((variableName: string): string[] => {
    if (!enableAutoComplete) return [];

    // Common suggestions based on variable type
    const suggestions: Record<string, string[]> = {
      'governing-law': ['United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France'],
      'contract-title': ['Service Agreement', 'Non-Disclosure Agreement', 'Employment Contract', 'Purchase Order', 'License Agreement'],
      'currency': ['USD', 'EUR', 'GBP', 'CAD', 'AUD'],
      'jurisdiction': ['Delaware', 'New York', 'California', 'Texas', 'Illinois']
    };

    return suggestions[variableName] || suggestions[variableName.toLowerCase()] || [];
  }, [enableAutoComplete]);

  // Handle variable value change
  const handleVariableChange = useCallback((variableId: string, value: any) => {
    const variable = enhancedDefinitions.find(def => def.id === variableId);
    if (!variable) return;

    const updatedVariables = { ...variables, [variable.name]: value };
    
    // Handle dependencies
    if (variable.dependencies) {
      // Update dependent variables if needed
      for (const depId of variable.dependencies) {
        const depVariable = enhancedDefinitions.find(def => def.id === depId);
        if (depVariable && !updatedVariables[depVariable.name]) {
          updatedVariables[depVariable.name] = depVariable.defaultValue;
        }
      }
    }

    onVariableUpdate(updatedVariables);
  }, [variables, enhancedDefinitions, onVariableUpdate]);

  // Handle bulk operations
  const handleBulkOperation = useCallback((operation: 'clear' | 'reset' | 'validate') => {
    const selectedIds = Array.from(state.selectedVariables);
    
    switch (operation) {
      case 'clear':
        const clearedVariables = { ...variables };
        selectedIds.forEach(id => {
          const variable = enhancedDefinitions.find(def => def.id === id);
          if (variable) {
            delete clearedVariables[variable.name];
          }
        });
        onVariableUpdate(clearedVariables);
        break;
        
      case 'reset':
        const resetVariables = { ...variables };
        selectedIds.forEach(id => {
          const variable = enhancedDefinitions.find(def => def.id === id);
          if (variable && variable.defaultValue !== undefined) {
            resetVariables[variable.name] = variable.defaultValue;
          }
        });
        onVariableUpdate(resetVariables);
        break;
        
      case 'validate':
        const errors = validatorService.validateAllVariables(variables, enhancedDefinitions);
        setState(prev => ({ 
          ...prev, 
          validationErrors: errors,
          showValidationPanel: true 
        }));
        break;
    }
    
    setState(prev => ({ 
      ...prev, 
      selectedVariables: new Set(),
      bulkEditMode: false 
    }));
  }, [state.selectedVariables, variables, enhancedDefinitions, onVariableUpdate]);

  // Export variables as JSON
  const exportVariables = useCallback(() => {
    const dataStr = JSON.stringify(variables, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'contract-variables.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }, [variables]);

  // Import variables from JSON
  const importVariables = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedVariables = JSON.parse(e.target?.result as string);
        onVariableUpdate({ ...variables, ...importedVariables });
      } catch (error) {
        console.error('Failed to import variables:', error);
      }
    };
    reader.readAsText(file);
  }, [variables, onVariableUpdate]);

  const hasValidationErrors = Object.keys(state.validationErrors).length > 0;
  const completionPercentage = Math.round(
    (Object.keys(variables).length / enhancedDefinitions.filter(def => def.required).length) * 100
  );

  return (
    <div className={`variable-manager ${className}`} id="variable-manager-main">
      {/* Header */}
      <div className="manager-header" id="manager-header">
        <div className="header-title">
          <h3>Variable Manager</h3>
          <div className="completion-indicator">
            <div className="completion-bar">
              <div 
                className="completion-fill" 
                style={{ width: `${Math.min(completionPercentage, 100)}%` }}
              ></div>
            </div>
            <span className="completion-text">
              {completionPercentage}% complete
            </span>
          </div>
        </div>

        <div className="header-actions">
          {showAdvancedFeatures && (
            <>
              <button
                onClick={() => setState(prev => ({ 
                  ...prev, 
                  bulkEditMode: !prev.bulkEditMode,
                  selectedVariables: new Set()
                }))}
                className={`btn btn-outline ${state.bulkEditMode ? 'active' : ''}`}
                disabled={readonly}
              >
                Bulk Edit
              </button>
              
              <button
                onClick={exportVariables}
                className="btn btn-outline"
                title="Export Variables"
              >
                📤
              </button>
              
              <label className="btn btn-outline" title="Import Variables">
                📥
                <input
                  type="file"
                  accept=".json"
                  onChange={importVariables}
                  style={{ display: 'none' }}
                  disabled={readonly}
                />
              </label>
            </>
          )}
          
          <button
            onClick={() => setState(prev => ({ 
              ...prev, 
              showValidationPanel: !prev.showValidationPanel 
            }))}
            className={`btn btn-outline ${hasValidationErrors ? 'error' : ''}`}
          >
            Validation {hasValidationErrors ? `(${Object.keys(state.validationErrors).length})` : '✓'}
          </button>
        </div>
      </div>

      {/* Search and Category Filter */}
      <div className="manager-filters" id="manager-filters">
        <div className="search-section">
          <input
            type="text"
            placeholder="Search variables..."
            value={state.searchTerm}
            onChange={(e) => setState(prev => ({ ...prev, searchTerm: e.target.value }))}
            className="search-input"
          />
        </div>

        <div className="category-tabs">
          {[
            { key: 'all', label: 'All Variables', count: enhancedDefinitions.length },
            { key: 'required', label: 'Required', count: categorizedVariables.required.length },
            { key: 'optional', label: 'Optional', count: categorizedVariables.optional.length },
            { key: 'parties', label: 'Parties', count: categorizedVariables.parties.length },
            { key: 'dates', label: 'Dates', count: categorizedVariables.dates.length },
            { key: 'financial', label: 'Financial', count: categorizedVariables.financial.length }
          ].map(category => (
            <button
              key={category.key}
              onClick={() => setState(prev => ({ 
                ...prev, 
                activeCategory: category.key as any 
              }))}
              className={`category-tab ${state.activeCategory === category.key ? 'active' : ''}`}
            >
              {category.label}
              <span className="category-count">({category.count})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Bulk Operations Bar */}
      {state.bulkEditMode && state.selectedVariables.size > 0 && (
        <div className="bulk-operations-bar" id="bulk-operations-bar">
          <span className="selected-count">
            {state.selectedVariables.size} variable{state.selectedVariables.size !== 1 ? 's' : ''} selected
          </span>
          
          <div className="bulk-actions">
            <button
              onClick={() => handleBulkOperation('reset')}
              className="btn btn-outline"
            >
              Reset to Default
            </button>
            <button
              onClick={() => handleBulkOperation('clear')}
              className="btn btn-outline"
            >
              Clear Values
            </button>
            <button
              onClick={() => handleBulkOperation('validate')}
              className="btn btn-outline"
            >
              Validate Selected
            </button>
          </div>
        </div>
      )}

      {/* Validation Panel */}
      {state.showValidationPanel && (
        <div className="validation-panel" id="validation-panel">
          <div className="panel-header">
            <h4>Validation Results</h4>
            <button
              onClick={() => setState(prev => ({ ...prev, showValidationPanel: false }))}
              className="panel-close"
            >
              ×
            </button>
          </div>
          
          <div className="validation-content">
            {hasValidationErrors ? (
              Object.entries(state.validationErrors).map(([variableName, errors]) => (
                <div key={variableName} className="validation-error-group">
                  <h5>{variableName}</h5>
                  {errors.map((error, index) => (
                    <div key={index} className="error-message">
                      ❌ {error}
                    </div>
                  ))}
                </div>
              ))
            ) : (
              <div className="validation-success">
                ✅ All variables are valid
              </div>
            )}
          </div>
        </div>
      )}

      {/* Variables Grid */}
      <div className="variables-grid" id="variables-grid">
        {filteredVariables.map(variable => (
          <div key={variable.id} className="variable-item">
            {state.bulkEditMode && (
              <div className="variable-checkbox">
                <input
                  type="checkbox"
                  checked={state.selectedVariables.has(variable.id)}
                  onChange={(e) => {
                    setState(prev => {
                      const newSelected = new Set(prev.selectedVariables);
                      if (e.target.checked) {
                        newSelected.add(variable.id);
                      } else {
                        newSelected.delete(variable.id);
                      }
                      return { ...prev, selectedVariables: newSelected };
                    });
                  }}
                />
              </div>
            )}
            
            <VariableEditor
              variable={variable}
              value={variables[variable.name]}
              onValueChange={(value) => handleVariableChange(variable.id, value)}
              readonly={readonly}
              validationErrors={state.validationErrors[variable.name] || []}
              autoCompleteSuggestions={getAutoCompleteSuggestions(variable.name)}
            />
          </div>
        ))}

        {filteredVariables.length === 0 && (
          <div className="empty-state">
            <h4>No variables found</h4>
            <p>Try adjusting your search or category filter.</p>
          </div>
        )}
      </div>

      {/* Variable Preview Panel */}
      {state.showPreviewPanel && (
        <div className="preview-panel" id="preview-panel">
          <div className="panel-header">
            <h4>Variable Preview</h4>
            <button
              onClick={() => setState(prev => ({ ...prev, showPreviewPanel: false }))}
              className="panel-close"
            >
              ×
            </button>
          </div>
          
          <div className="preview-content">
            <pre>{JSON.stringify(variables, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default VariableManager;