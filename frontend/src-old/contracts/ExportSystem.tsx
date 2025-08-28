/**
 * ExportSystem.tsx - Advanced Export System Component
 * 
 * Comprehensive export interface with format selection, styling options,
 * batch operations, and integration with document generation services.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  ContractDraft,
  ExportConfiguration,
  DocumentFormat,
  DocumentStyling,
  HeaderFooterConfig,
  WatermarkConfig,
  ExportSection,
  HeadingStyle,
  DocumentMargins,
  UUID
} from './types';
import { documentGeneratorService } from './utils/documentGenerator';

interface ExportSystemProps {
  contracts: ContractDraft[];
  selectedContractIds?: UUID[];
  onExport?: (format: DocumentFormat, contractIds: UUID[]) => void;
  onExportComplete?: (results: ExportResult[]) => void;
  enableBatchExport?: boolean;
  enableAdvancedOptions?: boolean;
  defaultFormat?: DocumentFormat;
  className?: string;
}

interface ExportResult {
  contractId: UUID;
  format: DocumentFormat;
  blob: Blob;
  filename: string;
  success: boolean;
  error?: string;
}

interface ExportState {
  activeTab: 'format' | 'styling' | 'sections' | 'metadata' | 'preview';
  selectedFormat: DocumentFormat;
  configuration: ExportConfiguration;
  isExporting: boolean;
  exportProgress: number;
  previewContent: string | null;
  batchMode: boolean;
  selectedContractIds: Set<UUID>;
}

interface FormatOption {
  format: DocumentFormat;
  name: string;
  description: string;
  icon: string;
  features: string[];
  recommended?: boolean;
}

interface StylePreset {
  name: string;
  description: string;
  styling: DocumentStyling;
}

const FORMAT_OPTIONS: FormatOption[] = [
  {
    format: 'PDF',
    name: 'PDF Document',
    description: 'Professional PDF format for final contracts and legal documents',
    icon: '📄',
    features: ['Digital signatures', 'Print-ready', 'Universal compatibility', 'Watermarks'],
    recommended: true
  },
  {
    format: 'DOCX',
    name: 'Word Document',
    description: 'Microsoft Word format for editing and collaboration',
    icon: '📝',
    features: ['Editable', 'Track changes', 'Comments', 'Version control']
  },
  {
    format: 'HTML',
    name: 'Web Document',
    description: 'HTML format for web publishing and responsive viewing',
    icon: '🌐',
    features: ['Responsive', 'Interactive', 'Web-friendly', 'Searchable']
  },
  {
    format: 'JSON',
    name: 'Data Export',
    description: 'Structured data format for system integration and analysis',
    icon: '📊',
    features: ['Structured data', 'API integration', 'Machine-readable', 'Metadata included']
  }
];

const STYLE_PRESETS: StylePreset[] = [
  {
    name: 'Professional Legal',
    description: 'Traditional legal document styling with formal typography',
    styling: {
      fontFamily: 'Times New Roman',
      fontSize: 12,
      lineSpacing: 1.15,
      margins: { top: 1, bottom: 1, left: 1.25, right: 1 },
      headingStyles: [
        { level: 1, fontSize: 16, bold: true, italic: false, spacing: { before: 18, after: 12 } },
        { level: 2, fontSize: 14, bold: true, italic: false, spacing: { before: 14, after: 8 } },
        { level: 3, fontSize: 12, bold: true, italic: false, spacing: { before: 10, after: 6 } }
      ],
      pageSize: 'US_LETTER',
      orientation: 'portrait'
    }
  },
  {
    name: 'Modern Business',
    description: 'Clean, modern styling for contemporary business contracts',
    styling: {
      fontFamily: 'Calibri',
      fontSize: 11,
      lineSpacing: 1.25,
      margins: { top: 1, bottom: 1, left: 1, right: 1 },
      headingStyles: [
        { level: 1, fontSize: 18, bold: true, italic: false, spacing: { before: 20, after: 14 } },
        { level: 2, fontSize: 14, bold: true, italic: false, spacing: { before: 16, after: 10 } },
        { level: 3, fontSize: 12, bold: true, italic: false, spacing: { before: 12, after: 8 } }
      ],
      pageSize: 'A4',
      orientation: 'portrait'
    }
  },
  {
    name: 'Compact',
    description: 'Space-efficient styling for shorter documents',
    styling: {
      fontFamily: 'Arial',
      fontSize: 10,
      lineSpacing: 1.1,
      margins: { top: 0.75, bottom: 0.75, left: 0.75, right: 0.75 },
      headingStyles: [
        { level: 1, fontSize: 14, bold: true, italic: false, spacing: { before: 12, after: 8 } },
        { level: 2, fontSize: 12, bold: true, italic: false, spacing: { before: 10, after: 6 } },
        { level: 3, fontSize: 11, bold: true, italic: false, spacing: { before: 8, after: 4 } }
      ],
      pageSize: 'A4',
      orientation: 'portrait'
    }
  }
];

export const ExportSystem: React.FC<ExportSystemProps> = ({
  contracts,
  selectedContractIds = [],
  onExport,
  onExportComplete,
  enableBatchExport = true,
  enableAdvancedOptions = true,
  defaultFormat = 'PDF',
  className = ''
}) => {
  const [state, setState] = useState<ExportState>({
    activeTab: 'format',
    selectedFormat: defaultFormat,
    configuration: {
      format: defaultFormat,
      includeComments: false,
      includeTrackChanges: true,
      includeMetadata: true,
      styling: STYLE_PRESETS[0].styling,
      sections: [
        { type: 'title_page', include: true },
        { type: 'toc', include: true },
        { type: 'contract_body', include: true },
        { type: 'signature_page', include: true },
        { type: 'appendix', include: false }
      ]
    },
    isExporting: false,
    exportProgress: 0,
    previewContent: null,
    batchMode: selectedContractIds.length > 1,
    selectedContractIds: new Set(selectedContractIds)
  });

  // Available contracts for export
  const availableContracts = useMemo(() => {
    return contracts.filter(contract => 
      state.selectedContractIds.size === 0 || 
      state.selectedContractIds.has(contract.id)
    );
  }, [contracts, state.selectedContractIds]);

  // Update configuration when format changes
  useEffect(() => {
    setState(prev => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        format: state.selectedFormat
      }
    }));
  }, [state.selectedFormat]);

  // Handle format selection
  const handleFormatChange = useCallback((format: DocumentFormat) => {
    setState(prev => ({
      ...prev,
      selectedFormat: format,
      configuration: {
        ...prev.configuration,
        format
      }
    }));
  }, []);

  // Handle style preset selection
  const handleStylePresetChange = useCallback((preset: StylePreset) => {
    setState(prev => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        styling: preset.styling
      }
    }));
  }, []);

  // Handle configuration updates
  const handleConfigurationChange = useCallback((updates: Partial<ExportConfiguration>) => {
    setState(prev => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        ...updates
      }
    }));
  }, []);

  // Handle section inclusion toggle
  const handleSectionToggle = useCallback((sectionType: string, include: boolean) => {
    setState(prev => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        sections: prev.configuration.sections.map(section =>
          section.type === sectionType ? { ...section, include } : section
        )
      }
    }));
  }, []);

  // Handle contract selection for batch export
  const handleContractSelection = useCallback((contractId: UUID, selected: boolean) => {
    setState(prev => {
      const newSelected = new Set(prev.selectedContractIds);
      if (selected) {
        newSelected.add(contractId);
      } else {
        newSelected.delete(contractId);
      }
      return {
        ...prev,
        selectedContractIds: newSelected,
        batchMode: newSelected.size > 1
      };
    });
  }, []);

  // Generate preview
  const generatePreview = useCallback(async () => {
    if (availableContracts.length === 0) return;

    const contract = availableContracts[0];
    try {
      const content = await documentGeneratorService.exporter.generateCompleteDocument(
        contract,
        state.configuration
      );
      setState(prev => ({ ...prev, previewContent: content }));
    } catch (error) {
      console.error('Preview generation failed:', error);
    }
  }, [availableContracts, state.configuration]);

  // Handle export
  const handleExport = useCallback(async () => {
    if (availableContracts.length === 0) return;

    setState(prev => ({ ...prev, isExporting: true, exportProgress: 0 }));
    
    const results: ExportResult[] = [];
    const contractsToExport = Array.from(state.selectedContractIds).length > 0
      ? contracts.filter(c => state.selectedContractIds.has(c.id))
      : availableContracts;

    try {
      for (let i = 0; i < contractsToExport.length; i++) {
        const contract = contractsToExport[i];
        
        try {
          const blob = await documentGeneratorService.exportContract(contract, state.configuration);
          const filename = `${contract.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.${state.selectedFormat.toLowerCase()}`;
          
          results.push({
            contractId: contract.id,
            format: state.selectedFormat,
            blob,
            filename,
            success: true
          });

          // Auto-download for single file exports
          if (contractsToExport.length === 1) {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          }
        } catch (error) {
          results.push({
            contractId: contract.id,
            format: state.selectedFormat,
            blob: new Blob(),
            filename: '',
            success: false,
            error: error instanceof Error ? error.message : 'Export failed'
          });
        }

        // Update progress
        setState(prev => ({
          ...prev,
          exportProgress: ((i + 1) / contractsToExport.length) * 100
        }));
      }

      // Handle batch download
      if (contractsToExport.length > 1) {
        await handleBatchDownload(results.filter(r => r.success));
      }

      // Notify completion
      onExportComplete?.(results);
      onExport?.(state.selectedFormat, contractsToExport.map(c => c.id));

    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setState(prev => ({ ...prev, isExporting: false, exportProgress: 0 }));
    }
  }, [availableContracts, contracts, state.configuration, state.selectedFormat, state.selectedContractIds, onExport, onExportComplete]);

  // Handle batch download (create ZIP file)
  const handleBatchDownload = useCallback(async (results: ExportResult[]) => {
    // In a real implementation, this would create a ZIP file
    // For now, we'll download files individually with a delay
    for (const result of results) {
      const url = URL.createObjectURL(result.blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      // Small delay between downloads
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }, []);

  return (
    <div className={`export-system ${className}`} id="export-system-main">
      {/* Header */}
      <div className="export-header" id="export-header">
        <div className="header-title">
          <h3>Export System</h3>
          <div className="export-info">
            {state.batchMode ? (
              <span>{state.selectedContractIds.size} contracts selected</span>
            ) : (
              <span>{availableContracts.length} contract{availableContracts.length !== 1 ? 's' : ''}</span>
            )}
          </div>
        </div>

        <div className="header-actions">
          {enableBatchExport && (
            <button
              onClick={() => setState(prev => ({ 
                ...prev, 
                batchMode: !prev.batchMode,
                selectedContractIds: new Set()
              }))}
              className={`btn btn-outline ${state.batchMode ? 'active' : ''}`}
            >
              Batch Mode
            </button>
          )}
          
          <button
            onClick={generatePreview}
            className="btn btn-outline"
            disabled={state.isExporting || availableContracts.length === 0}
          >
            Preview
          </button>
          
          <button
            onClick={handleExport}
            className="btn btn-primary"
            disabled={state.isExporting || availableContracts.length === 0}
          >
            {state.isExporting ? `Exporting... ${Math.round(state.exportProgress)}%` : 'Export'}
          </button>
        </div>
      </div>

      {/* Contract Selection (Batch Mode) */}
      {state.batchMode && enableBatchExport && (
        <div className="contract-selection" id="contract-selection">
          <h4>Select Contracts to Export</h4>
          <div className="contract-list">
            {contracts.map(contract => (
              <div key={contract.id} className="contract-item">
                <label className="contract-checkbox">
                  <input
                    type="checkbox"
                    checked={state.selectedContractIds.has(contract.id)}
                    onChange={(e) => handleContractSelection(contract.id, e.target.checked)}
                  />
                  <span className="contract-name">{contract.name}</span>
                  <span className="contract-meta">
                    {contract.status} • {contract.parties.length} parties • v{contract.version}
                  </span>
                </label>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export Tabs */}
      <div className="export-tabs" id="export-tabs">
        {[
          { key: 'format', label: 'Format', icon: '📄' },
          { key: 'styling', label: 'Styling', icon: '🎨' },
          { key: 'sections', label: 'Sections', icon: '📋' },
          { key: 'metadata', label: 'Metadata', icon: 'ℹ️' },
          { key: 'preview', label: 'Preview', icon: '👁️' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setState(prev => ({ ...prev, activeTab: tab.key as any }))}
            className={`tab ${state.activeTab === tab.key ? 'active' : ''}`}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content" id="export-tab-content">
        {/* Format Selection */}
        {state.activeTab === 'format' && (
          <div className="format-selection">
            <h4>Choose Export Format</h4>
            <div className="format-options">
              {FORMAT_OPTIONS.map(option => (
                <div
                  key={option.format}
                  className={`format-option ${state.selectedFormat === option.format ? 'selected' : ''}`}
                  onClick={() => handleFormatChange(option.format)}
                >
                  <div className="format-header">
                    <span className="format-icon">{option.icon}</span>
                    <div className="format-info">
                      <h5 className="format-name">
                        {option.name}
                        {option.recommended && <span className="recommended-badge">Recommended</span>}
                      </h5>
                      <p className="format-description">{option.description}</p>
                    </div>
                  </div>
                  
                  <div className="format-features">
                    {option.features.map(feature => (
                      <span key={feature} className="feature-tag">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Styling Options */}
        {state.activeTab === 'styling' && enableAdvancedOptions && (
          <div className="styling-options">
            <h4>Document Styling</h4>
            
            <div className="style-presets">
              <h5>Style Presets</h5>
              <div className="preset-options">
                {STYLE_PRESETS.map(preset => (
                  <button
                    key={preset.name}
                    onClick={() => handleStylePresetChange(preset)}
                    className="preset-option"
                  >
                    <strong>{preset.name}</strong>
                    <p>{preset.description}</p>
                  </button>
                ))}
              </div>
            </div>

            <div className="styling-controls">
              <div className="control-group">
                <label>Font Family:</label>
                <select
                  value={state.configuration.styling.fontFamily}
                  onChange={(e) => handleConfigurationChange({
                    styling: { ...state.configuration.styling, fontFamily: e.target.value }
                  })}
                >
                  <option value="Times New Roman">Times New Roman</option>
                  <option value="Arial">Arial</option>
                  <option value="Calibri">Calibri</option>
                  <option value="Georgia">Georgia</option>
                  <option value="Helvetica">Helvetica</option>
                </select>
              </div>

              <div className="control-group">
                <label>Font Size:</label>
                <input
                  type="number"
                  min="8"
                  max="16"
                  value={state.configuration.styling.fontSize}
                  onChange={(e) => handleConfigurationChange({
                    styling: { ...state.configuration.styling, fontSize: Number(e.target.value) }
                  })}
                />
              </div>

              <div className="control-group">
                <label>Line Spacing:</label>
                <select
                  value={state.configuration.styling.lineSpacing}
                  onChange={(e) => handleConfigurationChange({
                    styling: { ...state.configuration.styling, lineSpacing: Number(e.target.value) }
                  })}
                >
                  <option value={1.0}>Single</option>
                  <option value={1.15}>1.15</option>
                  <option value={1.25}>1.25</option>
                  <option value={1.5}>1.5</option>
                  <option value={2.0}>Double</option>
                </select>
              </div>

              <div className="control-group">
                <label>Page Size:</label>
                <select
                  value={state.configuration.styling.pageSize}
                  onChange={(e) => handleConfigurationChange({
                    styling: { ...state.configuration.styling, pageSize: e.target.value as any }
                  })}
                >
                  <option value="US_LETTER">US Letter</option>
                  <option value="A4">A4</option>
                  <option value="A3">A3</option>
                  <option value="LEGAL">Legal</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Section Selection */}
        {state.activeTab === 'sections' && (
          <div className="section-selection">
            <h4>Document Sections</h4>
            <div className="section-options">
              {state.configuration.sections.map(section => (
                <div key={section.type} className="section-option">
                  <label className="section-checkbox">
                    <input
                      type="checkbox"
                      checked={section.include}
                      onChange={(e) => handleSectionToggle(section.type, e.target.checked)}
                    />
                    <span className="section-name">
                      {section.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                  </label>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Metadata Options */}
        {state.activeTab === 'metadata' && enableAdvancedOptions && (
          <div className="metadata-options">
            <h4>Metadata & Options</h4>
            
            <div className="metadata-controls">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={state.configuration.includeComments}
                  onChange={(e) => handleConfigurationChange({ includeComments: e.target.checked })}
                />
                Include Comments
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={state.configuration.includeTrackChanges}
                  onChange={(e) => handleConfigurationChange({ includeTrackChanges: e.target.checked })}
                />
                Include Track Changes
              </label>

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={state.configuration.includeMetadata}
                  onChange={(e) => handleConfigurationChange({ includeMetadata: e.target.checked })}
                />
                Include Document Metadata
              </label>
            </div>

            <div className="watermark-section">
              <h5>Watermark (Optional)</h5>
              <div className="watermark-controls">
                <input
                  type="text"
                  placeholder="Watermark text..."
                  onChange={(e) => {
                    const watermark: WatermarkConfig = {
                      text: e.target.value,
                      opacity: 0.1,
                      rotation: -45,
                      fontSize: 72,
                      color: '#cccccc'
                    };
                    handleConfigurationChange({ 
                      watermark: e.target.value ? watermark : undefined 
                    });
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Preview */}
        {state.activeTab === 'preview' && (
          <div className="preview-section">
            <h4>Document Preview</h4>
            {state.previewContent ? (
              <div className="preview-content">
                <div 
                  className="preview-document"
                  dangerouslySetInnerHTML={{ __html: state.previewContent }}
                />
              </div>
            ) : (
              <div className="preview-placeholder">
                <p>Click "Preview" to generate a preview of your document.</p>
                <button
                  onClick={generatePreview}
                  className="btn btn-primary"
                  disabled={availableContracts.length === 0}
                >
                  Generate Preview
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Export Progress */}
      {state.isExporting && (
        <div className="export-progress" id="export-progress">
          <div className="progress-info">
            <span>Exporting contracts...</span>
            <span>{Math.round(state.exportProgress)}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${state.exportProgress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ExportSystem;