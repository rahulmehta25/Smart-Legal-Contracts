import React, { useState } from 'react';
import { Download, FileText, BarChart3, Filter, SortAsc, SortDesc, ExternalLink } from 'lucide-react';
import { saveAs } from 'file-saver';
import ConfidenceScore, { ConfidenceScoreGrid } from './ConfidenceScore';

const ResultsDisplay = ({ 
  results, 
  documentName, 
  onClauseSelect,
  isLoading = false 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [sortBy, setSortBy] = useState('confidence');
  const [sortOrder, setSortOrder] = useState('desc');
  const [filterType, setFilterType] = useState('all');

  // Mock data structure for when results are empty or for demo purposes
  const defaultResults = {
    summary: {
      totalClauses: 0,
      highConfidenceClauses: 0,
      avgConfidence: 0,
      documentComplexity: 'Low'
    },
    clauses: [],
    analysis: {
      enforceability: 0,
      clarity: 0,
      completeness: 0
    },
    recommendations: []
  };

  const analysisResults = results || defaultResults;

  // Filter and sort clauses
  const getFilteredClauses = () => {
    let filtered = analysisResults.clauses || [];
    
    if (filterType !== 'all') {
      filtered = filtered.filter(clause => clause.type === filterType);
    }

    return filtered.sort((a, b) => {
      const multiplier = sortOrder === 'desc' ? -1 : 1;
      switch (sortBy) {
        case 'confidence':
          return multiplier * (a.confidence - b.confidence);
        case 'type':
          return multiplier * a.type.localeCompare(b.type);
        case 'position':
          return multiplier * (a.startIndex - b.startIndex);
        default:
          return 0;
      }
    });
  };

  const exportResults = (format) => {
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `${documentName || 'document'}_arbitration_analysis_${timestamp}`;
    
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(analysisResults, null, 2)], { type: 'application/json' });
      saveAs(blob, `${filename}.json`);
    } else if (format === 'csv') {
      const csvContent = [
        ['Clause Type', 'Confidence', 'Start Position', 'End Position', 'Text Preview'],
        ...analysisResults.clauses.map(clause => [
          clause.type,
          clause.confidence,
          clause.startIndex,
          clause.endIndex,
          clause.text.substring(0, 100) + '...'
        ])
      ].map(row => row.join(',')).join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      saveAs(blob, `${filename}.csv`);
    }
  };

  const getClauseTypeStats = () => {
    const types = {};
    analysisResults.clauses.forEach(clause => {
      types[clause.type] = (types[clause.type] || 0) + 1;
    });
    return types;
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'clauses', label: 'Detected Clauses', icon: FileText },
    { id: 'analysis', label: 'Legal Analysis', icon: ExternalLink }
  ];

  if (isLoading) {
    return (
      <div id="results-loading-container" className="flex items-center justify-center h-64">
        <div id="results-loading-spinner" className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span id="results-loading-text" className="ml-2 text-gray-600">Analyzing document...</span>
      </div>
    );
  }

  return (
    <div id="results-display-container" className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div id="results-header" className="bg-gray-50 border-b border-gray-200 px-6 py-4">
        <div id="results-header-content" className="flex items-center justify-between">
          <div id="results-title-section">
            <h2 id="results-title" className="text-xl font-bold text-gray-900">
              Analysis Results
            </h2>
            {documentName && (
              <p id="results-document-name" className="text-sm text-gray-600 mt-1">
                Document: {documentName}
              </p>
            )}
          </div>
          <div id="results-export-controls" className="flex items-center space-x-2">
            <button
              id="export-json-button"
              onClick={() => exportResults('json')}
              className="flex items-center space-x-1 px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
            >
              <Download id="export-json-icon" className="h-4 w-4" />
              <span id="export-json-text">JSON</span>
            </button>
            <button
              id="export-csv-button"
              onClick={() => exportResults('csv')}
              className="flex items-center space-x-1 px-3 py-2 bg-success-600 text-white rounded-lg hover:bg-success-700 transition-colors text-sm"
            >
              <Download id="export-csv-icon" className="h-4 w-4" />
              <span id="export-csv-text">CSV</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div id="results-tabs-container" className="border-b border-gray-200">
        <nav id="results-tabs-nav" className="flex space-x-8 px-6">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center space-x-2 py-4 border-b-2 font-medium text-sm transition-colors
                  ${activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Icon id={`tab-icon-${tab.id}`} className="h-4 w-4" />
                <span id={`tab-label-${tab.id}`}>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div id="results-content" className="p-6">
        {activeTab === 'overview' && (
          <div id="overview-tab-content" className="space-y-6">
            {/* Summary Stats */}
            <div id="summary-stats-container" className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div id="total-clauses-stat" className="bg-primary-50 p-4 rounded-lg border border-primary-200">
                <div id="total-clauses-number" className="text-2xl font-bold text-primary-600">
                  {analysisResults.summary.totalClauses}
                </div>
                <div id="total-clauses-label" className="text-sm text-primary-700">Total Clauses</div>
              </div>
              <div id="high-confidence-stat" className="bg-success-50 p-4 rounded-lg border border-success-200">
                <div id="high-confidence-number" className="text-2xl font-bold text-success-600">
                  {analysisResults.summary.highConfidenceClauses}
                </div>
                <div id="high-confidence-label" className="text-sm text-success-700">High Confidence</div>
              </div>
              <div id="avg-confidence-stat" className="bg-warning-50 p-4 rounded-lg border border-warning-200">
                <div id="avg-confidence-number" className="text-2xl font-bold text-warning-600">
                  {analysisResults.summary.avgConfidence}%
                </div>
                <div id="avg-confidence-label" className="text-sm text-warning-700">Avg Confidence</div>
              </div>
              <div id="complexity-stat" className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <div id="complexity-level" className="text-2xl font-bold text-gray-600">
                  {analysisResults.summary.documentComplexity}
                </div>
                <div id="complexity-label" className="text-sm text-gray-700">Complexity</div>
              </div>
            </div>

            {/* Overall Confidence Score */}
            <div id="overall-confidence-container" className="bg-white border border-gray-200 rounded-lg p-6">
              <ConfidenceScore 
                score={analysisResults.summary.avgConfidence} 
                size="lg" 
                showLabel={true}
              />
            </div>

            {/* Clause Type Distribution */}
            <div id="clause-type-distribution" className="bg-white border border-gray-200 rounded-lg p-6">
              <h3 id="clause-distribution-title" className="text-lg font-semibold text-gray-900 mb-4">
                Clause Type Distribution
              </h3>
              <div id="clause-type-chart" className="space-y-3">
                {Object.entries(getClauseTypeStats()).map(([type, count]) => (
                  <div key={type} id={`clause-type-${type}`} className="flex items-center justify-between">
                    <span id={`clause-type-label-${type}`} className="text-sm font-medium text-gray-700 capitalize">
                      {type}
                    </span>
                    <div id={`clause-type-bar-container-${type}`} className="flex items-center space-x-2 flex-1 ml-4">
                      <div id={`clause-type-bar-bg-${type}`} className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          id={`clause-type-bar-fill-${type}`}
                          className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${(count / analysisResults.summary.totalClauses) * 100}%` }}
                        />
                      </div>
                      <span id={`clause-type-count-${type}`} className="text-sm text-gray-600 min-w-[2rem]">
                        {count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'clauses' && (
          <div id="clauses-tab-content" className="space-y-4">
            {/* Controls */}
            <div id="clauses-controls" className="flex items-center justify-between">
              <div id="clauses-filters" className="flex items-center space-x-4">
                <div id="filter-control" className="flex items-center space-x-2">
                  <Filter id="filter-icon" className="h-4 w-4 text-gray-500" />
                  <select
                    id="clause-type-filter"
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    className="border border-gray-300 rounded-lg px-3 py-1 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="all">All Types</option>
                    <option value="mandatory">Mandatory</option>
                    <option value="optional">Optional</option>
                    <option value="binding">Binding</option>
                    <option value="non-binding">Non-binding</option>
                  </select>
                </div>
                <div id="sort-control" className="flex items-center space-x-2">
                  <button
                    id="sort-order-button"
                    onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                    className="p-1 hover:bg-gray-100 rounded transition-colors"
                  >
                    {sortOrder === 'desc' ? 
                      <SortDesc id="sort-desc-icon" className="h-4 w-4 text-gray-500" /> : 
                      <SortAsc id="sort-asc-icon" className="h-4 w-4 text-gray-500" />
                    }
                  </button>
                  <select
                    id="sort-by-select"
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="border border-gray-300 rounded-lg px-3 py-1 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="confidence">Confidence</option>
                    <option value="type">Type</option>
                    <option value="position">Position</option>
                  </select>
                </div>
              </div>
              <div id="clauses-count" className="text-sm text-gray-600">
                Showing {getFilteredClauses().length} of {analysisResults.clauses.length} clauses
              </div>
            </div>

            {/* Clauses List */}
            <div id="clauses-list" className="space-y-3">
              {getFilteredClauses().map((clause, index) => (
                <div 
                  key={clause.id}
                  id={`clause-item-${clause.id}`}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => onClauseSelect && onClauseSelect(clause)}
                >
                  <div id={`clause-header-${clause.id}`} className="flex items-start justify-between mb-2">
                    <div id={`clause-info-${clause.id}`} className="flex-1">
                      <div id={`clause-type-badge-${clause.id}`} className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800 mb-2">
                        {clause.type}
                      </div>
                      <h4 id={`clause-title-${clause.id}`} className="font-medium text-gray-900">
                        Clause {index + 1}
                      </h4>
                    </div>
                    <div id={`clause-confidence-container-${clause.id}`} className="text-right">
                      <ConfidenceScore 
                        score={clause.confidence} 
                        size="sm" 
                        showLabel={false}
                        className="w-24"
                      />
                      <span id={`clause-confidence-text-${clause.id}`} className="text-xs text-gray-600 mt-1 block">
                        {clause.confidence}% confidence
                      </span>
                    </div>
                  </div>
                  <div id={`clause-preview-${clause.id}`} className="text-sm text-gray-700 bg-gray-50 p-3 rounded border-l-4 border-primary-400">
                    "{clause.text ? clause.text.substring(0, 200) + '...' : 'No preview available'}"
                  </div>
                  {clause.description && (
                    <div id={`clause-description-${clause.id}`} className="mt-2 text-sm text-gray-600">
                      <strong>Analysis:</strong> {clause.description}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {getFilteredClauses().length === 0 && (
              <div id="no-clauses-message" className="text-center py-8 text-gray-500">
                <FileText id="no-clauses-icon" className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p id="no-clauses-text">No clauses found matching your criteria.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analysis' && (
          <div id="analysis-tab-content" className="space-y-6">
            {/* Legal Analysis Scores */}
            <ConfidenceScoreGrid 
              title="Legal Analysis Metrics"
              scores={[
                { 
                  id: 'enforceability',
                  label: 'Enforceability',
                  description: 'Likelihood that clauses will be upheld in court',
                  score: analysisResults.analysis.enforceability 
                },
                { 
                  id: 'clarity',
                  label: 'Clarity',
                  description: 'How clearly the arbitration terms are defined',
                  score: analysisResults.analysis.clarity 
                },
                { 
                  id: 'completeness',
                  label: 'Completeness',
                  description: 'Coverage of essential arbitration elements',
                  score: analysisResults.analysis.completeness 
                }
              ]}
            />

            {/* Recommendations */}
            <div id="recommendations-container" className="bg-white border border-gray-200 rounded-lg p-6">
              <h3 id="recommendations-title" className="text-lg font-semibold text-gray-900 mb-4">
                Recommendations
              </h3>
              {analysisResults.recommendations && analysisResults.recommendations.length > 0 ? (
                <div id="recommendations-list" className="space-y-3">
                  {analysisResults.recommendations.map((rec, index) => (
                    <div 
                      key={index}
                      id={`recommendation-${index}`}
                      className="flex items-start space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg"
                    >
                      <div id={`recommendation-icon-${index}`} className="flex-shrink-0 mt-0.5">
                        <ExternalLink className="h-4 w-4 text-blue-600" />
                      </div>
                      <div id={`recommendation-content-${index}`} className="flex-1">
                        <p id={`recommendation-text-${index}`} className="text-sm text-blue-800">
                          {rec}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p id="no-recommendations-text" className="text-gray-500 italic">
                  No specific recommendations available for this document.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsDisplay;