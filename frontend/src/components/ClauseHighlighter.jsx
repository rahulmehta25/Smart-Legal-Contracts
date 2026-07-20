import React, { useState, useEffect } from 'react';
import { Search, Eye, EyeOff, ChevronDown, ChevronUp } from 'lucide-react';

const ClauseHighlighter = ({ 
  documentText, 
  detectedClauses = [], 
  onClauseSelect,
  searchTerm = '',
  onSearchTermChange 
}) => {
  const [highlightedText, setHighlightedText] = useState('');
  const [selectedClauseId, setSelectedClauseId] = useState(null);
  const [visibleClauses, setVisibleClauses] = useState({});
  const [searchMatches, setSearchMatches] = useState([]);
  const [isExpanded, setIsExpanded] = useState(true);

  // Initialize all clauses as visible
  useEffect(() => {
    const initialVisibility = {};
    detectedClauses.forEach(clause => {
      initialVisibility[clause.id] = true;
    });
    setVisibleClauses(initialVisibility);
  }, [detectedClauses]);

  // Generate highlighted text with clauses and search terms
  useEffect(() => {
    let text = documentText;
    let highlightedContent = text;
    let matches = [];

    // First, highlight detected clauses
    const visibleClausesList = detectedClauses.filter(clause => visibleClauses[clause.id]);
    
    // Sort clauses by start position (descending) to avoid position shifts during replacement
    const sortedClauses = [...visibleClausesList].sort((a, b) => b.startIndex - a.startIndex);

    sortedClauses.forEach(clause => {
      const beforeText = highlightedContent.substring(0, clause.startIndex);
      const clauseText = highlightedContent.substring(clause.startIndex, clause.endIndex);
      const afterText = highlightedContent.substring(clause.endIndex);

      const isSelected = selectedClauseId === clause.id;
      const highlightClass = `
        clause-highlight 
        ${isSelected ? 'bg-primary-200 border-primary-400' : getClauseTypeColor(clause.type)}
        border-l-4 px-2 py-1 rounded-r cursor-pointer transition-all duration-200 hover:shadow-md
      `.trim();

      highlightedContent = beforeText + 
        `<span id="clause-${clause.id}" class="${highlightClass}" data-clause-id="${clause.id}" title="${clause.type}: ${clause.confidence}% confidence">${clauseText}</span>` + 
        afterText;
    });

    // Then highlight search terms
    if (searchTerm && searchTerm.length > 2) {
      const searchRegex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      const searchMatches = [];
      let match;
      
      while ((match = searchRegex.exec(text)) !== null) {
        searchMatches.push({
          start: match.index,
          end: match.index + match[0].length,
          text: match[0]
        });
      }
      
      setSearchMatches(searchMatches);
      
      // Highlight search terms (avoiding clause highlights)
      highlightedContent = highlightedContent.replace(searchRegex, 
        '<span class="bg-yellow-300 font-semibold rounded px-1">$1</span>'
      );
    } else {
      setSearchMatches([]);
    }

    setHighlightedText(highlightedContent);
  }, [documentText, detectedClauses, visibleClauses, selectedClauseId, searchTerm]);

  const getClauseTypeColor = (type) => {
    const colors = {
      'mandatory': 'bg-danger-100 border-danger-400',
      'optional': 'bg-warning-100 border-warning-400',
      'binding': 'bg-primary-100 border-primary-400',
      'non-binding': 'bg-gray-100 border-gray-400',
      'default': 'bg-blue-100 border-blue-400'
    };
    return colors[type] || colors.default;
  };

  const handleClauseClick = (event) => {
    const clauseId = event.target.getAttribute('data-clause-id');
    if (clauseId) {
      setSelectedClauseId(clauseId);
      const clause = detectedClauses.find(c => c.id === clauseId);
      if (clause && onClauseSelect) {
        onClauseSelect(clause);
      }
    }
  };

  const toggleClauseVisibility = (clauseId) => {
    setVisibleClauses(prev => ({
      ...prev,
      [clauseId]: !prev[clauseId]
    }));
  };

  const toggleAllClauses = () => {
    const allVisible = Object.values(visibleClauses).every(visible => visible);
    const newVisibility = {};
    detectedClauses.forEach(clause => {
      newVisibility[clause.id] = !allVisible;
    });
    setVisibleClauses(newVisibility);
  };

  return (
    <div id="clause-highlighter-container" className="h-full flex flex-col">
      {/* Controls Panel */}
      <div id="clause-controls-panel" className="flex-shrink-0 bg-gray-50 border-b border-gray-200 p-4">
        <div id="clause-controls-header" className="flex items-center justify-between mb-4">
          <h3 id="document-viewer-title" className="text-lg font-semibold text-gray-900">
            Document Viewer
          </h3>
          <button
            id="toggle-viewer-button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <span id="toggle-viewer-text" className="text-sm">
              {isExpanded ? 'Collapse' : 'Expand'}
            </span>
            {isExpanded ? 
              <ChevronUp id="collapse-icon" className="h-4 w-4" /> : 
              <ChevronDown id="expand-icon" className="h-4 w-4" />
            }
          </button>
        </div>

        {isExpanded && (
          <>
            {/* Search Bar */}
            <div id="search-container" className="mb-4">
              <div id="search-input-wrapper" className="relative">
                <Search id="search-icon" className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  id="document-search-input"
                  type="text"
                  placeholder="Search within document..."
                  value={searchTerm}
                  onChange={(e) => onSearchTermChange && onSearchTermChange(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              {searchMatches.length > 0 && (
                <p id="search-results-count" className="text-sm text-gray-600 mt-1">
                  Found {searchMatches.length} matches
                </p>
              )}
            </div>

            {/* Clause Controls */}
            <div id="clause-controls" className="space-y-3">
              <div id="clause-controls-header-section" className="flex items-center justify-between">
                <span id="detected-clauses-label" className="text-sm font-medium text-gray-700">
                  Detected Clauses ({detectedClauses.length})
                </span>
                <button
                  id="toggle-all-clauses-button"
                  onClick={toggleAllClauses}
                  className="text-sm text-primary-600 hover:text-primary-700 transition-colors"
                >
                  {Object.values(visibleClauses).every(visible => visible) ? 'Hide All' : 'Show All'}
                </button>
              </div>

              <div id="clause-legend" className="grid grid-cols-2 gap-2">
                {detectedClauses.map((clause, index) => (
                  <div 
                    key={clause.id}
                    id={`clause-control-${clause.id}`}
                    className="flex items-center space-x-2"
                  >
                    <button
                      id={`toggle-clause-${clause.id}`}
                      onClick={() => toggleClauseVisibility(clause.id)}
                      className="p-1 rounded hover:bg-gray-200 transition-colors"
                    >
                      {visibleClauses[clause.id] ? 
                        <Eye id={`eye-icon-${clause.id}`} className="h-4 w-4 text-primary-600" /> : 
                        <EyeOff id={`eye-off-icon-${clause.id}`} className="h-4 w-4 text-gray-400" />
                      }
                    </button>
                    <div 
                      id={`clause-indicator-${clause.id}`}
                      className={`w-3 h-3 rounded border-2 ${getClauseTypeColor(clause.type)}`}
                    />
                    <span id={`clause-label-${clause.id}`} className="text-xs text-gray-600 truncate">
                      {clause.type} ({clause.confidence}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Document Content */}
      {isExpanded && (
        <div id="document-content-container" className="flex-1 overflow-auto p-6 bg-white">
          <div 
            id="highlighted-document-text"
            className="prose max-w-none text-sm leading-relaxed"
            onClick={handleClauseClick}
            dangerouslySetInnerHTML={{ 
              __html: highlightedText || '<p class="text-gray-500 italic">No document content available</p>' 
            }}
          />
        </div>
      )}

      {/* Selected Clause Info */}
      {selectedClauseId && (
        <div id="selected-clause-info" className="flex-shrink-0 bg-primary-50 border-t border-primary-200 p-4">
          <div id="selected-clause-header" className="flex items-center justify-between mb-2">
            <h4 id="selected-clause-title" className="font-medium text-primary-900">
              Selected Clause
            </h4>
            <button
              id="clear-selection-button"
              onClick={() => setSelectedClauseId(null)}
              className="text-primary-600 hover:text-primary-700 text-sm transition-colors"
            >
              Clear Selection
            </button>
          </div>
          {(() => {
            const selectedClause = detectedClauses.find(c => c.id === selectedClauseId);
            return selectedClause ? (
              <div id="selected-clause-details" className="space-y-1">
                <p id="selected-clause-type" className="text-sm">
                  <span className="font-medium">Type:</span> {selectedClause.type}
                </p>
                <p id="selected-clause-confidence" className="text-sm">
                  <span className="font-medium">Confidence:</span> {selectedClause.confidence}%
                </p>
                {selectedClause.description && (
                  <p id="selected-clause-description" className="text-sm">
                    <span className="font-medium">Description:</span> {selectedClause.description}
                  </p>
                )}
              </div>
            ) : null;
          })()}
        </div>
      )}
    </div>
  );
};

export default ClauseHighlighter;