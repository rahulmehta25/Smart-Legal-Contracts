import React, { useState, useRef, useEffect } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
  placeholder?: string;
  suggestions?: string[];
}

export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  loading = false,
  placeholder = "Search blockchain...",
  suggestions = []
}) => {
  const [query, setQuery] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Filter suggestions based on query
  useEffect(() => {
    if (query.trim() && suggestions.length > 0) {
      const filtered = suggestions.filter(suggestion =>
        suggestion.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredSuggestions(filtered);
      setShowSuggestions(filtered.length > 0);
    } else {
      setFilteredSuggestions([]);
      setShowSuggestions(false);
    }
    setSelectedSuggestion(-1);
  }, [query, suggestions]);

  // Handle search submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
      setShowSuggestions(false);
      setSelectedSuggestion(-1);
    }
  };

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    onSearch(suggestion);
    setShowSuggestions(false);
    setSelectedSuggestion(-1);
    inputRef.current?.blur();
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedSuggestion(prev => 
          prev < filteredSuggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedSuggestion(prev => prev > 0 ? prev - 1 : -1);
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedSuggestion >= 0) {
          handleSuggestionClick(filteredSuggestions[selectedSuggestion]);
        } else {
          handleSubmit(e);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedSuggestion(-1);
        inputRef.current?.blur();
        break;
    }
  };

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
        setSelectedSuggestion(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Get search type indicator
  const getSearchTypeIndicator = () => {
    if (!query.trim()) return null;

    if (query.length === 66 && query.startsWith('0x')) {
      return { type: 'Transaction Hash', color: 'text-blue-600' };
    } else if (query.length === 42 && query.startsWith('0x')) {
      return { type: 'Address', color: 'text-green-600' };
    } else if (/^\d+$/.test(query)) {
      return { type: 'Block Number', color: 'text-purple-600' };
    } else if (query.length === 64) {
      return { type: 'Document Hash', color: 'text-orange-600' };
    }
    return { type: 'General Search', color: 'text-gray-600' };
  };

  const searchTypeIndicator = getSearchTypeIndicator();

  return (
    <div id="search-bar-container" className="relative w-full max-w-2xl">
      <form id="search-form" onSubmit={handleSubmit} className="relative">
        <div id="search-input-container" className="relative">
          {/* Search Icon */}
          <div id="search-icon" className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            {loading ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            ) : (
              <svg
                className="h-5 w-5 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            )}
          </div>

          {/* Input Field */}
          <input
            id="search-input"
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              if (filteredSuggestions.length > 0) {
                setShowSuggestions(true);
              }
            }}
            placeholder={placeholder}
            className="block w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
            disabled={loading}
          />

          {/* Clear Button */}
          {query && (
            <button
              id="clear-search-btn"
              type="button"
              onClick={() => {
                setQuery('');
                setShowSuggestions(false);
                setSelectedSuggestion(-1);
                inputRef.current?.focus();
              }}
              className="absolute inset-y-0 right-10 flex items-center pr-2 text-gray-400 hover:text-gray-600"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}

          {/* Search Button */}
          <button
            id="search-submit-btn"
            type="submit"
            disabled={!query.trim() || loading}
            className="absolute inset-y-0 right-0 flex items-center pr-3 text-blue-600 hover:text-blue-800 disabled:text-gray-400"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        </div>

        {/* Search Type Indicator */}
        {searchTypeIndicator && (
          <div id="search-type-indicator" className="absolute top-full left-0 mt-1 text-xs">
            <span className={`${searchTypeIndicator.color} font-medium`}>
              {searchTypeIndicator.type}
            </span>
          </div>
        )}
      </form>

      {/* Suggestions Dropdown */}
      {showSuggestions && filteredSuggestions.length > 0 && (
        <div
          id="suggestions-dropdown"
          ref={suggestionsRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto"
        >
          {filteredSuggestions.map((suggestion, index) => (
            <button
              key={index}
              id={`suggestion-${index}`}
              type="button"
              onClick={() => handleSuggestionClick(suggestion)}
              className={`w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-b-0 ${
                index === selectedSuggestion ? 'bg-blue-50 text-blue-900' : 'text-gray-900'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm">{suggestion}</span>
                <span className="text-xs text-gray-500">
                  {suggestion.length === 66 && suggestion.startsWith('0x') ? 'Transaction' :
                   suggestion.length === 42 && suggestion.startsWith('0x') ? 'Address' :
                   /^\d+$/.test(suggestion) ? 'Block' : 'Hash'}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Search Examples */}
      <div id="search-examples" className="mt-3 text-xs text-gray-500">
        <span>Examples: </span>
        <button
          type="button"
          onClick={() => {
            setQuery('12345678');
            onSearch('12345678');
          }}
          className="text-blue-600 hover:text-blue-800 underline mr-2"
        >
          Block #12345678
        </button>
        <button
          type="button"
          onClick={() => {
            setQuery('0x742d35Cc6464C4532B05F98e7b0481BE0c5b4732');
            onSearch('0x742d35Cc6464C4532B05F98e7b0481BE0c5b4732');
          }}
          className="text-blue-600 hover:text-blue-800 underline mr-2"
        >
          Address
        </button>
        <button
          type="button"
          onClick={() => {
            setQuery('0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12');
            onSearch('0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef12');
          }}
          className="text-blue-600 hover:text-blue-800 underline"
        >
          Transaction Hash
        </button>
      </div>
    </div>
  );
};