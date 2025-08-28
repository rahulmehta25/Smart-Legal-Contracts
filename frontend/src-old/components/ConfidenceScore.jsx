import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const ConfidenceScore = ({ score, size = 'md', showLabel = true, className = '' }) => {
  // Convert score to percentage if it's between 0 and 1
  const percentage = score > 1 ? score : score * 100;
  
  const getScoreColor = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'danger';
  };

  const getScoreIcon = (score) => {
    if (score >= 70) return <TrendingUp id="confidence-trend-up-icon" className="h-4 w-4" />;
    if (score >= 40) return <Minus id="confidence-neutral-icon" className="h-4 w-4" />;
    return <TrendingDown id="confidence-trend-down-icon" className="h-4 w-4" />;
  };

  const colorVariant = getScoreColor(percentage);
  
  const sizeClasses = {
    sm: 'h-2 text-xs',
    md: 'h-3 text-sm',
    lg: 'h-4 text-base'
  };

  const colorClasses = {
    success: {
      bg: 'bg-success-500',
      text: 'text-success-700',
      border: 'border-success-200',
      bgLight: 'bg-success-50'
    },
    warning: {
      bg: 'bg-warning-500',
      text: 'text-warning-700',
      border: 'border-warning-200',
      bgLight: 'bg-warning-50'
    },
    danger: {
      bg: 'bg-danger-500',
      text: 'text-danger-700',
      border: 'border-danger-200',
      bgLight: 'bg-danger-50'
    }
  };

  return (
    <div id="confidence-score-container" className={`${className}`}>
      {showLabel && (
        <div id="confidence-score-header" className="flex items-center justify-between mb-2">
          <span id="confidence-score-label" className="text-sm font-medium text-gray-700">
            Confidence Score
          </span>
          <div id="confidence-score-value-container" className="flex items-center space-x-1">
            <span id="confidence-score-icon" className={colorClasses[colorVariant].text}>
              {getScoreIcon(percentage)}
            </span>
            <span id="confidence-score-percentage" className={`font-semibold ${colorClasses[colorVariant].text}`}>
              {percentage.toFixed(1)}%
            </span>
          </div>
        </div>
      )}
      
      <div id="confidence-score-progress-container" className="relative">
        <div 
          id="confidence-score-background"
          className={`w-full ${sizeClasses[size]} bg-gray-200 rounded-full overflow-hidden`}
        >
          <div
            id="confidence-score-fill"
            className={`${sizeClasses[size]} ${colorClasses[colorVariant].bg} rounded-full transition-all duration-500 ease-out`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
        
        {/* Confidence level indicator */}
        <div id="confidence-level-indicators" className="flex justify-between mt-1 text-xs text-gray-500">
          <span id="confidence-low-label">Low</span>
          <span id="confidence-medium-label">Medium</span>
          <span id="confidence-high-label">High</span>
        </div>
      </div>
      
      {/* Detailed breakdown for larger sizes */}
      {size === 'lg' && (
        <div id="confidence-score-details" className={`mt-3 p-3 rounded-lg border ${colorClasses[colorVariant].border} ${colorClasses[colorVariant].bgLight}`}>
          <div id="confidence-score-interpretation" className="text-sm">
            <span id="confidence-interpretation-label" className="font-medium text-gray-700">
              Interpretation: 
            </span>
            <span id="confidence-interpretation-text" className={`ml-1 ${colorClasses[colorVariant].text}`}>
              {percentage >= 80 && 'Very High Confidence - Strong arbitration clause detected'}
              {percentage >= 60 && percentage < 80 && 'High Confidence - Likely arbitration clause'}
              {percentage >= 40 && percentage < 60 && 'Medium Confidence - Possible arbitration clause'}
              {percentage < 40 && 'Low Confidence - Weak or no arbitration clause detected'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Variant for displaying multiple confidence scores
export const ConfidenceScoreGrid = ({ scores, title }) => {
  return (
    <div id="confidence-score-grid-container" className="space-y-4">
      {title && (
        <h3 id="confidence-score-grid-title" className="text-lg font-semibold text-gray-900">
          {title}
        </h3>
      )}
      <div id="confidence-score-grid" className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {scores.map((scoreData, index) => (
          <div 
            key={scoreData.id || index}
            id={`confidence-score-item-${index}`}
            className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm"
          >
            <div id={`score-item-header-${index}`} className="mb-3">
              <h4 id={`score-item-title-${index}`} className="font-medium text-gray-900">
                {scoreData.label}
              </h4>
              {scoreData.description && (
                <p id={`score-item-description-${index}`} className="text-sm text-gray-600 mt-1">
                  {scoreData.description}
                </p>
              )}
            </div>
            <ConfidenceScore 
              score={scoreData.score} 
              size="md" 
              showLabel={false}
              className="w-full"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConfidenceScore;