import React, { useState, useEffect } from 'react';

interface AlertProps {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  onDismiss?: () => void;
  autoHide?: boolean;
  autoHideDelay?: number;
  actions?: Array<{
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
  }>;
}

export const Alert: React.FC<AlertProps> = ({
  type,
  title,
  message,
  onDismiss,
  autoHide = false,
  autoHideDelay = 5000,
  actions = []
}) => {
  const [isVisible, setIsVisible] = useState(true);

  // Auto-hide functionality
  useEffect(() => {
    if (autoHide) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(() => {
          onDismiss?.();
        }, 300); // Wait for fade out animation
      }, autoHideDelay);

      return () => clearTimeout(timer);
    }
  }, [autoHide, autoHideDelay, onDismiss]);

  const getAlertStyles = () => {
    const baseStyles = "border-l-4 p-4 rounded-lg transition-all duration-300";
    
    switch (type) {
      case 'success':
        return `${baseStyles} bg-green-50 border-green-400 text-green-700`;
      case 'error':
        return `${baseStyles} bg-red-50 border-red-400 text-red-700`;
      case 'warning':
        return `${baseStyles} bg-yellow-50 border-yellow-400 text-yellow-700`;
      case 'info':
        return `${baseStyles} bg-blue-50 border-blue-400 text-blue-700`;
      default:
        return `${baseStyles} bg-gray-50 border-gray-400 text-gray-700`;
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'success':
        return (
          <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'info':
        return (
          <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
      default:
        return null;
    }
  };

  const handleDismiss = () => {
    setIsVisible(false);
    setTimeout(() => {
      onDismiss?.();
    }, 300);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      id="alert-container"
      className={`${getAlertStyles()} ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'}`}
    >
      <div id="alert-content" className="flex">
        {/* Icon */}
        <div id="alert-icon" className="flex-shrink-0">
          {getIcon()}
        </div>

        {/* Content */}
        <div id="alert-text" className="ml-3 flex-1">
          <h3 className="text-sm font-medium">{title}</h3>
          <div className="mt-1 text-sm">
            <p>{message}</p>
          </div>

          {/* Actions */}
          {actions.length > 0 && (
            <div id="alert-actions" className="mt-3 flex space-x-2">
              {actions.map((action, index) => (
                <button
                  key={index}
                  id={`alert-action-${index}`}
                  onClick={action.onClick}
                  className={`text-sm font-medium rounded-md px-3 py-1 transition-colors ${
                    action.variant === 'primary'
                      ? type === 'success'
                        ? 'bg-green-600 text-white hover:bg-green-700'
                        : type === 'error'
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : type === 'warning'
                        ? 'bg-yellow-600 text-white hover:bg-yellow-700'
                        : 'bg-blue-600 text-white hover:bg-blue-700'
                      : type === 'success'
                      ? 'text-green-600 hover:text-green-800 hover:bg-green-100'
                      : type === 'error'
                      ? 'text-red-600 hover:text-red-800 hover:bg-red-100'
                      : type === 'warning'
                      ? 'text-yellow-600 hover:text-yellow-800 hover:bg-yellow-100'
                      : 'text-blue-600 hover:text-blue-800 hover:bg-blue-100'
                  }`}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Dismiss Button */}
        {onDismiss && (
          <div id="alert-dismiss" className="ml-auto pl-3">
            <button
              onClick={handleDismiss}
              className="inline-flex text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <span className="sr-only">Dismiss</span>
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Progress Bar for Auto-hide */}
      {autoHide && (
        <div id="alert-progress" className="mt-3">
          <div className="w-full bg-gray-200 rounded-full h-1">
            <div
              className={`h-1 rounded-full transition-all ease-linear ${
                type === 'success' ? 'bg-green-400' :
                type === 'error' ? 'bg-red-400' :
                type === 'warning' ? 'bg-yellow-400' : 'bg-blue-400'
              }`}
              style={{
                width: '100%',
                animation: `shrink ${autoHideDelay}ms linear forwards`
              }}
            />
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes shrink {
          from {
            width: 100%;
          }
          to {
            width: 0%;
          }
        }
      `}</style>
    </div>
  );
};