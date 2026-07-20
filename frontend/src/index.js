import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Performance monitoring
if (process.env.NODE_ENV === 'development') {
  const reportWebVitals = (onPerfEntry) => {
    if (onPerfEntry && onPerfEntry instanceof Function) {
      import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
        getCLS(onPerfEntry);
        getFID(onPerfEntry);
        getFCP(onPerfEntry);
        getLCP(onPerfEntry);
        getTTFB(onPerfEntry);
      });
    }
  };
  
  // Report performance metrics in development
  reportWebVitals(console.log);
}

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error for monitoring
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div id="error-boundary-fallback" className="min-h-screen flex items-center justify-center bg-red-50">
          <div id="error-boundary-content" className="max-w-md mx-auto text-center p-6">
            <div id="error-icon-container" className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 mb-4">
              <svg id="error-icon" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h1 id="error-boundary-title" className="text-lg font-semibold text-gray-900 mb-2">
              Something went wrong
            </h1>
            <p id="error-boundary-message" className="text-sm text-gray-600 mb-4">
              The application encountered an unexpected error. Please refresh the page to try again.
            </p>
            <button
              id="error-boundary-refresh"
              onClick={() => window.location.reload()}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors"
            >
              Refresh Page
            </button>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details id="error-details" className="mt-4 text-left">
                <summary id="error-details-summary" className="cursor-pointer text-sm font-medium text-gray-700 mb-2">
                  Error Details (Development)
                </summary>
                <pre id="error-stack" className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-40">
                  {this.state.error && this.state.error.toString()}
                  <br />
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Create React root
const root = ReactDOM.createRoot(document.getElementById('react-root'));

// Render app with error boundary
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

// Accessibility: Announce page loads to screen readers
const announceToScreenReader = (message) => {
  const announcements = document.getElementById('accessibility-announcements');
  if (announcements) {
    announcements.textContent = message;
    setTimeout(() => {
      announcements.textContent = '';
    }, 1000);
  }
};

// Announce when app is ready
setTimeout(() => {
  announceToScreenReader('Arbitration Clause Detector application loaded and ready');
}, 1500);

// Handle offline/online status
const updateOnlineStatus = () => {
  const isOnline = navigator.onLine;
  if (!isOnline) {
    announceToScreenReader('Application is offline. Some features may not be available.');
  } else {
    announceToScreenReader('Application is back online.');
  }
};

window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);

// Keyboard navigation improvements
document.addEventListener('keydown', (event) => {
  // Focus management for better accessibility
  if (event.key === 'Tab') {
    document.body.classList.add('keyboard-navigation');
  }
});

document.addEventListener('mousedown', () => {
  document.body.classList.remove('keyboard-navigation');
});

// Theme detection and application
const detectSystemTheme = () => {
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.classList.add('dark-theme');
  } else {
    document.documentElement.classList.remove('dark-theme');
  }
};

// Listen for theme changes
if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectSystemTheme);
}

// Apply theme on load
detectSystemTheme();