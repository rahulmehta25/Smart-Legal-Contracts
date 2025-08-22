# Arbitration Clause Detector - Frontend

A modern React application for detecting and analyzing arbitration clauses in legal documents using AI-powered analysis.

## Features

### Core Functionality
- **Document Upload**: Drag-and-drop interface supporting PDF, DOCX, DOC, and TXT files
- **Real-time Analysis**: AI-powered detection of arbitration clauses with confidence scoring
- **Text Highlighting**: Interactive highlighting of detected clauses with type classification
- **Results Export**: Export analysis results in JSON and CSV formats
- **Search Functionality**: Search within uploaded documents
- **Side-by-side Comparison**: Split view for document and results

### UI Components

#### DocumentUploader (`/src/components/DocumentUploader.jsx`)
- Drag-and-drop file upload with validation
- Support for multiple file formats (PDF, DOCX, TXT)
- File size validation (max 10MB)
- Upload progress and status indicators
- Error handling with user-friendly messages

#### ResultsDisplay (`/src/components/ResultsDisplay.jsx`)
- Tabbed interface (Overview, Clauses, Legal Analysis)
- Interactive clause filtering and sorting
- Confidence score visualization
- Export functionality (JSON/CSV)
- Legal analysis metrics and recommendations

#### ClauseHighlighter (`/src/components/ClauseHighlighter.jsx`)
- Real-time text highlighting of detected clauses
- Color-coded clause types (mandatory, optional, binding, non-binding)
- Interactive clause selection and details
- Search within document functionality
- Toggle visibility for different clause types

#### ConfidenceScore (`/src/components/ConfidenceScore.jsx`)
- Visual confidence indicators with progress bars
- Color-coded confidence levels (High/Medium/Low)
- Detailed confidence score interpretation
- Grid layout for multiple scores

### Responsive Design
- **Mobile-first approach** with Tailwind CSS
- **Adaptive layouts** for desktop, tablet, and mobile
- **Accessible design** with ARIA labels and keyboard navigation
- **Dark mode support** (system preference detection)
- **High contrast mode** compatibility

### Accessibility Features
- Comprehensive ARIA labels and roles
- Keyboard navigation support
- Screen reader announcements
- Focus management and visual indicators
- Color contrast compliance (WCAG 2.1)

## Technology Stack

- **React 18** with hooks and modern patterns
- **Tailwind CSS** for responsive styling
- **Lucide React** for consistent iconography
- **React Dropzone** for file upload functionality
- **File Saver** for export capabilities
- **PostCSS** and **Autoprefixer** for CSS processing

## Getting Started

### Prerequisites
- Node.js 16+ and npm
- Modern web browser with JavaScript enabled

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm eject` - Eject from Create React App (one-way operation)

## Component Architecture

### App.jsx - Main Application
- Manages application state and navigation
- Coordinates between upload, processing, and results phases
- Handles file processing and analysis orchestration
- Responsive layout management

### State Management
- **currentStep**: Tracks upload → processing → results flow
- **uploadedFile**: Stores file reference and metadata
- **documentText**: Extracted text content for analysis
- **analysisResults**: AI analysis results with clauses and metrics
- **selectedClause**: Currently selected clause for detailed view
- **searchTerm**: Document search functionality
- **viewMode**: Controls split/document/results view modes

### Mock Data Structure
```javascript
{
  summary: {
    totalClauses: number,
    highConfidenceClauses: number,
    avgConfidence: number,
    documentComplexity: string
  },
  clauses: [
    {
      id: string,
      type: 'mandatory' | 'optional' | 'binding' | 'non-binding',
      confidence: number,
      startIndex: number,
      endIndex: number,
      text: string,
      description: string
    }
  ],
  analysis: {
    enforceability: number,
    clarity: number,
    completeness: number
  },
  recommendations: string[]
}
```

## Styling and Design

### Design System
- **Primary Color**: Blue (`#3b82f6`)
- **Success Color**: Green (`#22c55e`)
- **Warning Color**: Amber (`#f59e0b`)
- **Danger Color**: Red (`#ef4444`)
- **Typography**: Inter font family
- **Spacing**: Consistent 4px grid system

### Component Styling
- **Cards**: White background with subtle shadows
- **Buttons**: Consistent sizing with hover states
- **Forms**: Clean inputs with focus states
- **Badges**: Color-coded type indicators
- **Progress**: Animated confidence bars

### Responsive Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px  
- **Desktop**: > 1024px

## Performance Optimizations

### Code Splitting
- Lazy loading for large components
- Dynamic imports for utilities
- Bundle size optimization

### Rendering Performance
- React.memo for expensive components
- Callback memoization with useCallback
- Effect optimization with useEffect dependencies

### Accessibility Performance
- Efficient focus management
- Optimized screen reader announcements
- Reduced motion preferences support

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+
- **Mobile browsers**: iOS Safari 14+, Chrome Mobile 90+

## Integration Points

### Backend API Integration
The frontend is designed to integrate with a RAG-based arbitration clause detection system:

```javascript
// Example API integration
const analyzeDocument = async (documentText) => {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: documentText })
  });
  return response.json();
};
```

### File Processing
- PDF text extraction (requires backend service)
- DOCX parsing (requires backend service)
- TXT file reading (client-side)

## Deployment

### Production Build
```bash
npm run build
```

### Environment Variables
Create `.env` file for configuration:
```env
REACT_APP_API_URL=https://api.arbitration-detector.com
REACT_APP_VERSION=1.0.0
REACT_APP_ENVIRONMENT=production
```

### Static Hosting
The build output can be deployed to:
- **Netlify**: Drag-and-drop deployment
- **Vercel**: Git-based deployment
- **AWS S3**: Static website hosting
- **GitHub Pages**: Repository-based hosting

## Testing Strategy

### Unit Tests
- Component rendering tests
- State management tests
- Utility function tests
- Accessibility tests

### Integration Tests
- File upload workflows
- Analysis result rendering
- Export functionality
- Cross-component interactions

### End-to-End Tests
- Complete user workflows
- Cross-browser compatibility
- Performance benchmarks
- Accessibility compliance

## Contributing

### Code Standards
- **ESLint**: Enforced code quality rules
- **Prettier**: Consistent code formatting
- **Component naming**: PascalCase for components
- **File organization**: Feature-based structure

### Git Workflow
- Feature branches for new functionality
- Pull request reviews required
- Automated testing on PRs
- Semantic commit messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Create an issue in the project repository
- Review the troubleshooting guide
- Check the component documentation
- Contact the development team