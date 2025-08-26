# Arbitration Clause Detector - Frontend

AI-powered legal document analysis system for detecting arbitration clauses with 99%+ accuracy.

## Features

- **Document Upload**: Drag-and-drop interface for PDF, DOCX, and TXT files
- **Real-time Analysis**: WebSocket-powered live processing updates
- **Visual Highlighting**: Interactive PDF viewer with clause highlighting
- **Batch Processing**: Analyze multiple documents simultaneously
- **Admin Dashboard**: Comprehensive analytics and user management
- **Dark Mode**: System-aware theme switching
- **Responsive Design**: Optimized for desktop, tablet, and mobile

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at http://localhost:5173

### Environment Configuration

Create a `.env.local` file:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Application pages
├── services/      # API integration services
├── hooks/         # Custom React hooks
├── lib/           # Utility functions
└── types/         # TypeScript type definitions
```

## Technologies

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Shadcn/ui** - Component library
- **React Query** - Data fetching
- **React Router** - Routing
- **Recharts** - Data visualization

## Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

## API Integration

The frontend connects to the backend API at:
- REST API: `http://localhost:8000/api/v1`
- WebSocket: `ws://localhost:8000/ws`
- Documentation: `http://localhost:8000/docs`

## Deployment

### Production Build

```bash
npm run build
```

The build output will be in the `dist/` directory.

### Docker

```bash
docker build -t arbitration-frontend .
docker run -p 3000:3000 arbitration-frontend
```

## License

MIT