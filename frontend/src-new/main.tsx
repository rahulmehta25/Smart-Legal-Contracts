import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { validateEnvironment } from './config/env'

// Validate environment configuration on startup
validateEnvironment();

createRoot(document.getElementById("root")!).render(<App />);
