import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Arbitration Detector - Frontend Integration Demo',
  description: 'Modern React frontend demonstrating integration with FastAPI backend for arbitration clause detection.',
  keywords: ['arbitration', 'legal tech', 'document analysis', 'AI', 'FastAPI', 'Next.js'],
  authors: [{ name: 'Arbitration Team' }],
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" sizes="any" />
      </head>
      <body className={cn(inter.className, 'min-h-full bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-900')}>
        <div id="root-container" className="min-h-screen">
          <header id="main-header" className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 sticky top-0 z-50">
            <div id="header-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
              <div id="header-nav" className="flex items-center justify-between">
                <div id="logo-section" className="flex items-center space-x-3">
                  <div id="logo-icon" className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">A</span>
                  </div>
                  <div id="logo-text">
                    <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                      Arbitration Detector
                    </h1>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Frontend Integration Demo
                    </p>
                  </div>
                </div>
                <div id="header-status" className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Live Demo
                  </span>
                </div>
              </div>
            </div>
          </header>
          
          <main id="main-content" className="flex-1">
            {children}
          </main>
          
          <footer id="main-footer" className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm border-t border-gray-200 dark:border-gray-800 mt-auto">
            <div id="footer-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              <div id="footer-info" className="text-center text-sm text-gray-600 dark:text-gray-400">
                <p>
                  Arbitration Detector Frontend v1.0.0 • Built with Next.js 14 & React 18
                </p>
                <p className="mt-1">
                  Integrating with FastAPI Backend • WebSocket Support • TypeScript
                </p>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}