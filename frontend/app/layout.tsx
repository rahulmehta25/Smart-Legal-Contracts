import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';
import {
  Scale,
  Shield,
  FileSearch,
} from 'lucide-react';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Smart Legal Contracts | AI-Powered Arbitration Clause Detection',
  description: 'RAG-powered legal document analysis platform detecting arbitration clauses with 85%+ accuracy using advanced NLP and retrieval-augmented generation.',
  keywords: ['arbitration', 'legal tech', 'document analysis', 'AI', 'RAG', 'NLP', 'contract analysis'],
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0f172a',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full dark">
      <body className={cn(inter.className, 'min-h-full bg-slate-950 text-slate-100 antialiased')}>
        <div className="min-h-screen flex flex-col">
          <header className="bg-slate-900/80 backdrop-blur-xl border-b border-slate-800 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-9 h-9 bg-gradient-to-br from-violet-500 to-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-violet-500/20">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.031.352 5.988 5.988 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L18.75 4.971zm-16.5.52c.99-.203 1.99-.377 3-.52m0 0l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.989 5.989 0 01-2.031.352 5.989 5.989 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L5.25 4.971z" />
                    </svg>
                  </div>
                  <div>
                    <h1 className="text-lg font-bold text-white tracking-tight">Smart Legal Contracts</h1>
                    <p className="text-[10px] text-violet-400 font-medium tracking-widest uppercase">RAG-Powered Analysis</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="hidden sm:flex items-center space-x-2 text-xs text-slate-400 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700/50">
                    <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                    <span>System Online</span>
                  </div>
                  <div className="hidden md:flex items-center space-x-1 text-xs text-slate-500">
                    <span>v2.1.0</span>
                  </div>
                </div>
              </div>
            </div>
          </header>

          <main className="flex-1">
            {children}
          </main>

          <footer className="bg-slate-900/50 border-t border-slate-800/50 mt-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-500">
                <p>Smart Legal Contracts &copy; 2026 &middot; Enterprise Legal Intelligence</p>
                <div className="flex items-center gap-6">
                  <span>FastAPI + ChromaDB</span>
                  <span>Next.js 14</span>
                  <span>BERT NLP</span>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
