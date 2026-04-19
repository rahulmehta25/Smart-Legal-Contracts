import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';
import { Toaster } from '@/components/ui/toaster';
import { QueryProvider } from '@/components/providers/query-provider';
import { Navigation } from '@/components/layout/navigation';
import { Analytics } from '@vercel/analytics/next';
import { SpeedInsights } from '@vercel/speed-insights/next';
import { PostHogProvider } from '@/components/analytics/posthog-provider';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'Smart Legal Contracts | AI-Powered Arbitration Clause Detection',
  description: 'Detect arbitration clauses in legal documents with AI-powered analysis. Upload contracts and get instant risk assessments with 85%+ accuracy.',
  keywords: ['arbitration', 'legal tech', 'document analysis', 'AI', 'contract analysis', 'legal AI'],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#ffffff',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={cn(inter.className, 'min-h-screen bg-white text-gray-900 antialiased')}>
        <QueryProvider>
          <div className="min-h-screen flex flex-col">
            <Navigation />
            <main className="flex-1">
              {children}
            </main>
            <footer className="border-t border-gray-100 bg-gray-50/50">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
                  <p>Smart Legal Contracts. All rights reserved.</p>
                  <div className="flex items-center gap-6">
                    <span>v2.0.0</span>
                  </div>
                </div>
              </div>
            </footer>
          </div>
          <Toaster />
        </QueryProvider>
        <Analytics />
        <SpeedInsights />
        <PostHogProvider />
      </body>
    </html>
  );
}
