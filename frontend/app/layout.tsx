import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "@fontsource-variable/source-sans-3/wght.css";
import "@fontsource-variable/fraunces/wght.css";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Toaster } from "@/components/ui/toaster";
import { QueryProvider } from "@/components/providers/query-provider";
import { Navigation } from "@/components/layout/navigation";
import { SiteFooter } from "@/components/layout/site-footer";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { PostHogProvider } from "@/components/analytics/posthog-provider";

export const metadata: Metadata = {
  title: "Smart Legal Contracts | Arbitration clause detection",
  description:
    "Detect arbitration clauses in legal documents with AI analysis. Upload a contract or open the sample review for risk ratings and recommended next steps.",
  keywords: ["arbitration", "legal tech", "document analysis", "AI", "contract analysis", "legal AI"],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#F6F1E8",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <html lang="en">
      <body className={cn("min-h-screen bg-ivory font-sans text-ink antialiased")}>
        <QueryProvider>
          <div className="flex min-h-screen flex-col">
            <Navigation />
            <main className="flex-1">{children}</main>
            <SiteFooter />
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
