import Link from "next/link";
import { SAMPLE_DEMO_PATH } from "@/lib/sample-analysis";

export function SiteFooter() {
  return (
    <footer className="border-t border-rule bg-ink text-ivory">
      <div className="page-wrap py-12">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="font-serif text-2xl font-medium tracking-tight">Smart Legal</p>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-ivory/70">
              Portfolio review of arbitration language. Sample analysis stays available when live
              upload is not.
            </p>
          </div>
          <div className="flex flex-col gap-3 text-sm">
            <Link href={SAMPLE_DEMO_PATH} className="text-link text-ivory">
              Sample analysis
            </Link>
            <Link href="/upload" className="text-link text-ivory">
              Upload a document
            </Link>
            <Link href="/history" className="text-link text-ivory">
              History
            </Link>
          </div>
        </div>
        <div className="mt-10 flex flex-col gap-2 border-t border-ivory/15 pt-6 text-xs tracking-wide text-ivory/50 sm:flex-row sm:justify-between">
          <p>Smart Legal Contracts. All rights reserved.</p>
          <p>v2.0.0</p>
        </div>
      </div>
    </footer>
  );
}
