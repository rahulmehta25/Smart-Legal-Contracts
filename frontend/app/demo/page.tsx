"use client";

import { AnalysisResults } from "@/components/analysis/analysis-results";
import { SAMPLE_ANALYSIS, SAMPLE_DOCUMENT } from "@/lib/sample-analysis";

export default function DemoPage() {
  return (
    <AnalysisResults
      analysis={SAMPLE_ANALYSIS}
      document={SAMPLE_DOCUMENT}
      backHref="/"
      backLabel="Back to home"
      isSample
    />
  );
}
