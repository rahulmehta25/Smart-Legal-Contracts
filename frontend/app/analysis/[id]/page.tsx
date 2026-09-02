"use client";

import { useEffect } from "react";
import { useAnalysis, useDocument } from "@/lib/hooks";
import { Skeleton } from "@/components/ui/skeleton";
import { posthog } from "@/lib/posthog";
import { AnalysisResults } from "@/components/analysis/analysis-results";
import { SampleDemoCard } from "@/components/demo/sample-demo-card";
import { TextLink } from "@/components/ui/text-link";
import {
  SAMPLE_ANALYSIS,
  SAMPLE_DOCUMENT,
  isSampleAnalysisId,
} from "@/lib/sample-analysis";

function AnalysisResultsSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="h-4 w-96" />
      <div className="grid md:grid-cols-3 gap-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
      <Skeleton className="h-64" />
    </div>
  );
}

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const id = params.id;
  const sampleRequested = isSampleAnalysisId(id);
  const analysisId = Number.parseInt(id, 10);
  const shouldFetch = !sampleRequested && Number.isFinite(analysisId) && analysisId > 0;
  const { data: analysis, isLoading, error } = useAnalysis(shouldFetch ? analysisId : 0);
  const { data: document } = useDocument(shouldFetch ? analysis?.document_id || 0 : 0);

  useEffect(() => {
    if (sampleRequested) {
      posthog.capture?.("risk_memo_view", {
        analysis_id: "sample",
        narration_available: false,
        source: "sample_demo",
      });
      return;
    }
    if (!analysis) return;
    const narrationOn = !!posthog.getFeatureFlag?.("slc-risk-memo-narration");
    posthog.capture?.("risk_memo_view", {
      analysis_id: analysisId,
      narration_available: narrationOn,
    });
  }, [analysis, analysisId, sampleRequested]);

  if (sampleRequested) {
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

  if (!shouldFetch) {
    return (
      <div className="page-wrap py-16">
        <h1 className="display text-3xl">Analysis not found</h1>
        <p className="mt-3 max-w-lg text-base leading-relaxed text-ink-muted">
          The requested analysis could not be loaded. The upload API may be unavailable.
        </p>
        <p className="mt-6">
          <TextLink href="/history">Back to history</TextLink>
        </p>
        <SampleDemoCard
          className="mt-10"
          description="You can still open the sample MSA analysis without a live upload."
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="page-wrap py-16">
        <AnalysisResultsSkeleton />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="page-wrap py-16">
        <h1 className="display text-3xl">Analysis not found</h1>
        <p className="mt-3 max-w-lg text-base leading-relaxed text-ink-muted">
          The requested analysis could not be loaded. The upload API may be unavailable.
        </p>
        <p className="mt-6">
          <TextLink href="/history">Back to history</TextLink>
        </p>
        <SampleDemoCard
          className="mt-10"
          description="You can still open the sample MSA analysis without a live upload."
        />
      </div>
    );
  }

  return <AnalysisResults analysis={analysis} document={document} />;
}
