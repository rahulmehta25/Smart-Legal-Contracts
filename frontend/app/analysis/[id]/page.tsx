"use client";

import { use, useEffect } from "react";
import Link from "next/link";
import { useAnalysis, useDocument } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import { posthog } from "@/lib/posthog";
import { AnalysisResults } from "@/components/analysis/analysis-results";
import { SampleDemoCard } from "@/components/demo/sample-demo-card";
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

export default function AnalysisPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const sampleRequested = isSampleAnalysisId(id);
  const analysisId = Number.parseInt(id, 10);
  const shouldFetch = !sampleRequested && Number.isFinite(analysisId);
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

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnalysisResultsSkeleton />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card className="mb-6">
          <CardContent className="py-12 text-center">
            <AlertTriangle className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-gray-900 mb-1">Analysis not found</h3>
            <p className="text-sm text-gray-500 mb-4">
              The requested analysis could not be loaded. The upload API may be unavailable.
            </p>
            <Button asChild variant="outline">
              <Link href="/history">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to History
              </Link>
            </Button>
          </CardContent>
        </Card>
        <SampleDemoCard description="You can still open the sample MSA analysis without a live upload." />
      </div>
    );
  }

  return <AnalysisResults analysis={analysis} document={document} />;
}
