"use client";

import { useEffect } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { posthog } from "@/lib/posthog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TextLink } from "@/components/ui/text-link";
import { Reveal } from "@/components/ui/reveal";
import {
  computeRiskLevel,
  copyToClipboard,
  formatConfidence,
  formatDateTime,
  formatProcessingTime,
  getRiskBadgeVariant,
} from "@/lib/utils";
import { ClauseCard } from "@/components/analysis/clause-card";
import { SAMPLE_DEMO_PATH } from "@/lib/sample-analysis";
import type { ArbitrationAnalysis, Document } from "@/types/api";

interface AnalysisResultsProps {
  analysis: ArbitrationAnalysis;
  document?: Document | null;
  backHref?: string;
  backLabel?: string;
  isSample?: boolean;
}

export function AnalysisResults({
  analysis,
  document,
  backHref = "/history",
  backLabel = "Back",
  isSample = false,
}: AnalysisResultsProps) {
  const riskLevel = analysis.risk_level ?? computeRiskLevel(analysis);

  useEffect(() => {
    if (isSample) {
      posthog.capture?.("risk_memo_view", {
        analysis_id: "sample",
        narration_available: false,
        source: "sample_demo",
      });
      return;
    }
    const narrationOn = !!posthog.getFeatureFlag?.("slc-risk-memo-narration");
    posthog.capture?.("risk_memo_view", {
      analysis_id: analysis.id,
      narration_available: narrationOn,
    });
  }, [analysis.id, isSample]);

  const handleExport = async () => {
    const payload = {
      document: document?.filename ?? `Analysis #${analysis.id}`,
      analysis,
    };
    const copied = await copyToClipboard(JSON.stringify(payload, null, 2));
    if (copied) {
      toast.success("Analysis JSON copied to clipboard");
    } else {
      toast.error("Could not copy analysis JSON");
    }
  };

  return (
    <div>
      <section className="band band-ivory">
        <div className="page-wrap py-10 lg:py-14">
          <TextLink href={backHref}>{backLabel}</TextLink>

          {isSample && (
            <p className="mt-6 max-w-2xl text-sm leading-relaxed text-ink-muted">
              This walkthrough uses a canned SaaS MSA. It does not call the upload API.
            </p>
          )}

          <div className="mt-6 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="eyebrow">{isSample ? "Sample analysis" : "Analysis"}</p>
              <h1 className="display mt-3 text-3xl sm:text-4xl">
                {document?.filename || `Analysis #${analysis.id}`}
              </h1>
              <p className="mt-3 text-sm text-ink-muted">
                {formatDateTime(analysis.analyzed_at)}
                {" · "}
                {formatProcessingTime(analysis.processing_time_ms)}
                {document?.page_count ? ` · ${document.page_count} pages` : ""}
              </p>
            </div>
            <Button variant="link" onClick={handleExport}>
              Export JSON
            </Button>
          </div>
        </div>
      </section>

      <div className="flex justify-center band-ivory" aria-hidden>
        <div className="h-8 w-px bg-rule" />
      </div>

      <section className="band band-linen">
        <div className="page-wrap py-10">
          <Reveal>
            <div className="grid grid-cols-1 gap-10 sm:grid-cols-3 sm:gap-0">
              <div className="sm:pr-8">
                <p className="eyebrow">Risk level</p>
                <p className="mt-2 font-serif text-3xl font-medium capitalize tracking-tight text-ink">
                  {riskLevel}
                </p>
                <Badge variant={getRiskBadgeVariant(riskLevel)} className="mt-3">
                  {riskLevel} risk
                </Badge>
              </div>
              <div className="sm:border-l sm:border-rule sm:px-8">
                <p className="eyebrow">Clauses found</p>
                <p className="mt-2 font-serif text-3xl font-medium tracking-tight text-ink">
                  {analysis.clauses.length}
                </p>
              </div>
              <div className="sm:border-l sm:border-rule sm:pl-8">
                <p className="eyebrow">Confidence</p>
                <p className="mt-2 font-serif text-3xl font-medium tracking-tight text-ink">
                  {formatConfidence(analysis.confidence_score)}
                </p>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      <div className="flex justify-center band-linen" aria-hidden>
        <div className="h-8 w-px bg-rule" />
      </div>

      <section className="band band-ivory">
        <div className="page-wrap py-12 lg:py-16">
          <Tabs defaultValue="clauses">
            <TabsList className="mb-8 h-auto rounded-none bg-transparent p-0">
              <TabsTrigger
                value="clauses"
                className="rounded-none border-b-2 border-transparent bg-transparent px-0 pb-2 mr-8 shadow-none data-[state=active]:border-brass data-[state=active]:bg-transparent data-[state=active]:shadow-none"
              >
                Clauses ({analysis.clauses.length})
              </TabsTrigger>
              <TabsTrigger
                value="summary"
                className="rounded-none border-b-2 border-transparent bg-transparent px-0 pb-2 shadow-none data-[state=active]:border-brass data-[state=active]:bg-transparent data-[state=active]:shadow-none"
              >
                Summary
              </TabsTrigger>
            </TabsList>

            <TabsContent value="clauses">
              {analysis.clauses.length > 0 ? (
                <div>
                  {analysis.clauses.map((clause, index) => (
                    <ClauseCard key={clause.id} clause={clause} index={index} />
                  ))}
                </div>
              ) : (
                <div className="border-t border-b border-rule py-12">
                  <h3 className="font-serif text-2xl font-medium text-ink">
                    No arbitration clauses found
                  </h3>
                  <p className="mt-3 max-w-lg text-sm leading-relaxed text-ink-muted">
                    This document was analyzed with {formatConfidence(analysis.confidence_score)}{" "}
                    confidence and no arbitration-related clauses were detected.
                  </p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="summary">
              <div className="max-w-2xl border-t border-rule pt-8">
                <h3 className="font-serif text-2xl font-medium text-ink">Analysis summary</h3>
                <p className="mt-4 text-base leading-relaxed text-ink-muted whitespace-pre-wrap">
                  {analysis.analysis_summary || "No summary available."}
                </p>
                <dl className="mt-10 grid grid-cols-2 gap-6 text-sm">
                  <div>
                    <dt className="eyebrow">Version</dt>
                    <dd className="mt-1 text-ink">{analysis.analysis_version}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">Processing time</dt>
                    <dd className="mt-1 text-ink">{formatProcessingTime(analysis.processing_time_ms)}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">Document</dt>
                    <dd className="mt-1 text-ink">{isSample ? "sample" : analysis.document_id}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">Has arbitration</dt>
                    <dd className="mt-1 text-ink">{analysis.has_arbitration_clause ? "Yes" : "No"}</dd>
                  </div>
                </dl>
                {isSample ? (
                  <p className="mt-8">
                    <TextLink href={SAMPLE_DEMO_PATH}>Keep this sample open</TextLink>
                  </p>
                ) : (
                  <p className="mt-8">
                    <Link href="/upload" className="text-link">
                      Analyze another document
                    </Link>
                  </p>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </section>
    </div>
  );
}
