"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { useAnalyses, useDocuments } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TextLink } from "@/components/ui/text-link";
import { computeRiskLevel, formatConfidence, formatRelativeTime, getRiskBadgeVariant } from "@/lib/utils";
import { SampleDemoCard } from "@/components/demo/sample-demo-card";
import {
  SAMPLE_ANALYSIS,
  SAMPLE_DEMO_PATH,
  SAMPLE_DOCUMENT,
} from "@/lib/sample-analysis";

function HistorySkeleton() {
  return (
    <div className="space-y-4">
      {[...Array(5)].map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  );
}

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("all");
  const [arbitrationFilter, setArbitrationFilter] = useState<string>("all");

  const { data: analyses, isLoading: analysesLoading, isError: analysesError } = useAnalyses({ limit: 100 });
  const { data: documents, isLoading: documentsLoading } = useDocuments({ limit: 100 });

  const isLoading = analysesLoading || documentsLoading;

  const documentMap = useMemo(() => {
    const map = new Map<number, { filename: string }>();
    map.set(SAMPLE_DOCUMENT.id, SAMPLE_DOCUMENT);
    documents?.forEach((doc) => map.set(doc.id, doc));
    return map;
  }, [documents]);

  const liveAnalyses = useMemo(() => analyses ?? [], [analyses]);
  const allAnalyses = useMemo(() => [SAMPLE_ANALYSIS, ...liveAnalyses], [liveAnalyses]);

  const filteredAnalyses = useMemo(() => {
    return allAnalyses.filter((analysis) => {
      const document = documentMap.get(analysis.document_id);
      const filename = document?.filename || "";
      const riskLevel = computeRiskLevel(analysis);

      if (searchQuery && !filename.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      if (riskFilter !== "all" && riskLevel !== riskFilter) {
        return false;
      }

      if (arbitrationFilter === "yes" && !analysis.has_arbitration_clause) {
        return false;
      }
      if (arbitrationFilter === "no" && analysis.has_arbitration_clause) {
        return false;
      }

      return true;
    });
  }, [allAnalyses, documentMap, searchQuery, riskFilter, arbitrationFilter]);

  const stats = useMemo(() => {
    return {
      total: liveAnalyses.length,
      withArbitration: liveAnalyses.filter((a) => a.has_arbitration_clause).length,
      highRisk: liveAnalyses.filter((a) => computeRiskLevel(a) === "high").length,
    };
  }, [liveAnalyses]);

  return (
    <div className="page-wrap py-12 lg:py-16">
      <div className="mb-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <p className="eyebrow">Archive</p>
          <h1 className="display mt-3 text-3xl sm:text-4xl">Analysis history</h1>
          <p className="mt-3 max-w-xl text-base leading-relaxed text-ink-muted">
            View live analyses when the API is available. The sample review is always listed.
          </p>
        </div>
        <Button asChild>
          <Link href="/upload">New analysis</Link>
        </Button>
      </div>

      <div className="mb-10 grid grid-cols-1 gap-8 border-y border-rule py-8 sm:grid-cols-3 sm:gap-0">
        <div className="sm:pr-8">
          <p className="eyebrow">Total analyses</p>
          <p className="mt-2 font-serif text-3xl font-medium text-ink">{stats.total}</p>
        </div>
        <div className="sm:border-l sm:border-rule sm:px-8">
          <p className="eyebrow">With arbitration</p>
          <p className="mt-2 font-serif text-3xl font-medium text-ink">{stats.withArbitration}</p>
        </div>
        <div className="sm:border-l sm:border-rule sm:pl-8">
          <p className="eyebrow">High risk</p>
          <p className="mt-2 font-serif text-3xl font-medium text-ink">{stats.highRisk}</p>
        </div>
      </div>

      <SampleDemoCard className="mb-10" />

      {analysesError && (
        <p className="text-sm text-amber-700 mb-4">
          Live history could not be loaded. The sample analysis is still available.
        </p>
      )}

      <div className="border-y border-rule py-6">
        <div className="flex flex-col gap-3 sm:flex-row">
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1"
            aria-label="Search documents"
          />
          <div className="grid grid-cols-2 gap-3 sm:flex">
            <Select value={riskFilter} onValueChange={setRiskFilter}>
              <SelectTrigger className="w-full sm:w-[140px]" aria-label="Risk level">
                <SelectValue placeholder="Risk Level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Risks</SelectItem>
                <SelectItem value="high">High Risk</SelectItem>
                <SelectItem value="medium">Medium Risk</SelectItem>
                <SelectItem value="low">Low Risk</SelectItem>
              </SelectContent>
            </Select>

            <Select value={arbitrationFilter} onValueChange={setArbitrationFilter}>
              <SelectTrigger className="w-full sm:w-[160px]" aria-label="Arbitration">
                <SelectValue placeholder="Arbitration" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Documents</SelectItem>
                <SelectItem value="yes">Has Arbitration</SelectItem>
                <SelectItem value="no">No Arbitration</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {isLoading && filteredAnalyses.length === 0 ? (
        <div className="py-8">
          <HistorySkeleton />
        </div>
      ) : filteredAnalyses.length === 0 ? (
        <div className="border-b border-rule py-12">
          <h3 className="font-serif text-2xl font-medium text-ink">No analyses found</h3>
          <p className="mt-3 max-w-lg text-sm leading-relaxed text-ink-muted">
            {searchQuery || riskFilter !== "all" || arbitrationFilter !== "all"
              ? "Try adjusting your filters."
              : "Upload a document when the API is available, or open the sample analysis."}
          </p>
          <div className="mt-5 flex flex-wrap items-center gap-x-8 gap-y-3">
            <Button asChild>
              <Link href={SAMPLE_DEMO_PATH}>View Demo</Link>
            </Button>
            {!searchQuery && riskFilter === "all" && arbitrationFilter === "all" && (
              <TextLink href="/upload">Upload a document</TextLink>
            )}
          </div>
        </div>
      ) : (
        <>
          <ul className="md:hidden">
            {filteredAnalyses.map((analysis) => {
              const document = documentMap.get(analysis.document_id);
              const riskLevel = computeRiskLevel(analysis);
              const isSample = analysis === SAMPLE_ANALYSIS;
              const href = isSample ? SAMPLE_DEMO_PATH : `/analysis/${analysis.id}`;
              const filename = document?.filename || `Document #${analysis.document_id}`;

              return (
                <li key={isSample ? "sample" : analysis.id} className="border-b border-rule py-5">
                  <p className="filename-display font-serif text-lg font-medium text-ink">{filename}</p>
                  {isSample && (
                    <p className="mt-1 text-xs text-ink-muted">Sample, no upload required</p>
                  )}
                  <p className="mt-2 text-sm text-ink-muted">
                    <Badge variant={getRiskBadgeVariant(riskLevel)}>{riskLevel}</Badge>
                    <span className="ml-3">
                      {analysis.clauses?.length || 0} clauses · {formatConfidence(analysis.confidence_score)} ·{" "}
                      {formatRelativeTime(analysis.analyzed_at)}
                    </span>
                  </p>
                  <p className="mt-3">
                    <TextLink href={href}>View analysis</TextLink>
                  </p>
                </li>
              );
            })}
          </ul>

          <div className="hidden overflow-x-auto md:block">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Document</TableHead>
                  <TableHead>Risk Level</TableHead>
                  <TableHead>Clauses</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Analyzed</TableHead>
                  <TableHead className="w-[100px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAnalyses.map((analysis, index) => {
                  const document = documentMap.get(analysis.document_id);
                  const riskLevel = computeRiskLevel(analysis);
                  const isSample = analysis === SAMPLE_ANALYSIS;

                  return (
                    <TableRow
                      key={isSample ? "sample" : analysis.id}
                      className="transition-colors duration-150"
                      style={{
                        animation: `row-fade-in 0.3s ease-out ${index * 50}ms backwards`,
                      }}
                    >
                      <TableCell>
                        <div className="min-w-0">
                          <p className="filename-display font-medium text-sm text-ink">
                            {document?.filename || `Document #${analysis.document_id}`}
                          </p>
                          {isSample && (
                            <p className="text-xs text-ink-muted">Sample, no upload required</p>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={getRiskBadgeVariant(riskLevel)}>
                          {riskLevel}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm text-ink-muted">
                          {analysis.clauses?.length || 0}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm text-ink-muted">
                          {formatConfidence(analysis.confidence_score)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm text-ink-muted">
                          {formatRelativeTime(analysis.analyzed_at)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <TextLink href={isSample ? SAMPLE_DEMO_PATH : `/analysis/${analysis.id}`}>
                          View
                        </TextLink>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  );
}
