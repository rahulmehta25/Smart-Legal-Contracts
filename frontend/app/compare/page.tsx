"use client";

import { useState, useMemo } from "react";
import { useAnalyses, useDocuments } from "@/lib/hooks";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatConfidence } from "@/lib/utils";
import type { ArbitrationClause, RiskLevel } from "@/types/api";

function getRiskBadgeVariant(level: RiskLevel | undefined) {
  switch (level) {
    case "high":
      return "danger";
    case "medium":
      return "warning";
    case "low":
      return "success";
    default:
      return "secondary";
  }
}

function ClauseComparisonRow({
  clauseType,
  clauseA,
  clauseB,
}: {
  clauseType: string;
  clauseA?: ArbitrationClause;
  clauseB?: ArbitrationClause;
}) {
  const hasA = !!clauseA;
  const hasB = !!clauseB;

  return (
    <div className="grid grid-cols-1 gap-3 border-t border-rule py-5 last:border-b sm:grid-cols-3 sm:gap-4">
      <div className="font-serif text-lg text-ink">
        {clauseType.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
      </div>
      <div className="sm:text-center">
        {hasA ? (
          <div className="space-y-1">
            <Badge variant={getRiskBadgeVariant(clauseA.risk_level)}>
              {clauseA.risk_level}
            </Badge>
            <p className="text-xs text-ink-muted">
              {formatConfidence(clauseA.confidence_score)}
            </p>
          </div>
        ) : (
          <span className="text-sm text-ink-muted">Not found</span>
        )}
      </div>
      <div className="sm:text-center">
        {hasB ? (
          <div className="space-y-1">
            <Badge variant={getRiskBadgeVariant(clauseB.risk_level)}>
              {clauseB.risk_level}
            </Badge>
            <p className="text-xs text-ink-muted">
              {formatConfidence(clauseB.confidence_score)}
            </p>
          </div>
        ) : (
          <span className="text-sm text-ink-muted">Not found</span>
        )}
      </div>
    </div>
  );
}

export default function ComparePage() {
  const [documentAId, setDocumentAId] = useState<string>("");
  const [documentBId, setDocumentBId] = useState<string>("");

  const { data: analyses } = useAnalyses({ limit: 100 });
  const { data: documents } = useDocuments({ limit: 100 });

  const analysisA = analyses?.find((a) => a.id.toString() === documentAId);
  const analysisB = analyses?.find((a) => a.id.toString() === documentBId);

  const documentMap = useMemo(() => {
    if (!documents) return new Map();
    return new Map(documents.map((doc) => [doc.id, doc]));
  }, [documents]);

  const comparisonData = useMemo(() => {
    if (!analysisA || !analysisB) return null;

    const allClauseTypes = new Set<string>();
    analysisA.clauses?.forEach((c) => allClauseTypes.add(c.clause_type));
    analysisB.clauses?.forEach((c) => allClauseTypes.add(c.clause_type));

    const comparisons = Array.from(allClauseTypes).map((clauseType) => {
      const clauseA = analysisA.clauses?.find((c) => c.clause_type === clauseType);
      const clauseB = analysisB.clauses?.find((c) => c.clause_type === clauseType);
      return { clauseType, clauseA, clauseB };
    });

    const onlyInA = comparisons.filter((c) => c.clauseA && !c.clauseB).length;
    const onlyInB = comparisons.filter((c) => !c.clauseA && c.clauseB).length;
    const inBoth = comparisons.filter((c) => c.clauseA && c.clauseB).length;

    return { comparisons, onlyInA, onlyInB, inBoth };
  }, [analysisA, analysisB]);

  return (
    <div className="page-wrap py-12 lg:py-16">
      <div className="mb-10">
        <p className="eyebrow">Side by side</p>
        <h1 className="display mt-3 text-3xl sm:text-4xl">Compare documents</h1>
        <p className="mt-3 max-w-xl text-base leading-relaxed text-ink-muted">
          Select two analyzed documents to compare their arbitration clauses.
        </p>
      </div>

      <div className="border-y border-rule py-8">
        <p className="eyebrow">Documents</p>
        <div className="mt-5 grid gap-6 md:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm font-medium text-ink">
              Document A
            </label>
            <Select value={documentAId} onValueChange={setDocumentAId}>
              <SelectTrigger>
                <SelectValue placeholder="Select first document" />
              </SelectTrigger>
              <SelectContent>
                {analyses?.map((analysis) => {
                  const doc = documentMap.get(analysis.document_id);
                  return (
                    <SelectItem
                      key={analysis.id}
                      value={analysis.id.toString()}
                      disabled={analysis.id.toString() === documentBId}
                    >
                      {doc?.filename || `Document #${analysis.document_id}`}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-ink">
              Document B
            </label>
            <Select value={documentBId} onValueChange={setDocumentBId}>
              <SelectTrigger>
                <SelectValue placeholder="Select second document" />
              </SelectTrigger>
              <SelectContent>
                {analyses?.map((analysis) => {
                  const doc = documentMap.get(analysis.document_id);
                  return (
                    <SelectItem
                      key={analysis.id}
                      value={analysis.id.toString()}
                      disabled={analysis.id.toString() === documentAId}
                    >
                      {doc?.filename || `Document #${analysis.document_id}`}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {!analysisA || !analysisB ? (
        <div className="border-b border-rule py-12">
          <p className="font-serif text-2xl font-medium text-ink">Select two documents to compare</p>
          <p className="mt-3 max-w-lg text-sm leading-relaxed text-ink-muted">
            Choose a pair above to see which clauses appear in both reviews, and which appear in only one.
          </p>
        </div>
      ) : comparisonData ? (
        <>
          <div className="grid grid-cols-1 gap-8 border-b border-rule py-8 sm:grid-cols-3 sm:gap-0">
            <div className="sm:pr-8">
              <p className="eyebrow">Common clauses</p>
              <p className="mt-2 font-serif text-3xl font-medium text-ink">{comparisonData.inBoth}</p>
            </div>
            <div className="sm:border-l sm:border-rule sm:px-8">
              <p className="eyebrow">Only in document A</p>
              <p className="mt-2 font-serif text-3xl font-medium text-ink">{comparisonData.onlyInA}</p>
            </div>
            <div className="sm:border-l sm:border-rule sm:pl-8">
              <p className="eyebrow">Only in document B</p>
              <p className="mt-2 font-serif text-3xl font-medium text-ink">{comparisonData.onlyInB}</p>
            </div>
          </div>

          <div className="mt-10 hidden grid-cols-3 gap-4 sm:grid">
            <p className="eyebrow">Clause type</p>
            <div className="text-center">
              <p className="filename-display font-serif text-lg text-ink">
                {documentMap.get(analysisA.document_id)?.filename || "Document A"}
              </p>
              <p className="mt-1 text-xs text-ink-muted">
                {analysisA.clauses?.length || 0} clauses
              </p>
            </div>
            <div className="text-center">
              <p className="filename-display font-serif text-lg text-ink">
                {documentMap.get(analysisB.document_id)?.filename || "Document B"}
              </p>
              <p className="mt-1 text-xs text-ink-muted">
                {analysisB.clauses?.length || 0} clauses
              </p>
            </div>
          </div>

          {comparisonData.comparisons.length === 0 ? (
            <div className="border-y border-rule py-12">
              <p className="font-serif text-2xl font-medium text-ink">
                Neither document contains arbitration clauses
              </p>
            </div>
          ) : (
            <div className="mt-4">
              {comparisonData.comparisons.map(({ clauseType, clauseA, clauseB }) => (
                <ClauseComparisonRow
                  key={clauseType}
                  clauseType={clauseType}
                  clauseA={clauseA}
                  clauseB={clauseB}
                />
              ))}
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
