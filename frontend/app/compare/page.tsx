"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { useAnalyses, useDocuments } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  GitCompare,
  CheckCircle,
  Minus,
} from "lucide-react";
import { formatConfidence } from "@/lib/utils";
import { staggerContainer, staggerItem } from "@/components/ui/motion";
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
  index,
}: {
  clauseType: string;
  clauseA?: ArbitrationClause;
  clauseB?: ArbitrationClause;
  index: number;
}) {
  const hasA = !!clauseA;
  const hasB = !!clauseB;

  return (
    <motion.div
      className="grid grid-cols-3 gap-4 py-4 border-b border-gray-100 last:border-0"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
    >
      <div className="font-medium text-sm text-gray-700">
        {clauseType.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
      </div>
      <div className="text-center">
        {hasA ? (
          <div className="space-y-1">
            <Badge variant={getRiskBadgeVariant(clauseA.risk_level)}>
              {clauseA.risk_level}
            </Badge>
            <p className="text-xs text-gray-500">
              {formatConfidence(clauseA.confidence_score)}
            </p>
          </div>
        ) : (
          <span className="text-sm text-gray-400">Not found</span>
        )}
      </div>
      <div className="text-center">
        {hasB ? (
          <div className="space-y-1">
            <Badge variant={getRiskBadgeVariant(clauseB.risk_level)}>
              {clauseB.risk_level}
            </Badge>
            <p className="text-xs text-gray-500">
              {formatConfidence(clauseB.confidence_score)}
            </p>
          </div>
        ) : (
          <span className="text-sm text-gray-400">Not found</span>
        )}
      </div>
    </motion.div>
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
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Compare Documents</h1>
        <p className="text-gray-600">
          Select two analyzed documents to compare their arbitration clauses side-by-side.
        </p>
      </div>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="text-base">Select Documents</CardTitle>
          <CardDescription>Choose two documents to compare their analysis results.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
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
              <label className="block text-sm font-medium text-gray-700 mb-2">
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
        </CardContent>
      </Card>

      {!analysisA || !analysisB ? (
        <Card>
          <CardContent className="py-12 text-center">
            <GitCompare className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-gray-900 mb-1">Select documents to compare</h3>
            <p className="text-sm text-gray-500">
              Choose two documents above to see a side-by-side comparison of their clauses.
            </p>
          </CardContent>
        </Card>
      ) : comparisonData ? (
        <>
          <motion.div
            className="grid md:grid-cols-3 gap-4 mb-8"
            variants={staggerContainer}
            initial="hidden"
            animate="show"
          >
            <motion.div variants={staggerItem}>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
                      <CheckCircle className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Common Clauses</p>
                      <p className="text-lg font-semibold text-gray-900">{comparisonData.inBoth}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={staggerItem}>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-amber-50 flex items-center justify-center">
                      <Minus className="h-5 w-5 text-amber-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Only in Document A</p>
                      <p className="text-lg font-semibold text-gray-900">{comparisonData.onlyInA}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div variants={staggerItem}>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-purple-50 flex items-center justify-center">
                      <Minus className="h-5 w-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Only in Document B</p>
                      <p className="text-lg font-semibold text-gray-900">{comparisonData.onlyInB}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>

          <Card>
            <CardHeader className="pb-0">
              <div className="grid grid-cols-3 gap-4">
                <div className="font-medium text-sm text-gray-500">Clause Type</div>
                <div className="text-center">
                  <p className="font-medium text-sm text-gray-900">
                    {documentMap.get(analysisA.document_id)?.filename || "Document A"}
                  </p>
                  <p className="text-xs text-gray-500">
                    {analysisA.clauses?.length || 0} clauses
                  </p>
                </div>
                <div className="text-center">
                  <p className="font-medium text-sm text-gray-900">
                    {documentMap.get(analysisB.document_id)?.filename || "Document B"}
                  </p>
                  <p className="text-xs text-gray-500">
                    {analysisB.clauses?.length || 0} clauses
                  </p>
                </div>
              </div>
            </CardHeader>
            <Separator className="my-4" />
            <CardContent className="pt-0">
              {comparisonData.comparisons.length === 0 ? (
                <div className="py-8 text-center">
                  <CheckCircle className="h-12 w-12 text-emerald-400 mx-auto mb-4" />
                  <p className="text-sm text-gray-600">
                    Neither document contains arbitration clauses.
                  </p>
                </div>
              ) : (
                <div>
                  {comparisonData.comparisons.map(({ clauseType, clauseA, clauseB }, index) => (
                    <ClauseComparisonRow
                      key={clauseType}
                      clauseType={clauseType}
                      clauseA={clauseA}
                      clauseB={clauseB}
                      index={index}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}
