"use client";

import { use, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useAnalysis, useDocument } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ArrowLeft,
  FileText,
  Shield,
  Clock,
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
} from "lucide-react";
import {
  cn,
  formatDateTime,
  formatConfidence,
  formatProcessingTime,
  getRiskColor,
  getRiskBgColor,
  getRiskBorderColor,
  copyToClipboard,
} from "@/lib/utils";
import { toast } from "sonner";
import { staggerContainer, staggerItem, fadeInScale } from "@/components/ui/motion";
import type { ArbitrationClause, RiskLevel } from "@/types/api";

function ClauseCard({ clause, index }: { clause: ArbitrationClause; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);

  const riskLevel = clause.risk_level || "medium";

  return (
    <Card className={cn("border", getRiskBorderColor(riskLevel))}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left"
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Badge
                  variant={
                    riskLevel === "high" ? "danger" :
                    riskLevel === "medium" ? "warning" : "success"
                  }
                >
                  {riskLevel.toUpperCase()}
                </Badge>
              </motion.div>
              <div>
                <CardTitle className="text-base font-medium">
                  {clause.clause_type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                </CardTitle>
                {clause.section_reference && (
                  <CardDescription className="text-xs mt-0.5">
                    {clause.section_reference}
                  </CardDescription>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  {formatConfidence(clause.confidence_score)}
                </span>
                <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${clause.confidence_score * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.3 + index * 0.1, ease: "easeOut" }}
                  />
                </div>
              </div>
              {expanded ? (
                <ChevronUp className="h-4 w-4 text-gray-400" />
              ) : (
                <ChevronDown className="h-4 w-4 text-gray-400" />
              )}
            </div>
          </div>
        </CardHeader>
      </button>

      {expanded && (
        <>
          <Separator />
          <CardContent className="pt-4 space-y-4">
            <div className={cn("p-4 rounded-lg border-l-4", getRiskBgColor(riskLevel), getRiskBorderColor(riskLevel).replace("border-", "border-l-"))}>
              <p className="text-sm text-gray-700 italic leading-relaxed">
                &quot;{clause.clause_text}&quot;
              </p>
            </div>

            {clause.impact_summary && (
              <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <AlertTriangle className={cn("h-4 w-4 mt-0.5 flex-shrink-0", getRiskColor(riskLevel))} />
                <div>
                  <p className="text-xs font-medium text-gray-700 mb-0.5">Impact Assessment</p>
                  <p className="text-xs text-gray-600">{clause.impact_summary}</p>
                </div>
              </div>
            )}

            {clause.recommendations && clause.recommendations.length > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-700 mb-2">Recommendations</p>
                <ul className="space-y-1">
                  {clause.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                      <CheckCircle className="h-3 w-3 text-emerald-500 mt-0.5 flex-shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  copyToClipboard(clause.clause_text);
                  toast.success("Copied to clipboard");
                }}
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>
            </div>
          </CardContent>
        </>
      )}
    </Card>
  );
}

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
  const analysisId = parseInt(id, 10);
  const { data: analysis, isLoading, error } = useAnalysis(analysisId);
  const { data: document } = useDocument(analysis?.document_id || 0);

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
        <Card>
          <CardContent className="py-12 text-center">
            <AlertTriangle className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-gray-900 mb-1">Analysis not found</h3>
            <p className="text-sm text-gray-500 mb-4">
              The requested analysis could not be loaded.
            </p>
            <Button asChild variant="outline">
              <Link href="/history">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to History
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const riskLevel: RiskLevel = analysis.clauses.some((c) => c.risk_level === "high")
    ? "high"
    : analysis.clauses.some((c) => c.risk_level === "medium")
    ? "medium"
    : "low";

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center gap-4 mb-6">
        <Button variant="ghost" size="sm" asChild>
          <Link href="/history">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Link>
        </Button>
      </div>

      <div className="mb-8">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">
              {document?.filename || `Analysis #${analysis.id}`}
            </h1>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                {formatDateTime(analysis.analyzed_at)}
              </span>
              <span>{formatProcessingTime(analysis.processing_time_ms)}</span>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
      </div>

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
                <div className={cn("h-10 w-10 rounded-lg flex items-center justify-center", getRiskBgColor(riskLevel))}>
                  <Shield className={cn("h-5 w-5", getRiskColor(riskLevel))} />
                </div>
                <div>
                  <p className="text-xs text-gray-500">Risk Level</p>
                  <p className={cn("text-lg font-semibold capitalize", getRiskColor(riskLevel))}>
                    {riskLevel}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={staggerItem}>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
                  <FileText className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-xs text-gray-500">Clauses Found</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {analysis.clauses.length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={staggerItem}>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-emerald-50 flex items-center justify-center">
                  <CheckCircle className="h-5 w-5 text-emerald-600" />
                </div>
                <div>
                  <p className="text-xs text-gray-500">Confidence</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {formatConfidence(analysis.confidence_score)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      <Tabs defaultValue="clauses">
        <TabsList className="mb-4">
          <TabsTrigger value="clauses">Clauses ({analysis.clauses.length})</TabsTrigger>
          <TabsTrigger value="summary">Summary</TabsTrigger>
        </TabsList>

        <TabsContent value="clauses">
          {analysis.clauses.length > 0 ? (
            <div className="space-y-4">
              {analysis.clauses.map((clause, index) => (
                <motion.div
                  key={clause.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                >
                  <ClauseCard clause={clause} index={index} />
                </motion.div>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <CheckCircle className="h-12 w-12 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-sm font-medium text-gray-900 mb-1">
                  No Arbitration Clauses Found
                </h3>
                <p className="text-sm text-gray-500">
                  This document was analyzed with {formatConfidence(analysis.confidence_score)} confidence
                  and no arbitration-related clauses were detected.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="summary">
          <Card>
            <CardContent className="pt-6">
              <h3 className="font-medium text-gray-900 mb-3">Analysis Summary</h3>
              <p className="text-sm text-gray-600 leading-relaxed whitespace-pre-wrap">
                {analysis.analysis_summary || "No summary available."}
              </p>

              <Separator className="my-6" />

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Analysis Version</p>
                  <p className="font-medium">{analysis.analysis_version}</p>
                </div>
                <div>
                  <p className="text-gray-500">Processing Time</p>
                  <p className="font-medium">{formatProcessingTime(analysis.processing_time_ms)}</p>
                </div>
                <div>
                  <p className="text-gray-500">Document ID</p>
                  <p className="font-medium">{analysis.document_id}</p>
                </div>
                <div>
                  <p className="text-gray-500">Has Arbitration</p>
                  <p className="font-medium">{analysis.has_arbitration_clause ? "Yes" : "No"}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
