"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ArrowLeft,
  CheckCircle,
  Clock,
  Download,
  FileText,
  Shield,
} from "lucide-react";
import {
  cn,
  computeRiskLevel,
  copyToClipboard,
  formatConfidence,
  formatDateTime,
  formatProcessingTime,
  getRiskBgColor,
  getRiskColor,
} from "@/lib/utils";
import { staggerContainer, staggerItem } from "@/components/ui/motion";
import { ClauseCard } from "@/components/analysis/clause-card";
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
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center gap-4 mb-6">
        <Button variant="ghost" size="sm" asChild>
          <Link href={backHref}>
            <ArrowLeft className="h-4 w-4 mr-1" />
            {backLabel}
          </Link>
        </Button>
      </div>

      {isSample && (
        <div className="mb-6 rounded-lg border border-indigo-100 bg-indigo-50/70 px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">Sample analysis</Badge>
            <p className="text-sm text-indigo-900">
              This walkthrough uses a canned SaaS MSA. It does not call the upload API.
            </p>
          </div>
        </div>
      )}

      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">
              {document?.filename || `Analysis #${analysis.id}`}
            </h1>
            <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500">
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                {formatDateTime(analysis.analyzed_at)}
              </span>
              <span>{formatProcessingTime(analysis.processing_time_ms)}</span>
              {document?.page_count ? <span>{document.page_count} pages</span> : null}
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={handleExport}>
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
                <div
                  className={cn(
                    "h-10 w-10 rounded-lg flex items-center justify-center",
                    getRiskBgColor(riskLevel)
                  )}
                >
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
                  <p className="text-lg font-semibold text-gray-900">{analysis.clauses.length}</p>
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
                  This document was analyzed with {formatConfidence(analysis.confidence_score)}{" "}
                  confidence and no arbitration-related clauses were detected.
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
                  <p className="font-medium">{isSample ? "sample" : analysis.document_id}</p>
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
