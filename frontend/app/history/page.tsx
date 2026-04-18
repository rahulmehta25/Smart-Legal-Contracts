"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { useAnalyses, useDocuments } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
import {
  FileText,
  Search,
  ArrowRight,
  AlertTriangle,
  Upload,
} from "lucide-react";
import { cn, formatConfidence, formatRelativeTime } from "@/lib/utils";
import type { RiskLevel, ArbitrationAnalysis } from "@/types/api";

function HistorySkeleton() {
  return (
    <div className="space-y-4">
      {[...Array(5)].map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  );
}

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

function computeRiskLevel(analysis: ArbitrationAnalysis): RiskLevel {
  if (!analysis.clauses || analysis.clauses.length === 0) return "low";
  if (analysis.clauses.some((c) => c.risk_level === "high")) return "high";
  if (analysis.clauses.some((c) => c.risk_level === "medium")) return "medium";
  return "low";
}

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("all");
  const [arbitrationFilter, setArbitrationFilter] = useState<string>("all");

  const { data: analyses, isLoading: analysesLoading } = useAnalyses({ limit: 100 });
  const { data: documents, isLoading: documentsLoading } = useDocuments({ limit: 100 });

  const isLoading = analysesLoading || documentsLoading;

  const documentMap = useMemo(() => {
    if (!documents) return new Map();
    return new Map(documents.map((doc) => [doc.id, doc]));
  }, [documents]);

  const filteredAnalyses = useMemo(() => {
    if (!analyses) return [];

    return analyses.filter((analysis) => {
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
  }, [analyses, documentMap, searchQuery, riskFilter, arbitrationFilter]);

  const stats = useMemo(() => {
    if (!analyses) return { total: 0, withArbitration: 0, highRisk: 0 };
    return {
      total: analyses.length,
      withArbitration: analyses.filter((a) => a.has_arbitration_clause).length,
      highRisk: analyses.filter((a) => computeRiskLevel(a) === "high").length,
    };
  }, [analyses]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">Analysis History</h1>
          <p className="text-gray-600">View and manage your document analyses.</p>
        </div>
        <Button asChild>
          <Link href="/upload">
            <Upload className="h-4 w-4 mr-2" />
            New Analysis
          </Link>
        </Button>
      </div>

      <div className="grid md:grid-cols-3 gap-4 mb-8">
        <div className="animate-fade-in-up hover-lift stagger-1">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Analyses</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats.total}</p>
                </div>
                <div className="h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
                  <FileText className="h-5 w-5 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="animate-fade-in-up hover-lift stagger-2">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">With Arbitration</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats.withArbitration}</p>
                </div>
                <div className="h-10 w-10 rounded-lg bg-amber-50 flex items-center justify-center">
                  <AlertTriangle className="h-5 w-5 text-amber-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="animate-fade-in-up hover-lift stagger-3">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">High Risk</p>
                  <p className="text-2xl font-semibold text-gray-900">{stats.highRisk}</p>
                </div>
                <div className="h-10 w-10 rounded-lg bg-red-50 flex items-center justify-center">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <div className="p-6 pb-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="flex gap-2">
              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger className="w-[140px]">
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
                <SelectTrigger className="w-[160px]">
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
        <CardContent className="pt-0">
          {isLoading ? (
            <HistorySkeleton />
          ) : filteredAnalyses.length === 0 ? (
            <div className="py-12 text-center">
              <FileText className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-sm font-medium text-gray-900 mb-1">No analyses found</h3>
              <p className="text-sm text-gray-500 mb-4">
                {searchQuery || riskFilter !== "all" || arbitrationFilter !== "all"
                  ? "Try adjusting your filters."
                  : "Upload a document to get started."}
              </p>
              {!searchQuery && riskFilter === "all" && arbitrationFilter === "all" && (
                <Button asChild variant="outline">
                  <Link href="/upload">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Document
                  </Link>
                </Button>
              )}
            </div>
          ) : (
            <div className="border rounded-lg overflow-hidden">
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

                    return (
                      <TableRow
                        key={analysis.id}
                        className="transition-colors duration-150"
                        style={{
                          animation: `row-fade-in 0.3s ease-out ${index * 50}ms backwards`,
                        }}
                      >
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <div className="h-8 w-8 rounded bg-gray-100 flex items-center justify-center">
                              <FileText className="h-4 w-4 text-gray-400" />
                            </div>
                            <div className="min-w-0">
                              <p className="font-medium text-sm text-gray-900 truncate max-w-[200px]">
                                {document?.filename || `Document #${analysis.document_id}`}
                              </p>
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant={getRiskBadgeVariant(riskLevel)}>
                            {riskLevel}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-600">
                            {analysis.clauses?.length || 0}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-600">
                            {formatConfidence(analysis.confidence_score)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-500">
                            {formatRelativeTime(analysis.analyzed_at)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Button variant="ghost" size="sm" asChild>
                            <Link href={`/analysis/${analysis.id}`}>
                              View
                              <ArrowRight className="h-3 w-3 ml-1" />
                            </Link>
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
