"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Link from "next/link";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Upload,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  File,
  BarChart3,
  Play,
} from "lucide-react";
import { cn, formatFileSize, validateFile, ALLOWED_FILE_TYPES, formatConfidence } from "@/lib/utils";
import { useUploadDocument, useAnalyzeDocument } from "@/lib/hooks";
import { staggerContainer, staggerItem } from "@/components/ui/motion";
import env from "@/lib/env";
import type { RiskLevel } from "@/types/api";

interface BatchFile {
  file: File;
  status: "pending" | "uploading" | "analyzing" | "complete" | "error";
  progress: number;
  documentId?: number;
  analysisId?: number;
  error?: string;
  riskLevel?: RiskLevel;
  clauseCount?: number;
  confidence?: number;
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

export default function BatchPage() {
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const uploadDocument = useUploadDocument();
  const analyzeDocument = useAnalyzeDocument();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: BatchFile[] = acceptedFiles.map((file) => {
      const validation = validateFile(file, env.MAX_FILE_SIZE);
      if (!validation.valid) {
        return {
          file,
          status: "error" as const,
          progress: 0,
          error: validation.error,
        };
      }
      return {
        file,
        status: "pending" as const,
        progress: 0,
      };
    });

    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "application/msword": [".doc"],
      "text/plain": [".txt"],
    },
    maxSize: env.MAX_FILE_SIZE,
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFiles([]);
  };

  const processAllFiles = async () => {
    setIsProcessing(true);

    const pendingIndices = files
      .map((f, i) => ({ file: f, index: i }))
      .filter(({ file }) => file.status === "pending")
      .map(({ index }) => index);

    for (const index of pendingIndices) {
      const uploadedFile = files[index];
      if (!uploadedFile) continue;

      setFiles((prev) =>
        prev.map((f, i) =>
          i === index ? { ...f, status: "uploading" as const, progress: 0 } : f
        )
      );

      try {
        const result = await uploadDocument.mutateAsync({
          file: uploadedFile.file,
          onProgress: (progress) => {
            setFiles((prev) =>
              prev.map((f, i) => (i === index ? { ...f, progress } : f))
            );
          },
        });

        setFiles((prev) =>
          prev.map((f, i) =>
            i === index
              ? { ...f, status: "analyzing" as const, documentId: result.document_id, progress: 100 }
              : f
          )
        );

        const analysis = await analyzeDocument.mutateAsync({
          document_id: result.document_id,
        });

        const riskLevel: RiskLevel = analysis.clauses?.some((c) => c.risk_level === "high")
          ? "high"
          : analysis.clauses?.some((c) => c.risk_level === "medium")
          ? "medium"
          : "low";

        setFiles((prev) =>
          prev.map((f, i) =>
            i === index
              ? {
                  ...f,
                  status: "complete" as const,
                  analysisId: analysis.id,
                  riskLevel,
                  clauseCount: analysis.clauses?.length || 0,
                  confidence: analysis.confidence_score,
                }
              : f
          )
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : "Processing failed";
        setFiles((prev) =>
          prev.map((f, i) =>
            i === index ? { ...f, status: "error" as const, error: message } : f
          )
        );
      }
    }

    setIsProcessing(false);
    toast.success("Batch processing complete");
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;
  const completeCount = files.filter((f) => f.status === "complete").length;
  const errorCount = files.filter((f) => f.status === "error").length;
  const highRiskCount = files.filter((f) => f.riskLevel === "high").length;

  const overallProgress = files.length > 0
    ? ((completeCount + errorCount) / files.length) * 100
    : 0;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Batch Analysis</h1>
        <p className="text-gray-600">
          Upload multiple documents for batch processing and analysis.
        </p>
      </div>

      <Card className="mb-6">
        <CardContent className="pt-6">
          <motion.div
            animate={isDragActive ? { scale: 1.02 } : { scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
                isDragActive
                  ? "border-blue-400 bg-blue-50"
                  : "border-gray-200 hover:border-gray-300 hover:bg-gray-50 dropzone-idle"
              )}
            >
              <input {...getInputProps()} />
              <Upload
                className={cn(
                  "h-10 w-10 mx-auto mb-4 transition-colors",
                  isDragActive ? "text-blue-500" : "text-gray-400"
                )}
              />
              {isDragActive ? (
                <p className="text-blue-600 font-medium">Drop your files here</p>
              ) : (
                <>
                  <p className="text-gray-700 font-medium mb-1">
                    Drag and drop multiple files, or click to browse
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports {ALLOWED_FILE_TYPES.join(", ").toUpperCase()} up to{" "}
                    {formatFileSize(env.MAX_FILE_SIZE)} each
                  </p>
                </>
              )}
            </div>
          </motion.div>
        </CardContent>
      </Card>

      <AnimatePresence mode="popLayout">
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <motion.div
              className="grid md:grid-cols-4 gap-4 mb-6"
              variants={staggerContainer}
              initial="hidden"
              animate="show"
            >
              <motion.div variants={staggerItem}>
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-gray-500">Total Files</p>
                    <p className="text-2xl font-semibold text-gray-900">{files.length}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={staggerItem}>
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-gray-500">Completed</p>
                    <p className="text-2xl font-semibold text-emerald-600">{completeCount}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={staggerItem}>
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-gray-500">High Risk</p>
                    <p className="text-2xl font-semibold text-red-600">{highRiskCount}</p>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={staggerItem}>
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-gray-500">Errors</p>
                    <p className="text-2xl font-semibold text-gray-600">{errorCount}</p>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {isProcessing && (
              <Card className="mb-6">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">Processing...</span>
                    <span className="text-sm text-gray-500">
                      {completeCount + errorCount} / {files.length}
                    </span>
                  </div>
                  <Progress value={overallProgress} className="h-2" />
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Documents</CardTitle>
                    <CardDescription>{files.length} files queued</CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={clearAll} disabled={isProcessing}>
                      Clear All
                    </Button>
                    {pendingCount > 0 && (
                      <Button onClick={processAllFiles} disabled={isProcessing}>
                        {isProcessing ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Play className="mr-2 h-4 w-4" />
                        )}
                        Start Processing
                      </Button>
                    )}
                  </div>
                </div>
              </CardHeader>
              <Separator />
              <CardContent className="pt-0">
                <div className="border rounded-lg overflow-hidden mt-4">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>File</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Risk</TableHead>
                        <TableHead>Clauses</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead className="w-[100px]"></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {files.map((batchFile, index) => (
                        <TableRow
                          key={`${batchFile.file.name}-${index}`}
                          style={{
                            animation: `row-fade-in 0.3s ease-out ${index * 50}ms backwards`,
                          }}
                        >
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <File className="h-5 w-5 text-gray-400" />
                              <div className="min-w-0">
                                <p className="font-medium text-sm text-gray-900 truncate max-w-[200px]">
                                  {batchFile.file.name}
                                </p>
                                <p className="text-xs text-gray-500">
                                  {formatFileSize(batchFile.file.size)}
                                </p>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            {batchFile.status === "pending" && (
                              <Badge variant="secondary">Pending</Badge>
                            )}
                            {batchFile.status === "uploading" && (
                              <Badge variant="secondary">
                                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                                Uploading
                              </Badge>
                            )}
                            {batchFile.status === "analyzing" && (
                              <Badge variant="secondary">
                                <Loader2 className="h-3 w-3 animate-spin mr-1" />
                                Analyzing
                              </Badge>
                            )}
                            {batchFile.status === "complete" && (
                              <Badge variant="success">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                Complete
                              </Badge>
                            )}
                            {batchFile.status === "error" && (
                              <Badge variant="danger">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                Error
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            {batchFile.riskLevel && (
                              <Badge variant={getRiskBadgeVariant(batchFile.riskLevel)}>
                                {batchFile.riskLevel}
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            {batchFile.clauseCount !== undefined && (
                              <span className="text-sm">{batchFile.clauseCount}</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {batchFile.confidence !== undefined && (
                              <span className="text-sm">{formatConfidence(batchFile.confidence)}</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              {batchFile.status === "complete" && batchFile.analysisId && (
                                <Button variant="ghost" size="sm" asChild>
                                  <Link href={`/analysis/${batchFile.analysisId}`}>
                                    View
                                  </Link>
                                </Button>
                              )}
                              {(batchFile.status === "pending" || batchFile.status === "error") && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() => removeFile(index)}
                                  disabled={isProcessing}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {files.length === 0 && (
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <BarChart3 className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-sm font-medium text-gray-900 mb-1">No files queued</h3>
              <p className="text-sm text-gray-500">
                Upload multiple documents to process them in batch.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
