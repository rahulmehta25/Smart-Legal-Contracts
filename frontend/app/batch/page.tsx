"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { TextLink } from "@/components/ui/text-link";
import { cn, formatFileSize, validateFile, ALLOWED_FILE_TYPES, formatConfidence } from "@/lib/utils";
import { useUploadDocument, useAnalyzeDocument } from "@/lib/hooks";
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

function statusLabel(status: BatchFile["status"]) {
  switch (status) {
    case "pending":
      return "Pending";
    case "uploading":
      return "Uploading";
    case "analyzing":
      return "Analyzing";
    case "complete":
      return "Complete";
    case "error":
      return "Error";
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
    <div className="page-wrap py-12 lg:py-16">
      <div className="mb-10">
        <p className="eyebrow">Portfolio</p>
        <h1 className="display mt-3 text-3xl sm:text-4xl">Batch analysis</h1>
        <p className="mt-3 max-w-xl text-base leading-relaxed text-ink-muted">
          Upload multiple documents for batch processing and analysis.
        </p>
      </div>

      <div
        {...getRootProps()}
        className={cn(
          "cursor-pointer border-y border-rule py-12 text-center hover-short",
          isDragActive && "bg-linen"
        )}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="font-serif text-2xl font-medium text-ink">Drop your files here</p>
        ) : (
          <>
            <p className="font-serif text-2xl font-medium text-ink">
              Drag files here, or click to browse
            </p>
            <p className="mt-3 text-sm text-ink-muted">
              Supports {ALLOWED_FILE_TYPES.join(", ").toUpperCase()} up to{" "}
              {formatFileSize(env.MAX_FILE_SIZE)} each
            </p>
          </>
        )}
      </div>

      <AnimatePresence mode="popLayout">
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="mt-10 grid grid-cols-2 gap-8 border-y border-rule py-8 sm:grid-cols-4 sm:gap-0">
              <div className="sm:pr-8">
                <p className="eyebrow">Total files</p>
                <p className="mt-2 font-serif text-3xl font-medium text-ink">{files.length}</p>
              </div>
              <div className="sm:border-l sm:border-rule sm:px-8">
                <p className="eyebrow">Completed</p>
                <p className="mt-2 font-serif text-3xl font-medium text-ink">{completeCount}</p>
              </div>
              <div className="sm:border-l sm:border-rule sm:px-8">
                <p className="eyebrow">High risk</p>
                <p className="mt-2 font-serif text-3xl font-medium text-ink">{highRiskCount}</p>
              </div>
              <div className="sm:border-l sm:border-rule sm:pl-8">
                <p className="eyebrow">Errors</p>
                <p className="mt-2 font-serif text-3xl font-medium text-ink">{errorCount}</p>
              </div>
            </div>

            {isProcessing && (
              <div className="border-b border-rule py-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-ink">Processing</span>
                  <span className="text-sm text-ink-muted">
                    {completeCount + errorCount} / {files.length}
                  </span>
                </div>
                <Progress value={overallProgress} className="h-1" />
              </div>
            )}

            <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="eyebrow">Queue</p>
                <h2 className="mt-2 font-serif text-2xl font-medium text-ink">
                  {files.length} files queued
                </h2>
              </div>
              <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
                <Button variant="link" onClick={clearAll} disabled={isProcessing}>
                  Clear all
                </Button>
                {pendingCount > 0 && (
                  <Button onClick={processAllFiles} disabled={isProcessing}>
                    {isProcessing ? "Processing" : "Start processing"}
                  </Button>
                )}
              </div>
            </div>

            <ul className="mt-4">
              {files.map((batchFile, index) => (
                <li
                  key={`${batchFile.file.name}-${index}`}
                  className="border-t border-rule py-5 last:border-b"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <p className="filename-display font-serif text-lg font-medium text-ink">
                        {batchFile.file.name}
                      </p>
                      <p className="mt-1 text-sm text-ink-muted">
                        {formatFileSize(batchFile.file.size)} · {statusLabel(batchFile.status)}
                        {batchFile.clauseCount !== undefined ? ` · ${batchFile.clauseCount} clauses` : ""}
                        {batchFile.confidence !== undefined ? ` · ${formatConfidence(batchFile.confidence)}` : ""}
                      </p>
                      {batchFile.error && (
                        <p className="mt-1 text-sm text-red-800">{batchFile.error}</p>
                      )}
                    </div>
                    <div className="flex flex-wrap items-center gap-4">
                      {batchFile.riskLevel && (
                        <Badge variant={getRiskBadgeVariant(batchFile.riskLevel)}>
                          {batchFile.riskLevel}
                        </Badge>
                      )}
                      {batchFile.status === "complete" && batchFile.analysisId && (
                        <TextLink href={`/analysis/${batchFile.analysisId}`}>View</TextLink>
                      )}
                      {(batchFile.status === "pending" || batchFile.status === "error") && (
                        <button
                          type="button"
                          className="text-sm text-ink-muted underline decoration-rule underline-offset-4 hover:text-ink"
                          onClick={() => removeFile(index)}
                          disabled={isProcessing}
                        >
                          Remove
                        </button>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>

      {files.length === 0 && (
        <div className="border-b border-rule py-12">
          <p className="font-serif text-2xl font-medium text-ink">No files queued</p>
          <p className="mt-3 max-w-lg text-sm leading-relaxed text-ink-muted">
            Add several contracts above to process them as a portfolio.
          </p>
        </div>
      )}
    </div>
  );
}
