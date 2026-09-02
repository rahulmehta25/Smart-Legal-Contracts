"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Upload,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  File,
  ArrowRight,
} from "lucide-react";
import { cn, formatFileSize, validateFile, ALLOWED_FILE_TYPES } from "@/lib/utils";
import { useUploadDocument, useAnalyzeDocument } from "@/lib/hooks";
import { SampleDemoCard } from "@/components/demo/sample-demo-card";
import env from "@/lib/env";

interface UploadedFile {
  file: File;
  status: "pending" | "uploading" | "analyzing" | "complete" | "error";
  progress: number;
  documentId?: number;
  analysisId?: number;
  error?: string;
}

export default function UploadPage() {
  const router = useRouter();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const uploadDocument = useUploadDocument();
  const analyzeDocument = useAnalyzeDocument();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => {
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

  const processFile = async (fileIndex: number) => {
    const uploadedFile = files[fileIndex];
    if (!uploadedFile || uploadedFile.status !== "pending") return;

    setFiles((prev) =>
      prev.map((f, i) =>
        i === fileIndex ? { ...f, status: "uploading" as const, progress: 0 } : f
      )
    );

    try {
      const result = await uploadDocument.mutateAsync({
        file: uploadedFile.file,
        onProgress: (progress) => {
          setFiles((prev) =>
            prev.map((f, i) => (i === fileIndex ? { ...f, progress } : f))
          );
        },
      });

      setFiles((prev) =>
        prev.map((f, i) =>
          i === fileIndex
            ? { ...f, status: "analyzing" as const, documentId: result.document_id, progress: 100 }
            : f
        )
      );

      const analysis = await analyzeDocument.mutateAsync({
        document_id: result.document_id,
      });

      setFiles((prev) =>
        prev.map((f, i) =>
          i === fileIndex
            ? { ...f, status: "complete" as const, analysisId: analysis.id }
            : f
        )
      );

      toast.success(`Analysis complete for ${uploadedFile.file.name}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      setFiles((prev) =>
        prev.map((f, i) =>
          i === fileIndex
            ? {
                ...f,
                status: "error" as const,
                error: `${message} You can still open the sample analysis.`,
              }
            : f
        )
      );
      toast.error(message);
    }
  };

  const processAllFiles = async () => {
    const pendingFiles = files
      .map((f, i) => ({ file: f, index: i }))
      .filter(({ file }) => file.status === "pending");

    for (const { index } of pendingFiles) {
      await processFile(index);
    }
  };

  const pendingCount = files.filter((f) => f.status === "pending").length;
  const completeCount = files.filter((f) => f.status === "complete").length;

  return (
    <div className="page-wrap py-12 lg:py-16">
      <div className="mb-10">
        <p className="eyebrow">Review</p>
        <h1 className="display mt-3 text-3xl sm:text-4xl">Upload documents</h1>
        <p className="mt-3 max-w-xl text-base leading-relaxed text-ink-muted">
          Analyze a contract for arbitration clauses. If live upload fails, the sample analysis
          still works.
        </p>
      </div>

      <SampleDemoCard className="mb-10" />

      <div
        {...getRootProps()}
        className={cn(
          "mb-10 cursor-pointer border border-dashed p-10 text-center hover-short",
          isDragActive ? "border-brass bg-linen" : "border-rule hover:border-ink/40 dropzone-idle"
        )}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="font-serif text-xl text-ink">Drop your files here</p>
        ) : (
          <>
            <p className="font-serif text-xl text-ink">Drag and drop files, or click to browse</p>
            <p className="mt-2 text-sm text-ink-muted">
              Supports {ALLOWED_FILE_TYPES.join(", ").toUpperCase()} up to{" "}
              {formatFileSize(env.MAX_FILE_SIZE)}
            </p>
          </>
        )}
      </div>

      <AnimatePresence mode="popLayout">
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Files ({files.length})</CardTitle>
                    <CardDescription>
                      {completeCount} analyzed, {pendingCount} pending
                    </CardDescription>
                  </div>
                  {pendingCount > 0 && (
                    <Button onClick={processAllFiles} disabled={uploadDocument.isPending || analyzeDocument.isPending}>
                      {uploadDocument.isPending || analyzeDocument.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Upload className="mr-2 h-4 w-4" />
                      )}
                      Analyze All
                    </Button>
                  )}
                </div>
              </CardHeader>
              <Separator />
              <CardContent className="pt-4">
                <div className="space-y-3">
                  <AnimatePresence mode="popLayout">
                    {files.map((uploadedFile, index) => (
                      <motion.div
                        key={`${uploadedFile.file.name}-${uploadedFile.file.size}-${uploadedFile.file.lastModified}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                        layout
                        className="flex items-center gap-4 p-3 rounded-lg border border-gray-100 bg-gray-50/50"
                      >
                        <div className="flex-shrink-0">
                          <File className="h-8 w-8 text-gray-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-sm text-gray-900 truncate">
                              {uploadedFile.file.name}
                            </span>
                            <span className="text-xs text-gray-500">
                              {formatFileSize(uploadedFile.file.size)}
                            </span>
                          </div>
                          {uploadedFile.status === "uploading" && (
                            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                              <motion.div
                                className="h-full bg-brass"
                                initial={{ width: 0 }}
                                animate={{ width: `${uploadedFile.progress}%` }}
                                transition={{ type: "spring", stiffness: 100, damping: 20 }}
                              />
                            </div>
                          )}
                          {uploadedFile.status === "analyzing" && (
                            <div className="flex items-center gap-2 text-xs text-ink-muted">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Analyzing document...
                            </div>
                          )}
                          {uploadedFile.status === "error" && (
                            <p className="text-xs text-red-600">{uploadedFile.error}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {uploadedFile.status === "pending" && (
                            <Badge variant="secondary">Pending</Badge>
                          )}
                          {uploadedFile.status === "uploading" && (
                            <Badge variant="secondary">
                              <Loader2 className="h-3 w-3 animate-spin mr-1" />
                              Uploading
                            </Badge>
                          )}
                          {uploadedFile.status === "analyzing" && (
                            <Badge variant="secondary">
                              <Loader2 className="h-3 w-3 animate-spin mr-1" />
                              Analyzing
                            </Badge>
                          )}
                          {uploadedFile.status === "complete" && (
                            <>
                              <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.3, type: "spring" }}
                              >
                                <Badge variant="success">
                                  <CheckCircle className="h-3 w-3 mr-1" />
                                  Complete
                                </Badge>
                              </motion.div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => router.push(`/analysis/${uploadedFile.analysisId}`)}
                              >
                                View
                                <ArrowRight className="h-3 w-3 ml-1" />
                              </Button>
                            </>
                          )}
                          {uploadedFile.status === "error" && (
                            <Badge variant="danger">
                              <AlertCircle className="h-3 w-3 mr-1" />
                              Error
                            </Badge>
                          )}
                          {(uploadedFile.status === "pending" || uploadedFile.status === "error") && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => removeFile(index)}
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {files.length === 0 && (
        <div className="border-t border-rule py-12">
          <h3 className="font-serif text-xl font-medium text-ink">No files uploaded</h3>
          <p className="mt-2 text-sm text-ink-muted">
            Upload a document when the API is available, or open the sample analysis above.
          </p>
        </div>
      )}
    </div>
  );
}
