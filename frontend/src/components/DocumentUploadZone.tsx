import React, { useCallback, useState, DragEvent, ChangeEvent } from 'react';
import { Upload, FileText, X, CheckCircle, AlertTriangle, FileIcon, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface UploadedFile {
  id: string;
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
}

interface DocumentUploadZoneProps {
  onFileUpload?: (files: File[]) => void;
  onFileRemove?: (fileId: string) => void;
  acceptedTypes?: string[];
  maxFileSize?: number; // in bytes
  maxFiles?: number;
  className?: string;
}

export const DocumentUploadZone: React.FC<DocumentUploadZoneProps> = ({
  onFileUpload,
  onFileRemove,
  acceptedTypes = ['.pdf', '.doc', '.docx', '.txt'],
  maxFileSize = 10 * 1024 * 1024, // 10MB
  maxFiles = 5,
  className = ''
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);

  const processFiles = useCallback((files: File[]) => {
    const validFiles: File[] = [];
    
    // Validate files
    for (const file of files) {
      if (uploadedFiles.length + validFiles.length >= maxFiles) {
        break;
      }

      // Check file size
      if (file.size > maxFileSize) {
        console.error(`File ${file.name} exceeds maximum size`);
        continue;
      }

      // Check file type
      const extension = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!acceptedTypes.includes(extension)) {
        console.error(`File ${file.name} has unsupported type`);
        continue;
      }

      validFiles.push(file);
    }

    // Add valid files to upload queue
    validFiles.forEach((file) => {
      const fileId = `${Date.now()}-${Math.random()}`;
      const uploadFile: UploadedFile = {
        id: fileId,
        file,
        progress: 0,
        status: 'uploading'
      };

      setUploadedFiles(prev => [...prev, uploadFile]);

      // Simulate upload progress
      simulateUpload(fileId);
    });

    if (onFileUpload && validFiles.length > 0) {
      onFileUpload(validFiles);
    }
  }, [onFileUpload, acceptedTypes, maxFileSize, maxFiles, uploadedFiles.length]);

  const simulateUpload = async (fileId: string) => {
    // Simulate upload progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setUploadedFiles(prev => 
        prev.map(file => 
          file.id === fileId 
            ? { ...file, progress } 
            : file
        )
      );
    }

    // Mark as completed
    setUploadedFiles(prev => 
      prev.map(file => 
        file.id === fileId 
          ? { ...file, status: 'completed' } 
          : file
      )
    );
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
    if (onFileRemove) {
      onFileRemove(fileId);
    }
  };

  // Native drag and drop handlers
  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, [processFiles]);

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      processFiles(files);
    }
  }, [processFiles]);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return <FileText className="w-8 h-8 text-red-500" />;
      case 'doc':
      case 'docx':
        return <FileIcon className="w-8 h-8 text-blue-500" />;
      case 'txt':
        return <FileText className="w-8 h-8 text-gray-500" />;
      default:
        return <FileIcon className="w-8 h-8 text-gray-500" />;
    }
  };

  return (
    <div id="document-upload-zone" className={`space-y-6 ${className}`}>
      {/* Main Drop Zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
          transition-all duration-300 hover:bg-muted/50
          ${isDragActive 
            ? 'border-primary bg-primary/5 scale-[1.02]' 
            : 'border-border hover:border-primary/50'
          }
        `}
      >
        <input
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          id="file-upload-input"
        />
        
        {/* Upload Icon with Animation */}
        <div className={`mb-6 transition-transform duration-300 ${isDragActive ? 'scale-110' : ''}`}>
          <Upload className={`w-16 h-16 mx-auto ${isDragActive ? 'text-primary' : 'text-muted-foreground'}`} />
        </div>

        <div className="space-y-4">
          <h3 className="text-2xl font-semibold">
            {isDragActive ? 'Drop files here' : 'Upload Legal Documents'}
          </h3>
          
          <p className="text-muted-foreground max-w-sm mx-auto">
            Drag and drop your legal documents here, or click to browse
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-6">
            <Button variant="default" size="lg" className="btn-primary">
              <Upload className="w-5 h-5 mr-2" />
              Choose Files
            </Button>
            <span className="text-sm text-muted-foreground">
              or drag and drop
            </span>
          </div>

          {/* File Requirements */}
          <div className="text-xs text-muted-foreground mt-4 space-y-1">
            <p>Supported formats: {acceptedTypes.join(', ')}</p>
            <p>Maximum file size: {formatFileSize(maxFileSize)}</p>
            <p>Maximum files: {maxFiles}</p>
          </div>
        </div>

        {/* Drag Overlay */}
        {isDragActive && (
          <div className="absolute inset-0 bg-primary/10 rounded-2xl flex items-center justify-center backdrop-blur-sm">
            <div className="text-center">
              <CheckCircle className="w-12 h-12 text-primary mx-auto mb-2" />
              <p className="text-lg font-medium text-primary">Drop to upload</p>
            </div>
          </div>
        )}
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div id="uploaded-files-list" className="space-y-4">
          <h4 className="text-lg font-medium">Uploaded Files</h4>
          
          <div className="space-y-3">
            {uploadedFiles.map((uploadFile) => (
              <div
                key={uploadFile.id}
                className="professional-card p-4 flex items-center space-x-4 animate-slide-up"
              >
                {/* File Icon */}
                <div className="flex-shrink-0">
                  {getFileIcon(uploadFile.file.name)}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{uploadFile.file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(uploadFile.file.size)}
                  </p>
                  
                  {/* Progress Bar */}
                  {uploadFile.status === 'uploading' && (
                    <div className="mt-2">
                      <Progress value={uploadFile.progress} className="h-2" />
                      <p className="text-xs text-muted-foreground mt-1">
                        {uploadFile.progress}% uploaded
                      </p>
                    </div>
                  )}
                </div>

                {/* Status Icon */}
                <div className="flex-shrink-0">
                  {uploadFile.status === 'uploading' && (
                    <Loader2 className="w-5 h-5 animate-spin text-primary" />
                  )}
                  {uploadFile.status === 'completed' && (
                    <CheckCircle className="w-5 h-5 text-success" />
                  )}
                  {uploadFile.status === 'error' && (
                    <AlertTriangle className="w-5 h-5 text-destructive" />
                  )}
                </div>

                {/* Remove Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFile(uploadFile.id)}
                  className="flex-shrink-0 h-8 w-8 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Warnings/Errors */}
      {uploadedFiles.some(f => f.status === 'error') && (
        <Alert id="upload-errors" variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Some files failed to upload. Please try again or check the file format and size.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default DocumentUploadZone;