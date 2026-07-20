import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';

const DocumentUploader = ({ onFileUpload, isProcessing }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, success, error

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setUploadStatus('error');
      return;
    }

    setUploadStatus('uploading');
    const file = acceptedFiles[0];
    
    // Simulate file processing
    setTimeout(() => {
      setUploadedFiles([file]);
      setUploadStatus('success');
      onFileUpload(file);
    }, 1000);
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const removeFile = () => {
    setUploadedFiles([]);
    setUploadStatus('idle');
  };

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
        return <div id="upload-spinner" className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>;
      case 'success':
        return <CheckCircle id="upload-success-icon" className="h-5 w-5 text-success-600" />;
      case 'error':
        return <AlertCircle id="upload-error-icon" className="h-5 w-5 text-danger-600" />;
      default:
        return null;
    }
  };

  const getStatusText = () => {
    switch (uploadStatus) {
      case 'uploading':
        return 'Uploading...';
      case 'success':
        return 'Upload successful';
      case 'error':
        return 'Upload failed';
      default:
        return '';
    }
  };

  return (
    <div id="document-uploader-container" className="w-full max-w-2xl mx-auto">
      <div id="upload-header" className="mb-4">
        <h2 id="uploader-title" className="text-2xl font-bold text-gray-900 mb-2">
          Document Upload
        </h2>
        <p id="uploader-description" className="text-gray-600">
          Upload your legal document to detect arbitration clauses. Supports PDF, DOCX, and TXT files.
        </p>
      </div>

      {uploadedFiles.length === 0 ? (
        <div
          {...getRootProps()}
          id="dropzone-area"
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-primary-500 bg-primary-50' 
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
            }
            ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} id="file-input" disabled={isProcessing} />
          <div id="upload-icon-container" className="flex justify-center mb-4">
            <Upload id="upload-icon" className="h-12 w-12 text-gray-400" />
          </div>
          <div id="upload-text-container">
            <p id="drag-drop-text" className="text-lg font-medium text-gray-900 mb-2">
              {isDragActive ? 'Drop your document here' : 'Drag & drop your document here'}
            </p>
            <p id="browse-text" className="text-gray-600 mb-4">
              or <span className="text-primary-600 font-medium">browse</span> to choose a file
            </p>
            <p id="file-formats-text" className="text-sm text-gray-500">
              Supports PDF, DOCX, DOC, and TXT files (max 10MB)
            </p>
          </div>
        </div>
      ) : (
        <div id="uploaded-file-container" className="border border-gray-200 rounded-lg p-4 bg-white">
          <div id="file-item" className="flex items-center justify-between">
            <div id="file-info" className="flex items-center space-x-3">
              <File id="file-icon" className="h-8 w-8 text-primary-600" />
              <div id="file-details">
                <p id="file-name" className="font-medium text-gray-900">
                  {uploadedFiles[0].name}
                </p>
                <p id="file-size" className="text-sm text-gray-500">
                  {(uploadedFiles[0].size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <div id="file-actions" className="flex items-center space-x-2">
              {getStatusIcon()}
              <span id="upload-status-text" className="text-sm text-gray-600">
                {getStatusText()}
              </span>
              <button
                id="remove-file-button"
                onClick={removeFile}
                className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                disabled={isProcessing}
              >
                <X id="remove-file-icon" className="h-4 w-4 text-gray-500" />
              </button>
            </div>
          </div>
        </div>
      )}

      {uploadStatus === 'error' && (
        <div id="error-message-container" className="mt-4 p-4 bg-danger-50 border border-danger-200 rounded-lg">
          <div id="error-message" className="flex items-center space-x-2">
            <AlertCircle id="error-icon" className="h-5 w-5 text-danger-600" />
            <span id="error-text" className="text-danger-700">
              Failed to upload file. Please check the file format and size.
            </span>
          </div>
        </div>
      )}

      {isProcessing && uploadedFiles.length > 0 && (
        <div id="processing-indicator" className="mt-4 p-4 bg-primary-50 border border-primary-200 rounded-lg">
          <div id="processing-content" className="flex items-center space-x-2">
            <div id="processing-spinner" className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
            <span id="processing-text" className="text-primary-700">
              Analyzing document for arbitration clauses...
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentUploader;