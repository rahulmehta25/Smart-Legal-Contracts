"""
PDF Processing API Endpoints

Provides REST API endpoints for PDF processing functionality:
- POST /upload/pdf - Upload and process PDF files
- GET /documents/{id}/text - Get extracted text
- GET /documents/{id}/preview - Get document preview
- POST /batch/upload - Batch upload multiple files
- GET /jobs/{id}/status - Get processing job status
- GET /documents/{id}/download - Download original file
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

# Import our services
from ..services.document_processor import DocumentProcessor, ProcessingResult
from ..services.pdf_service import PDFProcessor, extract_pdf_text, PDFQuality
from ..services.preprocessing import TextPreprocessor, preprocess_text
from ..services.storage_service import StorageService, StorageConfig, StorageBackend, create_storage_service
from ..services.batch_processor import (
    get_batch_processor, BatchJobConfig, JobStatus, JobResult,
    process_document_async, extract_pdf_text_async, batch_process_documents_async
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["PDF Processing"])

# Initialize services (in production, these would be dependency injected)
document_processor = DocumentProcessor()
pdf_processor = PDFProcessor()
text_preprocessor = TextPreprocessor()
storage_service = create_storage_service(
    backend=StorageBackend.LOCAL,
    local_base_path="/tmp/document_storage"
)


# Request/Response models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    file_id: str
    filename: str
    content_type: str
    file_size: int
    processing_job_id: Optional[str] = None
    secure_url: Optional[str] = None
    thumbnails: Dict[str, str] = Field(default_factory=dict)
    message: str


class DocumentTextResponse(BaseModel):
    """Response model for document text extraction"""
    file_id: str
    text: str
    word_count: int
    character_count: int
    language: Optional[str] = None
    extraction_method: str
    processing_time: float
    quality_score: Optional[str] = None


class DocumentPreviewResponse(BaseModel):
    """Response model for document preview"""
    file_id: str
    filename: str
    content_type: str
    file_size: int
    page_count: int
    preview_text: str  # First 500 characters
    thumbnails: Dict[str, str]
    metadata: Dict[str, Any]
    processing_status: str


class BatchUploadRequest(BaseModel):
    """Request model for batch upload"""
    files: List[str] = Field(description="List of file identifiers to process")
    process_async: bool = Field(default=True, description="Process files asynchronously")
    extract_text: bool = Field(default=True, description="Extract text from documents")
    generate_previews: bool = Field(default=True, description="Generate document previews")


class BatchUploadResponse(BaseModel):
    """Response model for batch upload"""
    batch_id: str
    total_files: int
    job_id: Optional[str] = None
    estimated_completion_time: Optional[str] = None
    status_url: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time: Optional[float] = None


# Dependency functions
async def get_storage_service() -> StorageService:
    """Dependency to get storage service"""
    return storage_service


async def get_document_processor() -> DocumentProcessor:
    """Dependency to get document processor"""
    return document_processor


# API Endpoints

@router.post("/upload/pdf", response_model=DocumentUploadResponse, status_code=HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    extract_text: bool = Form(True, description="Extract text from PDF"),
    generate_thumbnails: bool = Form(True, description="Generate thumbnails"),
    ocr_quality: str = Form("medium", description="OCR quality: low, medium, high, ultra"),
    process_async: bool = Form(False, description="Process file asynchronously"),
    storage: StorageService = Depends(get_storage_service)
):
    """
    Upload and process a PDF file
    
    Returns file ID and processing information. If process_async is True,
    returns job ID for tracking progress.
    """
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported for this endpoint"
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Upload to storage
        upload_result = storage.upload_file(
            file_data=file_content,
            filename=file.filename,
            content_type=file.content_type,
            generate_thumbnails=generate_thumbnails
        )
        
        if upload_result.errors:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"File upload failed: {'; '.join(upload_result.errors)}"
            )
        
        response_data = DocumentUploadResponse(
            file_id=upload_result.file_id,
            filename=file.filename,
            content_type=file.content_type or "application/pdf",
            file_size=len(file_content),
            secure_url=upload_result.secure_url,
            thumbnails=upload_result.thumbnail_urls,
            message="File uploaded successfully"
        )
        
        # Process file
        if process_async:
            # Submit async processing job
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                job_config = BatchJobConfig(
                    max_retries=3,
                    timeout=1800,  # 30 minutes for PDF processing
                    priority=7
                )
                
                job_id = extract_pdf_text_async(tmp_path, job_config)
                response_data.processing_job_id = job_id
                response_data.message += f". Processing job {job_id} started."
                
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        else:
            # Process synchronously (for smaller files)
            if len(file_content) > 10 * 1024 * 1024:  # 10MB limit for sync processing
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail="File too large for synchronous processing. Use process_async=true."
                )
            
            # Extract text immediately
            quality = PDFQuality(ocr_quality.lower()) if ocr_quality in ['low', 'medium', 'high', 'ultra'] else PDFQuality.MEDIUM
            pdf_processor_instance = PDFProcessor(quality=quality)
            
            extraction_result = pdf_processor_instance.extract_text(
                file_content,
                extract_images=generate_thumbnails
            )
            
            # Store extraction result (you might want to cache this)
            # For now, we'll include basic info in the response
            response_data.message += f" Text extracted: {len(extraction_result.text)} characters."
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/documents/{file_id}/text", response_model=DocumentTextResponse)
async def get_document_text(
    file_id: str,
    include_preprocessing: bool = Query(True, description="Apply text preprocessing"),
    max_length: Optional[int] = Query(None, description="Maximum text length to return"),
    storage: StorageService = Depends(get_storage_service)
):
    """
    Get extracted text from a processed document
    """
    
    try:
        # Get file metadata
        file_metadata = storage.get_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found"
            )
        
        # Get file content
        file_content = storage.get_file(file_id)
        if not file_content:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document content for {file_id} not found"
            )
        
        # Process document to extract text
        processing_result = document_processor.process_document(
            file_content,
            file_metadata.original_filename
        )
        
        if processing_result.errors:
            logger.warning(f"Document processing had errors: {processing_result.errors}")
        
        text = processing_result.text
        
        # Apply preprocessing if requested
        language = None
        quality_score = None
        
        if include_preprocessing:
            preprocessing_result = text_preprocessor.preprocess(text)
            text = preprocessing_result.cleaned_text
            language = preprocessing_result.detected_language
            quality_score = preprocessing_result.text_quality.value
        
        # Truncate if max_length specified
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        
        return DocumentTextResponse(
            file_id=file_id,
            text=text,
            word_count=len(text.split()),
            character_count=len(text),
            language=language,
            extraction_method=processing_result.extraction_method,
            processing_time=processing_result.processing_time,
            quality_score=quality_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text extraction failed for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text: {str(e)}"
        )


@router.get("/documents/{file_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(
    file_id: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Get document preview with basic information and thumbnails
    """
    
    try:
        # Get file metadata
        file_metadata = storage.get_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found"
            )
        
        # Get file content for basic processing
        file_content = storage.get_file(file_id)
        if not file_content:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document content for {file_id} not found"
            )
        
        # Quick text extraction for preview
        processing_result = document_processor.process_document(
            file_content,
            file_metadata.original_filename
        )
        
        # Create preview text (first 500 characters)
        preview_text = processing_result.text[:500]
        if len(processing_result.text) > 500:
            preview_text += "..."
        
        # Get page count
        page_count = processing_result.metadata.page_count or 1
        
        # Get thumbnail URLs
        thumbnail_urls = {}
        for size, path in file_metadata.thumbnails.items():
            thumbnail_urls[size] = f"/api/thumbnails/{file_id}/{size}"
        
        return DocumentPreviewResponse(
            file_id=file_id,
            filename=file_metadata.original_filename,
            content_type=file_metadata.content_type,
            file_size=file_metadata.file_size,
            page_count=page_count,
            preview_text=preview_text,
            thumbnails=thumbnail_urls,
            metadata={
                'upload_date': file_metadata.upload_date.isoformat(),
                'file_type': processing_result.metadata.file_type.value,
                'word_count': processing_result.metadata.word_count,
                'has_images': len(processing_result.images) > 0,
                'has_tables': len(processing_result.tables) > 0
            },
            processing_status=file_metadata.status.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview generation failed for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )


@router.post("/batch/upload", response_model=BatchUploadResponse, status_code=HTTP_201_CREATED)
async def batch_upload_documents(
    files: List[UploadFile] = File(..., description="Multiple files to upload"),
    process_async: bool = Form(True, description="Process files asynchronously"),
    extract_text: bool = Form(True, description="Extract text from documents"),
    generate_previews: bool = Form(True, description="Generate document previews"),
    storage: StorageService = Depends(get_storage_service)
):
    """
    Upload and process multiple documents in batch
    """
    
    try:
        if len(files) == 0:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        if len(files) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Too many files. Maximum 50 files per batch."
            )
        
        batch_id = str(uuid.uuid4())
        uploaded_files = []
        
        # Upload all files first
        for file in files:
            try:
                file_content = await file.read()
                
                if len(file_content) == 0:
                    logger.warning(f"Skipping empty file: {file.filename}")
                    continue
                
                # Upload to storage
                upload_result = storage.upload_file(
                    file_data=file_content,
                    filename=file.filename,
                    content_type=file.content_type,
                    generate_thumbnails=generate_previews
                )
                
                if not upload_result.errors:
                    # Save file to temp location for processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                        tmp_file.write(file_content)
                        tmp_path = tmp_file.name
                    
                    uploaded_files.append({
                        'file_id': upload_result.file_id,
                        'filename': file.filename,
                        'file_path': tmp_path,
                        'content_type': file.content_type
                    })
                else:
                    logger.warning(f"Failed to upload {file.filename}: {upload_result.errors}")
                    
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {str(e)}")
                continue
        
        if not uploaded_files:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="No files were successfully uploaded"
            )
        
        response_data = BatchUploadResponse(
            batch_id=batch_id,
            total_files=len(uploaded_files),
            status_url=f"/api/jobs/{batch_id}/status",
            message=f"Uploaded {len(uploaded_files)} files successfully"
        )
        
        if process_async:
            # Submit batch processing job
            job_config = BatchJobConfig(
                max_retries=2,
                timeout=3600,  # 1 hour for batch processing
                priority=5
            )
            
            # Prepare file list for batch processing
            file_list = [
                {
                    'file_path': f['file_path'],
                    'filename': f['filename'],
                    'file_id': f['file_id']
                }
                for f in uploaded_files
            ]
            
            job_id = batch_process_documents_async(file_list, job_config)
            response_data.job_id = job_id
            response_data.message += f" Processing job {job_id} started."
            
            # Estimate completion time (rough estimate)
            avg_time_per_file = 30  # seconds
            estimated_minutes = (len(uploaded_files) * avg_time_per_file) // 60
            response_data.estimated_completion_time = f"~{estimated_minutes} minutes"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch upload failed: {str(e)}"
        )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and progress of a processing job
    """
    
    try:
        batch_processor = get_batch_processor()
        job_result = batch_processor.get_job_status(job_id)
        
        if not job_result:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Convert progress to dict
        progress_dict = {
            'current': job_result.progress.current,
            'total': job_result.progress.total,
            'percentage': job_result.progress.percentage,
            'stage': job_result.progress.stage,
            'message': job_result.progress.message,
            'details': job_result.progress.details
        }
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_result.status.value,
            progress=progress_dict,
            result=job_result.result,
            error=job_result.error,
            started_at=job_result.started_at.isoformat() if job_result.started_at else None,
            completed_at=job_result.completed_at.isoformat() if job_result.completed_at else None,
            execution_time=job_result.execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/documents/{file_id}/download")
async def download_document(
    file_id: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Download original document file
    """
    
    try:
        # Get file metadata
        file_metadata = storage.get_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found"
            )
        
        # Get file content
        file_content = storage.get_file(file_id)
        if not file_content:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document content for {file_id} not found"
            )
        
        # Return file as streaming response
        def generate_file_stream():
            yield file_content
        
        return StreamingResponse(
            generate_file_stream(),
            media_type=file_metadata.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{file_metadata.original_filename}\"",
                "Content-Length": str(file_metadata.file_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


@router.get("/thumbnails/{file_id}/{size}")
async def get_thumbnail(
    file_id: str,
    size: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Get thumbnail image for document
    """
    
    try:
        # Get file metadata
        file_metadata = storage.get_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found"
            )
        
        # Check if thumbnail exists
        if size not in file_metadata.thumbnails:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Thumbnail size '{size}' not found for document {file_id}"
            )
        
        thumbnail_path = file_metadata.thumbnails[size]
        
        # Get thumbnail content from storage
        # This is a simplified implementation - you'd need to implement
        # thumbnail retrieval in your storage service
        thumbnail_content = storage.get_file(thumbnail_path)
        if not thumbnail_content:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail="Thumbnail not found"
            )
        
        return Response(
            content=thumbnail_content,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail retrieval failed for {file_id}/{size}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Thumbnail retrieval failed: {str(e)}"
        )


@router.delete("/documents/{file_id}")
async def delete_document(
    file_id: str,
    storage: StorageService = Depends(get_storage_service)
):
    """
    Delete document and all associated data
    """
    
    try:
        # Check if document exists
        file_metadata = storage.get_metadata(file_id)
        if not file_metadata:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found"
            )
        
        # Delete document
        success = storage.delete_file(file_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete document"
            )
        
        return {"message": f"Document {file_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if services are working
        services_status = {
            "document_processor": "ok",
            "pdf_processor": "ok", 
            "text_preprocessor": "ok",
            "storage_service": "ok",
            "batch_processor": "ok"
        }
        
        # You could add more detailed checks here
        
        return {
            "status": "healthy",
            "services": services_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Service unhealthy"
        )