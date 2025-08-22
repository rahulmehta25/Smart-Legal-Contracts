from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
import aiofiles
import os
import uuid
from datetime import datetime

from app.db.database import get_db
from app.services.document_service import DocumentService
from app.models.document import (
    Document, DocumentCreate, DocumentResponse, 
    DocumentUploadResponse, DocumentChunkResponse
)

router = APIRouter(prefix="/documents", tags=["documents"])
document_service = DocumentService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        
        # Detect content type and decode
        if file.content_type == "application/pdf":
            # For PDF files, you would use PyPDF2 or similar
            # For now, assume text content
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file content")
        else:
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File must be text-based")
        
        # Create document
        document_data = DocumentCreate(
            filename=file.filename,
            content=text_content,
            content_type=file.content_type or "text/plain"
        )
        
        # Save to database
        document = document_service.create_document(db, document_data)
        
        # Process document
        processing_result = document_service.process_document(db, document.id)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document.id,
            filename=document.filename,
            chunks_created=processing_result["chunks_created"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@router.post("/", response_model=DocumentResponse)
async def create_document(
    document_data: DocumentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a document from text content
    """
    try:
        document = document_service.create_document(db, document_data)
        return DocumentResponse.from_orm(document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")


@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Process a document through the RAG pipeline
    """
    try:
        result = document_service.process_document(db, document_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    processed_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get list of documents with pagination
    """
    try:
        documents = document_service.get_documents(
            db, skip=skip, limit=limit, processed_only=processed_only
        )
        return [DocumentResponse.from_orm(doc) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific document by ID
    """
    try:
        document = document_service.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse.from_orm(document)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@router.get("/{document_id}/chunks", response_model=List[DocumentChunkResponse])
async def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a document
    """
    try:
        # Verify document exists
        document = document_service.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks = document_service.get_document_chunks(db, document_id)
        return [DocumentChunkResponse.from_orm(chunk) for chunk in chunks]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")


@router.get("/search/")
async def search_documents(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Search documents using vector similarity
    """
    try:
        results = document_service.search_documents(db, query, limit)
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all its associated data
    """
    try:
        success = document_service.delete_document(db, document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Reprocess an existing document
    """
    try:
        result = document_service.reprocess_document(db, document_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")


@router.get("/stats/overview")
async def get_document_statistics(
    db: Session = Depends(get_db)
):
    """
    Get document statistics and overview
    """
    try:
        stats = document_service.get_document_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")