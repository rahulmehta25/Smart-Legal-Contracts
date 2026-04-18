"""
Batch Analysis Service

Handles batch processing of multiple documents for arbitration clause detection:
- Parallel processing with configurable concurrency
- Progress tracking
- Partial failure handling
- Audit trail integration
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from sqlalchemy.orm import Session

from app.services.analysis_service import AnalysisService
from app.services.document_service import DocumentService
from app.rag.pipeline import RAGPipeline
from app.schemas.analysis import (
    BatchAnalysisRequest,
    BatchAnalysisItem,
    BatchAnalysisResponse,
    BatchAnalysisItemResult,
    ArbitrationAnalysisResponse,
    ConfidenceLevel,
)
from app.core.exceptions import (
    DocumentNotFoundError,
    AnalysisError,
    DocumentNotProcessedError,
)


@dataclass
class BatchAnalysisProgress:
    """Progress tracking for batch analysis"""
    total: int
    completed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)


class BatchAnalysisService:
    """
    Service for batch document analysis
    """

    def __init__(self, max_workers: int = 4):
        self.analysis_service = AnalysisService()
        self.document_service = DocumentService()
        self.rag_pipeline = RAGPipeline()
        self.max_workers = max_workers

    async def analyze_batch(
        self,
        db: Session,
        request: BatchAnalysisRequest,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> BatchAnalysisResponse:
        """
        Analyze multiple documents or texts in batch

        Args:
            db: Database session
            request: Batch analysis request
            user_id: User ID for audit trail
            request_id: Request ID for tracing

        Returns:
            BatchAnalysisResponse with results for each item
        """
        start_time = time.time()
        progress = BatchAnalysisProgress(total=len(request.items))

        logger.info(
            f"Starting batch analysis: {len(request.items)} items, "
            f"user={user_id}, request={request_id}"
        )

        results: List[BatchAnalysisItemResult] = []

        if request.parallel and len(request.items) > 1:
            # Process in parallel
            results = await self._process_parallel(
                db, request, progress, user_id, request_id
            )
        else:
            # Process sequentially
            results = await self._process_sequential(
                db, request, progress, user_id, request_id
            )

        total_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Batch analysis complete: {progress.successful}/{progress.total} successful, "
            f"{progress.failed} failed, {total_time}ms"
        )

        return BatchAnalysisResponse(
            total=progress.total,
            successful=progress.successful,
            failed=progress.failed,
            results=results,
            total_processing_time_ms=total_time
        )

    async def _process_parallel(
        self,
        db: Session,
        request: BatchAnalysisRequest,
        progress: BatchAnalysisProgress,
        user_id: Optional[str],
        request_id: Optional[str]
    ) -> List[BatchAnalysisItemResult]:
        """Process items in parallel using thread pool"""
        results = []

        # Create tasks for each item
        tasks = []
        for i, item in enumerate(request.items):
            task = asyncio.create_task(
                self._analyze_single_item(
                    db, item, request.min_confidence_threshold,
                    request.include_context, user_id, request_id, i
                )
            )
            tasks.append(task)

        # Wait for all tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.max_workers)

        async def limited_task(task):
            async with semaphore:
                return await task

        completed = await asyncio.gather(
            *[limited_task(t) for t in tasks],
            return_exceptions=True
        )

        for result in completed:
            if isinstance(result, Exception):
                progress.failed += 1
                progress.errors.append(str(result))
                results.append(BatchAnalysisItemResult(
                    reference_id=None,
                    document_id=None,
                    success=False,
                    error=str(result),
                    result=None
                ))
            else:
                if result.success:
                    progress.successful += 1
                else:
                    progress.failed += 1
                results.append(result)
            progress.completed += 1

        return results

    async def _process_sequential(
        self,
        db: Session,
        request: BatchAnalysisRequest,
        progress: BatchAnalysisProgress,
        user_id: Optional[str],
        request_id: Optional[str]
    ) -> List[BatchAnalysisItemResult]:
        """Process items sequentially"""
        results = []

        for i, item in enumerate(request.items):
            try:
                result = await self._analyze_single_item(
                    db, item, request.min_confidence_threshold,
                    request.include_context, user_id, request_id, i
                )
                if result.success:
                    progress.successful += 1
                else:
                    progress.failed += 1
                results.append(result)
            except Exception as e:
                progress.failed += 1
                progress.errors.append(str(e))
                results.append(BatchAnalysisItemResult(
                    reference_id=item.reference_id,
                    document_id=item.document_id,
                    success=False,
                    error=str(e),
                    result=None
                ))
            progress.completed += 1

        return results

    async def _analyze_single_item(
        self,
        db: Session,
        item: BatchAnalysisItem,
        min_confidence: float,
        include_context: bool,
        user_id: Optional[str],
        request_id: Optional[str],
        index: int
    ) -> BatchAnalysisItemResult:
        """Analyze a single batch item"""
        try:
            if item.document_id:
                # Analyze existing document
                result = await self._analyze_document(
                    db, item.document_id, min_confidence, include_context
                )
                return BatchAnalysisItemResult(
                    reference_id=item.reference_id,
                    document_id=item.document_id,
                    success=True,
                    error=None,
                    result=result
                )
            elif item.text:
                # Quick analysis of text
                result = await self._analyze_text(
                    item.text, min_confidence, include_context
                )
                return BatchAnalysisItemResult(
                    reference_id=item.reference_id,
                    document_id=None,
                    success=True,
                    error=None,
                    result=result
                )
            else:
                raise ValueError("Either document_id or text must be provided")

        except DocumentNotFoundError as e:
            return BatchAnalysisItemResult(
                reference_id=item.reference_id,
                document_id=item.document_id,
                success=False,
                error=f"Document not found: {item.document_id}",
                result=None
            )
        except DocumentNotProcessedError as e:
            return BatchAnalysisItemResult(
                reference_id=item.reference_id,
                document_id=item.document_id,
                success=False,
                error=f"Document not processed: {item.document_id}",
                result=None
            )
        except Exception as e:
            logger.error(f"Batch item {index} failed: {e}")
            return BatchAnalysisItemResult(
                reference_id=item.reference_id,
                document_id=item.document_id,
                success=False,
                error=str(e),
                result=None
            )

    async def _analyze_document(
        self,
        db: Session,
        document_id: int,
        min_confidence: float,
        include_context: bool
    ) -> ArbitrationAnalysisResponse:
        """Analyze an existing document"""
        # Check document exists and is processed
        document = self.document_service.get_document(db, document_id)
        if not document:
            raise DocumentNotFoundError(document_id)
        if not document.is_processed:
            raise DocumentNotProcessedError(document_id)

        # Perform analysis
        analysis_result = self.rag_pipeline.analyze_document_for_arbitration(document_id)

        # Convert to response
        confidence_level = self._get_confidence_level(analysis_result.confidence_score)

        return ArbitrationAnalysisResponse(
            id=0,  # Not stored
            document_id=document_id,
            has_arbitration_clause=analysis_result.has_arbitration_clause,
            confidence_score=analysis_result.confidence_score,
            confidence_level=confidence_level,
            analysis_summary=analysis_result.summary,
            analyzed_at=datetime.utcnow(),
            analysis_version="1.0",
            processing_time_ms=analysis_result.processing_time_ms,
            clauses=[],  # Simplified for batch
            metadata=analysis_result.metadata
        )

    async def _analyze_text(
        self,
        text: str,
        min_confidence: float,
        include_context: bool
    ) -> ArbitrationAnalysisResponse:
        """Quick analysis of raw text"""
        analysis_result = self.rag_pipeline.quick_text_analysis(text)

        confidence_level = self._get_confidence_level(analysis_result.confidence_score)

        return ArbitrationAnalysisResponse(
            id=0,
            document_id=0,
            has_arbitration_clause=analysis_result.has_arbitration_clause,
            confidence_score=analysis_result.confidence_score,
            confidence_level=confidence_level,
            analysis_summary=analysis_result.summary,
            analyzed_at=datetime.utcnow(),
            analysis_version="1.0",
            processing_time_ms=analysis_result.processing_time_ms,
            clauses=[],
            metadata=analysis_result.metadata
        )

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        elif score > 0:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.NONE


# Singleton instance
_batch_service: Optional[BatchAnalysisService] = None


def get_batch_analysis_service() -> BatchAnalysisService:
    """Get batch analysis service singleton"""
    global _batch_service
    if _batch_service is None:
        _batch_service = BatchAnalysisService()
    return _batch_service
