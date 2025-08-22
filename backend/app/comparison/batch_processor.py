"""
Batch processing engine for multiple document comparisons.
Handles bulk analysis, template deviation detection, and compliance checking.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import logging
from pathlib import Path
import hashlib

from .diff_engine import AdvancedDiffEngine, ComparisonResult
from .semantic_comparison import SemanticComparisonEngine
from .legal_change_detector import LegalChangeDetector
from .version_tracker import DocumentVersionTracker


class ProcessingStatus(Enum):
    """Status of batch processing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComparisonType(Enum):
    """Types of comparisons to perform."""
    STANDARD = "standard"
    SEMANTIC = "semantic"
    LEGAL = "legal"
    FULL = "full"
    TEMPLATE_DEVIATION = "template_deviation"
    COMPLIANCE = "compliance"


@dataclass
class DocumentInfo:
    """Information about a document to be processed."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    doc_type: str = "text"


@dataclass
class ComparisonJob:
    """A comparison job between two documents."""
    job_id: str
    doc_a: DocumentInfo
    doc_b: DocumentInfo
    comparison_types: List[ComparisonType]
    priority: int = 5
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJobResult:
    """Result of a batch processing job."""
    job_id: str
    status: ProcessingStatus
    comparison_result: Optional[ComparisonResult]
    semantic_analysis: Optional[Dict[str, Any]] = None
    legal_analysis: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProcessingReport:
    """Complete report of batch processing results."""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    processing_time: float
    results: List[BatchJobResult]
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    compliance_report: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """
    Main batch processing engine for document comparisons.
    Supports parallel processing, job queues, and progress tracking.
    """
    
    def __init__(self, max_workers: int = 4, enable_caching: bool = True):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to cache comparison results
        """
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # Processing engines
        self.diff_engine = AdvancedDiffEngine()
        self.semantic_engine = SemanticComparisonEngine()
        self.legal_detector = LegalChangeDetector()
        
        # Job management
        self.job_queue: List[ComparisonJob] = []
        self.active_jobs: Dict[str, ComparisonJob] = {}
        self.completed_jobs: Dict[str, BatchJobResult] = {}
        
        # Caching
        self.comparison_cache: Dict[str, Any] = {}
        
        # Statistics
        self.processing_stats = {
            'total_jobs_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(self, jobs: List[ComparisonJob], 
                           progress_callback: Optional[Callable] = None) -> BatchProcessingReport:
        """
        Process a batch of comparison jobs.
        
        Args:
            jobs: List of comparison jobs to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Batch processing report with results
        """
        batch_id = self._generate_batch_id()
        start_time = datetime.now()
        
        self.logger.info(f"Starting batch processing {batch_id} with {len(jobs)} jobs")
        
        # Sort jobs by priority
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        # Process jobs with thread pool
        results = []
        completed_count = 0
        failed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._process_single_job, job): job 
                for job in sorted_jobs
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.status == ProcessingStatus.COMPLETED:
                        completed_count += 1
                    else:
                        failed_count += 1
                        
                    # Progress callback
                    if progress_callback:
                        progress = (completed_count + failed_count) / len(jobs) * 100
                        progress_callback(progress, result)
                        
                except Exception as e:
                    self.logger.error(f"Job {job.job_id} failed with error: {str(e)}")
                    failed_result = BatchJobResult(
                        job_id=job.job_id,
                        status=ProcessingStatus.FAILED,
                        comparison_result=None,
                        error_message=str(e)
                    )
                    results.append(failed_result)
                    failed_count += 1
        
        # Calculate processing time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Generate summary statistics
        summary_stats = self._generate_batch_statistics(results)
        
        # Generate compliance report if applicable
        compliance_report = self._generate_compliance_report(results, jobs)
        
        report = BatchProcessingReport(
            batch_id=batch_id,
            total_jobs=len(jobs),
            completed_jobs=completed_count,
            failed_jobs=failed_count,
            processing_time=total_time,
            results=results,
            summary_statistics=summary_stats,
            compliance_report=compliance_report
        )
        
        self.logger.info(f"Batch processing {batch_id} completed in {total_time:.2f}s")
        return report
    
    def process_document_set(self, documents: List[DocumentInfo],
                           comparison_type: ComparisonType = ComparisonType.STANDARD,
                           template_doc: Optional[DocumentInfo] = None) -> BatchProcessingReport:
        """
        Process a set of documents, comparing each against a template or each other.
        
        Args:
            documents: List of documents to process
            comparison_type: Type of comparison to perform
            template_doc: Optional template document for deviation analysis
            
        Returns:
            Batch processing report
        """
        jobs = []
        
        if template_doc:
            # Compare each document against template
            for doc in documents:
                job = ComparisonJob(
                    job_id=f"template_comparison_{doc.doc_id}",
                    doc_a=template_doc,
                    doc_b=doc,
                    comparison_types=[comparison_type],
                    metadata={'comparison_mode': 'template_deviation'}
                )
                jobs.append(job)
        else:
            # Compare documents pairwise
            for i, doc_a in enumerate(documents):
                for j, doc_b in enumerate(documents[i+1:], i+1):
                    job = ComparisonJob(
                        job_id=f"pairwise_{doc_a.doc_id}_{doc_b.doc_id}",
                        doc_a=doc_a,
                        doc_b=doc_b,
                        comparison_types=[comparison_type],
                        metadata={'comparison_mode': 'pairwise'}
                    )
                    jobs.append(job)
        
        return asyncio.run(self.process_batch(jobs))
    
    def analyze_template_deviations(self, template_content: str,
                                  document_variants: List[DocumentInfo]) -> Dict[str, Any]:
        """
        Analyze how documents deviate from a standard template.
        
        Args:
            template_content: Standard template content
            document_variants: List of document variants to analyze
            
        Returns:
            Deviation analysis report
        """
        template_doc = DocumentInfo(
            doc_id="template",
            content=template_content,
            metadata={'type': 'template'}
        )
        
        jobs = []
        for variant in document_variants:
            job = ComparisonJob(
                job_id=f"deviation_{variant.doc_id}",
                doc_a=template_doc,
                doc_b=variant,
                comparison_types=[ComparisonType.TEMPLATE_DEVIATION, ComparisonType.SEMANTIC],
                metadata={'analysis_type': 'template_deviation'}
            )
            jobs.append(job)
        
        batch_report = asyncio.run(self.process_batch(jobs))
        
        # Analyze deviations
        deviations = []
        common_changes = {}
        
        for result in batch_report.results:
            if result.status == ProcessingStatus.COMPLETED:
                deviation_analysis = self._analyze_single_deviation(
                    result.comparison_result, result.semantic_analysis
                )
                deviations.append(deviation_analysis)
                
                # Track common changes
                for change in deviation_analysis.get('significant_changes', []):
                    change_key = self._normalize_change_for_grouping(change)
                    common_changes[change_key] = common_changes.get(change_key, 0) + 1
        
        return {
            'template_analysis': {
                'total_variants': len(document_variants),
                'analyzed_variants': len(deviations),
                'average_deviation_score': sum(d.get('deviation_score', 0) for d in deviations) / max(len(deviations), 1)
            },
            'individual_deviations': deviations,
            'common_deviations': [
                {'change': change, 'frequency': freq, 'percentage': freq / len(deviations) * 100}
                for change, freq in sorted(common_changes.items(), key=lambda x: x[1], reverse=True)
            ],
            'recommendations': self._generate_template_recommendations(common_changes, deviations)
        }
    
    def compliance_check(self, documents: List[DocumentInfo],
                        compliance_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check documents for compliance with specified rules.
        
        Args:
            documents: Documents to check
            compliance_rules: Rules to check against
            
        Returns:
            Compliance report
        """
        compliance_results = []
        
        for doc in documents:
            doc_compliance = self._check_document_compliance(doc, compliance_rules)
            compliance_results.append(doc_compliance)
        
        # Generate overall compliance report
        total_docs = len(documents)
        compliant_docs = len([r for r in compliance_results if r['is_compliant']])
        
        return {
            'summary': {
                'total_documents': total_docs,
                'compliant_documents': compliant_docs,
                'compliance_rate': compliant_docs / total_docs * 100 if total_docs > 0 else 0,
                'total_violations': sum(len(r['violations']) for r in compliance_results)
            },
            'document_results': compliance_results,
            'violation_categories': self._categorize_violations(compliance_results),
            'recommendations': self._generate_compliance_recommendations(compliance_results)
        }
    
    def _process_single_job(self, job: ComparisonJob) -> BatchJobResult:
        """Process a single comparison job."""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(job)
            if self.enable_caching and cache_key in self.comparison_cache:
                self.processing_stats['cache_hits'] += 1
                cached_result = self.comparison_cache[cache_key]
                return BatchJobResult(
                    job_id=job.job_id,
                    status=ProcessingStatus.COMPLETED,
                    comparison_result=cached_result['comparison'],
                    semantic_analysis=cached_result.get('semantic'),
                    legal_analysis=cached_result.get('legal'),
                    processing_time=0.0,  # Cached result
                    metadata={'cached': True}
                )
            
            self.processing_stats['cache_misses'] += 1
            
            # Perform comparisons based on requested types
            results = {}
            
            if ComparisonType.STANDARD in job.comparison_types or ComparisonType.FULL in job.comparison_types:
                comparison_result = self.diff_engine.compare_documents(
                    job.doc_a.content, job.doc_b.content
                )
                results['comparison'] = comparison_result
            
            if ComparisonType.SEMANTIC in job.comparison_types or ComparisonType.FULL in job.comparison_types:
                semantic_diffs = self.semantic_engine.compare_documents_semantically(
                    job.doc_a.content, job.doc_b.content
                )
                results['semantic'] = {
                    'semantic_diffs': semantic_diffs,
                    'overall_similarity': self.semantic_engine.analyzer.compute_semantic_similarity(
                        job.doc_a.content, job.doc_b.content
                    )
                }
            
            if ComparisonType.LEGAL in job.comparison_types or ComparisonType.FULL in job.comparison_types:
                if 'comparison' in results:
                    legal_analysis = self.legal_detector.analyze_legal_changes(
                        results['comparison'].differences,
                        results.get('semantic', {}).get('semantic_diffs', []),
                        job.doc_a.content,
                        job.doc_b.content
                    )
                    results['legal'] = legal_analysis
            
            # Handle special comparison types
            if ComparisonType.TEMPLATE_DEVIATION in job.comparison_types:
                if 'comparison' not in results:
                    comparison_result = self.diff_engine.compare_documents(
                        job.doc_a.content, job.doc_b.content
                    )
                    results['comparison'] = comparison_result
                
                deviation_analysis = self._analyze_template_deviation(
                    results['comparison'], job.doc_a.content, job.doc_b.content
                )
                results['template_deviation'] = deviation_analysis
            
            # Cache results
            if self.enable_caching:
                self.comparison_cache[cache_key] = results
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.processing_stats['total_jobs_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['average_processing_time'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['total_jobs_processed']
            )
            
            return BatchJobResult(
                job_id=job.job_id,
                status=ProcessingStatus.COMPLETED,
                comparison_result=results.get('comparison'),
                semantic_analysis=results.get('semantic'),
                legal_analysis=results.get('legal'),
                processing_time=processing_time,
                metadata={'template_deviation': results.get('template_deviation')}
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error processing job {job.job_id}: {str(e)}")
            
            return BatchJobResult(
                job_id=job.job_id,
                status=ProcessingStatus.FAILED,
                comparison_result=None,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _generate_cache_key(self, job: ComparisonJob) -> str:
        """Generate cache key for a comparison job."""
        content_hash_a = hashlib.md5(job.doc_a.content.encode()).hexdigest()
        content_hash_b = hashlib.md5(job.doc_b.content.encode()).hexdigest()
        comparison_types_str = ','.join(sorted([ct.value for ct in job.comparison_types]))
        
        cache_string = f"{content_hash_a}_{content_hash_b}_{comparison_types_str}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"batch_{timestamp}_{hash(timestamp) % 10000}"
    
    def _generate_batch_statistics(self, results: List[BatchJobResult]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.status == ProcessingStatus.COMPLETED]
        
        if not successful_results:
            return {'error': 'No successful comparisons'}
        
        # Calculate similarity statistics
        similarities = []
        total_changes = []
        processing_times = []
        
        for result in successful_results:
            if result.comparison_result:
                similarities.append(result.comparison_result.similarity_score)
                total_changes.append(len(result.comparison_result.differences))
            processing_times.append(result.processing_time)
        
        return {
            'successful_jobs': len(successful_results),
            'failed_jobs': len(results) - len(successful_results),
            'average_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'average_changes': sum(total_changes) / len(total_changes) if total_changes else 0,
            'total_processing_time': sum(processing_times),
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'cache_hit_rate': (
                self.processing_stats['cache_hits'] / 
                (self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']) * 100
                if (self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']) > 0 else 0
            )
        }
    
    def _generate_compliance_report(self, results: List[BatchJobResult], 
                                  jobs: List[ComparisonJob]) -> Optional[Dict[str, Any]]:
        """Generate compliance report if applicable."""
        compliance_jobs = [j for j in jobs if ComparisonType.COMPLIANCE in j.comparison_types]
        
        if not compliance_jobs:
            return None
        
        # This would be expanded based on specific compliance requirements
        return {
            'compliance_jobs_processed': len(compliance_jobs),
            'compliance_summary': 'Compliance analysis completed'
        }
    
    def _analyze_single_deviation(self, comparison_result: ComparisonResult,
                                semantic_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze deviation for a single document."""
        if not comparison_result:
            return {'error': 'No comparison result available'}
        
        # Calculate deviation score based on similarity
        deviation_score = 1.0 - comparison_result.similarity_score
        
        # Categorize changes
        significant_changes = []
        minor_changes = []
        
        for diff in comparison_result.differences:
            change_info = {
                'type': diff.diff_type.value,
                'content': diff.new_content[:100] if diff.new_content else diff.old_content[:100],
                'confidence': diff.confidence,
                'position': diff.new_position
            }
            
            # Classify as significant or minor based on size and confidence
            if len(diff.new_content or diff.old_content) > 50 and diff.confidence > 0.8:
                significant_changes.append(change_info)
            else:
                minor_changes.append(change_info)
        
        return {
            'deviation_score': deviation_score,
            'total_changes': len(comparison_result.differences),
            'significant_changes': significant_changes,
            'minor_changes': minor_changes,
            'similarity_score': comparison_result.similarity_score,
            'semantic_similarity': semantic_analysis.get('overall_similarity', 0) if semantic_analysis else None
        }
    
    def _normalize_change_for_grouping(self, change: Dict[str, Any]) -> str:
        """Normalize change for grouping common changes."""
        # Simple normalization - could be enhanced
        return f"{change['type']}_{len(change['content'])}_chars"
    
    def _generate_template_recommendations(self, common_changes: Dict[str, int], 
                                         deviations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for template improvements."""
        recommendations = []
        
        # Find most common changes
        most_common = sorted(common_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for change, frequency in most_common:
            if frequency / len(deviations) > 0.5:  # More than 50% of documents
                recommendations.append(
                    f"Consider updating template: '{change}' appears in {frequency}/{len(deviations)} documents"
                )
        
        # Check for high deviation scores
        high_deviation_docs = [d for d in deviations if d.get('deviation_score', 0) > 0.5]
        if len(high_deviation_docs) / len(deviations) > 0.3:
            recommendations.append(
                f"High deviation detected in {len(high_deviation_docs)} documents - template may need revision"
            )
        
        return recommendations
    
    def _check_document_compliance(self, doc: DocumentInfo, 
                                 compliance_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single document for compliance."""
        violations = []
        
        # This would be expanded based on specific compliance rules
        # For now, just check basic requirements
        
        content = doc.content.lower()
        
        # Check for required clauses
        required_clauses = compliance_rules.get('required_clauses', [])
        for clause in required_clauses:
            if clause.lower() not in content:
                violations.append({
                    'type': 'missing_clause',
                    'description': f"Required clause '{clause}' not found",
                    'severity': 'high'
                })
        
        # Check for prohibited terms
        prohibited_terms = compliance_rules.get('prohibited_terms', [])
        for term in prohibited_terms:
            if term.lower() in content:
                violations.append({
                    'type': 'prohibited_term',
                    'description': f"Prohibited term '{term}' found",
                    'severity': 'medium'
                })
        
        return {
            'document_id': doc.doc_id,
            'is_compliant': len(violations) == 0,
            'violations': violations,
            'compliance_score': max(0, 100 - len(violations) * 10)  # Simple scoring
        }
    
    def _categorize_violations(self, compliance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize violations across all documents."""
        categories = {}
        
        for result in compliance_results:
            for violation in result['violations']:
                v_type = violation['type']
                if v_type not in categories:
                    categories[v_type] = {'count': 0, 'documents': set()}
                categories[v_type]['count'] += 1
                categories[v_type]['documents'].add(result['document_id'])
        
        # Convert sets to lists for JSON serialization
        for category in categories.values():
            category['documents'] = list(category['documents'])
        
        return categories
    
    def _generate_compliance_recommendations(self, compliance_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for compliance improvements."""
        recommendations = []
        
        # Find most common violations
        violation_counts = {}
        for result in compliance_results:
            for violation in result['violations']:
                v_type = violation['type']
                violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        for violation_type, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True):
            if count > len(compliance_results) * 0.3:  # More than 30% of documents
                recommendations.append(
                    f"Address '{violation_type}' violations - found in {count} documents"
                )
        
        return recommendations
    
    def _analyze_template_deviation(self, comparison_result: ComparisonResult,
                                  template_content: str, variant_content: str) -> Dict[str, Any]:
        """Analyze template deviation for template comparison type."""
        return self._analyze_single_deviation(comparison_result, None)


# Utility functions

def process_document_batch(documents: List[Dict[str, Any]], 
                          comparison_type: str = "standard") -> Dict[str, Any]:
    """
    Quick utility to process a batch of documents.
    
    Args:
        documents: List of document dictionaries with 'id' and 'content'
        comparison_type: Type of comparison ("standard", "semantic", "legal", "full")
        
    Returns:
        Batch processing results
    """
    processor = BatchProcessor()
    
    doc_infos = []
    for doc in documents:
        doc_info = DocumentInfo(
            doc_id=doc['id'],
            content=doc['content'],
            metadata=doc.get('metadata', {})
        )
        doc_infos.append(doc_info)
    
    comparison_enum = ComparisonType(comparison_type.lower())
    report = processor.process_document_set(doc_infos, comparison_enum)
    
    return {
        'batch_id': report.batch_id,
        'summary': {
            'total_jobs': report.total_jobs,
            'completed_jobs': report.completed_jobs,
            'failed_jobs': report.failed_jobs,
            'processing_time': report.processing_time
        },
        'results': [
            {
                'job_id': r.job_id,
                'status': r.status.value,
                'similarity_score': r.comparison_result.similarity_score if r.comparison_result else 0,
                'changes_count': len(r.comparison_result.differences) if r.comparison_result else 0,
                'processing_time': r.processing_time
            }
            for r in report.results
        ],
        'statistics': report.summary_statistics
    }