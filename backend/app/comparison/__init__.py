"""
Advanced AI-powered document comparison system.

This package provides comprehensive document comparison capabilities including:
- Advanced diff algorithms (Myers, character/word/paragraph level)
- Semantic analysis using NLP and transformer models  
- Legal change detection and risk assessment
- Git-like version control for documents
- Three-way merging with conflict resolution
- Batch processing for multiple documents
- Rich visualization components
- PDF export functionality

Main Components:
- DiffEngine: Core comparison algorithms
- SemanticComparison: NLP-powered semantic analysis
- LegalChangeDetector: Legal significance and risk assessment
- VersionTracker: Git-like version control
- MergeEngine: Three-way document merging
- BatchProcessor: Batch processing capabilities
- Visualizers: Various visualization components

Quick Start:
    from backend.app.comparison import compare_documents_advanced
    
    result = compare_documents_advanced(
        old_document="Original content...",
        new_document="Modified content...",
        include_semantic=True,
        include_legal=True
    )
"""

from .diff_engine import (
    AdvancedDiffEngine,
    MyersAlgorithm, 
    FuzzyMatcher,
    TextPreprocessor,
    DiffResult,
    DiffType,
    DiffLevel,
    ComparisonResult,
    compare_documents_simple,
    get_similarity_score,
    find_moved_sections
)

from .semantic_comparison import (
    SemanticComparisonEngine,
    SemanticAnalyzer,
    SemanticDiff,
    SemanticChangeType,
    ConceptEntity,
    analyze_document_semantics,
    compare_documents_with_semantics
)

from .legal_change_detector import (
    LegalChangeDetector,
    LegalRisk,
    LegalRiskLevel,
    LegalChangeCategory,
    ChangeImpact,
    LegalAnalysisResult,
    analyze_legal_risk_simple,
    get_legal_change_patterns
)

from .version_tracker import (
    DocumentVersionTracker,
    DocumentVersion,
    DocumentBranch, 
    MergeResult,
    MergeConflict,
    Author,
    VersionStatus,
    create_document_repository,
    load_document_repository
)

from .merge_engine import (
    ThreeWayMergeEngine,
    MergeStrategy,
    ConflictResolution,
    merge_documents_simple
)

from .batch_processor import (
    BatchProcessor,
    BatchProcessingReport,
    ComparisonJob,
    BatchJobResult,
    DocumentInfo,
    ComparisonType,
    ProcessingStatus,
    process_document_batch
)

# Import visualization components
from .visualizers import (
    SideBySideVisualizer,
    InlineDiffVisualizer,
    ChangeHeatmapVisualizer,
    TimelineVisualizer,
    RelationshipGraphVisualizer,
    PDFExporter
)

__version__ = "1.0.0"

__all__ = [
    # Core diff engine
    "AdvancedDiffEngine",
    "MyersAlgorithm",
    "FuzzyMatcher", 
    "TextPreprocessor",
    "DiffResult",
    "DiffType",
    "DiffLevel",
    "ComparisonResult",
    "compare_documents_simple",
    "get_similarity_score",
    "find_moved_sections",
    
    # Semantic comparison
    "SemanticComparisonEngine",
    "SemanticAnalyzer",
    "SemanticDiff",
    "SemanticChangeType",
    "ConceptEntity",
    "analyze_document_semantics",
    "compare_documents_with_semantics",
    
    # Legal analysis
    "LegalChangeDetector",
    "LegalRisk",
    "LegalRiskLevel",
    "LegalChangeCategory",
    "ChangeImpact", 
    "LegalAnalysisResult",
    "analyze_legal_risk_simple",
    "get_legal_change_patterns",
    
    # Version control
    "DocumentVersionTracker",
    "DocumentVersion",
    "DocumentBranch",
    "MergeResult",
    "MergeConflict",
    "Author",
    "VersionStatus",
    "create_document_repository",
    "load_document_repository",
    
    # Merging
    "ThreeWayMergeEngine",
    "MergeStrategy",
    "ConflictResolution",
    "merge_documents_simple",
    
    # Batch processing
    "BatchProcessor",
    "BatchProcessingReport",
    "ComparisonJob",
    "BatchJobResult",
    "DocumentInfo", 
    "ComparisonType",
    "ProcessingStatus",
    "process_document_batch",
    
    # Visualization
    "SideBySideVisualizer",
    "InlineDiffVisualizer",
    "ChangeHeatmapVisualizer",
    "TimelineVisualizer",
    "RelationshipGraphVisualizer",
    "PDFExporter",
    
    # High-level functions
    "compare_documents_advanced",
    "analyze_document_changes",
    "create_comparison_report"
]


def compare_documents_advanced(old_document: str, new_document: str,
                             include_semantic: bool = False,
                             include_legal: bool = False,
                             comparison_levels: list = None) -> dict:
    """
    Advanced document comparison with multiple analysis types.
    
    Args:
        old_document: Original document content
        new_document: Modified document content  
        include_semantic: Whether to include semantic analysis
        include_legal: Whether to include legal risk analysis
        comparison_levels: List of diff levels to analyze
        
    Returns:
        Comprehensive comparison results dictionary
    """
    # Core diff analysis
    diff_engine = AdvancedDiffEngine()
    if comparison_levels:
        levels = [DiffLevel(level) for level in comparison_levels]
    else:
        levels = [DiffLevel.WORD, DiffLevel.PARAGRAPH]
        
    comparison_result = diff_engine.compare_documents(old_document, new_document, levels)
    
    results = {
        'comparison_result': comparison_result,
        'similarity_score': comparison_result.similarity_score,
        'total_changes': len(comparison_result.differences),
        'statistics': comparison_result.statistics
    }
    
    # Semantic analysis
    if include_semantic:
        semantic_engine = SemanticComparisonEngine()
        semantic_diffs = semantic_engine.compare_documents_semantically(old_document, new_document)
        results['semantic_analysis'] = {
            'semantic_diffs': semantic_diffs,
            'semantic_summary': compare_documents_with_semantics(old_document, new_document)
        }
    
    # Legal risk analysis
    if include_legal:
        legal_detector = LegalChangeDetector()
        semantic_diffs = results.get('semantic_analysis', {}).get('semantic_diffs', [])
        legal_analysis = legal_detector.analyze_legal_changes(
            comparison_result.differences,
            semantic_diffs,
            old_document,
            new_document
        )
        results['legal_analysis'] = legal_analysis
    
    return results


def analyze_document_changes(documents: list, analysis_type: str = "full") -> dict:
    """
    Analyze changes across multiple documents.
    
    Args:
        documents: List of document dictionaries with 'id' and 'content'
        analysis_type: Type of analysis ("standard", "semantic", "legal", "full")
        
    Returns:
        Batch analysis results
    """
    return process_document_batch(documents, analysis_type)


def create_comparison_report(old_document: str, new_document: str,
                           report_format: str = "html",
                           include_visualizations: bool = True) -> str:
    """
    Create a comprehensive comparison report.
    
    Args:
        old_document: Original document content
        new_document: Modified document content
        report_format: Output format ("html", "pdf", "json")
        include_visualizations: Whether to include visualizations
        
    Returns:
        Generated report content
    """
    # Perform comprehensive analysis
    results = compare_documents_advanced(
        old_document, new_document,
        include_semantic=True,
        include_legal=True
    )
    
    if report_format == "html":
        if include_visualizations:
            # Create side-by-side visualization
            from .visualizers.side_by_side import create_side_by_side_view
            return create_side_by_side_view(
                old_document, new_document, 
                results['comparison_result']
            )
        else:
            # Simple HTML report
            return f"""
            <html>
            <body>
                <h1>Document Comparison Report</h1>
                <p>Similarity Score: {results['similarity_score']:.1%}</p>
                <p>Total Changes: {results['total_changes']}</p>
                <h2>Statistics</h2>
                <pre>{json.dumps(results['statistics'], indent=2)}</pre>
            </body>
            </html>
            """
    
    elif report_format == "pdf":
        from .visualizers.pdf_export import export_comparison_pdf
        pdf_buffer = export_comparison_pdf(
            old_document, new_document, results['comparison_result']
        )
        return pdf_buffer.getvalue()
    
    elif report_format == "json":
        # Convert results to JSON-serializable format
        import json
        return json.dumps(results, default=str, indent=2)
    
    else:
        raise ValueError(f"Unsupported report format: {report_format}")


# Additional utility functions for common use cases

def quick_similarity_check(doc1: str, doc2: str) -> float:
    """Quick similarity check between two documents."""
    return get_similarity_score(doc1, doc2)


def find_document_moves(doc1: str, doc2: str) -> list:
    """Find sections that have been moved between documents."""
    return find_moved_sections(doc1, doc2)


def detect_legal_risks(old_doc: str, new_doc: str) -> dict:
    """Quick legal risk detection."""
    return analyze_legal_risk_simple(old_doc, new_doc)