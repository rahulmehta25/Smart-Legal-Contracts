from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
from loguru import logger

from app.rag.text_processor import LegalTextProcessor, TextChunk
from app.rag.embeddings import EmbeddingGenerator
from app.rag.retriever import ArbitrationRetriever, RetrievalResult
from app.db.vector_store import VectorStore
from app.models.analysis import ArbitrationAnalysis, ArbitrationClause


@dataclass
class AnalysisResult:
    """Complete arbitration analysis result"""
    has_arbitration_clause: bool
    confidence_score: float
    summary: str
    clauses: List[Dict[str, Any]]
    processing_time_ms: int
    metadata: Dict[str, Any]


class RAGPipeline:
    """
    Complete RAG pipeline for arbitration clause detection
    """
    
    def __init__(self):
        # Initialize components
        self.text_processor = LegalTextProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.retriever = ArbitrationRetriever(
            self.vector_store,
            self.embedding_generator,
            self.text_processor
        )
        
        # Analysis thresholds
        self.confidence_thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        }
        
        logger.info("RAG Pipeline initialized successfully")
    
    def process_document(self, 
                        document_id: int, 
                        text: str, 
                        chunk_size: int = 1000) -> Tuple[List[TextChunk], Dict[str, Any]]:
        """
        Process a document and store chunks in vector store
        
        Args:
            document_id: Unique document identifier
            text: Document text content
            chunk_size: Size of text chunks
            
        Returns:
            Tuple of (chunks, processing_metadata)
        """
        start_time = time.time()
        
        # Process document into chunks
        chunks, doc_metadata = self.text_processor.process_document(text, chunk_size)
        
        # Prepare data for vector store
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_indices = [chunk.chunk_index for chunk in chunks]
        start_chars = [chunk.start_char for chunk in chunks]
        end_chars = [chunk.end_char for chunk in chunks]
        
        # Store chunks in vector store
        embedding_ids = self.vector_store.add_document_chunks(
            chunks=chunk_texts,
            document_id=document_id,
            chunk_indices=chunk_indices,
            start_chars=start_chars,
            end_chars=end_chars
        )
        
        # Update chunks with embedding IDs
        for chunk, embedding_id in zip(chunks, embedding_ids):
            chunk.embedding_id = embedding_id
        
        processing_time = (time.time() - start_time) * 1000
        
        processing_metadata = {
            **doc_metadata,
            "processing_time_ms": processing_time,
            "embedding_ids": embedding_ids
        }
        
        logger.info(f"Document {document_id} processed: {len(chunks)} chunks in {processing_time:.2f}ms")
        
        return chunks, processing_metadata
    
    def analyze_document_for_arbitration(self, document_id: int) -> AnalysisResult:
        """
        Analyze a document for arbitration clauses
        
        Args:
            document_id: Document to analyze
            
        Returns:
            AnalysisResult with complete analysis
        """
        start_time = time.time()
        
        # Retrieve potential arbitration clauses
        arbitration_results = self.retriever.retrieve_arbitration_clauses(
            document_id=document_id,
            top_k=20,
            min_score=0.3
        )
        
        # Analyze results
        analysis = self._analyze_retrieval_results(arbitration_results)
        
        # Get document coverage analysis
        coverage_analysis = self.retriever.analyze_document_arbitration_coverage(document_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create final analysis result
        result = AnalysisResult(
            has_arbitration_clause=analysis["has_arbitration_clause"],
            confidence_score=analysis["confidence_score"],
            summary=analysis["summary"],
            clauses=analysis["clauses"],
            processing_time_ms=int(processing_time),
            metadata={
                "total_candidates": len(arbitration_results),
                "coverage_analysis": coverage_analysis,
                "analysis_method": "rag_pipeline",
                "confidence_level": analysis["confidence_level"]
            }
        )
        
        logger.info(f"Document {document_id} analysis completed: "
                   f"{'HAS' if result.has_arbitration_clause else 'NO'} arbitration clause "
                   f"(confidence: {result.confidence_score:.2f})")
        
        return result
    
    def quick_text_analysis(self, text: str) -> AnalysisResult:
        """
        Quick analysis of text without storing in vector store
        
        Args:
            text: Text to analyze
            
        Returns:
            AnalysisResult with analysis
        """
        start_time = time.time()
        
        # Process text
        chunks, doc_metadata = self.text_processor.process_document(text, chunk_size=800)
        
        # Analyze each chunk for arbitration signals
        all_signals = []
        arbitration_chunks = []
        
        for chunk in chunks:
            signals = self.text_processor.extract_arbitration_signals(chunk.content)
            all_signals.append(signals)
            
            if signals["arbitration_keywords_count"] > 1:
                arbitration_chunks.append({
                    "text": chunk.content,
                    "signals": signals,
                    "chunk_index": chunk.chunk_index,
                    "relevance_score": min(signals["arbitration_keywords_count"] / 5.0, 1.0)
                })
        
        # Calculate overall analysis
        analysis = self._analyze_signals_and_chunks(all_signals, arbitration_chunks)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResult(
            has_arbitration_clause=analysis["has_arbitration_clause"],
            confidence_score=analysis["confidence_score"],
            summary=analysis["summary"],
            clauses=analysis["clauses"],
            processing_time_ms=int(processing_time),
            metadata={
                "total_chunks": len(chunks),
                "arbitration_chunks": len(arbitration_chunks),
                "analysis_method": "quick_analysis",
                "confidence_level": analysis["confidence_level"]
            }
        )
    
    def _analyze_retrieval_results(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Analyze retrieval results to determine arbitration presence
        """
        if not results:
            return {
                "has_arbitration_clause": False,
                "confidence_score": 0.0,
                "summary": "No arbitration-related content found in document",
                "clauses": [],
                "confidence_level": "none"
            }
        
        # Analyze top results
        high_confidence_results = [r for r in results if r.final_score > 0.7]
        medium_confidence_results = [r for r in results if 0.5 < r.final_score <= 0.7]
        
        # Extract clauses with high arbitration signals
        detected_clauses = []
        overall_signals = {
            "binding_arbitration": False,
            "mandatory_arbitration": False,
            "class_action_waiver": False,
            "jury_waiver": False,
            "total_keywords": 0
        }
        
        for result in results[:10]:  # Top 10 results
            signals = result.arbitration_signals
            
            # Update overall signals
            if signals.get("binding_arbitration"):
                overall_signals["binding_arbitration"] = True
            if signals.get("mandatory_arbitration"):
                overall_signals["mandatory_arbitration"] = True
            if signals.get("class_action_waiver"):
                overall_signals["class_action_waiver"] = True
            if signals.get("jury_waiver"):
                overall_signals["jury_waiver"] = True
            
            overall_signals["total_keywords"] += signals.get("arbitration_keywords_count", 0)
            
            # Add high-quality clauses
            if result.final_score > 0.6:
                clause = {
                    "text": result.text[:500] + "..." if len(result.text) > 500 else result.text,
                    "relevance_score": result.final_score,
                    "signals": signals,
                    "chunk_id": result.chunk_id,
                    "type": self._classify_clause_type(signals)
                }
                detected_clauses.append(clause)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            high_confidence_results,
            medium_confidence_results,
            overall_signals
        )
        
        # Determine if arbitration clause exists
        has_arbitration = (
            len(high_confidence_results) > 0 or
            overall_signals["binding_arbitration"] or
            overall_signals["mandatory_arbitration"] or
            confidence_score > 0.5
        )
        
        # Generate summary
        summary = self._generate_analysis_summary(
            has_arbitration,
            confidence_score,
            overall_signals,
            len(detected_clauses)
        )
        
        # Determine confidence level
        if confidence_score >= self.confidence_thresholds["high_confidence"]:
            confidence_level = "high"
        elif confidence_score >= self.confidence_thresholds["medium_confidence"]:
            confidence_level = "medium"
        elif confidence_score >= self.confidence_thresholds["low_confidence"]:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return {
            "has_arbitration_clause": has_arbitration,
            "confidence_score": confidence_score,
            "summary": summary,
            "clauses": detected_clauses,
            "confidence_level": confidence_level
        }
    
    def _analyze_signals_and_chunks(self, 
                                  all_signals: List[Dict[str, Any]], 
                                  arbitration_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze arbitration signals from text chunks
        """
        if not arbitration_chunks:
            return {
                "has_arbitration_clause": False,
                "confidence_score": 0.0,
                "summary": "No arbitration content detected in text",
                "clauses": [],
                "confidence_level": "none"
            }
        
        # Aggregate signals
        overall_signals = {
            "binding_arbitration": any(s.get("binding_arbitration", False) for s in all_signals),
            "mandatory_arbitration": any(s.get("mandatory_arbitration", False) for s in all_signals),
            "class_action_waiver": any(s.get("class_action_waiver", False) for s in all_signals),
            "jury_waiver": any(s.get("jury_waiver", False) for s in all_signals),
            "total_keywords": sum(s.get("arbitration_keywords_count", 0) for s in all_signals)
        }
        
        # Calculate confidence based on signals and chunk quality
        base_confidence = min(len(arbitration_chunks) / 5.0, 0.7)  # More chunks = higher confidence
        
        signal_bonus = 0.0
        if overall_signals["binding_arbitration"]:
            signal_bonus += 0.2
        if overall_signals["mandatory_arbitration"]:
            signal_bonus += 0.15
        if overall_signals["class_action_waiver"]:
            signal_bonus += 0.1
        if overall_signals["jury_waiver"]:
            signal_bonus += 0.05
        
        confidence_score = min(base_confidence + signal_bonus, 1.0)
        
        # Determine arbitration presence
        has_arbitration = (
            confidence_score > 0.4 or
            overall_signals["binding_arbitration"] or
            overall_signals["mandatory_arbitration"]
        )
        
        # Prepare clause results
        clauses = []
        for chunk in arbitration_chunks[:5]:  # Top 5 chunks
            clause = {
                "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                "relevance_score": chunk["relevance_score"],
                "signals": chunk["signals"],
                "chunk_index": chunk["chunk_index"],
                "type": self._classify_clause_type(chunk["signals"])
            }
            clauses.append(clause)
        
        # Generate summary
        summary = self._generate_analysis_summary(
            has_arbitration,
            confidence_score,
            overall_signals,
            len(clauses)
        )
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "medium"
        elif confidence_score >= 0.4:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return {
            "has_arbitration_clause": has_arbitration,
            "confidence_score": confidence_score,
            "summary": summary,
            "clauses": clauses,
            "confidence_level": confidence_level
        }
    
    def _calculate_confidence_score(self, 
                                  high_conf: List[RetrievalResult],
                                  medium_conf: List[RetrievalResult],
                                  signals: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score
        """
        # Base score from retrieval results
        base_score = 0.0
        
        if high_conf:
            base_score += min(len(high_conf) * 0.2, 0.6)
        
        if medium_conf:
            base_score += min(len(medium_conf) * 0.1, 0.3)
        
        # Signal bonuses
        signal_bonus = 0.0
        if signals["binding_arbitration"]:
            signal_bonus += 0.25
        if signals["mandatory_arbitration"]:
            signal_bonus += 0.2
        if signals["class_action_waiver"]:
            signal_bonus += 0.15
        if signals["jury_waiver"]:
            signal_bonus += 0.1
        
        # Keyword density bonus
        keyword_bonus = min(signals["total_keywords"] / 20.0, 0.2)
        
        total_score = base_score + signal_bonus + keyword_bonus
        return min(total_score, 1.0)
    
    def _classify_clause_type(self, signals: Dict[str, Any]) -> str:
        """
        Classify the type of arbitration clause
        """
        if signals.get("binding_arbitration") and signals.get("mandatory_arbitration"):
            return "mandatory_binding_arbitration"
        elif signals.get("binding_arbitration"):
            return "binding_arbitration"
        elif signals.get("mandatory_arbitration"):
            return "mandatory_arbitration"
        elif signals.get("class_action_waiver"):
            return "class_action_waiver"
        elif signals.get("jury_waiver"):
            return "jury_waiver"
        else:
            return "general_arbitration"
    
    def _generate_analysis_summary(self, 
                                 has_arbitration: bool,
                                 confidence: float,
                                 signals: Dict[str, Any],
                                 num_clauses: int) -> str:
        """
        Generate human-readable analysis summary
        """
        if not has_arbitration:
            return "No arbitration clauses detected in the document."
        
        summary_parts = [
            f"Arbitration clauses detected with {confidence:.1%} confidence."
        ]
        
        if signals["binding_arbitration"]:
            summary_parts.append("Contains binding arbitration requirements.")
        
        if signals["mandatory_arbitration"]:
            summary_parts.append("Contains mandatory arbitration clauses.")
        
        if signals["class_action_waiver"]:
            summary_parts.append("Includes class action waivers.")
        
        if signals["jury_waiver"]:
            summary_parts.append("Contains jury trial waivers.")
        
        summary_parts.append(f"Found {num_clauses} relevant clause(s).")
        
        return " ".join(summary_parts)