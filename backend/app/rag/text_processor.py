import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import spacy
from loguru import logger


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    section_title: str = ""
    is_arbitration_relevant: bool = False
    token_count: int = 0


class LegalTextProcessor:
    """
    Specialized text processor for legal documents with focus on arbitration clauses
    """
    
    def __init__(self):
        # Load spaCy model for legal text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Arbitration-related keywords and phrases
        self.arbitration_keywords = {
            'binding_arbitration': [
                'binding arbitration', 'mandatory arbitration', 'compulsory arbitration',
                'final and binding arbitration', 'arbitration shall be binding'
            ],
            'arbitration_general': [
                'arbitration', 'arbitrator', 'arbitral', 'arbitrate', 'arbitrating',
                'arbitration proceedings', 'arbitration process', 'arbitral tribunal'
            ],
            'dispute_resolution': [
                'dispute resolution', 'resolution of disputes', 'settle disputes',
                'dispute settlement', 'alternative dispute resolution', 'adr'
            ],
            'class_action_waiver': [
                'class action waiver', 'collective action waiver', 'class action prohibition',
                'no class actions', 'waive class action', 'representative action waiver'
            ],
            'jury_waiver': [
                'jury trial waiver', 'waive jury trial', 'no jury trial',
                'jury trial prohibition', 'waiver of jury trial'
            ],
            'court_jurisdiction': [
                'exclusive jurisdiction', 'competent jurisdiction', 'courts of',
                'jurisdiction and venue', 'submit to jurisdiction'
            ]
        }
        
        # Legal section patterns
        self.section_patterns = [
            r'^\s*(?:section|sec\.?)\s*\d+',
            r'^\s*\d+\.\s*[A-Z]',
            r'^\s*[A-Z][A-Z\s]+:',
            r'^\s*\([a-z]\)',
            r'^\s*\([0-9]+\)',
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess legal text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Clean up special characters while preserving legal formatting
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\'\"\/\n]', ' ', text)
        
        return text.strip()
    
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect sections and headings in legal text
        """
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for section patterns
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    sections.append({
                        'title': line,
                        'line_number': i,
                        'start_char': text.find(line),
                        'type': 'section_header'
                    })
                    break
            
            # Check for all caps headings (common in legal docs)
            if len(line) > 5 and line.isupper() and not any(char.isdigit() for char in line):
                sections.append({
                    'title': line,
                    'line_number': i,
                    'start_char': text.find(line),
                    'type': 'heading'
                })
        
        return sections
    
    def chunk_text_by_sentences(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[TextChunk]:
        """
        Chunk text by sentences with overlap, optimized for legal text
        """
        if self.nlp is None:
            return self._simple_chunk_text(text, max_chunk_size, overlap)
        
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                # Create chunk
                chunk_end = current_start + len(current_chunk)
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    start_char=current_start,
                    end_char=chunk_end,
                    chunk_index=chunk_index,
                    token_count=len(current_chunk.split())
                )
                chunk.is_arbitration_relevant = self._is_arbitration_relevant(chunk.content)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                chunk_index += 1
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_start = chunk_end - len(overlap_text)
            else:
                if not current_chunk:
                    current_start = text.find(sentence)
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_index=chunk_index,
                token_count=len(current_chunk.split())
            )
            chunk.is_arbitration_relevant = self._is_arbitration_relevant(chunk.content)
            chunks.append(chunk)
        
        return chunks
    
    def _simple_chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[TextChunk]:
        """
        Simple text chunking when spaCy is not available
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_chunk_size - overlap):
            chunk_words = words[i:i + max_chunk_size]
            chunk_text = " ".join(chunk_words)
            
            start_char = text.find(chunk_words[0]) if chunk_words else 0
            end_char = start_char + len(chunk_text)
            
            chunk = TextChunk(
                content=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=len(chunks),
                token_count=len(chunk_words)
            )
            chunk.is_arbitration_relevant = self._is_arbitration_relevant(chunk.content)
            chunks.append(chunk)
        
        return chunks
    
    def _is_arbitration_relevant(self, text: str) -> bool:
        """
        Check if text chunk is relevant to arbitration
        """
        text_lower = text.lower()
        
        # Count keyword matches
        total_matches = 0
        for category, keywords in self.arbitration_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    total_matches += 1
        
        # Consider relevant if it has multiple arbitration keywords
        return total_matches >= 2
    
    def extract_arbitration_signals(self, text: str) -> Dict[str, Any]:
        """
        Extract specific arbitration-related signals from text
        """
        text_lower = text.lower()
        signals = {
            'binding_arbitration': False,
            'class_action_waiver': False,
            'jury_waiver': False,
            'mandatory_arbitration': False,
            'arbitration_keywords_count': 0,
            'found_keywords': []
        }
        
        # Check for specific signals
        for category, keywords in self.arbitration_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    signals['arbitration_keywords_count'] += 1
                    signals['found_keywords'].append(keyword)
                    
                    if category == 'binding_arbitration':
                        signals['binding_arbitration'] = True
                        if 'mandatory' in keyword.lower():
                            signals['mandatory_arbitration'] = True
                    elif category == 'class_action_waiver':
                        signals['class_action_waiver'] = True
                    elif category == 'jury_waiver':
                        signals['jury_waiver'] = True
        
        return signals
    
    def process_document(self, text: str, chunk_size: int = 1000) -> Tuple[List[TextChunk], Dict[str, Any]]:
        """
        Process entire document and return chunks with metadata
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Detect sections
        sections = self.detect_sections(cleaned_text)
        
        # Create chunks
        chunks = self.chunk_text_by_sentences(cleaned_text, chunk_size)
        
        # Add section information to chunks
        for chunk in chunks:
            chunk.section_title = self._find_section_for_chunk(chunk, sections)
        
        # Extract document-level arbitration signals
        doc_signals = self.extract_arbitration_signals(cleaned_text)
        
        metadata = {
            'total_chunks': len(chunks),
            'arbitration_relevant_chunks': sum(1 for c in chunks if c.is_arbitration_relevant),
            'sections_found': len(sections),
            'document_signals': doc_signals,
            'processed_length': len(cleaned_text)
        }
        
        logger.info(f"Processed document: {len(chunks)} chunks, {metadata['arbitration_relevant_chunks']} arbitration-relevant")
        
        return chunks, metadata
    
    def _find_section_for_chunk(self, chunk: TextChunk, sections: List[Dict[str, Any]]) -> str:
        """
        Find the section title for a given chunk
        """
        relevant_section = ""
        
        for section in sections:
            if section['start_char'] <= chunk.start_char:
                relevant_section = section['title']
            else:
                break
        
        return relevant_section