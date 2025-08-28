"""Document section detection and structure analysis."""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pdfplumber
import PyPDF2
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section in a document."""
    title: str
    content: str
    start_page: int
    end_page: int
    section_number: Optional[str] = None
    subsections: List['DocumentSection'] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'content': self.content[:500],  # Truncate for summary
            'start_page': self.start_page,
            'end_page': self.end_page,
            'section_number': self.section_number,
            'confidence': self.confidence,
            'subsections': [s.to_dict() for s in self.subsections]
        }

class DocumentStructureAnalyzer:
    """Analyzes document structure to identify sections."""
    
    def __init__(self):
        """Initialize document structure analyzer."""
        self.section_patterns = self._load_section_patterns()
        self.arbitration_indicators = [
            'arbitration', 'dispute resolution', 'binding arbitration',
            'class action waiver', 'dispute', 'claims', 'legal proceedings',
            'jury trial', 'mediation', 'litigation', 'court proceedings',
            'mandatory arbitration', 'alternative dispute resolution'
        ]
        
    def _load_section_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting section headers."""
        return [
            # Numbered sections (e.g., "1.", "1.1", "1.1.1")
            re.compile(r'^(?P<num>\d+\.?\d*\.?\d*)\s+(?P<title>[A-Z][A-Za-z\s,]+)', re.MULTILINE),
            # Lettered sections (e.g., "A.", "B.")
            re.compile(r'^(?P<letter>[A-Z]\.)\s+(?P<title>[A-Z][A-Za-z\s,]+)', re.MULTILINE),
            # Roman numerals (e.g., "I.", "II.", "III.")
            re.compile(r'^(?P<roman>[IVX]+\.)\s+(?P<title>[A-Z][A-Za-z\s,]+)', re.MULTILINE),
            # All caps headers
            re.compile(r'^(?P<title>[A-Z][A-Z\s]{3,})$', re.MULTILINE),
            # Markdown-style headers
            re.compile(r'^#{1,6}\s+(?P<title>.+)$', re.MULTILINE),
            # Bold headers in common formats
            re.compile(r'^\*\*(?P<title>[A-Z][A-Za-z\s,]+)\*\*', re.MULTILINE),
            # Section with colon
            re.compile(r'^(?P<title>[A-Z][A-Za-z\s,]+):(?:\s|$)', re.MULTILINE),
        ]
    
    def analyze_document(self, filepath: str) -> List[DocumentSection]:
        """
        Analyze document structure and identify sections.
        
        Args:
            filepath: Path to document (PDF, TXT, etc.)
            
        Returns:
            List of document sections with hierarchy
        """
        try:
            # Extract text based on file type
            if filepath.endswith('.pdf'):
                text, page_map = self._extract_pdf_text(filepath)
            elif filepath.endswith(('.txt', '.text')):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                page_map = {i: 1 for i in range(len(text))}
            elif filepath.endswith(('.doc', '.docx')):
                text, page_map = self._extract_docx_text(filepath)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            
            # Detect sections
            sections = self._detect_sections(text, page_map)
            
            # Build hierarchy
            sections = self._build_hierarchy(sections)
            
            # Score sections for arbitration likelihood
            sections = self._score_sections(sections)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return []
    
    def _extract_pdf_text(self, filepath: str) -> Tuple[str, Dict]:
        """Extract text from PDF with page mapping."""
        text_parts = []
        page_map = {}
        char_count = 0
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    
                    # Map character positions to pages
                    for i in range(len(page_text)):
                        page_map[char_count + i] = page_num
                    char_count += len(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            # Fallback to PyPDF2
            try:
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                        
                        for i in range(len(page_text)):
                            page_map[char_count + i] = page_num
                        char_count += len(page_text)
            except Exception as e2:
                logger.error(f"Both PDF extractors failed: {e2}")
                raise
        
        return '\n'.join(text_parts), page_map
    
    def _extract_docx_text(self, filepath: str) -> Tuple[str, Dict]:
        """Extract text from DOCX with page mapping."""
        try:
            import docx
            doc = docx.Document(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs])
            # Simple page mapping for DOCX (approximate)
            page_map = {i: (i // 3000) + 1 for i in range(len(text))}
            return text, page_map
        except ImportError:
            logger.error("python-docx not installed")
            raise ValueError("DOCX support requires python-docx package")
    
    def _detect_sections(self, text: str, page_map: Dict) -> List[DocumentSection]:
        """Detect sections in text using patterns."""
        sections = []
        matched_positions = set()
        
        for pattern in self.section_patterns:
            for match in pattern.finditer(text):
                # Skip if already matched at this position
                if match.start() in matched_positions:
                    continue
                matched_positions.add(match.start())
                
                # Extract title and section number
                groups = match.groupdict()
                title = groups.get('title', '').strip()
                if not title:
                    continue
                    
                section_num = (
                    groups.get('num') or 
                    groups.get('letter') or 
                    groups.get('roman') or 
                    None
                )
                
                # Find section content (until next section or end)
                start_idx = match.end()
                
                # Find next section
                next_section_idx = len(text)
                for next_pattern in self.section_patterns:
                    next_match = next_pattern.search(text, start_idx)
                    if next_match:
                        next_section_idx = min(next_section_idx, next_match.start())
                
                content = text[start_idx:next_section_idx].strip()
                
                # Get page numbers
                start_page = page_map.get(match.start(), 1)
                end_page = page_map.get(next_section_idx - 1, 1)
                
                sections.append(DocumentSection(
                    title=title,
                    content=content,
                    start_page=start_page,
                    end_page=end_page,
                    section_number=section_num,
                    subsections=[],
                    confidence=0.0
                ))
        
        # Sort sections by position
        sections.sort(key=lambda s: s.start_page)
        
        return sections
    
    def _build_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build hierarchical structure from flat sections."""
        if not sections:
            return sections
            
        root_sections = []
        section_stack = []
        
        for section in sections:
            if not section.section_number:
                # No number, add to root
                root_sections.append(section)
                section_stack = [section]
            else:
                # Determine depth based on section number
                depth = section.section_number.count('.')
                
                # Find appropriate parent
                while len(section_stack) > depth:
                    section_stack.pop()
                
                if section_stack:
                    # Add as subsection to last item in stack
                    section_stack[-1].subsections.append(section)
                else:
                    # Add to root
                    root_sections.append(section)
                
                section_stack.append(section)
        
        return root_sections
    
    def _score_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Score sections for likelihood of containing arbitration clauses."""
        for section in sections:
            score = 0.0
            title_lower = section.title.lower()
            content_lower = section.content.lower()[:1000]  # Check first 1000 chars
            
            # Check title
            for indicator in self.arbitration_indicators:
                if indicator in title_lower:
                    score += 0.5
                if indicator in content_lower:
                    score += 0.2
            
            # Check for legal section indicators
            legal_terms = ['terms', 'conditions', 'agreement', 'legal', 'contract', 'policy']
            if any(term in title_lower for term in legal_terms):
                score += 0.3
            
            # Specific high-confidence patterns
            if 'dispute resolution' in title_lower:
                score += 0.7
            if 'arbitration' in title_lower:
                score += 0.8
            
            section.confidence = min(1.0, score)
            
            # Recursively score subsections
            if section.subsections:
                section.subsections = self._score_sections(section.subsections)
        
        return sections
    
    def find_arbitration_sections(self, filepath: str, threshold: float = 0.5) -> List[DocumentSection]:
        """
        Find sections likely to contain arbitration clauses.
        
        Args:
            filepath: Path to document
            threshold: Confidence threshold
            
        Returns:
            List of sections likely containing arbitration clauses
        """
        all_sections = self.analyze_document(filepath)
        arbitration_sections = []
        
        def collect_relevant_sections(sections: List[DocumentSection]):
            for section in sections:
                if section.confidence >= threshold:
                    arbitration_sections.append(section)
                if section.subsections:
                    collect_relevant_sections(section.subsections)
        
        collect_relevant_sections(all_sections)
        
        # Sort by confidence
        arbitration_sections.sort(key=lambda x: x.confidence, reverse=True)
        
        return arbitration_sections
    
    def extract_full_text(self, filepath: str) -> str:
        """Extract full text from document."""
        if filepath.endswith('.pdf'):
            text, _ = self._extract_pdf_text(filepath)
        elif filepath.endswith(('.txt', '.text')):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif filepath.endswith(('.doc', '.docx')):
            text, _ = self._extract_docx_text(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        return text