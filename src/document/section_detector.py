import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pdfplumber
import PyPDF2

@dataclass
class DocumentSection:
    """Represents a section in a document"""
    title: str
    content: str
    start_page: int
    end_page: int
    section_number: Optional[str]
    subsections: List['DocumentSection']
    confidence: float

class DocumentStructureAnalyzer:
    def __init__(self):
        """Initialize document structure analyzer"""
        self.section_patterns = self._load_section_patterns()
        self.arbitration_indicators = [
            'arbitration', 'dispute resolution', 'binding arbitration',
            'class action waiver', 'dispute', 'claims', 'legal proceedings'
        ]
        
    def _load_section_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting section headers"""
        return [
            # Numbered sections
            re.compile(r'^(?P<num>\d+\.?\d*\.?\d*)\s+(?P<title>[A-Z][A-Za-z\s]+)', re.MULTILINE),
            # Lettered sections
            re.compile(r'^(?P<letter>[A-Z]\.)\s+(?P<title>[A-Z][A-Za-z\s]+)', re.MULTILINE),
            # All caps headers
            re.compile(r'^(?P<title>[A-Z][A-Z\s]{3,})$', re.MULTILINE),
            # Markdown-style headers
            re.compile(r'^#{1,6}\s+(?P<title>.+)$', re.MULTILINE),
        ]
    
    def analyze_document(self, filepath: str) -> List[DocumentSection]:
        """
        Analyze document structure and identify sections
        
        Args:
            filepath: Path to document (PDF, TXT, etc.)
            
        Returns:
            List of document sections with hierarchy
        """
        # Extract text based on file type
        if filepath.endswith('.pdf'):
            text, page_map = self._extract_pdf_text(filepath)
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            page_map = {i: 1 for i in range(len(text.split('\n')))}
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
        
        # Detect sections
        sections = self._detect_sections(text, page_map)
        
        # Build hierarchy
        sections = self._build_hierarchy(sections)
        
        # Score sections for arbitration likelihood
        sections = self._score_sections(sections)
        
        return sections
    
    def _extract_pdf_text(self, filepath: str) -> Tuple[str, Dict]:
        """Extract text from PDF with page mapping"""
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
            print(f"Error with pdfplumber: {e}")
            # Fallback to PyPDF2
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    
                    # Map character positions to pages
                    for i in range(len(page_text)):
                        page_map[char_count + i] = page_num
                    char_count += len(page_text)
        
        return '\n'.join(text_parts), page_map
    
    def _detect_sections(self, text: str, page_map: Dict) -> List[DocumentSection]:
        """Detect sections in text using patterns"""
        sections = []
        
        for pattern in self.section_patterns:
            for match in pattern.finditer(text):
                title = match.group('title').strip() if 'title' in match.groupdict() else ""
                section_num = match.group('num') if 'num' in match.groupdict() else None
                
                # Find section content (until next section or end)
                start_idx = match.end()
                
                # Find next section
                next_match = pattern.search(text, start_idx)
                end_idx = next_match.start() if next_match else len(text)
                
                content = text[start_idx:end_idx].strip()
                
                # Get page numbers
                start_page = page_map.get(match.start(), 1)
                end_page = page_map.get(end_idx - 1, 1)
                
                sections.append(DocumentSection(
                    title=title,
                    content=content,
                    start_page=start_page,
                    end_page=end_page,
                    section_number=section_num,
                    subsections=[],
                    confidence=0.0
                ))
        
        return sections
    
    def _build_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build hierarchical structure from flat sections"""
        # Simple hierarchy based on section numbers
        root_sections = []
        current_parent = None
        
        for section in sections:
            if section.section_number:
                depth = section.section_number.count('.')
                if depth == 0:
                    root_sections.append(section)
                    current_parent = section
                elif current_parent and depth > 0:
                    current_parent.subsections.append(section)
            else:
                root_sections.append(section)
        
        return root_sections
    
    def _score_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Score sections for likelihood of containing arbitration clauses"""
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
            if any(term in title_lower for term in ['terms', 'conditions', 'agreement', 'legal']):
                score += 0.3
            
            section.confidence = min(1.0, score)
            
            # Recursively score subsections
            if section.subsections:
                section.subsections = self._score_sections(section.subsections)
        
        return sections
    
    def find_arbitration_sections(self, filepath: str, threshold: float = 0.5) -> List[DocumentSection]:
        """
        Find sections likely to contain arbitration clauses
        
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

