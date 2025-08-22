"""
Legal Citation Parser with Bluebook Formatting Support

This module provides comprehensive legal citation parsing, validation, and formatting
according to various citation standards including Bluebook, ALWD, and international formats.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
import spacy
from transformers import pipeline
import json


class CitationType(Enum):
    """Types of legal citations."""
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    BOOK = "book"
    JOURNAL_ARTICLE = "journal_article"
    LAW_REVIEW = "law_review"
    NEWSPAPER = "newspaper"
    WEBSITE = "website"
    TREATY = "treaty"
    UNPUBLISHED = "unpublished"


class CitationFormat(Enum):
    """Citation formatting standards."""
    BLUEBOOK = "bluebook"
    ALWD = "alwd"
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    INTERNATIONAL = "international"


class Jurisdiction(Enum):
    """Legal jurisdictions for citation formatting."""
    US_FEDERAL = "us_federal"
    US_STATE = "us_state"
    UK = "uk"
    CANADA = "canada"
    AUSTRALIA = "australia"
    EU = "eu"
    INTERNATIONAL = "international"


@dataclass
class CitationComponents:
    """Core components of a legal citation."""
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    case_name: Optional[str] = None
    statute_title: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    parallel_citations: List[str] = None
    
    def __post_init__(self):
        if self.parallel_citations is None:
            self.parallel_citations = []


@dataclass
class ParsedCitation:
    """Fully parsed legal citation with metadata."""
    original_text: str
    citation_type: CitationType
    components: CitationComponents
    jurisdiction: Optional[Jurisdiction] = None
    is_valid: bool = True
    validation_errors: List[str] = None
    confidence_score: float = 1.0
    suggested_corrections: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.suggested_corrections is None:
            self.suggested_corrections = []


@dataclass
class FormattedCitation:
    """Citation formatted according to specific standard."""
    formatted_text: str
    citation_format: CitationFormat
    short_form: str
    id_form: str
    supra_form: Optional[str] = None
    footnote_format: Optional[str] = None


@dataclass
class CitationValidation:
    """Results of citation validation."""
    is_valid: bool
    validation_score: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    verified_online: bool = False
    last_verified: Optional[datetime] = None


class ReporterDatabase:
    """Database of legal reporters and their abbreviations."""
    
    def __init__(self):
        self.reporters = self._load_reporter_database()
        self.court_abbreviations = self._load_court_abbreviations()
        
    def _load_reporter_database(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive reporter database."""
        # In production, this would load from a comprehensive database
        return {
            # Federal Courts
            "U.S.": {
                "full_name": "United States Reports",
                "court": "Supreme Court",
                "jurisdiction": "federal",
                "years": "1790-present",
                "type": "official"
            },
            "S. Ct.": {
                "full_name": "Supreme Court Reporter",
                "court": "Supreme Court", 
                "jurisdiction": "federal",
                "years": "1882-present",
                "type": "commercial"
            },
            "F.3d": {
                "full_name": "Federal Reporter, Third Series",
                "court": "Courts of Appeals",
                "jurisdiction": "federal",
                "years": "1993-present",
                "type": "official"
            },
            "F.2d": {
                "full_name": "Federal Reporter, Second Series",
                "court": "Courts of Appeals",
                "jurisdiction": "federal", 
                "years": "1924-1993",
                "type": "official"
            },
            "F. Supp. 3d": {
                "full_name": "Federal Supplement, Third Series",
                "court": "District Courts",
                "jurisdiction": "federal",
                "years": "2014-present",
                "type": "official"
            },
            
            # State Courts (selected examples)
            "Cal. 4th": {
                "full_name": "California Reports, Fourth Series",
                "court": "California Supreme Court",
                "jurisdiction": "state",
                "state": "California",
                "years": "1991-present"
            },
            "N.Y.2d": {
                "full_name": "New York Reports, Second Series", 
                "court": "New York Court of Appeals",
                "jurisdiction": "state",
                "state": "New York",
                "years": "1956-present"
            },
            
            # Specialized Courts
            "Fed. Cl.": {
                "full_name": "Federal Claims Reporter",
                "court": "Court of Federal Claims",
                "jurisdiction": "federal",
                "specialization": "claims"
            },
            "B.R.": {
                "full_name": "Bankruptcy Reporter",
                "court": "Bankruptcy Courts",
                "jurisdiction": "federal",
                "specialization": "bankruptcy"
            },
            
            # Administrative
            "A.L.R.": {
                "full_name": "American Law Reports",
                "type": "annotation",
                "jurisdiction": "national"
            }
        }
    
    def _load_court_abbreviations(self) -> Dict[str, str]:
        """Load court abbreviations and their full names."""
        return {
            "S.D.N.Y.": "Southern District of New York",
            "E.D. Pa.": "Eastern District of Pennsylvania", 
            "9th Cir.": "Ninth Circuit Court of Appeals",
            "2d Cir.": "Second Circuit Court of Appeals",
            "D.C. Cir.": "District of Columbia Circuit",
            "Fed. Cir.": "Federal Circuit Court of Appeals",
            "Cal.": "California Supreme Court",
            "N.Y.": "New York Court of Appeals",
            "Fla.": "Florida Supreme Court",
            "Tex.": "Texas Supreme Court"
        }
    
    def get_reporter_info(self, reporter_abbrev: str) -> Optional[Dict[str, Any]]:
        """Get information about a reporter abbreviation."""
        return self.reporters.get(reporter_abbrev)
    
    def is_valid_reporter(self, reporter_abbrev: str) -> bool:
        """Check if reporter abbreviation is valid."""
        return reporter_abbrev in self.reporters
    
    def get_court_name(self, court_abbrev: str) -> Optional[str]:
        """Get full court name from abbreviation."""
        return self.court_abbreviations.get(court_abbrev)


class CitationPatterns:
    """Regular expression patterns for different citation types."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[CitationType, List[re.Pattern]]:
        """Compile regex patterns for citation recognition."""
        patterns = {}
        
        # Case citations
        patterns[CitationType.CASE] = [
            # Standard case citation: Volume Reporter Page (Court Year)
            re.compile(r'(\d+)\s+([A-Z][A-Za-z.\s]+[A-Z]\.?)\s+(\d+)(?:,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)', re.IGNORECASE),
            
            # Case with parallel citations
            re.compile(r'(\d+)\s+([A-Z][A-Za-z.\s]+[A-Z]\.?)\s+(\d+)(?:,\s*(\d+))?(?:,\s*(\d+)\s+([A-Z][A-Za-z.\s]+[A-Z]\.?)\s+(\d+))?\s*\(([^)]+)\s+(\d{4})\)', re.IGNORECASE),
            
            # Case name with citation
            re.compile(r'([A-Z][A-Za-z\s&,.]+)\s+v\.?\s+([A-Z][A-Za-z\s&,.]+),\s*(\d+)\s+([A-Z][A-Za-z.\s]+[A-Z]\.?)\s+(\d+)(?:,\s*(\d+))?\s*\(([^)]+)\s+(\d{4})\)', re.IGNORECASE),
        ]
        
        # Statute citations
        patterns[CitationType.STATUTE] = [
            # U.S.C. citations
            re.compile(r'(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+(?:\([a-z0-9]+\))*)', re.IGNORECASE),
            
            # State statute citations
            re.compile(r'([A-Z][A-Za-z.\s]+)\s*§?\s*(\d+(?:[.-]\d+)*(?:\([a-z0-9]+\))*)', re.IGNORECASE),
            
            # Code citations with year
            re.compile(r'([A-Z][A-Za-z.\s]+Code)\s*§?\s*(\d+(?:[.-]\d+)*(?:\([a-z0-9]+\))*)\s*\((\d{4})\)', re.IGNORECASE),
        ]
        
        # Constitutional citations
        patterns[CitationType.CONSTITUTIONAL] = [
            # U.S. Constitution
            re.compile(r'U\.?S\.?\s*Const\.?\s*art\.?\s*([IVX]+)(?:,\s*§\s*(\d+))?(?:,\s*cl\.?\s*(\d+))?', re.IGNORECASE),
            
            # U.S. Constitution amendments
            re.compile(r'U\.?S\.?\s*Const\.?\s*amend\.?\s*([IVX]+)', re.IGNORECASE),
            
            # State constitutions
            re.compile(r'([A-Z][A-Za-z.\s]+)\s*Const\.?\s*art\.?\s*([IVX]+)(?:,\s*§\s*(\d+))?', re.IGNORECASE),
        ]
        
        # Regulation citations
        patterns[CitationType.REGULATION] = [
            # C.F.R. citations
            re.compile(r'(\d+)\s+C\.?F\.?R\.?\s*§?\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
            
            # Federal Register
            re.compile(r'(\d+)\s+Fed\.?\s*Reg\.?\s*(\d+)(?:,\s*(\d+))?\s*\(([^)]+)\s*(\d{4})\)', re.IGNORECASE),
        ]
        
        # Law review citations
        patterns[CitationType.LAW_REVIEW] = [
            # Standard law review format
            re.compile(r'([A-Z][A-Za-z\s.,]+),\s*([^,]+),\s*(\d+)\s+([A-Z][A-Za-z.\s]+L\.?\s*Rev\.?)\s*(\d+)(?:,\s*(\d+))?\s*\((\d{4})\)', re.IGNORECASE),
            
            # Shorter law review format
            re.compile(r'(\d+)\s+([A-Z][A-Za-z.\s]+L\.?\s*Rev\.?)\s*(\d+)(?:,\s*(\d+))?\s*\((\d{4})\)', re.IGNORECASE),
        ]
        
        return patterns


class CitationParser:
    """Advanced legal citation parser with Bluebook formatting support."""
    
    def __init__(self):
        self.reporter_db = ReporterDatabase()
        self.patterns = CitationPatterns()
        self.nlp = spacy.load("en_core_web_sm")
        
        # Citation validation services
        self.validation_apis = {
            'google_scholar': 'https://scholar.google.com',
            'westlaw': 'https://1.next.westlaw.com',
            'lexis': 'https://advance.lexis.com'
        }
        
        self.logger = logging.getLogger(__name__)
    
    def parse_citations(self, text: str) -> List[ParsedCitation]:
        """Parse all citations found in text."""
        try:
            citations = []
            
            # First pass: find all potential citations using patterns
            potential_citations = self._find_potential_citations(text)
            
            # Second pass: validate and parse each citation
            for citation_text, citation_type in potential_citations:
                parsed = self._parse_single_citation(citation_text, citation_type)
                if parsed:
                    citations.append(parsed)
            
            # Third pass: identify and link parallel citations
            citations = self._link_parallel_citations(citations)
            
            self.logger.info(f"Successfully parsed {len(citations)} citations")
            return citations
            
        except Exception as e:
            self.logger.error(f"Citation parsing failed: {str(e)}")
            raise
    
    def _find_potential_citations(self, text: str) -> List[Tuple[str, CitationType]]:
        """Find potential citations in text using pattern matching."""
        potential_citations = []
        
        for citation_type, pattern_list in self.patterns.patterns.items():
            for pattern in pattern_list:
                matches = pattern.finditer(text)
                for match in matches:
                    citation_text = match.group(0)
                    potential_citations.append((citation_text, citation_type))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation_text, citation_type in potential_citations:
            if citation_text not in seen:
                seen.add(citation_text)
                unique_citations.append((citation_text, citation_type))
        
        return unique_citations
    
    def _parse_single_citation(self, citation_text: str, citation_type: CitationType) -> Optional[ParsedCitation]:
        """Parse a single citation based on its type."""
        try:
            if citation_type == CitationType.CASE:
                return self._parse_case_citation(citation_text)
            elif citation_type == CitationType.STATUTE:
                return self._parse_statute_citation(citation_text)
            elif citation_type == CitationType.CONSTITUTIONAL:
                return self._parse_constitutional_citation(citation_text)
            elif citation_type == CitationType.REGULATION:
                return self._parse_regulation_citation(citation_text)
            elif citation_type == CitationType.LAW_REVIEW:
                return self._parse_law_review_citation(citation_text)
            else:
                return self._parse_generic_citation(citation_text, citation_type)
                
        except Exception as e:
            self.logger.warning(f"Failed to parse citation '{citation_text}': {str(e)}")
            return None
    
    def _parse_case_citation(self, citation_text: str) -> Optional[ParsedCitation]:
        """Parse case citation with case name, volume, reporter, page, court, and year."""
        # Try different case citation patterns
        for pattern in self.patterns.patterns[CitationType.CASE]:
            match = pattern.match(citation_text)
            if match:
                groups = match.groups()
                
                # Extract components based on pattern structure
                if len(groups) >= 6:  # Standard case citation
                    components = CitationComponents(
                        volume=groups[0],
                        reporter=groups[1].strip(),
                        page=groups[2],
                        court=groups[4] if len(groups) > 4 else None,
                        year=int(groups[5]) if len(groups) > 5 else None
                    )
                    
                    # Validate reporter
                    if not self.reporter_db.is_valid_reporter(components.reporter):
                        # Try to find similar reporter
                        similar_reporter = self._find_similar_reporter(components.reporter)
                        if similar_reporter:
                            components.reporter = similar_reporter
                    
                    return ParsedCitation(
                        original_text=citation_text,
                        citation_type=CitationType.CASE,
                        components=components,
                        confidence_score=self._calculate_confidence_score(components)
                    )
        
        return None
    
    def _parse_statute_citation(self, citation_text: str) -> Optional[ParsedCitation]:
        """Parse statutory citation."""
        for pattern in self.patterns.patterns[CitationType.STATUTE]:
            match = pattern.match(citation_text)
            if match:
                groups = match.groups()
                
                components = CitationComponents(
                    statute_title=groups[0] if groups[0] else None,
                    section=groups[1] if len(groups) > 1 else None
                )
                
                # Handle U.S.C. citations specially
                if "U.S.C" in citation_text.upper():
                    components.statute_title = f"{groups[0]} U.S.C."
                
                return ParsedCitation(
                    original_text=citation_text,
                    citation_type=CitationType.STATUTE,
                    components=components,
                    confidence_score=self._calculate_confidence_score(components)
                )
        
        return None
    
    def _parse_constitutional_citation(self, citation_text: str) -> Optional[ParsedCitation]:
        """Parse constitutional citation."""
        for pattern in self.patterns.patterns[CitationType.CONSTITUTIONAL]:
            match = pattern.match(citation_text)
            if match:
                groups = match.groups()
                
                components = CitationComponents(
                    statute_title="U.S. Constitution" if "U.S." in citation_text else groups[0],
                    section=groups[1] if len(groups) > 1 else None
                )
                
                return ParsedCitation(
                    original_text=citation_text,
                    citation_type=CitationType.CONSTITUTIONAL,
                    components=components,
                    confidence_score=self._calculate_confidence_score(components)
                )
        
        return None
    
    def _parse_regulation_citation(self, citation_text: str) -> Optional[ParsedCitation]:
        """Parse regulation citation."""
        for pattern in self.patterns.patterns[CitationType.REGULATION]:
            match = pattern.match(citation_text)
            if match:
                groups = match.groups()
                
                components = CitationComponents(
                    volume=groups[0],
                    reporter="C.F.R." if "C.F.R" in citation_text else "Fed. Reg.",
                    section=groups[1] if len(groups) > 1 else None,
                    year=int(groups[-1]) if groups[-1] and groups[-1].isdigit() else None
                )
                
                return ParsedCitation(
                    original_text=citation_text,
                    citation_type=CitationType.REGULATION,
                    components=components,
                    confidence_score=self._calculate_confidence_score(components)
                )
        
        return None
    
    def _parse_law_review_citation(self, citation_text: str) -> Optional[ParsedCitation]:
        """Parse law review citation."""
        for pattern in self.patterns.patterns[CitationType.LAW_REVIEW]:
            match = pattern.match(citation_text)
            if match:
                groups = match.groups()
                
                components = CitationComponents(
                    volume=groups[2] if len(groups) > 2 else groups[0],
                    reporter=groups[3] if len(groups) > 3 else groups[1],
                    page=groups[4] if len(groups) > 4 else groups[2],
                    year=int(groups[-1]) if groups[-1] and groups[-1].isdigit() else None
                )
                
                return ParsedCitation(
                    original_text=citation_text,
                    citation_type=CitationType.LAW_REVIEW,
                    components=components,
                    confidence_score=self._calculate_confidence_score(components)
                )
        
        return None
    
    def _parse_generic_citation(self, citation_text: str, citation_type: CitationType) -> ParsedCitation:
        """Parse generic citation when specific parsing fails."""
        return ParsedCitation(
            original_text=citation_text,
            citation_type=citation_type,
            components=CitationComponents(),
            confidence_score=0.5,
            validation_errors=["Unable to parse citation components"]
        )
    
    def _find_similar_reporter(self, reporter: str) -> Optional[str]:
        """Find similar reporter abbreviation using fuzzy matching."""
        from difflib import get_close_matches
        
        close_matches = get_close_matches(
            reporter, 
            self.reporter_db.reporters.keys(), 
            n=1, 
            cutoff=0.8
        )
        
        return close_matches[0] if close_matches else None
    
    def _calculate_confidence_score(self, components: CitationComponents) -> float:
        """Calculate confidence score for parsed citation."""
        score = 0.0
        total_components = 0
        
        # Check presence of key components
        if components.volume:
            score += 0.2
        if components.reporter:
            score += 0.3
        if components.page:
            score += 0.2
        if components.year:
            score += 0.2
        if components.court:
            score += 0.1
        
        # Validate reporter if present
        if components.reporter and self.reporter_db.is_valid_reporter(components.reporter):
            score += 0.2
        
        return min(score, 1.0)
    
    def _link_parallel_citations(self, citations: List[ParsedCitation]) -> List[ParsedCitation]:
        """Identify and link parallel citations."""
        # Group citations by year and case characteristics
        grouped_citations = defaultdict(list)
        
        for citation in citations:
            if citation.citation_type == CitationType.CASE and citation.components.year:
                key = (citation.components.year, citation.components.case_name)
                grouped_citations[key].append(citation)
        
        # Link citations in the same group as parallels
        for group in grouped_citations.values():
            if len(group) > 1:
                # Find the "primary" citation (typically official reporter)
                primary_idx = 0
                for i, citation in enumerate(group):
                    reporter_info = self.reporter_db.get_reporter_info(citation.components.reporter)
                    if reporter_info and reporter_info.get('type') == 'official':
                        primary_idx = i
                        break
                
                # Add parallel citations to each citation in the group
                for i, citation in enumerate(group):
                    parallels = [c.original_text for j, c in enumerate(group) if j != i]
                    citation.components.parallel_citations.extend(parallels)
        
        return citations
    
    def format_citation(
        self, 
        citation: ParsedCitation, 
        format_type: CitationFormat = CitationFormat.BLUEBOOK,
        short_form: bool = False
    ) -> FormattedCitation:
        """Format citation according to specified standard."""
        try:
            if format_type == CitationFormat.BLUEBOOK:
                return self._format_bluebook(citation, short_form)
            elif format_type == CitationFormat.ALWD:
                return self._format_alwd(citation, short_form)
            elif format_type == CitationFormat.APA:
                return self._format_apa(citation, short_form)
            else:
                return self._format_bluebook(citation, short_form)  # Default to Bluebook
                
        except Exception as e:
            self.logger.error(f"Citation formatting failed: {str(e)}")
            raise
    
    def _format_bluebook(self, citation: ParsedCitation, short_form: bool = False) -> FormattedCitation:
        """Format citation according to Bluebook rules."""
        c = citation.components
        
        if citation.citation_type == CitationType.CASE:
            if short_form:
                # Short form: Case Name, Volume Reporter Page
                formatted = f"{c.case_name or 'Case'}, {c.volume} {c.reporter} {c.page}"
                short = formatted
                id_form = f"Id."
            else:
                # Full form: Case Name, Volume Reporter Page (Court Year)
                court_year = f"({c.court} {c.year})" if c.court and c.year else f"({c.year})"
                formatted = f"{c.case_name or 'Case'}, {c.volume} {c.reporter} {c.page} {court_year}"
                short = f"{c.case_name or 'Case'}, {c.volume} {c.reporter} at {c.page}"
                id_form = f"Id. at {c.page}"
        
        elif citation.citation_type == CitationType.STATUTE:
            # Statute: Title § Section (Year)
            formatted = f"{c.statute_title} § {c.section}"
            if c.year:
                formatted += f" ({c.year})"
            short = formatted
            id_form = "Id."
        
        elif citation.citation_type == CitationType.CONSTITUTIONAL:
            # Constitutional: U.S. Const. art. I, § 8
            formatted = f"{c.statute_title}"
            if c.section:
                formatted += f" § {c.section}"
            short = formatted
            id_form = "Id."
        
        elif citation.citation_type == CitationType.REGULATION:
            # Regulation: Volume C.F.R. § Section (Year)
            formatted = f"{c.volume} {c.reporter} § {c.section}"
            if c.year:
                formatted += f" ({c.year})"
            short = formatted
            id_form = "Id."
        
        else:
            formatted = citation.original_text
            short = formatted
            id_form = "Id."
        
        return FormattedCitation(
            formatted_text=formatted,
            citation_format=CitationFormat.BLUEBOOK,
            short_form=short,
            id_form=id_form,
            supra_form=f"{c.case_name or 'supra'}, supra note" if citation.citation_type == CitationType.CASE else None
        )
    
    def _format_alwd(self, citation: ParsedCitation, short_form: bool = False) -> FormattedCitation:
        """Format citation according to ALWD rules."""
        # ALWD is similar to Bluebook with some differences
        bluebook_format = self._format_bluebook(citation, short_form)
        
        # Apply ALWD-specific rules
        formatted_text = bluebook_format.formatted_text
        
        # ALWD uses different abbreviations for some reporters
        alwd_replacements = {
            "F.3d": "F.3d",
            "F.2d": "F.2d", 
            "F. Supp.": "F. Supp.",
            "S. Ct.": "S. Ct."
        }
        
        for bluebook_abbrev, alwd_abbrev in alwd_replacements.items():
            formatted_text = formatted_text.replace(bluebook_abbrev, alwd_abbrev)
        
        return FormattedCitation(
            formatted_text=formatted_text,
            citation_format=CitationFormat.ALWD,
            short_form=bluebook_format.short_form,
            id_form=bluebook_format.id_form
        )
    
    def _format_apa(self, citation: ParsedCitation, short_form: bool = False) -> FormattedCitation:
        """Format citation according to APA rules."""
        c = citation.components
        
        if citation.citation_type == CitationType.CASE:
            # APA: Case Name, Volume Reporter Page (Court Year)
            formatted = f"{c.case_name}, {c.volume} {c.reporter} {c.page} ({c.court} {c.year})"
            short = f"{c.case_name} ({c.year})"
            
        elif citation.citation_type == CitationType.STATUTE:
            # APA: Title § Section (Year)
            formatted = f"{c.statute_title} § {c.section} ({c.year})"
            short = formatted
        
        else:
            formatted = citation.original_text
            short = formatted
        
        return FormattedCitation(
            formatted_text=formatted,
            citation_format=CitationFormat.APA,
            short_form=short,
            id_form=formatted
        )
    
    async def validate_citation(self, citation: ParsedCitation) -> CitationValidation:
        """Validate citation accuracy and existence."""
        try:
            validation = CitationValidation(
                is_valid=True,
                validation_score=1.0,
                errors=[],
                warnings=[],
                suggestions=[]
            )
            
            # Basic validation checks
            errors = []
            warnings = []
            suggestions = []
            
            # Check required components
            if citation.citation_type == CitationType.CASE:
                if not citation.components.volume:
                    errors.append("Missing volume number")
                if not citation.components.reporter:
                    errors.append("Missing reporter abbreviation")
                if not citation.components.page:
                    errors.append("Missing page number")
                if not citation.components.year:
                    warnings.append("Missing year")
            
            # Validate reporter abbreviation
            if citation.components.reporter:
                if not self.reporter_db.is_valid_reporter(citation.components.reporter):
                    similar = self._find_similar_reporter(citation.components.reporter)
                    if similar:
                        suggestions.append(f"Did you mean '{similar}'?")
                    else:
                        errors.append(f"Unknown reporter abbreviation: {citation.components.reporter}")
            
            # Check year reasonableness
            if citation.components.year:
                current_year = datetime.now().year
                if citation.components.year > current_year:
                    errors.append("Year cannot be in the future")
                elif citation.components.year < 1600:
                    warnings.append("Very old year, please verify")
            
            # Online verification (if enabled)
            try:
                online_verified = await self._verify_citation_online(citation)
                validation.verified_online = online_verified
                validation.last_verified = datetime.now()
            except Exception as e:
                warnings.append(f"Could not verify online: {str(e)}")
            
            # Calculate validation score
            validation_score = 1.0
            validation_score -= len(errors) * 0.3
            validation_score -= len(warnings) * 0.1
            validation_score = max(0.0, validation_score)
            
            validation.is_valid = len(errors) == 0
            validation.validation_score = validation_score
            validation.errors = errors
            validation.warnings = warnings
            validation.suggestions = suggestions
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Citation validation failed: {str(e)}")
            return CitationValidation(
                is_valid=False,
                validation_score=0.0,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )
    
    async def _verify_citation_online(self, citation: ParsedCitation) -> bool:
        """Attempt to verify citation exists online."""
        # This would implement actual online verification
        # For demo purposes, return True for valid-looking citations
        return (citation.components.volume and 
                citation.components.reporter and 
                citation.components.page)
    
    def extract_citations_from_document(self, document_text: str) -> Dict[str, Any]:
        """Extract and analyze all citations from a document."""
        try:
            # Parse all citations
            citations = self.parse_citations(document_text)
            
            # Analyze citation patterns
            citation_analysis = self._analyze_citation_patterns(citations)
            
            # Group citations by type
            citations_by_type = defaultdict(list)
            for citation in citations:
                citations_by_type[citation.citation_type].append(citation)
            
            # Generate citation table
            citation_table = self._generate_citation_table(citations)
            
            # Check for citation errors
            citation_errors = []
            for citation in citations:
                if citation.validation_errors:
                    citation_errors.extend(citation.validation_errors)
            
            return {
                'total_citations': len(citations),
                'citations_by_type': {k.value: len(v) for k, v in citations_by_type.items()},
                'citation_analysis': citation_analysis,
                'citation_table': citation_table,
                'validation_errors': citation_errors,
                'parsed_citations': [asdict(c) for c in citations]
            }
            
        except Exception as e:
            self.logger.error(f"Document citation extraction failed: {str(e)}")
            raise
    
    def _analyze_citation_patterns(self, citations: List[ParsedCitation]) -> Dict[str, Any]:
        """Analyze patterns in citation usage."""
        if not citations:
            return {}
        
        # Temporal analysis
        years = [c.components.year for c in citations if c.components.year]
        year_analysis = {
            'earliest_year': min(years) if years else None,
            'latest_year': max(years) if years else None,
            'year_range': max(years) - min(years) if len(years) > 1 else 0,
            'avg_year': sum(years) / len(years) if years else None
        }
        
        # Reporter analysis
        reporters = [c.components.reporter for c in citations if c.components.reporter]
        reporter_counts = defaultdict(int)
        for reporter in reporters:
            reporter_counts[reporter] += 1
        
        # Jurisdiction analysis
        jurisdictions = defaultdict(int)
        for citation in citations:
            if citation.components.reporter:
                reporter_info = self.reporter_db.get_reporter_info(citation.components.reporter)
                if reporter_info:
                    jurisdiction = reporter_info.get('jurisdiction', 'unknown')
                    jurisdictions[jurisdiction] += 1
        
        return {
            'temporal_analysis': year_analysis,
            'most_cited_reporters': dict(sorted(reporter_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]),
            'jurisdiction_distribution': dict(jurisdictions),
            'average_confidence': sum(c.confidence_score for c in citations) / len(citations)
        }
    
    def _generate_citation_table(self, citations: List[ParsedCitation]) -> List[Dict[str, Any]]:
        """Generate a citation table for the document."""
        table = []
        
        for i, citation in enumerate(citations, 1):
            table_entry = {
                'number': i,
                'original_text': citation.original_text,
                'type': citation.citation_type.value,
                'formatted_bluebook': self.format_citation(citation, CitationFormat.BLUEBOOK).formatted_text,
                'confidence_score': citation.confidence_score,
                'has_errors': bool(citation.validation_errors)
            }
            table.append(table_entry)
        
        return table
    
    def suggest_citation_improvements(self, citations: List[ParsedCitation]) -> List[Dict[str, Any]]:
        """Suggest improvements for citation formatting and accuracy."""
        suggestions = []
        
        for citation in citations:
            citation_suggestions = []
            
            # Check for common formatting issues
            if citation.citation_type == CitationType.CASE:
                # Suggest parallel citations
                if not citation.components.parallel_citations:
                    citation_suggestions.append("Consider adding parallel citations")
                
                # Check court abbreviation
                if citation.components.court:
                    full_court_name = self.reporter_db.get_court_name(citation.components.court)
                    if not full_court_name:
                        citation_suggestions.append(f"Verify court abbreviation: {citation.components.court}")
            
            # Check confidence score
            if citation.confidence_score < 0.8:
                citation_suggestions.append("Low confidence in parsing - please verify manually")
            
            # Add validation errors as suggestions
            if citation.validation_errors:
                citation_suggestions.extend(citation.validation_errors)
            
            if citation_suggestions:
                suggestions.append({
                    'citation': citation.original_text,
                    'suggestions': citation_suggestions
                })
        
        return suggestions
    
    def batch_format_citations(
        self, 
        citations: List[ParsedCitation], 
        format_type: CitationFormat = CitationFormat.BLUEBOOK
    ) -> List[FormattedCitation]:
        """Format multiple citations in batch."""
        formatted_citations = []
        
        for citation in citations:
            try:
                formatted = self.format_citation(citation, format_type)
                formatted_citations.append(formatted)
            except Exception as e:
                self.logger.warning(f"Failed to format citation '{citation.original_text}': {str(e)}")
                # Add error citation
                formatted_citations.append(FormattedCitation(
                    formatted_text=citation.original_text,
                    citation_format=format_type,
                    short_form=citation.original_text,
                    id_form="Id."
                ))
        
        return formatted_citations