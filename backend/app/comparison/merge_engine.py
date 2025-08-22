"""
Advanced three-way merge engine for document merging.
Handles intelligent conflict resolution and document structure preservation.
"""

import re
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict
import difflib

from .diff_engine import AdvancedDiffEngine, DiffResult, DiffType, DiffLevel
from .semantic_comparison import SemanticComparisonEngine, SemanticDiff, SemanticChangeType


class MergeStrategy(Enum):
    """Different merge strategies available."""
    AUTO = "auto"                    # Automatic resolution when possible
    FAVOR_THEIRS = "favor_theirs"   # Prefer their changes in conflicts
    FAVOR_OURS = "favor_ours"       # Prefer our changes in conflicts  
    MANUAL = "manual"               # Require manual resolution
    SEMANTIC = "semantic"           # Use semantic analysis for resolution
    LEGAL_SAFE = "legal_safe"       # Prioritize legal safety


class ConflictResolution(Enum):
    """Types of conflict resolution."""
    AUTO_RESOLVED = "auto_resolved"
    MANUAL_REQUIRED = "manual_required"
    SEMANTIC_MERGE = "semantic_merge"
    STRUCTURAL_MERGE = "structural_merge"


@dataclass
class MergeConflict:
    """Represents a merge conflict between documents."""
    conflict_id: str
    conflict_type: str
    base_content: str
    ours_content: str
    theirs_content: str
    position: Tuple[int, int]
    context: str
    severity: str  # "low", "medium", "high", "critical"
    auto_resolvable: bool
    suggested_resolution: str
    resolution_rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeChunk:
    """Represents a chunk of content in a merge."""
    chunk_type: str  # "unchanged", "ours", "theirs", "conflict"
    content: str
    base_content: str = ""
    position: Tuple[int, int] = (0, 0)
    conflict: Optional[MergeConflict] = None


@dataclass
class MergeResult:
    """Result of a three-way merge operation."""
    success: bool
    merged_content: str
    conflicts: List[MergeConflict]
    auto_resolved_conflicts: List[MergeConflict]
    merge_strategy_used: MergeStrategy
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentStructureAnalyzer:
    """Analyzes document structure for intelligent merging."""
    
    @staticmethod
    def identify_sections(content: str) -> List[Dict[str, Any]]:
        """
        Identify sections and structure in document content.
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of sections with metadata
        """
        sections = []
        lines = content.split('\n')
        current_section = None
        
        # Patterns for identifying headers/sections
        header_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown_header'),      # Markdown headers
            (r'^(\d+\.)+\s+(.+)$', 'numbered_section'),   # Numbered sections
            (r'^([A-Z][^.]+):$', 'colon_header'),         # Colon headers
            (r'^([A-Z\s]+)$', 'caps_header'),             # All caps headers
        ]
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches any header pattern
            is_header = False
            for pattern, header_type in header_patterns:
                match = re.match(pattern, line)
                if match:
                    # Start new section
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'title': line,
                        'type': header_type,
                        'start_line': line_num,
                        'end_line': line_num,
                        'content': [],
                        'subsections': []
                    }
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_section['content'].append(line)
                current_section['end_line'] = line_num
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    @staticmethod
    def find_section_matches(sections_base: List[Dict], 
                           sections_ours: List[Dict],
                           sections_theirs: List[Dict]) -> Dict[str, Any]:
        """
        Find matching sections between three document versions.
        
        Returns:
            Dictionary mapping section relationships
        """
        def section_similarity(sec1: Dict, sec2: Dict) -> float:
            """Calculate similarity between two sections."""
            title_sim = difflib.SequenceMatcher(
                None, sec1['title'].lower(), sec2['title'].lower()
            ).ratio()
            
            content_sim = 0.0
            if sec1['content'] and sec2['content']:
                content_sim = difflib.SequenceMatcher(
                    None, ' '.join(sec1['content']), ' '.join(sec2['content'])
                ).ratio()
            
            return (title_sim * 0.7 + content_sim * 0.3)
        
        matches = {
            'base_to_ours': {},
            'base_to_theirs': {},
            'ours_to_theirs': {},
            'unmatched_base': [],
            'unmatched_ours': [],
            'unmatched_theirs': []
        }
        
        # Find best matches between base and ours
        used_ours = set()
        for i, base_sec in enumerate(sections_base):
            best_match = -1
            best_sim = 0.3  # Minimum similarity threshold
            
            for j, ours_sec in enumerate(sections_ours):
                if j in used_ours:
                    continue
                    
                sim = section_similarity(base_sec, ours_sec)
                if sim > best_sim:
                    best_sim = sim
                    best_match = j
            
            if best_match != -1:
                matches['base_to_ours'][i] = best_match
                used_ours.add(best_match)
        
        # Similar matching for base to theirs and ours to theirs
        used_theirs = set()
        for i, base_sec in enumerate(sections_base):
            best_match = -1
            best_sim = 0.3
            
            for j, theirs_sec in enumerate(sections_theirs):
                if j in used_theirs:
                    continue
                    
                sim = section_similarity(base_sec, theirs_sec)
                if sim > best_sim:
                    best_sim = sim
                    best_match = j
            
            if best_match != -1:
                matches['base_to_theirs'][i] = best_match
                used_theirs.add(best_match)
        
        return matches


class ConflictResolver:
    """Resolves merge conflicts using various strategies."""
    
    def __init__(self):
        self.semantic_engine = SemanticComparisonEngine()
    
    def resolve_conflict(self, conflict: MergeConflict, 
                        strategy: MergeStrategy) -> Tuple[str, ConflictResolution]:
        """
        Resolve a merge conflict using the specified strategy.
        
        Args:
            conflict: The conflict to resolve
            strategy: Strategy to use for resolution
            
        Returns:
            Tuple of (resolved_content, resolution_type)
        """
        if strategy == MergeStrategy.AUTO:
            return self._auto_resolve_conflict(conflict)
        elif strategy == MergeStrategy.FAVOR_OURS:
            return conflict.ours_content, ConflictResolution.AUTO_RESOLVED
        elif strategy == MergeStrategy.FAVOR_THEIRS:
            return conflict.theirs_content, ConflictResolution.AUTO_RESOLVED
        elif strategy == MergeStrategy.SEMANTIC:
            return self._semantic_resolve_conflict(conflict)
        elif strategy == MergeStrategy.LEGAL_SAFE:
            return self._legal_safe_resolve_conflict(conflict)
        else:
            return "", ConflictResolution.MANUAL_REQUIRED
    
    def _auto_resolve_conflict(self, conflict: MergeConflict) -> Tuple[str, ConflictResolution]:
        """Attempt to automatically resolve a conflict."""
        base = conflict.base_content.strip()
        ours = conflict.ours_content.strip()
        theirs = conflict.theirs_content.strip()
        
        # If one side is unchanged from base, choose the changed side
        if base == ours and base != theirs:
            return theirs, ConflictResolution.AUTO_RESOLVED
        elif base == theirs and base != ours:
            return ours, ConflictResolution.AUTO_RESOLVED
        
        # If changes are additive (both sides add different content)
        if self._is_additive_change(base, ours, theirs):
            merged = self._merge_additive_changes(base, ours, theirs)
            return merged, ConflictResolution.AUTO_RESOLVED
        
        # If changes are compatible formatting/style changes
        if self._are_compatible_changes(ours, theirs):
            merged = self._merge_compatible_changes(ours, theirs)
            return merged, ConflictResolution.AUTO_RESOLVED
        
        return "", ConflictResolution.MANUAL_REQUIRED
    
    def _semantic_resolve_conflict(self, conflict: MergeConflict) -> Tuple[str, ConflictResolution]:
        """Resolve conflict using semantic analysis."""
        # Compare semantic meaning of both changes
        ours_vs_base = self.semantic_engine.analyzer.compute_semantic_similarity(
            conflict.base_content, conflict.ours_content
        )
        theirs_vs_base = self.semantic_engine.analyzer.compute_semantic_similarity(
            conflict.base_content, conflict.theirs_content
        )
        
        # If one change preserves meaning better, prefer it
        if abs(ours_vs_base - theirs_vs_base) > 0.2:
            if ours_vs_base > theirs_vs_base:
                return conflict.ours_content, ConflictResolution.SEMANTIC_MERGE
            else:
                return conflict.theirs_content, ConflictResolution.SEMANTIC_MERGE
        
        # Try to merge semantically compatible changes
        if ours_vs_base > 0.7 and theirs_vs_base > 0.7:
            # Both changes preserve meaning - try to combine
            return self._combine_semantic_changes(
                conflict.base_content, conflict.ours_content, conflict.theirs_content
            ), ConflictResolution.SEMANTIC_MERGE
        
        return "", ConflictResolution.MANUAL_REQUIRED
    
    def _legal_safe_resolve_conflict(self, conflict: MergeConflict) -> Tuple[str, ConflictResolution]:
        """Resolve conflict prioritizing legal safety."""
        # Legal patterns that indicate risk
        risky_patterns = [
            r'shall not', r'cannot', r'prohibited', r'forbidden',
            r'unlimited', r'liable for all', r'indemnify',
            r'immediately due', r'terminate.*for cause'
        ]
        
        def has_risky_pattern(text: str) -> bool:
            text_lower = text.lower()
            return any(re.search(pattern, text_lower) for pattern in risky_patterns)
        
        ours_risky = has_risky_pattern(conflict.ours_content)
        theirs_risky = has_risky_pattern(conflict.theirs_content)
        
        # Prefer less risky option
        if ours_risky and not theirs_risky:
            return conflict.theirs_content, ConflictResolution.AUTO_RESOLVED
        elif theirs_risky and not ours_risky:
            return conflict.ours_content, ConflictResolution.AUTO_RESOLVED
        
        # If both or neither are risky, require manual review
        return "", ConflictResolution.MANUAL_REQUIRED
    
    def _is_additive_change(self, base: str, ours: str, theirs: str) -> bool:
        """Check if both changes are additive (don't remove content)."""
        return (len(ours) >= len(base) and len(theirs) >= len(base) and
                base in ours and base in theirs)
    
    def _merge_additive_changes(self, base: str, ours: str, theirs: str) -> str:
        """Merge two additive changes."""
        # Simple approach: combine both additions
        ours_addition = ours.replace(base, "", 1).strip()
        theirs_addition = theirs.replace(base, "", 1).strip()
        
        if ours_addition and theirs_addition:
            return f"{base} {ours_addition} {theirs_addition}".strip()
        elif ours_addition:
            return ours
        else:
            return theirs
    
    def _are_compatible_changes(self, ours: str, theirs: str) -> bool:
        """Check if changes are compatible (e.g., different formatting)."""
        # Remove whitespace and compare
        ours_normalized = re.sub(r'\s+', ' ', ours.strip())
        theirs_normalized = re.sub(r'\s+', ' ', theirs.strip())
        
        return ours_normalized == theirs_normalized
    
    def _merge_compatible_changes(self, ours: str, theirs: str) -> str:
        """Merge compatible changes, preferring better formatting."""
        # Prefer version with better formatting (more consistent whitespace)
        if ours.count('\n\n') > theirs.count('\n\n'):
            return ours
        else:
            return theirs
    
    def _combine_semantic_changes(self, base: str, ours: str, theirs: str) -> str:
        """Attempt to combine semantically compatible changes."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP techniques
        
        # For now, concatenate unique additions from both sides
        words_base = set(base.lower().split())
        words_ours = set(ours.lower().split())
        words_theirs = set(theirs.lower().split())
        
        new_words_ours = words_ours - words_base
        new_words_theirs = words_theirs - words_base
        
        if new_words_ours and new_words_theirs and not (new_words_ours & new_words_theirs):
            # Both added different words - combine them
            return f"{ours} {' '.join(new_words_theirs - words_ours)}"
        
        # Default to longer version if combination not possible
        return ours if len(ours) > len(theirs) else theirs


class ThreeWayMergeEngine:
    """
    Advanced three-way merge engine for documents.
    Handles structural and semantic merging with conflict resolution.
    """
    
    def __init__(self):
        self.diff_engine = AdvancedDiffEngine()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.conflict_resolver = ConflictResolver()
    
    def merge_documents(self, base_content: str, ours_content: str, 
                       theirs_content: str, strategy: MergeStrategy = MergeStrategy.AUTO) -> MergeResult:
        """
        Perform three-way merge of documents.
        
        Args:
            base_content: Common ancestor content
            ours_content: Our version content
            theirs_content: Their version content
            strategy: Merge strategy to use
            
        Returns:
            MergeResult with merged content and conflicts
        """
        # Analyze document structures
        base_sections = self.structure_analyzer.identify_sections(base_content)
        ours_sections = self.structure_analyzer.identify_sections(ours_content)
        theirs_sections = self.structure_analyzer.identify_sections(theirs_content)
        
        # Find section matches
        section_matches = self.structure_analyzer.find_section_matches(
            base_sections, ours_sections, theirs_sections
        )
        
        # Perform structural merge if sections are well-defined
        if len(base_sections) > 1 and section_matches:
            return self._structural_merge(
                base_sections, ours_sections, theirs_sections,
                section_matches, strategy
            )
        else:
            # Fall back to line-based merge
            return self._line_based_merge(
                base_content, ours_content, theirs_content, strategy
            )
    
    def _structural_merge(self, base_sections: List[Dict], 
                         ours_sections: List[Dict],
                         theirs_sections: List[Dict],
                         section_matches: Dict[str, Any],
                         strategy: MergeStrategy) -> MergeResult:
        """Perform structure-aware merge."""
        merged_sections = []
        conflicts = []
        auto_resolved = []
        
        # Process matched sections
        for base_idx, ours_idx in section_matches['base_to_ours'].items():
            theirs_idx = section_matches['base_to_theirs'].get(base_idx)
            
            if theirs_idx is not None:
                # Section exists in all three versions
                section_result = self._merge_section(
                    base_sections[base_idx],
                    ours_sections[ours_idx],
                    theirs_sections[theirs_idx],
                    strategy
                )
                merged_sections.append(section_result['content'])
                conflicts.extend(section_result['conflicts'])
                auto_resolved.extend(section_result['auto_resolved'])
            else:
                # Section only in base and ours - it was deleted in theirs
                if strategy in [MergeStrategy.FAVOR_OURS, MergeStrategy.AUTO]:
                    merged_sections.append(self._format_section(ours_sections[ours_idx]))
                # Otherwise, section is omitted (deleted)
        
        # Process sections only in theirs
        used_theirs = set(section_matches['base_to_theirs'].values())
        for i, section in enumerate(theirs_sections):
            if i not in used_theirs:
                # New section in theirs
                merged_sections.append(self._format_section(section))
        
        merged_content = '\n\n'.join(merged_sections)
        
        return MergeResult(
            success=len(conflicts) == 0,
            merged_content=merged_content,
            conflicts=conflicts,
            auto_resolved_conflicts=auto_resolved,
            merge_strategy_used=strategy,
            statistics={
                'total_sections': len(merged_sections),
                'conflicts_count': len(conflicts),
                'auto_resolved_count': len(auto_resolved)
            }
        )
    
    def _merge_section(self, base_section: Dict, ours_section: Dict,
                      theirs_section: Dict, strategy: MergeStrategy) -> Dict[str, Any]:
        """Merge a single section from three versions."""
        base_content = self._format_section(base_section)
        ours_content = self._format_section(ours_section)
        theirs_content = self._format_section(theirs_section)
        
        # Use line-based merge for section content
        section_merge = self._line_based_merge(
            base_content, ours_content, theirs_content, strategy
        )
        
        return {
            'content': section_merge.merged_content,
            'conflicts': section_merge.conflicts,
            'auto_resolved': section_merge.auto_resolved_conflicts
        }
    
    def _format_section(self, section: Dict) -> str:
        """Format a section back to text."""
        lines = [section['title']]
        lines.extend(section['content'])
        return '\n'.join(lines)
    
    def _line_based_merge(self, base_content: str, ours_content: str,
                         theirs_content: str, strategy: MergeStrategy) -> MergeResult:
        """Perform line-based three-way merge."""
        # Split into lines for processing
        base_lines = base_content.split('\n')
        ours_lines = ours_content.split('\n')
        theirs_lines = theirs_content.split('\n')
        
        # Get diffs from base to each version
        base_to_ours = self.diff_engine.compare_documents(
            '\n'.join(base_lines), '\n'.join(ours_lines)
        )
        base_to_theirs = self.diff_engine.compare_documents(
            '\n'.join(base_lines), '\n'.join(theirs_lines)
        )
        
        # Build merge chunks
        chunks = self._build_merge_chunks(
            base_lines, ours_lines, theirs_lines,
            base_to_ours.differences, base_to_theirs.differences
        )
        
        # Resolve conflicts in chunks
        conflicts = []
        auto_resolved = []
        merged_lines = []
        
        for chunk in chunks:
            if chunk.chunk_type == "conflict":
                resolved_content, resolution_type = self.conflict_resolver.resolve_conflict(
                    chunk.conflict, strategy
                )
                
                if resolution_type == ConflictResolution.MANUAL_REQUIRED:
                    conflicts.append(chunk.conflict)
                    # Add conflict markers
                    merged_lines.append("<<<<<<< OURS")
                    merged_lines.append(chunk.conflict.ours_content)
                    merged_lines.append("=======")
                    merged_lines.append(chunk.conflict.theirs_content)
                    merged_lines.append(">>>>>>> THEIRS")
                else:
                    auto_resolved.append(chunk.conflict)
                    merged_lines.append(resolved_content)
            else:
                merged_lines.append(chunk.content)
        
        return MergeResult(
            success=len(conflicts) == 0,
            merged_content='\n'.join(merged_lines),
            conflicts=conflicts,
            auto_resolved_conflicts=auto_resolved,
            merge_strategy_used=strategy,
            statistics={
                'total_chunks': len(chunks),
                'conflict_chunks': len([c for c in chunks if c.chunk_type == "conflict"]),
                'auto_resolved_count': len(auto_resolved)
            }
        )
    
    def _build_merge_chunks(self, base_lines: List[str], ours_lines: List[str],
                           theirs_lines: List[str], ours_diffs: List[DiffResult],
                           theirs_diffs: List[DiffResult]) -> List[MergeChunk]:
        """Build merge chunks from diff results."""
        chunks = []
        
        # Convert line-based positions for diffs
        ours_changes = self._map_diffs_to_lines(ours_diffs, base_lines)
        theirs_changes = self._map_diffs_to_lines(theirs_diffs, base_lines)
        
        # Build chunks by walking through base content
        line_num = 0
        while line_num < len(base_lines):
            # Check if current line has changes
            ours_change = self._find_change_at_line(ours_changes, line_num)
            theirs_change = self._find_change_at_line(theirs_changes, line_num)
            
            if ours_change and theirs_change:
                # Conflict: both sides changed this area
                conflict = self._create_conflict(
                    base_lines, ours_lines, theirs_lines,
                    line_num, ours_change, theirs_change
                )
                chunks.append(MergeChunk(
                    chunk_type="conflict",
                    content="",
                    conflict=conflict
                ))
                # Skip past the conflicting region
                line_num = max(ours_change.get('end_line', line_num),
                              theirs_change.get('end_line', line_num)) + 1
            elif ours_change:
                # Only ours changed
                chunks.append(MergeChunk(
                    chunk_type="ours",
                    content='\n'.join(ours_lines[
                        ours_change['start_line']:ours_change['end_line']+1
                    ])
                ))
                line_num = ours_change['end_line'] + 1
            elif theirs_change:
                # Only theirs changed
                chunks.append(MergeChunk(
                    chunk_type="theirs",
                    content='\n'.join(theirs_lines[
                        theirs_change['start_line']:theirs_change['end_line']+1
                    ])
                ))
                line_num = theirs_change['end_line'] + 1
            else:
                # No changes - use base content
                chunks.append(MergeChunk(
                    chunk_type="unchanged",
                    content=base_lines[line_num]
                ))
                line_num += 1
        
        return chunks
    
    def _map_diffs_to_lines(self, diffs: List[DiffResult], base_lines: List[str]) -> List[Dict]:
        """Map character-based diffs to line numbers."""
        changes = []
        
        for diff in diffs:
            # Convert character positions to line numbers
            start_line = base_lines[:diff.old_position[0]].count('\n')
            end_line = base_lines[:diff.old_position[1]].count('\n')
            
            changes.append({
                'start_line': start_line,
                'end_line': end_line,
                'diff': diff
            })
        
        return changes
    
    def _find_change_at_line(self, changes: List[Dict], line_num: int) -> Optional[Dict]:
        """Find change affecting a specific line number."""
        for change in changes:
            if change['start_line'] <= line_num <= change['end_line']:
                return change
        return None
    
    def _create_conflict(self, base_lines: List[str], ours_lines: List[str],
                        theirs_lines: List[str], line_num: int,
                        ours_change: Dict, theirs_change: Dict) -> MergeConflict:
        """Create a merge conflict object."""
        base_content = '\n'.join(base_lines[
            min(ours_change['start_line'], theirs_change['start_line']):
            max(ours_change['end_line'], theirs_change['end_line']) + 1
        ])
        
        ours_content = '\n'.join(ours_lines[
            ours_change['start_line']:ours_change['end_line'] + 1
        ])
        
        theirs_content = '\n'.join(theirs_lines[
            theirs_change['start_line']:theirs_change['end_line'] + 1
        ])
        
        return MergeConflict(
            conflict_id=f"conflict_{line_num}",
            conflict_type="content_conflict",
            base_content=base_content,
            ours_content=ours_content,
            theirs_content=theirs_content,
            position=(line_num, line_num + 1),
            context=self._get_conflict_context(base_lines, line_num),
            severity="medium",
            auto_resolvable=True,
            suggested_resolution="",
            resolution_rationale=""
        )
    
    def _get_conflict_context(self, lines: List[str], line_num: int, 
                             context_size: int = 3) -> str:
        """Get context around a conflict."""
        start = max(0, line_num - context_size)
        end = min(len(lines), line_num + context_size + 1)
        context_lines = lines[start:end]
        return '\n'.join(context_lines)


# Utility functions

def merge_documents_simple(base: str, ours: str, theirs: str, 
                          strategy: str = "auto") -> Dict[str, Any]:
    """
    Simple three-way document merge function.
    
    Args:
        base: Base/common ancestor document
        ours: Our version of the document
        theirs: Their version of the document
        strategy: Merge strategy ("auto", "favor_ours", "favor_theirs", etc.)
        
    Returns:
        Dictionary with merge results
    """
    merge_engine = ThreeWayMergeEngine()
    strategy_enum = MergeStrategy(strategy.lower())
    
    result = merge_engine.merge_documents(base, ours, theirs, strategy_enum)
    
    return {
        'success': result.success,
        'merged_content': result.merged_content,
        'conflicts_count': len(result.conflicts),
        'auto_resolved_count': len(result.auto_resolved_conflicts),
        'strategy_used': result.merge_strategy_used.value,
        'conflicts': [
            {
                'id': c.conflict_id,
                'type': c.conflict_type,
                'severity': c.severity,
                'ours': c.ours_content,
                'theirs': c.theirs_content,
                'auto_resolvable': c.auto_resolvable
            }
            for c in result.conflicts
        ]
    }