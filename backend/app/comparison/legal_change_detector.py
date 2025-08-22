"""
Legal change detection and risk assessment system.
Analyzes legal significance of document changes and assesses potential risks.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np

from .diff_engine import DiffResult, DiffType, DiffLevel
from .semantic_comparison import SemanticDiff, SemanticChangeType


class LegalRiskLevel(Enum):
    """Risk levels for legal changes."""
    CRITICAL = "critical"      # Immediate legal review required
    HIGH = "high"             # Significant legal implications
    MEDIUM = "medium"         # Moderate legal impact
    LOW = "low"              # Minor legal relevance
    INFORMATIONAL = "informational"  # No legal significance


class LegalChangeCategory(Enum):
    """Categories of legal changes."""
    LIABILITY = "liability"
    OBLIGATIONS = "obligations"
    RIGHTS = "rights"
    REMEDIES = "remedies"
    TERMINATION = "termination"
    PAYMENT = "payment"
    JURISDICTION = "jurisdiction"
    DISPUTE_RESOLUTION = "dispute_resolution"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPLIANCE = "compliance"
    INDEMNIFICATION = "indemnification"
    FORCE_MAJEURE = "force_majeure"
    GOVERNING_LAW = "governing_law"


class ChangeImpact(Enum):
    """Impact direction of legal changes."""
    FAVORABLE = "favorable"      # Change benefits the analyzing party
    UNFAVORABLE = "unfavorable"  # Change is detrimental
    NEUTRAL = "neutral"         # No clear benefit/detriment
    UNCLEAR = "unclear"         # Impact cannot be determined


@dataclass
class LegalRisk:
    """Represents a legal risk identified in document changes."""
    risk_level: LegalRiskLevel
    category: LegalChangeCategory
    impact: ChangeImpact
    description: str
    recommendation: str
    affected_clause: str
    old_language: str
    new_language: str
    position: Tuple[int, int]
    confidence: float
    precedent_references: List[str] = field(default_factory=list)
    regulatory_concerns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalAnalysisResult:
    """Complete legal analysis of document changes."""
    document_pair_id: str
    analysis_timestamp: datetime
    overall_risk_level: LegalRiskLevel
    total_risks: int
    risks_by_category: Dict[LegalChangeCategory, int]
    risks_by_level: Dict[LegalRiskLevel, int]
    detailed_risks: List[LegalRisk]
    hidden_changes: List[Dict[str, Any]]
    summary: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LegalPatternLibrary:
    """Library of legal patterns and their significance."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive legal patterns."""
        return {
            # Liability patterns
            'liability_expansion': {
                'patterns': [
                    r'shall be liable for all',
                    r'responsible for any and all',
                    r'liable for indirect.*damages',
                    r'liable for consequential.*damages',
                    r'unlimited liability',
                    r'liability.*not limited'
                ],
                'category': LegalChangeCategory.LIABILITY,
                'risk_level': LegalRiskLevel.CRITICAL,
                'description': 'Expansion of liability exposure'
            },
            'liability_limitation': {
                'patterns': [
                    r'liability.*limited to',
                    r'shall not be liable for',
                    r'excludes.*liability',
                    r'no liability for indirect',
                    r'liability cap',
                    r'maximum liability'
                ],
                'category': LegalChangeCategory.LIABILITY,
                'risk_level': LegalRiskLevel.HIGH,
                'description': 'Limitation of liability provisions'
            },
            
            # Obligation patterns
            'obligation_strengthening': {
                'patterns': [
                    r'must.*immediately',
                    r'shall.*without delay',
                    r'required to.*within',
                    r'obligation to.*ensure',
                    r'duty to.*maintain',
                    r'covenant.*to perform'
                ],
                'category': LegalChangeCategory.OBLIGATIONS,
                'risk_level': LegalRiskLevel.HIGH,
                'description': 'Strengthening of contractual obligations'
            },
            'obligation_relaxation': {
                'patterns': [
                    r'may.*in its discretion',
                    r'best efforts.*instead of',
                    r'reasonable.*instead of',
                    r'subject to.*availability'
                ],
                'category': LegalChangeCategory.OBLIGATIONS,
                'risk_level': LegalRiskLevel.MEDIUM,
                'description': 'Relaxation of obligations'
            },
            
            # Rights patterns
            'rights_expansion': {
                'patterns': [
                    r'right to.*terminate',
                    r'entitled to.*damages',
                    r'may.*suspend',
                    r'right to.*audit',
                    r'authorized to.*access'
                ],
                'category': LegalChangeCategory.RIGHTS,
                'risk_level': LegalRiskLevel.MEDIUM,
                'description': 'Expansion of rights'
            },
            
            # Payment and financial patterns
            'payment_acceleration': {
                'patterns': [
                    r'immediately due.*upon',
                    r'payable.*on demand',
                    r'accelerated payment',
                    r'payment within.*days reduced',
                    r'interest.*on overdue'
                ],
                'category': LegalChangeCategory.PAYMENT,
                'risk_level': LegalRiskLevel.HIGH,
                'description': 'Payment acceleration or penalties'
            },
            
            # Termination patterns
            'termination_expansion': {
                'patterns': [
                    r'terminate.*for convenience',
                    r'terminate.*without cause',
                    r'immediate termination.*upon',
                    r'termination.*for any reason',
                    r'right to terminate.*at any time'
                ],
                'category': LegalChangeCategory.TERMINATION,
                'risk_level': LegalRiskLevel.HIGH,
                'description': 'Expansion of termination rights'
            },
            
            # Dispute resolution patterns
            'arbitration_mandatory': {
                'patterns': [
                    r'shall.*arbitrate',
                    r'mandatory.*arbitration',
                    r'binding.*arbitration',
                    r'disputes.*resolved by.*arbitration',
                    r'waive.*right to jury trial'
                ],
                'category': LegalChangeCategory.DISPUTE_RESOLUTION,
                'risk_level': LegalRiskLevel.HIGH,
                'description': 'Mandatory arbitration clauses'
            },
            
            # Confidentiality patterns
            'confidentiality_expansion': {
                'patterns': [
                    r'confidential.*in perpetuity',
                    r'confidentiality.*survives termination',
                    r'non-disclosure.*indefinite',
                    r'confidential.*includes.*all'
                ],
                'category': LegalChangeCategory.CONFIDENTIALITY,
                'risk_level': LegalRiskLevel.MEDIUM,
                'description': 'Expansion of confidentiality requirements'
            },
            
            # Indemnification patterns
            'indemnification_expansion': {
                'patterns': [
                    r'indemnify.*against all',
                    r'hold harmless.*from any',
                    r'defend.*against.*claims',
                    r'indemnification.*includes.*fees'
                ],
                'category': LegalChangeCategory.INDEMNIFICATION,
                'risk_level': LegalRiskLevel.CRITICAL,
                'description': 'Expansion of indemnification obligations'
            },
            
            # Jurisdiction patterns
            'jurisdiction_change': {
                'patterns': [
                    r'exclusive jurisdiction',
                    r'governed by.*law of',
                    r'jurisdiction.*courts of',
                    r'venue.*shall be'
                ],
                'category': LegalChangeCategory.JURISDICTION,
                'risk_level': LegalRiskLevel.MEDIUM,
                'description': 'Changes to governing law or jurisdiction'
            }
        }


class HiddenChangeDetector:
    """Detects subtle or hidden changes that may have legal significance."""
    
    @staticmethod
    def detect_subtle_modifications(old_text: str, new_text: str) -> List[Dict[str, Any]]:
        """
        Detect subtle modifications that might be overlooked.
        
        Args:
            old_text: Original text
            new_text: Modified text
            
        Returns:
            List of subtle changes detected
        """
        hidden_changes = []
        
        # Detect single word changes in critical phrases
        critical_phrases = [
            'shall', 'must', 'may', 'will', 'should', 'can', 'cannot',
            'all', 'any', 'some', 'none', 'every', 'each',
            'immediately', 'promptly', 'reasonable', 'best efforts',
            'and', 'or', 'not', 'unless', 'if', 'when', 'where'
        ]
        
        old_words = old_text.lower().split()
        new_words = new_text.lower().split()
        
        # Find single word substitutions
        if len(old_words) == len(new_words):
            for i, (old_word, new_word) in enumerate(zip(old_words, new_words)):
                if old_word != new_word:
                    if old_word in critical_phrases or new_word in critical_phrases:
                        hidden_changes.append({
                            'type': 'critical_word_change',
                            'old_word': old_word,
                            'new_word': new_word,
                            'position': i,
                            'context': ' '.join(old_words[max(0, i-3):i+4]),
                            'risk_level': 'high',
                            'description': f'Critical word changed from "{old_word}" to "{new_word}"'
                        })
        
        # Detect punctuation changes that affect meaning
        punctuation_changes = HiddenChangeDetector._detect_punctuation_changes(old_text, new_text)
        hidden_changes.extend(punctuation_changes)
        
        # Detect number changes
        number_changes = HiddenChangeDetector._detect_number_changes(old_text, new_text)
        hidden_changes.extend(number_changes)
        
        return hidden_changes
    
    @staticmethod
    def _detect_punctuation_changes(old_text: str, new_text: str) -> List[Dict[str, Any]]:
        """Detect legally significant punctuation changes."""
        changes = []
        
        # Find critical punctuation patterns
        critical_patterns = [
            (r'(\w+),\s*(\w+)', r'\1\s+\2', 'comma_removal'),
            (r'(\w+)\s+(\w+)', r'\1,\s*\2', 'comma_addition'),
            (r'(\w+);\s*(\w+)', r'\1\.\s*\2', 'semicolon_to_period'),
            (r'(\w+)\.\s*(\w+)', r'\1;\s*\2', 'period_to_semicolon')
        ]
        
        for old_pattern, new_pattern, change_type in critical_patterns:
            if re.search(old_pattern, old_text) and re.search(new_pattern, new_text):
                changes.append({
                    'type': change_type,
                    'description': f'Punctuation change: {change_type.replace("_", " ")}',
                    'risk_level': 'medium',
                    'details': 'Punctuation changes can significantly alter legal meaning'
                })
        
        return changes
    
    @staticmethod
    def _detect_number_changes(old_text: str, new_text: str) -> List[Dict[str, Any]]:
        """Detect changes in numbers, dates, or time periods."""
        changes = []
        
        # Extract numbers from both texts
        old_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', old_text)
        new_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', new_text)
        
        if old_numbers != new_numbers:
            changes.append({
                'type': 'numerical_change',
                'old_numbers': old_numbers,
                'new_numbers': new_numbers,
                'risk_level': 'high',
                'description': 'Numbers in document have changed'
            })
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
        old_dates = re.findall(date_pattern, old_text)
        new_dates = re.findall(date_pattern, new_text)
        
        if old_dates != new_dates:
            changes.append({
                'type': 'date_change',
                'old_dates': old_dates,
                'new_dates': new_dates,
                'risk_level': 'high',
                'description': 'Date values have changed'
            })
        
        return changes


class LegalChangeDetector:
    """
    Main detector for legal changes and risk assessment.
    """
    
    def __init__(self):
        self.pattern_library = LegalPatternLibrary()
        self.hidden_detector = HiddenChangeDetector()
        
    def analyze_legal_changes(self, diff_results: List[DiffResult], 
                            semantic_diffs: List[SemanticDiff] = None,
                            old_document: str = "", 
                            new_document: str = "") -> LegalAnalysisResult:
        """
        Analyze legal significance of document changes.
        
        Args:
            diff_results: Results from diff engine
            semantic_diffs: Results from semantic comparison
            old_document: Original document content
            new_document: Modified document content
            
        Returns:
            Complete legal analysis results
        """
        legal_risks = []
        
        # Analyze surface-level changes
        for diff in diff_results:
            risks = self._analyze_diff_for_legal_risk(diff, old_document, new_document)
            legal_risks.extend(risks)
        
        # Analyze semantic changes if provided
        if semantic_diffs:
            semantic_risks = self._analyze_semantic_changes(semantic_diffs)
            legal_risks.extend(semantic_risks)
        
        # Detect hidden changes
        hidden_changes = self.hidden_detector.detect_subtle_modifications(
            old_document, new_document
        )
        
        # Calculate overall risk level
        overall_risk = self._calculate_overall_risk(legal_risks)
        
        # Categorize risks
        risks_by_category = defaultdict(int)
        risks_by_level = defaultdict(int)
        
        for risk in legal_risks:
            risks_by_category[risk.category] += 1
            risks_by_level[risk.risk_level] += 1
        
        # Generate summary and recommendations
        summary = self._generate_summary(legal_risks, hidden_changes)
        recommendations = self._generate_recommendations(legal_risks, overall_risk)
        
        return LegalAnalysisResult(
            document_pair_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_timestamp=datetime.now(),
            overall_risk_level=overall_risk,
            total_risks=len(legal_risks),
            risks_by_category=dict(risks_by_category),
            risks_by_level=dict(risks_by_level),
            detailed_risks=legal_risks,
            hidden_changes=hidden_changes,
            summary=summary,
            recommendations=recommendations
        )
    
    def _analyze_diff_for_legal_risk(self, diff: DiffResult, 
                                   old_doc: str, new_doc: str) -> List[LegalRisk]:
        """Analyze a single diff result for legal risks."""
        risks = []
        
        # Get context around the change
        context = self._get_extended_context(diff, old_doc, new_doc)
        
        # Check against legal patterns
        for pattern_name, pattern_info in self.pattern_library.patterns.items():
            for pattern in pattern_info['patterns']:
                # Check if pattern appears in new content (addition/modification)
                if (diff.new_content and 
                    re.search(pattern, diff.new_content.lower(), re.IGNORECASE)):
                    
                    # Determine impact direction
                    impact = self._determine_change_impact(
                        pattern_name, diff.old_content, diff.new_content
                    )
                    
                    risk = LegalRisk(
                        risk_level=pattern_info['risk_level'],
                        category=pattern_info['category'],
                        impact=impact,
                        description=pattern_info['description'],
                        recommendation=self._generate_risk_recommendation(pattern_info),
                        affected_clause=context.get('clause_title', 'Unknown'),
                        old_language=diff.old_content,
                        new_language=diff.new_content,
                        position=diff.old_position,
                        confidence=self._calculate_pattern_confidence(diff, pattern),
                        metadata={
                            'pattern_matched': pattern_name,
                            'diff_type': diff.diff_type.value,
                            'context': context
                        }
                    )
                    risks.append(risk)
                
                # Check if pattern was removed (deletion)
                elif (diff.old_content and 
                      re.search(pattern, diff.old_content.lower(), re.IGNORECASE) and
                      diff.diff_type == DiffType.DELETION):
                    
                    risk = LegalRisk(
                        risk_level=LegalRiskLevel.HIGH,
                        category=pattern_info['category'],
                        impact=ChangeImpact.UNFAVORABLE,  # Assuming removal is unfavorable
                        description=f"Removal of {pattern_info['description']}",
                        recommendation=f"Review removal of {pattern_info['description']}",
                        affected_clause=context.get('clause_title', 'Unknown'),
                        old_language=diff.old_content,
                        new_language=diff.new_content,
                        position=diff.old_position,
                        confidence=0.8,
                        metadata={
                            'pattern_matched': pattern_name,
                            'diff_type': 'removal',
                            'context': context
                        }
                    )
                    risks.append(risk)
        
        return risks
    
    def _analyze_semantic_changes(self, semantic_diffs: List[SemanticDiff]) -> List[LegalRisk]:
        """Analyze semantic changes for legal risks."""
        risks = []
        
        for semantic_diff in semantic_diffs:
            # Map semantic change types to legal risks
            risk_level, category = self._map_semantic_to_legal_risk(semantic_diff)
            
            if risk_level != LegalRiskLevel.INFORMATIONAL:
                risk = LegalRisk(
                    risk_level=risk_level,
                    category=category,
                    impact=self._determine_semantic_impact(semantic_diff),
                    description=f"Semantic change: {semantic_diff.change_type.value}",
                    recommendation=semantic_diff.explanation,
                    affected_clause="Semantic analysis",
                    old_language=semantic_diff.old_segment,
                    new_language=semantic_diff.new_segment,
                    position=semantic_diff.position,
                    confidence=semantic_diff.confidence,
                    metadata={
                        'semantic_change_type': semantic_diff.change_type.value,
                        'intent_change_score': semantic_diff.intent_change_score,
                        'semantic_similarity': semantic_diff.semantic_similarity
                    }
                )
                risks.append(risk)
        
        return risks
    
    def _get_extended_context(self, diff: DiffResult, 
                            old_doc: str, new_doc: str) -> Dict[str, Any]:
        """Get extended context around a difference."""
        context = {}
        
        # Try to identify the clause or section
        doc_to_analyze = old_doc if diff.old_content else new_doc
        
        if doc_to_analyze:
            # Look for section headers before the change
            start_pos = max(0, diff.old_position[0] - 1000)
            context_text = doc_to_analyze[start_pos:diff.old_position[1]]
            
            # Find potential clause titles (lines that are short and may be headers)
            lines = context_text.split('\n')
            for i in reversed(range(len(lines))):
                line = lines[i].strip()
                if line and len(line) < 100 and ':' not in line:
                    # Potential header
                    context['clause_title'] = line
                    break
        
        return context
    
    def _determine_change_impact(self, pattern_name: str, old_content: str, 
                               new_content: str) -> ChangeImpact:
        """Determine the impact direction of a change."""
        # This is a simplified implementation - in practice, this would be more sophisticated
        unfavorable_patterns = [
            'liability_expansion', 'obligation_strengthening', 'payment_acceleration',
            'termination_expansion', 'indemnification_expansion'
        ]
        
        favorable_patterns = [
            'liability_limitation', 'obligation_relaxation'
        ]
        
        if pattern_name in unfavorable_patterns:
            return ChangeImpact.UNFAVORABLE
        elif pattern_name in favorable_patterns:
            return ChangeImpact.FAVORABLE
        else:
            return ChangeImpact.NEUTRAL
    
    def _generate_risk_recommendation(self, pattern_info: Dict) -> str:
        """Generate recommendation for a risk pattern."""
        category = pattern_info['category']
        risk_level = pattern_info['risk_level']
        
        base_recommendations = {
            LegalChangeCategory.LIABILITY: "Review liability exposure with legal counsel",
            LegalChangeCategory.OBLIGATIONS: "Assess feasibility of meeting new obligations",
            LegalChangeCategory.PAYMENT: "Review financial impact of payment terms",
            LegalChangeCategory.TERMINATION: "Evaluate termination clause implications",
            LegalChangeCategory.DISPUTE_RESOLUTION: "Consider dispute resolution preferences"
        }
        
        base = base_recommendations.get(category, "Review change with legal counsel")
        
        if risk_level == LegalRiskLevel.CRITICAL:
            return f"URGENT: {base}. Immediate attention required."
        elif risk_level == LegalRiskLevel.HIGH:
            return f"HIGH PRIORITY: {base}"
        else:
            return base
    
    def _calculate_pattern_confidence(self, diff: DiffResult, pattern: str) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = 0.7
        
        # Higher confidence for exact matches
        if pattern.lower() in diff.new_content.lower():
            base_confidence = 0.9
        
        # Adjust based on context length
        if len(diff.new_content) > 100:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _map_semantic_to_legal_risk(self, semantic_diff: SemanticDiff) -> Tuple[LegalRiskLevel, LegalChangeCategory]:
        """Map semantic change types to legal risk levels and categories."""
        change_type = semantic_diff.change_type
        
        mapping = {
            SemanticChangeType.MEANING_NEGATED: (LegalRiskLevel.CRITICAL, LegalChangeCategory.OBLIGATIONS),
            SemanticChangeType.MEANING_SHIFTED: (LegalRiskLevel.HIGH, LegalChangeCategory.OBLIGATIONS),
            SemanticChangeType.MEANING_STRENGTHENED: (LegalRiskLevel.MEDIUM, LegalChangeCategory.OBLIGATIONS),
            SemanticChangeType.MEANING_WEAKENED: (LegalRiskLevel.MEDIUM, LegalChangeCategory.OBLIGATIONS),
            SemanticChangeType.CONCEPT_ADDED: (LegalRiskLevel.MEDIUM, LegalChangeCategory.RIGHTS),
            SemanticChangeType.CONCEPT_REMOVED: (LegalRiskLevel.HIGH, LegalChangeCategory.RIGHTS),
            SemanticChangeType.RELATIONSHIP_CHANGED: (LegalRiskLevel.HIGH, LegalChangeCategory.OBLIGATIONS),
            SemanticChangeType.MEANING_PRESERVED: (LegalRiskLevel.INFORMATIONAL, LegalChangeCategory.OBLIGATIONS)
        }
        
        return mapping.get(change_type, (LegalRiskLevel.MEDIUM, LegalChangeCategory.OBLIGATIONS))
    
    def _determine_semantic_impact(self, semantic_diff: SemanticDiff) -> ChangeImpact:
        """Determine impact of semantic changes."""
        if semantic_diff.intent_change_score > 0.7:
            return ChangeImpact.UNFAVORABLE
        elif semantic_diff.intent_change_score < 0.3:
            return ChangeImpact.NEUTRAL
        else:
            return ChangeImpact.UNCLEAR
    
    def _calculate_overall_risk(self, risks: List[LegalRisk]) -> LegalRiskLevel:
        """Calculate overall risk level from individual risks."""
        if not risks:
            return LegalRiskLevel.INFORMATIONAL
        
        risk_scores = {
            LegalRiskLevel.CRITICAL: 4,
            LegalRiskLevel.HIGH: 3,
            LegalRiskLevel.MEDIUM: 2,
            LegalRiskLevel.LOW: 1,
            LegalRiskLevel.INFORMATIONAL: 0
        }
        
        # Calculate weighted average risk score
        total_score = sum(risk_scores[risk.risk_level] for risk in risks)
        avg_score = total_score / len(risks)
        
        # Map back to risk level
        if avg_score >= 3.5:
            return LegalRiskLevel.CRITICAL
        elif avg_score >= 2.5:
            return LegalRiskLevel.HIGH
        elif avg_score >= 1.5:
            return LegalRiskLevel.MEDIUM
        elif avg_score >= 0.5:
            return LegalRiskLevel.LOW
        else:
            return LegalRiskLevel.INFORMATIONAL
    
    def _generate_summary(self, risks: List[LegalRisk], 
                         hidden_changes: List[Dict[str, Any]]) -> str:
        """Generate summary of legal analysis."""
        if not risks and not hidden_changes:
            return "No significant legal risks detected in document changes."
        
        risk_counts = defaultdict(int)
        for risk in risks:
            risk_counts[risk.risk_level] += 1
        
        summary_parts = []
        
        if risk_counts[LegalRiskLevel.CRITICAL]:
            summary_parts.append(f"{risk_counts[LegalRiskLevel.CRITICAL]} critical risk(s)")
        if risk_counts[LegalRiskLevel.HIGH]:
            summary_parts.append(f"{risk_counts[LegalRiskLevel.HIGH]} high risk(s)")
        if risk_counts[LegalRiskLevel.MEDIUM]:
            summary_parts.append(f"{risk_counts[LegalRiskLevel.MEDIUM]} medium risk(s)")
        
        summary = f"Analysis identified {', '.join(summary_parts)} in document changes."
        
        if hidden_changes:
            summary += f" Additionally, {len(hidden_changes)} subtle change(s) detected."
        
        return summary
    
    def _generate_recommendations(self, risks: List[LegalRisk], 
                                overall_risk: LegalRiskLevel) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if overall_risk in [LegalRiskLevel.CRITICAL, LegalRiskLevel.HIGH]:
            recommendations.append("Immediate legal review recommended before accepting changes")
        
        # Category-specific recommendations
        category_counts = defaultdict(int)
        for risk in risks:
            category_counts[risk.category] += 1
        
        if category_counts[LegalChangeCategory.LIABILITY] > 0:
            recommendations.append("Review liability allocation and insurance coverage")
        
        if category_counts[LegalChangeCategory.PAYMENT] > 0:
            recommendations.append("Assess financial impact of payment term changes")
        
        if category_counts[LegalChangeCategory.TERMINATION] > 0:
            recommendations.append("Evaluate termination clause implications on business continuity")
        
        if not recommendations:
            recommendations.append("Changes appear to have minimal legal impact")
        
        return recommendations


# Utility functions

def analyze_legal_risk_simple(old_document: str, new_document: str) -> Dict[str, Any]:
    """
    Simple legal risk analysis function.
    
    Args:
        old_document: Original document content
        new_document: Modified document content
        
    Returns:
        Dictionary with risk analysis results
    """
    from .diff_engine import AdvancedDiffEngine
    
    # Perform diff analysis
    diff_engine = AdvancedDiffEngine()
    comparison_result = diff_engine.compare_documents(old_document, new_document)
    
    # Perform legal analysis
    detector = LegalChangeDetector()
    legal_analysis = detector.analyze_legal_changes(
        comparison_result.differences, 
        old_document=old_document,
        new_document=new_document
    )
    
    return {
        'overall_risk_level': legal_analysis.overall_risk_level.value,
        'total_risks': legal_analysis.total_risks,
        'summary': legal_analysis.summary,
        'recommendations': legal_analysis.recommendations,
        'detailed_analysis': legal_analysis
    }


def get_legal_change_patterns() -> Dict[str, List[str]]:
    """Get all legal change patterns for reference."""
    library = LegalPatternLibrary()
    patterns_by_category = defaultdict(list)
    
    for pattern_name, pattern_info in library.patterns.items():
        category = pattern_info['category'].value
        patterns_by_category[category].extend(pattern_info['patterns'])
    
    return dict(patterns_by_category)