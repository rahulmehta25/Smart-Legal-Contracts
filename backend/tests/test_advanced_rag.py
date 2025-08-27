"""
Tests for Advanced RAG Pipeline and AI Capabilities
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

from app.rag.advanced_rag_pipeline import (
    AdvancedRAGPipeline,
    LegalEntity,
    ContractClause,
    RiskAssessment,
    SemanticSearchResult,
    NaturalLanguageInterface
)
from app.rag.intelligent_extraction import (
    IntelligentClauseExtractor,
    ExtractedClause,
    ContractStructure
)
from app.rag.confidence_scoring import (
    ConfidenceScoringEngine,
    RiskAssessmentEngine,
    ConfidenceScore,
    RiskScore,
    ClauseRiskProfile
)


class TestAdvancedRAGPipeline:
    """Test suite for Advanced RAG Pipeline"""
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create RAG pipeline instance"""
        return AdvancedRAGPipeline()
    
    @pytest.fixture
    def sample_document(self):
        """Sample legal document for testing"""
        return {
            'id': 'test_doc_1',
            'content': """
                1. ARBITRATION AGREEMENT
                
                Any dispute arising out of or relating to this Agreement shall be resolved 
                through binding arbitration administered by JAMS in accordance with its 
                Commercial Arbitration Rules. The arbitration shall be held in Delaware.
                
                2. LIMITATION OF LIABILITY
                
                In no event shall either party be liable for any indirect, incidental, 
                special, or consequential damages, including lost profits, even if advised 
                of the possibility of such damages.
                
                3. CONFIDENTIALITY
                
                Each party agrees to maintain the confidentiality of the other party's 
                proprietary information and shall not disclose such information to any 
                third party without prior written consent.
            """,
            'metadata': {
                'type': 'commercial_agreement',
                'date': '2024-01-20',
                'parties': ['Company A', 'Company B']
            }
        }
    
    def test_single_document_analysis(self, rag_pipeline, sample_document):
        """Test single document analysis"""
        result = rag_pipeline.analyze_single_document(sample_document)
        
        assert result is not None
        assert 'document_id' in result
        assert result['document_id'] == 'test_doc_1'
        assert 'clauses' in result
        assert len(result['clauses']) > 0
        assert 'risk_assessment' in result
        assert 'arbitration_analysis' in result
    
    @pytest.mark.asyncio
    async def test_multiple_document_analysis(self, rag_pipeline):
        """Test multiple document analysis with cross-referencing"""
        documents = [
            {
                'id': 'doc1',
                'content': 'This agreement shall be governed by Delaware law.',
                'metadata': {}
            },
            {
                'id': 'doc2',
                'content': 'This agreement shall be governed by California law.',
                'metadata': {}
            }
        ]
        
        result = await rag_pipeline.analyze_multiple_documents(
            documents, cross_reference=True
        )
        
        assert 'individual_analyses' in result
        assert len(result['individual_analyses']) == 2
        assert 'cross_document_analysis' in result
        assert 'consolidated_insights' in result
    
    def test_semantic_search(self, rag_pipeline, sample_document):
        """Test semantic search functionality"""
        documents = [sample_document]
        query = "What are the dispute resolution mechanisms?"
        
        results = rag_pipeline.perform_semantic_search(
            query=query,
            documents=documents,
            top_k=5
        )
        
        assert len(results) > 0
        assert isinstance(results[0], SemanticSearchResult)
        assert results[0].relevance_score > 0
        assert 'arbitration' in results[0].content.lower()
    
    def test_clause_extraction(self, rag_pipeline, sample_document):
        """Test intelligent clause extraction"""
        clauses = rag_pipeline.extract_intelligent_clauses(
            sample_document['content']
        )
        
        assert len(clauses) > 0
        assert any(c.clause_type == 'arbitration' for c in clauses)
        assert any(c.clause_type == 'limitation' for c in clauses)
        assert any(c.clause_type == 'confidentiality' for c in clauses)
        
        # Check clause details
        arbitration_clause = next(
            (c for c in clauses if c.clause_type == 'arbitration'), None
        )
        assert arbitration_clause is not None
        assert arbitration_clause.confidence_score > 0.5
        assert len(arbitration_clause.legal_implications) > 0
    
    def test_risk_scoring(self, rag_pipeline, sample_document):
        """Test risk scoring algorithms"""
        clauses = rag_pipeline.extract_intelligent_clauses(
            sample_document['content']
        )
        
        risk_assessment = rag_pipeline.calculate_risk_score(
            sample_document['content'], clauses
        )
        
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0 <= risk_assessment.overall_risk_score <= 100
        assert len(risk_assessment.risk_factors) > 0
        assert len(risk_assessment.mitigation_strategies) > 0
        assert risk_assessment.confidence > 0
    
    def test_natural_language_interface(self, rag_pipeline, sample_document):
        """Test natural language query interface"""
        nl_interface = rag_pipeline.create_natural_language_interface(
            [sample_document]
        )
        
        # Test factual question
        response = nl_interface.query(
            "What type of arbitration is specified?"
        )
        assert 'JAMS' in str(response)
        
        # Test clause search
        response = nl_interface.query(
            "Show me the limitation of liability clause"
        )
        assert 'limitation' in str(response).lower()
    
    def test_arbitration_suggestions(self, rag_pipeline):
        """Test AI-powered arbitration clause suggestions"""
        context = {
            'jurisdiction': 'Delaware',
            'contract_type': 'commercial',
            'party_sophistication': 'both_sophisticated'
        }
        requirements = ['expedited', 'confidential']
        
        suggestions = rag_pipeline.generate_arbitration_suggestions(
            context, requirements
        )
        
        assert len(suggestions) > 0
        assert all('clause' in s for s in suggestions)
        assert any('expedited' in s['clause'].lower() for s in suggestions)
        assert all('suitability_score' in s for s in suggestions)


class TestIntelligentClauseExtractor:
    """Test suite for Intelligent Clause Extractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create clause extractor instance"""
        return IntelligentClauseExtractor()
    
    @pytest.fixture
    def sample_contract(self):
        """Sample contract text"""
        return """
            PURCHASE AGREEMENT
            
            This Purchase Agreement ("Agreement") is entered into as of January 1, 2024,
            between Buyer Corp ("Buyer") and Seller Inc ("Seller").
            
            1. PAYMENT TERMS
            Buyer shall pay Seller the purchase price of $100,000 within thirty (30) days
            of delivery. Payment shall be made via wire transfer.
            
            2. TERMINATION
            Either party may terminate this Agreement upon thirty (30) days written notice
            if the other party materially breaches any provision hereof.
            
            3. GOVERNING LAW
            This Agreement shall be governed by the laws of the State of Delaware,
            without regard to its conflict of laws provisions.
        """
    
    def test_extract_all_clauses(self, extractor, sample_contract):
        """Test comprehensive clause extraction"""
        clauses = extractor.extract_all_clauses(sample_contract)
        
        assert len(clauses) > 0
        assert all(isinstance(c, ExtractedClause) for c in clauses)
        
        # Check clause types
        clause_types = {c.type for c in clauses}
        assert 'payment' in clause_types
        assert 'termination' in clause_types
        assert 'governing_law' in clause_types
    
    def test_entity_extraction(self, extractor, sample_contract):
        """Test legal entity extraction"""
        clauses = extractor.extract_all_clauses(sample_contract)
        
        # Check entities in clauses
        all_entities = []
        for clause in clauses:
            all_entities.extend(clause.entities)
        
        entity_names = {e['text'] for e in all_entities}
        assert 'Buyer Corp' in entity_names or 'Buyer' in entity_names
        assert 'Seller Inc' in entity_names or 'Seller' in entity_names
    
    def test_obligation_extraction(self, extractor, sample_contract):
        """Test obligation and condition extraction"""
        clauses = extractor.extract_all_clauses(sample_contract)
        
        payment_clause = next(
            (c for c in clauses if c.type == 'payment'), None
        )
        assert payment_clause is not None
        assert len(payment_clause.obligations) > 0
        assert any('shall pay' in o.get('trigger', '') 
                  for o in payment_clause.obligations)
    
    def test_temporal_extraction(self, extractor, sample_contract):
        """Test temporal element extraction"""
        clauses = extractor.extract_all_clauses(sample_contract)
        
        # Check for temporal elements
        temporal_clauses = [c for c in clauses if c.temporal_elements]
        assert len(temporal_clauses) > 0
        
        # Check specific temporal elements
        payment_clause = next(
            (c for c in clauses if c.type == 'payment'), None
        )
        assert payment_clause is not None
        assert any('thirty (30) days' in t.get('text', '') 
                  for t in payment_clause.temporal_elements)
    
    def test_cross_reference_detection(self, extractor):
        """Test cross-reference detection"""
        contract_with_refs = """
            1. DEFINITIONS
            Terms used in this Agreement shall have the meanings set forth in Section 1.
            
            2. PAYMENT
            Payment terms are as specified in Section 3.2 and Exhibit A.
            
            3. TERMINATION
            3.1 This Agreement may be terminated as provided in Section 2.
            3.2 Upon termination, obligations under Section 2 shall survive.
        """
        
        clauses = extractor.extract_all_clauses(contract_with_refs)
        
        # Check cross-references
        clauses_with_refs = [c for c in clauses if c.cross_references]
        assert len(clauses_with_refs) > 0
        
        all_refs = []
        for clause in clauses_with_refs:
            all_refs.extend(clause.cross_references)
        
        assert any('Section' in ref for ref in all_refs)
    
    def test_clarity_and_enforceability_scoring(self, extractor, sample_contract):
        """Test clarity and enforceability scoring"""
        clauses = extractor.extract_all_clauses(sample_contract)
        
        for clause in clauses:
            assert 0 <= clause.clarity_score <= 1
            assert 0 <= clause.enforceability_score <= 1
            
            # Well-written clauses should have decent scores
            if clause.type == 'governing_law':
                assert clause.clarity_score > 0.5
                assert clause.enforceability_score > 0.6


class TestConfidenceScoringEngine:
    """Test suite for Confidence Scoring Engine"""
    
    @pytest.fixture
    def scoring_engine(self):
        """Create confidence scoring engine"""
        return ConfidenceScoringEngine()
    
    @pytest.fixture
    def analysis_result(self):
        """Sample analysis result"""
        return {
            'has_arbitration_clause': True,
            'clauses': [
                {'text': 'Binding arbitration clause...', 'type': 'arbitration'},
                {'text': 'Limitation of liability...', 'type': 'limitation'}
            ]
        }
    
    @pytest.fixture
    def evidence(self):
        """Sample evidence for scoring"""
        return [
            {
                'relevance': 0.9,
                'keyword_count': 5,
                'entity_count': 3,
                'pattern_match': True
            },
            {
                'relevance': 0.7,
                'keyword_count': 3,
                'entity_count': 2,
                'pattern_match': True
            }
        ]
    
    def test_confidence_score_calculation(
        self, scoring_engine, analysis_result, evidence
    ):
        """Test confidence score calculation"""
        score = scoring_engine.calculate_confidence_score(
            analysis_result, evidence
        )
        
        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.overall_score <= 1
        assert len(score.component_scores) > 0
        assert len(score.factors) > 0
        assert score.explanation != ""
        assert score.reliability_indicator in [
            'highly_reliable', 'reliable', 'moderately_reliable', 'low_reliability'
        ]
    
    def test_confidence_interval(
        self, scoring_engine, analysis_result, evidence
    ):
        """Test confidence interval calculation"""
        score = scoring_engine.calculate_confidence_score(
            analysis_result, evidence
        )
        
        lower, upper = score.confidence_interval
        assert 0 <= lower <= score.overall_score <= upper <= 1
        assert upper - lower < 0.5  # Reasonable interval width
    
    def test_confidence_factors(
        self, scoring_engine, analysis_result
    ):
        """Test confidence factor identification"""
        # Test with low evidence
        low_evidence = [{'relevance': 0.3, 'keyword_count': 1}]
        score = scoring_engine.calculate_confidence_score(
            analysis_result, low_evidence
        )
        
        negative_factors = [f for f in score.factors if f['type'] == 'negative']
        assert len(negative_factors) > 0
        assert any('insufficient_evidence' in f.get('factor', '') 
                  for f in negative_factors)
        
        # Test with high evidence
        high_evidence = [
            {'relevance': 0.95, 'keyword_count': 10, 'pattern_match': True}
            for _ in range(5)
        ]
        score = scoring_engine.calculate_confidence_score(
            analysis_result, high_evidence
        )
        
        positive_factors = [f for f in score.factors if f['type'] == 'positive']
        assert len(positive_factors) > 0


class TestRiskAssessmentEngine:
    """Test suite for Risk Assessment Engine"""
    
    @pytest.fixture
    def risk_engine(self):
        """Create risk assessment engine"""
        return RiskAssessmentEngine()
    
    @pytest.fixture
    def document(self):
        """Sample document for risk assessment"""
        return {
            'id': 'test_doc',
            'content': 'Full document text...',
            'metadata': {'type': 'service_agreement'}
        }
    
    @pytest.fixture
    def clauses(self):
        """Sample clauses for risk assessment"""
        return [
            {
                'id': 'clause_1',
                'clause_type': 'arbitration',
                'text': 'Binding arbitration with class action waiver...',
                'confidence_score': 0.9
            },
            {
                'id': 'clause_2',
                'clause_type': 'limitation_liability',
                'text': 'No liability for consequential damages...',
                'confidence_score': 0.85
            },
            {
                'id': 'clause_3',
                'clause_type': 'termination',
                'text': 'May terminate at any time without cause...',
                'confidence_score': 0.8
            }
        ]
    
    @pytest.fixture
    def context(self):
        """Context for risk assessment"""
        return {
            'jurisdiction': 'California',
            'party_type': 'consumer',
            'industry': 'technology'
        }
    
    def test_document_risk_assessment(
        self, risk_engine, document, clauses, context
    ):
        """Test comprehensive document risk assessment"""
        risk_score = risk_engine.assess_document_risk(
            document, clauses, context
        )
        
        assert isinstance(risk_score, RiskScore)
        assert 0 <= risk_score.overall_risk <= 1
        assert risk_score.risk_category in [
            'Critical Risk - Immediate Action Required',
            'High Risk - Significant Concerns',
            'Medium Risk - Moderate Concerns',
            'Low Risk - Standard Precautions'
        ]
        assert len(risk_score.risk_factors) > 0
        assert len(risk_score.recommendations) > 0
    
    def test_clause_risk_profiling(self, risk_engine, clauses, context):
        """Test individual clause risk profiling"""
        for clause in clauses:
            risk_profile = risk_engine._assess_clause_risk(clause, context)
            
            assert isinstance(risk_profile, ClauseRiskProfile)
            assert risk_profile.risk_level in ['critical', 'high', 'medium', 'low']
            assert 0 <= risk_profile.risk_score <= 1
            assert isinstance(risk_profile.vulnerability_points, list)
            assert isinstance(risk_profile.fallback_positions, list)
    
    def test_jurisdiction_specific_risk(self, risk_engine, context):
        """Test jurisdiction-specific risk assessment"""
        # Non-compete in California
        non_compete_clause = {
            'clause_type': 'non_compete',
            'text': 'Employee shall not compete for 2 years...'
        }
        
        ca_context = {'jurisdiction': 'California'}
        risk_profile = risk_engine._assess_clause_risk(
            non_compete_clause, ca_context
        )
        
        assert risk_profile.risk_score < 0.5  # Should be low risk in CA
        assert any('unenforceable in California' in concern.lower() 
                  for concern in risk_profile.enforceability_concerns)
        
        # Same clause in Texas
        tx_context = {'jurisdiction': 'Texas'}
        risk_profile = risk_engine._assess_clause_risk(
            non_compete_clause, tx_context
        )
        
        assert risk_profile.risk_score > 0.5  # Should be higher risk in TX
    
    def test_mitigation_prioritization(
        self, risk_engine, document, clauses, context
    ):
        """Test risk mitigation prioritization"""
        risk_score = risk_engine.assess_document_risk(
            document, clauses, context
        )
        
        priorities = risk_score.mitigation_priority
        assert len(priorities) > 0
        
        # Check priority ordering
        for i, priority in enumerate(priorities):
            assert priority['priority'] == i + 1
            assert priority['urgency'] in ['immediate', 'short-term', 'long-term']
            assert 'action' in priority
    
    def test_financial_impact_assessment(
        self, risk_engine, document, clauses, context
    ):
        """Test financial impact assessment"""
        risk_score = risk_engine.assess_document_risk(
            document, clauses, context
        )
        
        financial_impact = risk_score.financial_impact
        assert 'maximum_liability' in financial_impact
        assert 'payment_obligations' in financial_impact
        assert isinstance(financial_impact['payment_obligations'], list)


class TestIntegration:
    """Integration tests for the complete RAG system"""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete RAG system"""
        return {
            'pipeline': AdvancedRAGPipeline(),
            'extractor': IntelligentClauseExtractor(),
            'confidence': ConfidenceScoringEngine(),
            'risk': RiskAssessmentEngine()
        }
    
    def test_end_to_end_document_analysis(self, complete_system):
        """Test complete end-to-end document analysis flow"""
        document = {
            'id': 'integration_test',
            'content': """
                SERVICE AGREEMENT
                
                This Service Agreement is between Client and Provider.
                
                1. SERVICES: Provider shall deliver consulting services.
                
                2. PAYMENT: Client shall pay $10,000 monthly.
                
                3. ARBITRATION: All disputes shall be resolved through
                   binding arbitration under JAMS rules. Class actions waived.
                
                4. LIABILITY: Provider's liability limited to fees paid.
                
                5. TERMINATION: Either party may terminate with 30 days notice.
            """,
            'metadata': {'type': 'service_agreement'}
        }
        
        # Extract clauses
        clauses = complete_system['extractor'].extract_all_clauses(
            document['content']
        )
        assert len(clauses) > 0
        
        # Perform risk assessment
        clause_dicts = [
            {'clause_type': c.type, 'text': c.text} for c in clauses
        ]
        risk_score = complete_system['risk'].assess_document_risk(
            document, clause_dicts, {}
        )
        assert risk_score.overall_risk > 0
        
        # Calculate confidence
        analysis_result = {'clauses': clause_dicts}
        evidence = [{'relevance': 0.8} for _ in clauses]
        confidence = complete_system['confidence'].calculate_confidence_score(
            analysis_result, evidence
        )
        assert confidence.overall_score > 0
        
        # Verify integration
        assert len(clauses) > 0
        assert risk_score.overall_risk > 0
        assert confidence.overall_score > 0
    
    @pytest.mark.asyncio
    async def test_multi_document_comparison(self, complete_system):
        """Test multi-document comparison and cross-referencing"""
        documents = [
            {
                'id': 'doc1',
                'content': 'Agreement with binding arbitration in New York.',
                'metadata': {}
            },
            {
                'id': 'doc2',
                'content': 'Agreement with litigation in California courts.',
                'metadata': {}
            }
        ]
        
        result = await complete_system['pipeline'].analyze_multiple_documents(
            documents, cross_reference=True
        )
        
        # Check for conflict detection
        cross_analysis = result.get('cross_document_analysis', {})
        conflicts = cross_analysis.get('conflicting_provisions', [])
        
        # Should detect jurisdiction/dispute resolution conflicts
        assert len(result['individual_analyses']) == 2
        assert 'consolidated_insights' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])