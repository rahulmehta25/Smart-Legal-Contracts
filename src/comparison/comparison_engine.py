from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database.schema import ArbitrationClauseDB, VectorStore, Base
from models.legal_bert_detector import LegalBERTDetector
import numpy as np

class ClauseComparisonEngine:
    def __init__(self, db_url: str = "sqlite:///arbitration_clauses.db"):
        """Initialize comparison engine with database"""
        self.engine = create_engine(db_url)
        self.bert_detector = LegalBERTDetector()
        self.vector_store = VectorStore()
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Load existing vectors
        try:
            self.vector_store.load("data/clause_vectors")
        except:
            print("No existing vector store found. Starting fresh.")
    
    def add_clause_to_database(self, clause: Dict) -> str:
        """Add new clause to comparison database"""
        with Session(self.engine) as session:
            # Generate embedding
            embedding = self.bert_detector._get_embedding(clause['text'])
            embedding_np = embedding.cpu().numpy()
            
            # Add to database
            db_clause = ArbitrationClauseDB(
                company_name=clause.get('company', 'Unknown'),
                industry=clause.get('industry', 'Unknown'),
                document_type=clause.get('doc_type', 'TOS'),
                clause_text=clause['text'],
                clause_summary=clause.get('summary', ''),
                key_provisions=clause.get('provisions', []),
                enforceability_score=clause.get('enforceability', 0.5),
                risk_score=clause.get('risk', 0.5),
                jurisdiction=clause.get('jurisdiction', 'US'),
                metadata=clause.get('metadata', {})
            )
            
            session.add(db_clause)
            session.commit()
            
            # Add to vector store
            clause_id = str(db_clause.id)
            self.vector_store.add_clause(clause_id, embedding_np)
            
            # Update database with vector reference
            db_clause.vector_id = clause_id
            session.commit()
            
            return clause_id
    
    def compare_clause(self, input_clause: str, top_k: int = 10) -> Dict:
        """
        Compare input clause with database
        
        Returns:
            Comparison results with similar clauses and analysis
        """
        # Generate embedding for input clause
        embedding = self.bert_detector._get_embedding(input_clause)
        embedding_np = embedding.cpu().numpy()
        
        # Find similar clauses
        similar_clauses = self.vector_store.search_similar(embedding_np, top_k)
        
        # Fetch clause details from database
        with Session(self.engine) as session:
            results = []
            for clause_id, similarity_score in similar_clauses:
                db_clause = session.query(ArbitrationClauseDB).filter_by(
                    vector_id=clause_id
                ).first()
                
                if db_clause:
                    results.append({
                        'company': db_clause.company_name,
                        'industry': db_clause.industry,
                        'document_type': db_clause.document_type,
                        'similarity': similarity_score,
                        'summary': db_clause.clause_summary,
                        'provisions': db_clause.key_provisions,
                        'enforceability': db_clause.enforceability_score,
                        'risk_score': db_clause.risk_score
                    })
        
        # Analyze differences and similarities
        analysis = self._analyze_comparison(input_clause, results)
        
        return {
            'similar_clauses': results[:5],  # Top 5 most similar
            'analysis': analysis,
            'statistics': self._calculate_statistics(results)
        }
    
    def _analyze_comparison(self, input_clause: str, similar_clauses: List[Dict]) -> Dict:
        """Analyze comparison results"""
        analysis = {
            'unique_aspects': [],
            'common_provisions': [],
            'risk_assessment': '',
            'recommendations': []
        }
        
        if not similar_clauses:
            analysis['risk_assessment'] = 'Unable to assess - no similar clauses found'
            return analysis
        
        # Calculate average scores
        avg_enforceability = np.mean([c['enforceability'] for c in similar_clauses])
        avg_risk = np.mean([c['risk_score'] for c in similar_clauses])
        
        # Common provisions
        all_provisions = []
        for clause in similar_clauses:
            all_provisions.extend(clause.get('provisions', []))
        
        from collections import Counter
        provision_counts = Counter(all_provisions)
        analysis['common_provisions'] = [
            p for p, count in provision_counts.most_common(5) 
            if count >= len(similar_clauses) * 0.3
        ]
        
        # Risk assessment
        if avg_risk > 0.7:
            analysis['risk_assessment'] = 'High risk - similar to aggressive arbitration clauses'
        elif avg_risk > 0.4:
            analysis['risk_assessment'] = 'Moderate risk - standard arbitration terms'
        else:
            analysis['risk_assessment'] = 'Low risk - relatively favorable terms'
        
        # Recommendations
        if avg_enforceability < 0.5:
            analysis['recommendations'].append(
                'Similar clauses have low enforceability - consider challenging'
            )
        
        input_lower = input_clause.lower()
        if 'class action waiver' in input_lower and \
           'Class action waiver' not in analysis['common_provisions']:
            analysis['unique_aspects'].append('Contains uncommon class action waiver')
        
        if 'opt-out' not in input_lower and avg_risk > 0.6:
            analysis['recommendations'].append(
                'Consider negotiating for opt-out provision'
            )
        
        return analysis
    
    def _calculate_statistics(self, similar_clauses: List[Dict]) -> Dict:
        """Calculate statistics from comparison"""
        if not similar_clauses:
            return {}
        
        industries = [c['industry'] for c in similar_clauses]
        from collections import Counter
        industry_dist = Counter(industries)
        
        return {
            'average_enforceability': np.mean([c['enforceability'] for c in similar_clauses]),
            'average_risk': np.mean([c['risk_score'] for c in similar_clauses]),
            'industry_distribution': dict(industry_dist),
            'total_similar_clauses': len(similar_clauses)
        }
