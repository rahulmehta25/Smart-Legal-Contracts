"""Comparison engine for analyzing arbitration clauses."""
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
import numpy as np
import logging
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

class ClauseComparisonEngine:
    """Engine for comparing and analyzing arbitration clauses."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize comparison engine with database."""
        from ..database.schema import DatabaseManager, VectorStore
        from ..models.legal_bert_detector import LegalBERTDetector
        
        self.db_manager = DatabaseManager(db_url)
        self.bert_detector = LegalBERTDetector()
        self.vector_store = VectorStore()
        
        logger.info("Comparison engine initialized")
    
    def add_clause_to_database(self, clause: Dict) -> str:
        """
        Add new clause to comparison database.
        
        Args:
            clause: Dictionary with clause information
            
        Returns:
            Clause ID
        """
        try:
            # Generate embedding
            clause_text = clause.get('text', clause.get('clause_text', ''))
            if not clause_text:
                raise ValueError("No clause text provided")
            
            embedding = self.bert_detector._get_embedding(clause_text)
            embedding_np = embedding.cpu().numpy()
            
            # Generate vector ID
            import uuid
            vector_id = str(uuid.uuid4())
            
            # Prepare database entry
            db_clause_data = {
                'company_name': clause.get('company', 'Unknown'),
                'industry': clause.get('industry', 'Unknown'),
                'document_type': clause.get('doc_type', clause.get('document_type', 'TOS')),
                'clause_text': clause_text,
                'clause_summary': clause.get('summary', self._generate_summary(clause_text)),
                'key_provisions': clause.get('provisions', clause.get('key_provisions', [])),
                'enforceability_score': clause.get('enforceability', 0.5),
                'risk_score': clause.get('risk', clause.get('risk_score', 0.5)),
                'jurisdiction': clause.get('jurisdiction', 'US'),
                'vector_id': vector_id,
                'metadata': clause.get('metadata', {}),
                'date_effective': clause.get('date_effective')
            }
            
            # Add to database
            clause_id = self.db_manager.add_clause(db_clause_data)
            
            # Add to vector store
            self.vector_store.add_clause(vector_id, embedding_np)
            
            # Save vector store
            self.vector_store.save()
            
            logger.info(f"Added clause {clause_id} with vector {vector_id}")
            return str(clause_id)
            
        except Exception as e:
            logger.error(f"Error adding clause to database: {e}")
            raise
    
    def compare_clause(self, input_clause: str, top_k: int = 10) -> Dict:
        """
        Compare input clause with database.
        
        Args:
            input_clause: Text of clause to compare
            top_k: Number of similar clauses to return
            
        Returns:
            Comparison results with similar clauses and analysis
        """
        try:
            # Generate embedding for input clause
            embedding = self.bert_detector._get_embedding(input_clause)
            embedding_np = embedding.cpu().numpy()
            
            # Find similar clauses
            similar_clauses = self.vector_store.search_similar(embedding_np, top_k)
            
            # Fetch clause details from database
            session = self.db_manager.get_session()
            results = []
            
            for vector_id, similarity_score in similar_clauses:
                from ..database.schema import ArbitrationClauseDB
                db_clause = session.query(ArbitrationClauseDB).filter_by(
                    vector_id=vector_id
                ).first()
                
                if db_clause:
                    results.append({
                        'id': db_clause.id,
                        'company': db_clause.company_name,
                        'industry': db_clause.industry,
                        'document_type': db_clause.document_type,
                        'similarity': similarity_score,
                        'summary': db_clause.clause_summary,
                        'provisions': db_clause.key_provisions or [],
                        'enforceability': db_clause.enforceability_score,
                        'risk_score': db_clause.risk_score,
                        'jurisdiction': db_clause.jurisdiction,
                        'date_added': db_clause.date_added.isoformat() if db_clause.date_added else None
                    })
            
            session.close()
            
            # Analyze differences and similarities
            analysis = self._analyze_comparison(input_clause, results)
            
            # Calculate statistics
            statistics = self._calculate_statistics(results)
            
            return {
                'similar_clauses': results[:5],  # Top 5 most similar
                'all_results': results,
                'analysis': analysis,
                'statistics': statistics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing clause: {e}")
            return {
                'similar_clauses': [],
                'all_results': [],
                'analysis': {'error': str(e)},
                'statistics': {},
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_comparison(self, input_clause: str, similar_clauses: List[Dict]) -> Dict:
        """Analyze comparison results."""
        analysis = {
            'unique_aspects': [],
            'common_provisions': [],
            'risk_assessment': '',
            'enforceability_assessment': '',
            'recommendations': [],
            'industry_insights': [],
            'jurisdiction_analysis': {}
        }
        
        if not similar_clauses:
            analysis['risk_assessment'] = 'Unable to assess - no similar clauses found'
            analysis['recommendations'].append('Consider adding this clause to the database for future comparisons')
            return analysis
        
        # Calculate average scores
        avg_enforceability = np.mean([c['enforceability'] for c in similar_clauses if c['enforceability'] is not None])
        avg_risk = np.mean([c['risk_score'] for c in similar_clauses if c['risk_score'] is not None])
        
        # Analyze common provisions
        all_provisions = []
        for clause in similar_clauses:
            all_provisions.extend(clause.get('provisions', []))
        
        provision_counts = Counter(all_provisions)
        analysis['common_provisions'] = [
            {'provision': p, 'frequency': count / len(similar_clauses)}
            for p, count in provision_counts.most_common(10)
            if count >= len(similar_clauses) * 0.3
        ]
        
        # Risk assessment
        if avg_risk > 0.7:
            analysis['risk_assessment'] = 'HIGH RISK - Similar to aggressive arbitration clauses'
            analysis['recommendations'].append('Consider negotiating for more favorable terms')
        elif avg_risk > 0.4:
            analysis['risk_assessment'] = 'MODERATE RISK - Standard arbitration terms'
            analysis['recommendations'].append('Review specific provisions carefully')
        else:
            analysis['risk_assessment'] = 'LOW RISK - Relatively favorable terms'
        
        # Enforceability assessment
        if avg_enforceability > 0.8:
            analysis['enforceability_assessment'] = 'HIGHLY ENFORCEABLE - Courts likely to uphold'
        elif avg_enforceability > 0.5:
            analysis['enforceability_assessment'] = 'MODERATELY ENFORCEABLE - Some provisions may be challenged'
        else:
            analysis['enforceability_assessment'] = 'QUESTIONABLE ENFORCEABILITY - May not hold up in court'
            analysis['recommendations'].append('Similar clauses have been successfully challenged')
        
        # Check for unique aspects
        input_lower = input_clause.lower()
        unique_checks = [
            ('class action waiver', 'Contains class action waiver'),
            ('opt-out', 'Includes opt-out provision'),
            ('opt out', 'Includes opt-out provision'),
            ('confidential', 'Contains confidentiality requirements'),
            ('jury trial', 'Waives jury trial rights'),
            ('30 days', 'Has 30-day opt-out period'),
            ('60 days', 'Has 60-day opt-out period'),
        ]
        
        for check_text, description in unique_checks:
            if check_text in input_lower:
                # Check if rare among similar clauses
                provision_found = False
                for clause in similar_clauses[:3]:
                    if any(check_text in str(p).lower() for p in clause.get('provisions', [])):
                        provision_found = True
                        break
                
                if not provision_found:
                    analysis['unique_aspects'].append(description)
        
        # Industry insights
        industries = [c['industry'] for c in similar_clauses if c['industry']]
        if industries:
            industry_counts = Counter(industries)
            top_industries = industry_counts.most_common(3)
            for industry, count in top_industries:
                analysis['industry_insights'].append({
                    'industry': industry,
                    'frequency': count / len(similar_clauses),
                    'message': f"Common in {industry} industry"
                })
        
        # Jurisdiction analysis
        jurisdictions = [c['jurisdiction'] for c in similar_clauses if c['jurisdiction']]
        if jurisdictions:
            jurisdiction_counts = Counter(jurisdictions)
            analysis['jurisdiction_analysis'] = {
                'most_common': jurisdiction_counts.most_common(1)[0][0] if jurisdiction_counts else 'Unknown',
                'distribution': dict(jurisdiction_counts)
            }
        
        # Specific recommendations
        if 'opt-out' not in input_lower and 'opt out' not in input_lower and avg_risk > 0.6:
            analysis['recommendations'].append('Consider negotiating for an opt-out provision')
        
        if 'class action waiver' in input_lower:
            analysis['recommendations'].append('Class action waivers may not be enforceable in all jurisdictions')
        
        if avg_enforceability < 0.5:
            analysis['recommendations'].append('Similar clauses have questionable enforceability - consider legal review')
        
        return analysis
    
    def _calculate_statistics(self, similar_clauses: List[Dict]) -> Dict:
        """Calculate statistics from comparison."""
        if not similar_clauses:
            return {
                'total_similar_clauses': 0,
                'message': 'No similar clauses found in database'
            }
        
        # Filter out None values
        enforceability_scores = [c['enforceability'] for c in similar_clauses if c['enforceability'] is not None]
        risk_scores = [c['risk_score'] for c in similar_clauses if c['risk_score'] is not None]
        
        statistics = {
            'total_similar_clauses': len(similar_clauses),
            'average_similarity': np.mean([c['similarity'] for c in similar_clauses]),
            'max_similarity': max([c['similarity'] for c in similar_clauses]),
            'min_similarity': min([c['similarity'] for c in similar_clauses]),
        }
        
        if enforceability_scores:
            statistics['average_enforceability'] = np.mean(enforceability_scores)
            statistics['enforceability_std'] = np.std(enforceability_scores)
        
        if risk_scores:
            statistics['average_risk'] = np.mean(risk_scores)
            statistics['risk_std'] = np.std(risk_scores)
        
        # Industry distribution
        industries = [c['industry'] for c in similar_clauses if c['industry'] and c['industry'] != 'Unknown']
        if industries:
            industry_dist = Counter(industries)
            statistics['industry_distribution'] = dict(industry_dist.most_common(5))
        
        # Document type distribution
        doc_types = [c['document_type'] for c in similar_clauses if c['document_type']]
        if doc_types:
            doc_type_dist = Counter(doc_types)
            statistics['document_type_distribution'] = dict(doc_type_dist)
        
        # Date range
        dates = [c['date_added'] for c in similar_clauses if c.get('date_added')]
        if dates:
            statistics['oldest_clause'] = min(dates)
            statistics['newest_clause'] = max(dates)
        
        return statistics
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the clause."""
        # Simple extractive summary
        summary = text[:200].strip()
        summary = ' '.join(summary.split())  # Normalize whitespace
        
        if len(text) > 200:
            # Find end of sentence
            for punct in ['.', '!', '?', ';']:
                idx = summary.rfind(punct)
                if idx > 100:
                    summary = summary[:idx+1]
                    break
            else:
                summary += "..."
        
        return summary
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the comparison database."""
        session = self.db_manager.get_session()
        try:
            from ..database.schema import ArbitrationClauseDB
            from sqlalchemy import func
            
            total_clauses = session.query(func.count(ArbitrationClauseDB.id)).scalar()
            
            # Get unique counts
            unique_companies = session.query(func.count(func.distinct(ArbitrationClauseDB.company_name))).scalar()
            unique_industries = session.query(func.count(func.distinct(ArbitrationClauseDB.industry))).scalar()
            unique_jurisdictions = session.query(func.count(func.distinct(ArbitrationClauseDB.jurisdiction))).scalar()
            
            # Get averages
            avg_risk = session.query(func.avg(ArbitrationClauseDB.risk_score)).scalar()
            avg_enforceability = session.query(func.avg(ArbitrationClauseDB.enforceability_score)).scalar()
            
            # Get vector store stats
            vector_stats = self.vector_store.get_stats()
            
            return {
                'database': {
                    'total_clauses': total_clauses or 0,
                    'unique_companies': unique_companies or 0,
                    'unique_industries': unique_industries or 0,
                    'unique_jurisdictions': unique_jurisdictions or 0,
                    'average_risk_score': float(avg_risk) if avg_risk else 0.0,
                    'average_enforceability': float(avg_enforceability) if avg_enforceability else 0.0
                },
                'vector_store': vector_stats,
                'status': 'operational'
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
        finally:
            session.close()
    
    def bulk_import_clauses(self, clauses: List[Dict]) -> Dict:
        """
        Bulk import clauses to the database.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Import statistics
        """
        success_count = 0
        error_count = 0
        errors = []
        
        for i, clause in enumerate(clauses):
            try:
                self.add_clause_to_database(clause)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append({
                    'index': i,
                    'error': str(e),
                    'clause': clause.get('company', 'Unknown')
                })
                logger.error(f"Error importing clause {i}: {e}")
        
        # Save vector store after bulk import
        self.vector_store.save()
        
        return {
            'total': len(clauses),
            'success': success_count,
            'errors': error_count,
            'error_details': errors[:10]  # Return first 10 errors
        }