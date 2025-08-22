import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BusinessMetrics:
    """
    Comprehensive business metrics calculation for legal document processing platform
    """
    
    @staticmethod
    def calculate_document_processing_volume(processed_docs: List[Dict]) -> Dict[str, Any]:
        """
        Calculate document processing volume metrics
        
        :param processed_docs: List of processed document records
        :return: Dictionary of volume metrics
        """
        total_docs = len(processed_docs)
        metrics = {
            'total_documents': total_docs,
            'daily_avg_documents': total_docs / 30,  # Assuming 30-day period
            'hourly_avg_documents': total_docs / (30 * 24),
            'processing_rate': total_docs / len(set(doc['timestamp'].date() for doc in processed_docs))
        }
        return metrics
    
    @staticmethod
    def calculate_clause_metrics(clauses: List[Dict]) -> Dict[str, Any]:
        """
        Analyze clause-related business metrics
        
        :param clauses: List of clause records
        :return: Dictionary of clause metrics
        """
        clause_types = {}
        for clause in clauses:
            clause_type = clause.get('type', 'undefined')
            clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
        
        return {
            'total_clauses': len(clauses),
            'clause_type_distribution': clause_types,
            'unique_clause_types': len(set(clause.get('type') for clause in clauses))
        }
    
    @staticmethod
    def calculate_user_engagement(user_activities: List[Dict]) -> Dict[str, Any]:
        """
        Calculate user engagement metrics
        
        :param user_activities: List of user activity records
        :return: Dictionary of engagement metrics
        """
        active_users = len(set(activity['user_id'] for activity in user_activities))
        total_interactions = len(user_activities)
        
        return {
            'total_active_users': active_users,
            'total_interactions': total_interactions,
            'interactions_per_user': total_interactions / active_users if active_users > 0 else 0,
            'daily_active_users': len(set(activity['user_id'] for activity in user_activities if activity['timestamp'].date() == pd.Timestamp.now().date()))
        }
    
    @staticmethod
    def calculate_api_usage(api_logs: List[Dict]) -> Dict[str, Any]:
        """
        Analyze API usage metrics
        
        :param api_logs: List of API call logs
        :return: Dictionary of API usage metrics
        """
        endpoints = {}
        for log in api_logs:
            endpoint = log.get('endpoint', 'undefined')
            endpoints[endpoint] = endpoints.get(endpoint, 0) + 1
        
        return {
            'total_api_calls': len(api_logs),
            'endpoint_distribution': endpoints,
            'unique_endpoints': len(set(log.get('endpoint') for log in api_logs))
        }