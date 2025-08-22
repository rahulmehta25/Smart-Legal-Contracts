import numpy as np
import pandas as pd
from typing import Dict, Any

class CustomerHealthScorer:
    def __init__(self):
        # Configurable weights for health score components
        self.weights = {
            'usage_frequency': 0.3,
            'feature_adoption': 0.2,
            'support_interactions': 0.15,
            'payment_history': 0.15,
            'nps_score': 0.2
        }
    
    def calculate_health_score(self, customer_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive customer health score
        
        :param customer_data: Dictionary containing customer metrics
        :return: Health score between 0-100
        """
        try:
            # Normalize and weight each component
            health_components = {
                'usage_frequency': self._normalize_usage(customer_data.get('usage_frequency', 0)),
                'feature_adoption': self._normalize_adoption(customer_data.get('feature_adoption', [])),
                'support_interactions': self._normalize_support(customer_data.get('support_tickets', [])),
                'payment_history': self._normalize_payments(customer_data.get('payment_history', [])),
                'nps_score': customer_data.get('nps_score', 0)
            }
            
            # Calculate weighted health score
            health_score = sum(
                health_components[key] * self.weights[key] 
                for key in self.weights
            )
            
            return max(0, min(100, health_score))
        
        except Exception as e:
            print(f"Error calculating health score: {e}")
            return 50  # Default neutral score
    
    def _normalize_usage(self, usage):
        """Normalize usage frequency"""
        return min(max(usage / 10, 0), 1) * 100
    
    def _normalize_adoption(self, features):
        """Normalize feature adoption"""
        return len(features) / 10 * 100
    
    def _normalize_support(self, tickets):
        """Normalize support interactions"""
        # Lower number of tickets is better
        return max(100 - (len(tickets) * 10), 0)
    
    def _normalize_payments(self, payment_history):
        """Normalize payment history"""
        # Assume payment_history is a list of boolean or numeric values
        return sum(payment_history) / len(payment_history) * 100 if payment_history else 0
    
    def get_health_category(self, health_score):
        """Categorize health score"""
        if health_score >= 80:
            return "Healthy"
        elif health_score >= 50:
            return "At Risk"
        else:
            return "Critical"

# Example usage
if __name__ == "__main__":
    scorer = CustomerHealthScorer()
    sample_customer = {
        'usage_frequency': 7,
        'feature_adoption': ['feature1', 'feature2', 'feature3'],
        'support_tickets': [],
        'payment_history': [1, 1, 1, 1],
        'nps_score': 8
    }
    
    health_score = scorer.calculate_health_score(sample_customer)
    print(f"Customer Health Score: {health_score}")
    print(f"Health Category: {scorer.get_health_category(health_score)}")