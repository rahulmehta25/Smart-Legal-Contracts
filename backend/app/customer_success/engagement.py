import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class EngagementTracker:
    def __init__(self, customer_data: List[Dict[str, Any]]):
        """
        Initialize engagement tracker with customer interaction data
        
        :param customer_data: List of customer interaction dictionaries
        """
        self.df = pd.DataFrame(customer_data)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    
    def calculate_engagement_score(self, customer_id: str, days: int = 30) -> float:
        """
        Calculate comprehensive engagement score for a customer
        
        :param customer_id: Unique customer identifier
        :param days: Number of days to analyze
        :return: Engagement score (0-100)
        """
        recent_data = self.df[
            (self.df['customer_id'] == customer_id) & 
            (self.df['timestamp'] > datetime.now() - timedelta(days=days))
        ]
        
        # Components of engagement score
        feature_variety = len(recent_data['feature'].unique()) / 10 * 30
        interaction_frequency = recent_data.groupby('feature').size().mean() * 10
        recency_weight = self._calculate_recency_weight(recent_data)
        product_depth = self._calculate_product_depth(recent_data)
        
        engagement_score = (
            feature_variety + 
            interaction_frequency + 
            recency_weight + 
            product_depth
        )
        
        return min(max(engagement_score, 0), 100)
    
    def _calculate_recency_weight(self, customer_data: pd.DataFrame) -> float:
        """
        Calculate weight based on how recently the customer has been active
        
        :param customer_data: Customer's recent interaction data
        :return: Recency weight
        """
        if customer_data.empty:
            return 0
        
        days_since_last_interaction = (datetime.now() - customer_data['timestamp'].max()).days
        recency_score = max(20 - days_since_last_interaction, 0)
        return recency_score
    
    def _calculate_product_depth(self, customer_data: pd.DataFrame) -> float:
        """
        Measure customer's depth of product usage
        
        :param customer_data: Customer's recent interaction data
        :return: Product depth score
        """
        total_features = 10  # Total available features
        used_features = len(customer_data['feature'].unique())
        return (used_features / total_features) * 20
    
    def segment_customers_by_engagement(self) -> Dict[str, List[str]]:
        """
        Segment customers into engagement tiers
        
        :return: Dictionary of customer segments
        """
        customer_segments = {
            'champions': [],
            'at_risk': [],
            'hibernating': []
        }
        
        for customer_id in self.df['customer_id'].unique():
            engagement_score = self.calculate_engagement_score(customer_id)
            
            if engagement_score >= 80:
                customer_segments['champions'].append(customer_id)
            elif 40 <= engagement_score < 80:
                customer_segments['at_risk'].append(customer_id)
            else:
                customer_segments['hibernating'].append(customer_id)
        
        return customer_segments
    
    def engagement_progression(self, customer_id: str, interval: str = 'M') -> pd.Series:
        """
        Track customer's engagement progression over time
        
        :param customer_id: Unique customer identifier
        :param interval: Time interval for grouping (default: monthly)
        :return: Engagement scores over time
        """
        customer_data = self.df[self.df['customer_id'] == customer_id]
        
        engagement_progression = customer_data.groupby(
            pd.Grouper(key='timestamp', freq=interval)
        ).apply(
            lambda x: self.calculate_engagement_score(customer_id, days=len(x))
        )
        
        return engagement_progression
    
    def recommend_actions(self, customer_id: str) -> Dict[str, str]:
        """
        Generate personalized engagement recommendations
        
        :param customer_id: Unique customer identifier
        :return: Dictionary of recommended actions
        """
        engagement_score = self.calculate_engagement_score(customer_id)
        customer_data = self.df[self.df['customer_id'] == customer_id]
        
        unused_features = set(self.df['feature'].unique()) - set(customer_data['feature'].unique())
        
        recommendations = {
            'engagement_level': 'Low' if engagement_score < 40 else 'Medium' if engagement_score < 80 else 'High',
            'suggested_features': list(unused_features)[:3],
            'communication_frequency': 'Weekly' if engagement_score < 40 else 'Bi-weekly' if engagement_score < 80 else 'Monthly',
            'intervention_level': 'High' if engagement_score < 40 else 'Medium' if engagement_score < 80 else 'Low'
        }
        
        return recommendations

# Example usage
if __name__ == "__main__":
    sample_data = [
        {'customer_id': 'cust1', 'feature': 'dashboard', 'timestamp': datetime.now() - timedelta(days=5)},
        {'customer_id': 'cust1', 'feature': 'reporting', 'timestamp': datetime.now() - timedelta(days=10)},
        {'customer_id': 'cust2', 'feature': 'onboarding', 'timestamp': datetime.now() - timedelta(days=30)}
    ]
    
    tracker = EngagementTracker(sample_data)
    
    print("Engagement Score for cust1:", tracker.calculate_engagement_score('cust1'))
    print("Customer Segments:", tracker.segment_customers_by_engagement())
    print("Recommendations for cust1:", tracker.recommend_actions('cust1'))