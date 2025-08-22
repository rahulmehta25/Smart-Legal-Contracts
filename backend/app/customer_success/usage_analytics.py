import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

class UsageAnalytics:
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize usage analytics with customer interaction data
        
        :param data: List of dictionaries containing usage events
        """
        self.df = pd.DataFrame(data)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    
    def feature_adoption_rate(self, feature_list: List[str]) -> Dict[str, float]:
        """
        Calculate adoption rates for specific features
        
        :param feature_list: List of features to analyze
        :return: Dictionary of feature adoption rates
        """
        adoption_rates = {}
        total_customers = len(self.df['customer_id'].unique())
        
        for feature in feature_list:
            feature_users = self.df[self.df['feature'] == feature]['customer_id'].nunique()
            adoption_rates[feature] = feature_users / total_customers * 100
        
        return adoption_rates
    
    def engagement_trend(self, days: int = 30) -> pd.DataFrame:
        """
        Analyze user engagement trends over specified period
        
        :param days: Number of days to analyze
        :return: DataFrame with daily engagement metrics
        """
        recent_data = self.df[self.df['timestamp'] > datetime.now() - timedelta(days=days)]
        
        daily_engagement = recent_data.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
            'customer_id': 'nunique',
            'feature': 'count'
        }).rename(columns={
            'customer_id': 'active_users',
            'feature': 'total_interactions'
        })
        
        return daily_engagement
    
    def time_to_first_value(self, onboarding_features: List[str]) -> float:
        """
        Calculate average time to first value for new customers
        
        :param onboarding_features: List of key onboarding features
        :return: Average time to first value in hours
        """
        customer_first_value = {}
        
        for customer in self.df['customer_id'].unique():
            customer_data = self.df[self.df['customer_id'] == customer]
            first_onboarding_event = customer_data[
                customer_data['feature'].isin(onboarding_features)
            ]['timestamp'].min()
            
            if first_onboarding_event:
                customer_first_value[customer] = first_onboarding_event
        
        first_value_times = [
            (time - datetime.now()).total_seconds() / 3600 
            for time in customer_first_value.values()
        ]
        
        return np.mean(first_value_times) if first_value_times else 0
    
    def cohort_analysis(self, cohort_feature: str) -> pd.DataFrame:
        """
        Perform cohort analysis based on a specific feature
        
        :param cohort_feature: Feature to use for cohort grouping
        :return: Cohort retention DataFrame
        """
        cohort_data = self.df.groupby([
            pd.Grouper(key='timestamp', freq='M'),
            cohort_feature
        ]).agg({
            'customer_id': 'nunique'
        }).reset_index()
        
        return cohort_data

# Example usage
if __name__ == "__main__":
    sample_data = [
        {'customer_id': 'cust1', 'feature': 'onboarding', 'timestamp': datetime.now() - timedelta(days=10)},
        {'customer_id': 'cust1', 'feature': 'dashboard', 'timestamp': datetime.now() - timedelta(days=5)},
        {'customer_id': 'cust2', 'feature': 'onboarding', 'timestamp': datetime.now() - timedelta(days=15)},
        {'customer_id': 'cust2', 'feature': 'reporting', 'timestamp': datetime.now() - timedelta(days=3)}
    ]
    
    analytics = UsageAnalytics(sample_data)
    
    print("Feature Adoption Rates:")
    print(analytics.feature_adoption_rate(['onboarding', 'dashboard', 'reporting']))
    
    print("\nEngagement Trend:")
    print(analytics.engagement_trend())
    
    print("\nTime to First Value:")
    print(analytics.time_to_first_value(['onboarding']))