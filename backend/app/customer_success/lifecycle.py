import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class CustomerLifecycleManager:
    def __init__(self):
        """
        Initialize customer lifecycle tracking
        """
        self.customers = {}
        self.lifecycle_stages = {
            'acquisition': ['initial_contact', 'demo', 'trial'],
            'onboarding': ['setup', 'first_value', 'training'],
            'engagement': ['active_usage', 'feature_adoption', 'expansion'],
            'retention': ['stable', 'at_risk', 'renewal_consideration'],
            'upsell': ['upgrade_potential', 'cross_sell_opportunity'],
            'advocacy': ['referral', 'case_study', 'community_leader']
        }
    
    def add_customer(self, customer_id: str, details: Dict[str, Any]):
        """
        Add a new customer to lifecycle tracking
        
        :param customer_id: Unique customer identifier
        :param details: Customer initial details
        """
        details['created_at'] = details.get('created_at', datetime.now())
        details['current_stage'] = 'acquisition'
        details['stage_history'] = [{'stage': 'acquisition', 'entered_at': details['created_at']}]
        
        self.customers[customer_id] = details
    
    def update_customer_stage(self, customer_id: str, new_stage: str):
        """
        Update customer's lifecycle stage
        
        :param customer_id: Unique customer identifier
        :param new_stage: New stage for the customer
        """
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        current_customer = self.customers[customer_id]
        old_stage = current_customer.get('current_stage', 'unknown')
        
        # Validate stage transition
        if not self._is_valid_stage_transition(old_stage, new_stage):
            raise ValueError(f"Invalid stage transition from {old_stage} to {new_stage}")
        
        current_customer['current_stage'] = new_stage
        current_customer['stage_history'].append({
            'stage': new_stage,
            'entered_at': datetime.now()
        })
    
    def _is_valid_stage_transition(self, old_stage: str, new_stage: str) -> bool:
        """
        Validate stage transitions
        
        :param old_stage: Previous stage
        :param new_stage: New stage
        :return: Whether transition is valid
        """
        stage_graph = {
            'acquisition': ['onboarding'],
            'onboarding': ['engagement'],
            'engagement': ['retention', 'upsell'],
            'retention': ['upsell', 'advocacy'],
            'upsell': ['advocacy'],
            'advocacy': []
        }
        
        return new_stage in stage_graph.get(old_stage, [])
    
    def calculate_stage_duration(self, customer_id: str, stage: Optional[str] = None) -> Dict[str, timedelta]:
        """
        Calculate duration spent in each stage or a specific stage
        
        :param customer_id: Unique customer identifier
        :param stage: Optional specific stage to calculate
        :return: Dictionary of stage durations
        """
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.customers[customer_id]
        stage_history = customer['stage_history']
        
        stage_durations = {}
        
        for i in range(len(stage_history) - 1):
            current_entry = stage_history[i]
            next_entry = stage_history[i + 1]
            
            duration = next_entry['entered_at'] - current_entry['entered_at']
            
            if stage is None or current_entry['stage'] == stage:
                stage_durations[current_entry['stage']] = duration
        
        return stage_durations
    
    def predict_next_stage(self, customer_id: str) -> str:
        """
        Predict the most likely next stage for a customer
        
        :param customer_id: Unique customer identifier
        :return: Predicted next stage
        """
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.customers[customer_id]
        current_stage = customer['current_stage']
        
        # Simple prediction based on stage transition rules
        stage_prediction_map = {
            'acquisition': 'onboarding',
            'onboarding': 'engagement',
            'engagement': 'retention',
            'retention': 'upsell',
            'upsell': 'advocacy',
            'advocacy': 'advocacy'  # End stage
        }
        
        return stage_prediction_map[current_stage]
    
    def generate_stage_recommendations(self, customer_id: str) -> Dict[str, Any]:
        """
        Generate recommendations based on current lifecycle stage
        
        :param customer_id: Unique customer identifier
        :return: Stage-specific recommendations
        """
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.customers[customer_id]
        current_stage = customer['current_stage']
        
        recommendations = {
            'acquisition': {
                'actions': ['send_welcome_email', 'schedule_initial_demo'],
                'communication_frequency': 'weekly',
                'key_metric': 'conversion_rate'
            },
            'onboarding': {
                'actions': ['provide_setup_assistance', 'offer_training_sessions'],
                'communication_frequency': 'bi-weekly',
                'key_metric': 'time_to_first_value'
            },
            'engagement': {
                'actions': ['suggest_advanced_features', 'share_best_practices'],
                'communication_frequency': 'monthly',
                'key_metric': 'feature_adoption_rate'
            },
            'retention': {
                'actions': ['proactive_check-ins', 'renewal_preparation'],
                'communication_frequency': 'quarterly',
                'key_metric': 'customer_health_score'
            },
            'upsell': {
                'actions': ['propose_upgraded_plan', 'highlight_premium_features'],
                'communication_frequency': 'monthly',
                'key_metric': 'expansion_revenue'
            },
            'advocacy': {
                'actions': ['request_referral', 'invite_to_case_study'],
                'communication_frequency': 'quarterly',
                'key_metric': 'referral_rate'
            }
        }
        
        return recommendations.get(current_stage, {})

# Example usage
if __name__ == "__main__":
    lifecycle_manager = CustomerLifecycleManager()
    
    # Add a sample customer
    lifecycle_manager.add_customer('cust1', {
        'name': 'Acme Corp',
        'industry': 'Technology',
        'created_at': datetime.now() - timedelta(days=30)
    })
    
    # Simulate stage progression
    customer_id = 'cust1'
    
    print("Initial Stage:", lifecycle_manager.customers[customer_id]['current_stage'])
    
    # Progress through stages
    stages_to_progress = ['onboarding', 'engagement', 'retention']
    for stage in stages_to_progress:
        lifecycle_manager.update_customer_stage(customer_id, stage)
        print(f"Progressed to {stage}")
    
    # Get recommendations
    recommendations = lifecycle_manager.generate_stage_recommendations(customer_id)
    print("\nRecommendations:", recommendations)
    
    # Predict next stage
    print("\nPredicted Next Stage:", lifecycle_manager.predict_next_stage(customer_id))