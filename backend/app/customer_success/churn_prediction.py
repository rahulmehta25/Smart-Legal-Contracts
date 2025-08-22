import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, List

class ChurnPredictor:
    def __init__(self):
        """
        Initialize churn prediction model
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def prepare_features(self, customer_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare and transform customer data for churn prediction
        
        :param customer_data: List of customer dictionaries
        :return: Dictionary with prepared training and testing data
        """
        df = pd.DataFrame(customer_data)
        
        # Feature engineering
        df['total_usage'] = df['feature_count'] * df['usage_frequency']
        df['support_load'] = df['support_tickets'] / df['total_months']
        df['payment_reliability'] = df['successful_payments'] / df['total_payments']
        
        # Select relevant features
        features = [
            'total_usage', 'support_load', 'payment_reliability', 
            'account_age', 'nps_score', 'feature_adoption_rate'
        ]
        
        X = df[features]
        y = df['churned']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def train(self, customer_data: List[Dict[str, Any]]):
        """
        Train churn prediction model
        
        :param customer_data: List of customer dictionaries
        """
        prepared_data = self.prepare_features(customer_data)
        
        self.model.fit(
            prepared_data['X_train'], 
            prepared_data['y_train']
        )
        
        # Evaluate model
        y_pred = self.model.predict(prepared_data['X_test'])
        print("Model Performance:")
        print(classification_report(
            prepared_data['y_test'], 
            y_pred
        ))
    
    def predict_churn_probability(self, customer: Dict[str, Any]) -> float:
        """
        Predict churn probability for a single customer
        
        :param customer: Customer data dictionary
        :return: Churn probability (0-1)
        """
        # Prepare customer features
        customer_df = pd.DataFrame([customer])
        customer_df['total_usage'] = customer_df['feature_count'] * customer_df['usage_frequency']
        customer_df['support_load'] = customer_df['support_tickets'] / customer_df['total_months']
        customer_df['payment_reliability'] = customer_df['successful_payments'] / customer_df['total_payments']
        
        features = [
            'total_usage', 'support_load', 'payment_reliability', 
            'account_age', 'nps_score', 'feature_adoption_rate'
        ]
        
        customer_features = customer_df[features]
        customer_features_scaled = self.scaler.transform(customer_features)
        
        # Predict churn probability
        churn_proba = self.model.predict_proba(customer_features_scaled)[0][1]
        return churn_proba
    
    def risk_segments(self, customers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Segment customers by churn risk
        
        :param customers: List of customer dictionaries
        :return: Dictionary of risk-segmented customers
        """
        risk_groups = {
            'low_risk': [],
            'medium_risk': [],
            'high_risk': []
        }
        
        for customer in customers:
            churn_probability = self.predict_churn_probability(customer)
            
            if churn_probability < 0.3:
                risk_groups['low_risk'].append({**customer, 'churn_probability': churn_probability})
            elif churn_probability < 0.7:
                risk_groups['medium_risk'].append({**customer, 'churn_probability': churn_probability})
            else:
                risk_groups['high_risk'].append({**customer, 'churn_probability': churn_probability})
        
        return risk_groups

# Example usage
if __name__ == "__main__":
    sample_customers = [
        {
            'feature_count': 5,
            'usage_frequency': 0.7,
            'support_tickets': 2,
            'total_months': 12,
            'successful_payments': 11,
            'total_payments': 12,
            'account_age': 365,
            'nps_score': 8,
            'feature_adoption_rate': 0.6,
            'churned': 0
        },
        # Add more sample customers
    ]
    
    predictor = ChurnPredictor()
    predictor.train(sample_customers)
    
    # Example prediction
    test_customer = sample_customers[0]
    print("Churn Probability:", predictor.predict_churn_probability(test_customer))
    
    # Risk segmentation
    customers = [...]  # More customer data
    risk_segments = predictor.risk_segments(customers)
    print("Risk Segments:", risk_segments)