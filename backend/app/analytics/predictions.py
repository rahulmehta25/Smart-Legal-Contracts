import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any

class TrendPredictor:
    """
    Advanced predictive analytics for legal document processing
    """
    
    @staticmethod
    def forecast_document_volume(historical_data: List[Dict], forecast_periods: int = 30) -> Dict[str, Any]:
        """
        Forecast future document processing volume
        
        :param historical_data: List of historical document processing records
        :param forecast_periods: Number of periods to forecast
        :return: Forecast dictionary with predictions
        """
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['day_number'] = range(len(df))
        
        X = df[['day_number']]
        y = df['volume']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Linear Regression Forecast
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Random Forest Forecast for more complex trends
        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(X_train, y_train)
        
        # Generate future predictions
        future_days = np.arange(df['day_number'].max() + 1, df['day_number'].max() + forecast_periods + 1)
        lr_forecast = lr_model.predict(future_days.reshape(-1, 1))
        rf_forecast = rf_model.predict(future_days.reshape(-1, 1))
        
        return {
            'linear_regression_forecast': lr_forecast.tolist(),
            'random_forest_forecast': rf_forecast.tolist(),
            'model_accuracy': {
                'linear_regression_r2': lr_model.score(X_test, y_test),
                'random_forest_r2': rf_model.score(X_test, y_test)
            }
        }
    
    @staticmethod
    def predict_clause_adoption(historical_clauses: List[Dict]) -> Dict[str, Any]:
        """
        Predict future clause adoption trends
        
        :param historical_clauses: List of historical clause usage records
        :return: Clause adoption prediction dictionary
        """
        df = pd.DataFrame(historical_clauses)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        clause_trends = {}
        for clause_type in df['type'].unique():
            clause_data = df[df['type'] == clause_type]
            cumulative_adoption = clause_data.groupby(clause_data['timestamp'].dt.to_period('M')).size().cumsum()
            
            clause_trends[clause_type] = {
                'total_adoption': cumulative_adoption.iloc[-1],
                'monthly_growth_rate': cumulative_adoption.pct_change().mean()
            }
        
        return clause_trends
    
    @staticmethod
    def churn_prediction(user_activity: List[Dict], churn_threshold: int = 30) -> Dict[str, Any]:
        """
        Predict user churn based on activity patterns
        
        :param user_activity: List of user activity records
        :param churn_threshold: Days of inactivity to consider as churned
        :return: Churn prediction metrics
        """
        df = pd.DataFrame(user_activity)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        user_last_activity = df.groupby('user_id')['timestamp'].max()
        current_time = df['timestamp'].max()
        
        churned_users = user_last_activity[current_time - user_last_activity > pd.Timedelta(days=churn_threshold)]
        
        return {
            'total_users': len(user_last_activity),
            'churned_users': len(churned_users),
            'churn_rate': len(churned_users) / len(user_last_activity),
            'avg_days_to_churn': (current_time - user_last_activity).mean().days
        }