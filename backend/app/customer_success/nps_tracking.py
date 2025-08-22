import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class NPSTracker:
    def __init__(self):
        """
        Initialize NPS tracking system
        """
        self.surveys = []
    
    def record_survey(self, survey_data: Dict[str, Any]):
        """
        Record individual NPS survey response
        
        :param survey_data: Dictionary containing survey details
        """
        required_fields = ['customer_id', 'score', 'timestamp', 'comments']
        
        # Validate survey data
        for field in required_fields:
            if field not in survey_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Normalize timestamp
        survey_data['timestamp'] = pd.to_datetime(survey_data.get('timestamp', datetime.now()))
        
        self.surveys.append(survey_data)
    
    def calculate_nps(self, period: int = 30) -> float:
        """
        Calculate Net Promoter Score for a given period
        
        :param period: Number of days to analyze
        :return: NPS score
        """
        recent_surveys = [
            survey for survey in self.surveys 
            if survey['timestamp'] > datetime.now() - timedelta(days=period)
        ]
        
        if not recent_surveys:
            return 0
        
        # NPS categorization
        def categorize_nps(score):
            if score <= 6:
                return 'detractor'
            elif score <= 8:
                return 'passive'
            else:
                return 'promoter'
        
        # Add NPS category to each survey
        for survey in recent_surveys:
            survey['nps_category'] = categorize_nps(survey['score'])
        
        # Calculate NPS
        df = pd.DataFrame(recent_surveys)
        promoters = len(df[df['nps_category'] == 'promoter'])
        detractors = len(df[df['nps_category'] == 'detractor'])
        total_responses = len(df)
        
        nps = ((promoters - detractors) / total_responses) * 100
        return nps
    
    def sentiment_analysis(self, period: int = 30) -> Dict[str, Any]:
        """
        Perform sentiment analysis on survey comments
        
        :param period: Number of days to analyze
        :return: Sentiment analysis results
        """
        recent_surveys = [
            survey for survey in self.surveys 
            if survey['timestamp'] > datetime.now() - timedelta(days=period)
        ]
        
        # Simple keyword-based sentiment analysis
        sentiment_keywords = {
            'positive': ['great', 'awesome', 'love', 'amazing', 'fantastic'],
            'negative': ['bad', 'terrible', 'worst', 'disappointing', 'frustrating']
        }
        
        sentiment_results = {
            'total_comments': 0,
            'positive_comments': 0,
            'negative_comments': 0,
            'neutral_comments': 0,
            'top_positive_words': {},
            'top_negative_words': {}
        }
        
        for survey in recent_surveys:
            if 'comments' in survey and survey['comments']:
                sentiment_results['total_comments'] += 1
                comment_lower = survey['comments'].lower()
                
                # Count sentiment keywords
                positive_matches = sum(1 for word in sentiment_keywords['positive'] if word in comment_lower)
                negative_matches = sum(1 for word in sentiment_keywords['negative'] if word in comment_lower)
                
                if positive_matches > negative_matches:
                    sentiment_results['positive_comments'] += 1
                elif negative_matches > positive_matches:
                    sentiment_results['negative_comments'] += 1
                else:
                    sentiment_results['neutral_comments'] += 1
        
        return sentiment_results
    
    def customer_feedback_trends(self, period: int = 90) -> pd.DataFrame:
        """
        Analyze customer feedback trends over time
        
        :param period: Number of days to analyze
        :return: DataFrame with trend data
        """
        recent_surveys = [
            survey for survey in self.surveys 
            if survey['timestamp'] > datetime.now() - timedelta(days=period)
        ]
        
        df = pd.DataFrame(recent_surveys)
        df['nps_category'] = df['score'].apply(lambda x: 'Promoter' if x > 8 else 'Passive' if x > 6 else 'Detractor')
        
        # Group by week and calculate NPS
        weekly_nps = df.groupby(pd.Grouper(key='timestamp', freq='W')).apply(
            lambda x: ((len(x[x['nps_category'] == 'Promoter']) - len(x[x['nps_category'] == 'Detractor'])) / len(x)) * 100
        ).reset_index()
        
        weekly_nps.columns = ['Week', 'NPS']
        return weekly_nps

# Example usage
if __name__ == "__main__":
    nps_tracker = NPSTracker()
    
    # Record sample surveys
    surveys = [
        {
            'customer_id': 'cust1', 
            'score': 9, 
            'timestamp': datetime.now() - timedelta(days=10),
            'comments': 'Great product, love the features!'
        },
        {
            'customer_id': 'cust2', 
            'score': 6, 
            'timestamp': datetime.now() - timedelta(days=5),
            'comments': 'Some features are disappointing'
        }
    ]
    
    for survey in surveys:
        nps_tracker.record_survey(survey)
    
    print("NPS Score:", nps_tracker.calculate_nps())
    print("Sentiment Analysis:", nps_tracker.sentiment_analysis())
    print("Feedback Trends:\n", nps_tracker.customer_feedback_trends())