import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta

class DataAggregator:
    """
    Advanced data aggregation and grouping for business intelligence
    """
    
    @staticmethod
    def aggregate_by_time(data: List[Dict], time_column: str, aggregate_column: str, 
                           granularity: str = 'daily') -> pd.DataFrame:
        """
        Aggregate data by time periods
        
        :param data: List of dictionaries with data
        :param time_column: Column to use for time-based aggregation
        :param aggregate_column: Column to aggregate
        :param granularity: Time granularity ('daily', 'weekly', 'monthly')
        :return: Aggregated DataFrame
        """
        df = pd.DataFrame(data)
        df[time_column] = pd.to_datetime(df[time_column])
        
        if granularity == 'daily':
            df['period'] = df[time_column].dt.date
        elif granularity == 'weekly':
            df['period'] = df[time_column].dt.to_period('W')
        elif granularity == 'monthly':
            df['period'] = df[time_column].dt.to_period('M')
        
        return df.groupby('period')[aggregate_column].agg(['count', 'mean', 'sum'])
    
    @staticmethod
    def rolling_window_analysis(data: List[Dict], window_size: int = 30) -> Dict[str, Any]:
        """
        Perform rolling window analysis on time series data
        
        :param data: List of dictionaries with time series data
        :param window_size: Size of rolling window in days
        :return: Dictionary with rolling window metrics
        """
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        rolling_metrics = {
            'volume_trend': df.rolling(window=window_size).size().mean(),
            'volatility': df.rolling(window=window_size)['value'].std(),
            'moving_average': df.rolling(window=window_size)['value'].mean()
        }
        
        return rolling_metrics
    
    @staticmethod
    def segment_analysis(data: List[Dict], segment_column: str) -> Dict[str, Any]:
        """
        Perform segmentation analysis
        
        :param data: List of dictionaries
        :param segment_column: Column to use for segmentation
        :return: Segmentation metrics
        """
        df = pd.DataFrame(data)
        
        segment_metrics = {}
        for segment, segment_data in df.groupby(segment_column):
            segment_metrics[segment] = {
                'count': len(segment_data),
                'avg_value': segment_data['value'].mean(),
                'total_value': segment_data['value'].sum()
            }
        
        return segment_metrics