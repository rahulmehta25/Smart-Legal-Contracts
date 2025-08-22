import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

class ClauseEvolutionTracker:
    def __init__(self, data):
        """
        Initialize Clause Evolution Tracker
        
        Args:
            data (pd.DataFrame): DataFrame with clause information
        """
        self.data = data
        self.clause_encoder = LabelEncoder()
        
    def track_changes(self, time_column='timestamp', clause_column='clause_type'):
        """
        Analyze clause changes over time
        
        Returns:
            dict: Clause evolution metrics
        """
        # Convert timestamp to datetime
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        
        # Encode clause types
        self.data['clause_encoded'] = self.clause_encoder.fit_transform(self.data[clause_column])
        
        # Temporal clustering of clause changes
        time_grouped = self.data.groupby([pd.Grouper(key=time_column, freq='M'), clause_column]).size().unstack()
        
        # Trend analysis
        trends = time_grouped.apply(lambda x: stats.linregress(x.index.astype(int), x.values)[0], axis=0)
        
        return {
            'clause_trends': trends.to_dict(),
            'frequency_matrix': time_grouped.to_dict(),
            'unique_clauses': self.clause_encoder.classes_
        }
    
    def detect_emerging_patterns(self, window_size=3):
        """
        Detect emerging clause patterns
        
        Args:
            window_size (int): Number of periods to consider for emergence
        
        Returns:
            list: Emerging clause types
        """
        # Group by time and calculate rolling frequencies
        freq_matrix = self.data.groupby([pd.Grouper(key='timestamp', freq='M'), 'clause_type']).size().unstack()
        
        # Calculate percentage change
        pct_change = freq_matrix.pct_change(periods=window_size)
        
        # Identify emerging clauses (significant increase)
        emerging_clauses = pct_change[pct_change.mean() > 0.5].columns.tolist()
        
        return emerging_clauses
    
    def complexity_score(self, text_column='clause_text'):
        """
        Calculate clause complexity metrics
        
        Returns:
            pd.Series: Complexity scores for each clause
        """
        def calculate_complexity(text):
            return {
                'word_count': len(text.split()),
                'unique_words': len(set(text.split())),
                'average_word_length': np.mean([len(word) for word in text.split()])
            }
        
        complexity_metrics = self.data[text_column].apply(calculate_complexity)
        
        return pd.DataFrame(complexity_metrics.tolist())

# Example usage
if __name__ == '__main__':
    # Sample data generation for demonstration
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2022-01-01', periods=100),
        'clause_type': np.random.choice(['Arbitration', 'Mediation', 'Litigation'], 100),
        'clause_text': ['Sample clause text'] * 100
    })
    
    tracker = ClauseEvolutionTracker(sample_data)
    evolution_results = tracker.track_changes()
    emerging_patterns = tracker.detect_emerging_patterns()
    complexity_scores = tracker.complexity_score()
    
    print("Clause Evolution Results:", evolution_results)
    print("Emerging Clause Patterns:", emerging_patterns)
    print("Clause Complexity Scores:\n", complexity_scores)