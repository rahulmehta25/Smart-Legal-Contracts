import numpy as np
import pandas as pd
from typing import List, Dict, Any
import requests

class MarketIntelligence:
    """
    Advanced market analysis and competitive intelligence
    """
    
    @staticmethod
    def analyze_competitor_clauses(competitor_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze competitor clause usage and trends
        
        :param competitor_data: List of competitor clause records
        :return: Competitive clause analysis
        """
        df = pd.DataFrame(competitor_data)
        
        clause_frequency = df['clause_type'].value_counts()
        clause_complexity = df.groupby('clause_type')['complexity_score'].mean()
        
        return {
            'clause_frequency': clause_frequency.to_dict(),
            'clause_complexity': clause_complexity.to_dict(),
            'market_concentration': {
                'top_3_clauses': clause_frequency.nlargest(3).to_dict(),
                'emerging_clauses': clause_frequency[clause_frequency < clause_frequency.quantile(0.25)].to_dict()
            }
        }
    
    @staticmethod
    def estimate_market_share(industry_data: List[Dict], company_data: Dict) -> Dict[str, Any]:
        """
        Estimate market share based on industry and company data
        
        :param industry_data: List of industry-wide records
        :param company_data: Company's specific performance data
        :return: Market share estimation
        """
        industry_df = pd.DataFrame(industry_data)
        total_market_volume = industry_df['total_value'].sum()
        company_market_volume = company_data.get('total_value', 0)
        
        return {
            'total_market_volume': total_market_volume,
            'company_market_volume': company_market_volume,
            'estimated_market_share': (company_market_volume / total_market_volume) * 100 if total_market_volume > 0 else 0,
            'market_growth_rate': industry_df['total_value'].pct_change().mean() * 100
        }
    
    @staticmethod
    def pricing_optimization(competitor_pricing: List[Dict], company_features: Dict) -> Dict[str, Any]:
        """
        Analyze pricing strategies and optimize pricing
        
        :param competitor_pricing: List of competitor pricing records
        :param company_features: Company's feature set and performance
        :return: Pricing recommendations
        """
        pricing_df = pd.DataFrame(competitor_pricing)
        
        # Calculate pricing metrics
        avg_price = pricing_df['price'].mean()
        price_std = pricing_df['price'].std()
        
        # Determine pricing position based on features
        feature_score = sum(company_features.get('feature_weights', {}).values())
        
        pricing_recommendation = {
            'market_avg_price': avg_price,
            'price_standard_deviation': price_std,
            'suggested_price_range': {
                'lower_bound': avg_price - price_std,
                'upper_bound': avg_price + price_std
            },
            'feature_score': feature_score,
            'recommended_strategy': 'premium' if feature_score > np.percentile(pricing_df['feature_score'], 75) else 'competitive'
        }
        
        return pricing_recommendation
    
    @staticmethod
    def track_industry_trends(data_sources: List[str]) -> Dict[str, Any]:
        """
        Track and aggregate industry trends from multiple sources
        
        :param data_sources: List of URLs or data source endpoints
        :return: Aggregated industry trend insights
        """
        trend_data = {}
        for source in data_sources:
            try:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    trend_data[source] = response.json()
            except Exception as e:
                trend_data[source] = {'error': str(e)}
        
        return {
            'trend_sources': list(trend_data.keys()),
            'aggregated_trends': trend_data
        }