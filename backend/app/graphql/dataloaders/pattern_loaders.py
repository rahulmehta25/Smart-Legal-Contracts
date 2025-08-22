"""
DataLoaders for Pattern entities
"""

from typing import List, Dict, Any
from aiodataloader import DataLoader
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from ..types import Pattern as PatternType
from ...db.models import Pattern, Detection


class PatternDataLoader(DataLoader):
    """DataLoader for Pattern entities"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[int]) -> List[PatternType]:
        """Batch load patterns by IDs"""
        try:
            patterns = self.session.query(Pattern).filter(
                Pattern.id.in_(keys)
            ).all()
            
            pattern_map = {
                pattern.id: self._convert_to_graphql_type(pattern)
                for pattern in patterns
            }
            
            return [pattern_map.get(key) for key in keys]
            
        except Exception as e:
            return [None] * len(keys)
    
    def _convert_to_graphql_type(self, pattern: Pattern) -> PatternType:
        """Convert SQLAlchemy Pattern to GraphQL PatternType"""
        return PatternType(
            id=str(pattern.id),
            pattern_name=pattern.pattern_name,
            pattern_text=pattern.pattern_text,
            pattern_type=pattern.pattern_type,
            category=pattern.category,
            language=pattern.language,
            effectiveness_score=pattern.effectiveness_score,
            usage_count=pattern.usage_count,
            last_used=pattern.last_used,
            is_active=pattern.is_active,
            created_by=pattern.created_by,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at,
            # Computed fields
            is_effective=pattern.is_effective,
            average_confidence_score=self._calculate_avg_confidence(pattern)
        )
    
    def _calculate_avg_confidence(self, pattern: Pattern) -> float:
        """Calculate average confidence score for pattern detections"""
        try:
            # Get average confidence score from detections using this pattern
            avg_confidence = self.session.query(func.avg(Detection.confidence_score)).filter(
                Detection.pattern_id == pattern.id
            ).scalar()
            
            return float(avg_confidence) if avg_confidence else None
            
        except Exception:
            return None


class PatternsByTypeDataLoader(DataLoader):
    """DataLoader for Patterns by type"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, pattern_types: List[str]) -> List[List[PatternType]]:
        """Batch load patterns by types"""
        try:
            patterns = self.session.query(Pattern).filter(
                Pattern.pattern_type.in_(pattern_types),
                Pattern.is_active == True
            ).order_by(Pattern.effectiveness_score.desc()).all()
            
            # Group patterns by type
            patterns_by_type = {}
            for pattern in patterns:
                if pattern.pattern_type not in patterns_by_type:
                    patterns_by_type[pattern.pattern_type] = []
                patterns_by_type[pattern.pattern_type].append(
                    PatternDataLoader(self.session)._convert_to_graphql_type(pattern)
                )
            
            return [patterns_by_type.get(pattern_type, []) for pattern_type in pattern_types]
            
        except Exception as e:
            return [[] for _ in pattern_types]


class PatternsByCategoryDataLoader(DataLoader):
    """DataLoader for Patterns by category"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, categories: List[str]) -> List[List[PatternType]]:
        """Batch load patterns by categories"""
        try:
            patterns = self.session.query(Pattern).filter(
                Pattern.category.in_(categories),
                Pattern.is_active == True
            ).order_by(Pattern.effectiveness_score.desc()).all()
            
            # Group patterns by category
            patterns_by_category = {}
            for pattern in patterns:
                if pattern.category not in patterns_by_category:
                    patterns_by_category[pattern.category] = []
                patterns_by_category[pattern.category].append(
                    PatternDataLoader(self.session)._convert_to_graphql_type(pattern)
                )
            
            return [patterns_by_category.get(category, []) for category in categories]
            
        except Exception as e:
            return [[] for _ in categories]


class ActivePatternsDataLoader(DataLoader):
    """DataLoader for Active Patterns"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[str]) -> List[List[PatternType]]:
        """Batch load all active patterns"""
        try:
            patterns = self.session.query(Pattern).filter(
                Pattern.is_active == True
            ).order_by(Pattern.effectiveness_score.desc()).all()
            
            pattern_list = [
                PatternDataLoader(self.session)._convert_to_graphql_type(pattern)
                for pattern in patterns
            ]
            
            # Return same list for all keys (active patterns are global)
            return [pattern_list for _ in keys]
            
        except Exception as e:
            return [[] for _ in keys]


class MostUsedPatternsDataLoader(DataLoader):
    """DataLoader for Most Used Patterns"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, limits: List[int]) -> List[List[PatternType]]:
        """Batch load most used patterns with limit"""
        try:
            results = []
            
            for limit in limits:
                patterns = self.session.query(Pattern).filter(
                    Pattern.is_active == True
                ).order_by(Pattern.usage_count.desc()).limit(limit).all()
                
                pattern_list = [
                    PatternDataLoader(self.session)._convert_to_graphql_type(pattern)
                    for pattern in patterns
                ]
                
                results.append(pattern_list)
            
            return results
            
        except Exception as e:
            return [[] for _ in limits]


class PatternStatsDataLoader(DataLoader):
    """DataLoader for Pattern statistics"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Batch load pattern statistics"""
        try:
            total_patterns = self.session.query(Pattern).count()
            active_patterns = self.session.query(Pattern).filter(
                Pattern.is_active == True
            ).count()
            
            # Calculate average effectiveness
            avg_effectiveness = self.session.query(func.avg(Pattern.effectiveness_score)).filter(
                Pattern.is_active == True
            ).scalar()
            
            # Get most used patterns (top 10)
            most_used = self.session.query(Pattern).filter(
                Pattern.is_active == True
            ).order_by(Pattern.usage_count.desc()).limit(10).all()
            
            # Get patterns by type
            patterns_by_type = self.session.query(
                Pattern.pattern_type,
                func.count(Pattern.id).label('count'),
                func.avg(Pattern.effectiveness_score).label('avg_effectiveness')
            ).filter(Pattern.is_active == True).group_by(Pattern.pattern_type).all()
            
            # Get patterns by category
            patterns_by_category = self.session.query(
                Pattern.category,
                func.count(Pattern.id).label('count'),
                func.avg(Pattern.effectiveness_score).label('avg_effectiveness')
            ).filter(Pattern.is_active == True).group_by(Pattern.category).all()
            
            stats = {
                'total_patterns': total_patterns,
                'active_patterns': active_patterns,
                'average_effectiveness': float(avg_effectiveness) if avg_effectiveness else 0.0,
                'most_used_patterns': [
                    PatternDataLoader(self.session)._convert_to_graphql_type(pattern)
                    for pattern in most_used
                ],
                'patterns_by_type': [
                    {
                        'type': row.pattern_type,
                        'count': row.count,
                        'average_effectiveness': float(row.avg_effectiveness)
                    }
                    for row in patterns_by_type
                ],
                'patterns_by_category': [
                    {
                        'category': row.category,
                        'count': row.count,
                        'average_effectiveness': float(row.avg_effectiveness)
                    }
                    for row in patterns_by_category
                ]
            }
            
            return [stats for _ in keys]
            
        except Exception as e:
            return [{} for _ in keys]


class PatternEffectivenessDataLoader(DataLoader):
    """DataLoader for Pattern effectiveness tracking"""
    
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
    
    async def batch_load_fn(self, pattern_ids: List[int]) -> List[Dict[str, Any]]:
        """Batch load pattern effectiveness metrics"""
        try:
            results = []
            
            for pattern_id in pattern_ids:
                # Get detection count and average confidence for this pattern
                detection_stats = self.session.query(
                    func.count(Detection.id).label('detection_count'),
                    func.avg(Detection.confidence_score).label('avg_confidence'),
                    func.count(Detection.id).filter(Detection.is_validated == True).label('validated_count')
                ).filter(Detection.pattern_id == pattern_id).first()
                
                # Get pattern info
                pattern = self.session.query(Pattern).get(pattern_id)
                
                if pattern and detection_stats:
                    effectiveness_data = {
                        'pattern_id': pattern_id,
                        'usage_count': pattern.usage_count,
                        'detection_count': detection_stats.detection_count or 0,
                        'average_confidence': float(detection_stats.avg_confidence) if detection_stats.avg_confidence else 0.0,
                        'validated_count': detection_stats.validated_count or 0,
                        'validation_rate': (detection_stats.validated_count / detection_stats.detection_count) if detection_stats.detection_count > 0 else 0.0,
                        'effectiveness_score': pattern.effectiveness_score
                    }
                else:
                    effectiveness_data = {
                        'pattern_id': pattern_id,
                        'usage_count': 0,
                        'detection_count': 0,
                        'average_confidence': 0.0,
                        'validated_count': 0,
                        'validation_rate': 0.0,
                        'effectiveness_score': 0.0
                    }
                
                results.append(effectiveness_data)
            
            return results
            
        except Exception as e:
            return [{} for _ in pattern_ids]


def create_pattern_loaders(session: Session) -> Dict[str, DataLoader]:
    """Create all pattern-related DataLoaders"""
    return {
        'pattern': PatternDataLoader(session),
        'patterns_by_type': PatternsByTypeDataLoader(session),
        'patterns_by_category': PatternsByCategoryDataLoader(session),
        'active_patterns': ActivePatternsDataLoader(session),
        'most_used_patterns': MostUsedPatternsDataLoader(session),
        'pattern_stats': PatternStatsDataLoader(session),
        'pattern_effectiveness': PatternEffectivenessDataLoader(session),
    }