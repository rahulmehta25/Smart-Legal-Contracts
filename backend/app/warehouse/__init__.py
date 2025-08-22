"""
Data Warehouse Management System

Provides comprehensive data warehouse capabilities including:
- Snowflake/BigQuery integration
- Dimensional modeling (star/snowflake schemas)
- Slowly changing dimensions
- Aggregation tables
- Partitioning strategies
- Query optimization
- Data mart creation
- OLAP cube management
"""

from .dimensional_modeling import DimensionalModeler
from .warehouse_manager import WarehouseManager
from .query_optimizer import WarehouseQueryOptimizer
from .aggregation_manager import AggregationManager
from .partition_manager import PartitionManager

__all__ = [
    'DimensionalModeler',
    'WarehouseManager',
    'WarehouseQueryOptimizer',
    'AggregationManager',
    'PartitionManager'
]

__version__ = '1.0.0'