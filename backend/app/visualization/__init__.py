"""
Advanced Data Visualization and Reporting System

This module provides enterprise-grade data visualization capabilities including:
- Interactive charts and graphs
- Real-time streaming visualizations  
- Dynamic dashboard creation
- Multi-format report generation
- Data warehouse integration
- ETL/ELT pipeline support

Key Features:
- 3D clause relationship graphs
- Geographical risk heatmaps
- Time series analysis
- Sankey diagrams for clause flow
- Network graphs for entity relationships
- Word clouds for key terms
- Treemaps for hierarchical data
- AR/VR data exploration capabilities
"""

from .charts import ChartEngine
from .dashboards import DashboardManager
from .reports import ReportBuilder
from .export import ExportManager
from .streaming import StreamingVisualizer

__all__ = [
    'ChartEngine',
    'DashboardManager', 
    'ReportBuilder',
    'ExportManager',
    'StreamingVisualizer'
]

__version__ = '1.0.0'